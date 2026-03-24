# -*- coding: utf-8 -*-
"""
Walk-forward 回测：多股票面板数据的滚动训练/测试 + 组合回测。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

from config import INITIAL_CAPITAL, MAX_WEIGHT_PER_STOCK, REBALANCE_DAYS
from models import train_predictor, predict, train_ensemble, predict_ensemble
from portfolio import run_portfolio_backtest
from features.build import FEATURE_NAMES


def _segment_dates(dates: pd.DatetimeIndex, train_days: int, test_days: int):
    dates = pd.Index(sorted(dates.unique()))
    i = 0
    n = len(dates)
    while i + train_days + test_days <= n:
        train_range = dates[i : i + train_days]
        test_range = dates[i + train_days : i + train_days + test_days]
        yield train_range, test_range
        i += test_days


def walk_forward_panel(
    df: pd.DataFrame,
    model_type: str = "ridge",
    top_n: int = 10,
    train_days: int = 252 * 3,
    test_days: int = 252 // 2,
    market_exposure_series: Optional[pd.Series] = None,
    rebalance_days: Optional[int] = None,
    use_predicted_signal: bool = False,
) -> Dict[str, Any]:
    """
    对多股票面板数据进行 walk-forward：
      - df: 含 date, ticker, close, ret, 特征列, target
      - 每个窗口：用过去 train_days 训练模型，在后面 test_days 上预测并做组合回测
    返回整体的资金曲线与指标。
    """
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    all_dates = pd.to_datetime(df["date"])
    has_sprtrn = "sprtrn" in df.columns

    global_eq: pd.Series | None = None
    benchmark_compound = 1.0  # (1+r1)*(1+r2)*...
    all_test_dates = []  # 收集所有测试日，用于算同池等权基准
    for idx, (train_range, test_range) in enumerate(_segment_dates(all_dates, train_days, test_days)):
        all_test_dates.extend(test_range.tolist())
        train_mask = df["date"].isin(train_range)
        test_mask = df["date"].isin(test_range)
        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()
        if train_df.empty or test_df.empty:
            continue

        # 训练/预测分数：
        # - 默认：用历史特征训练模型，预测 target（未来收益）
        # - 开启 use_predicted_signal：用 predicted_RET 校准到 target，输出校准后的分数用于排序
        y_train = train_df["target"]
        if use_predicted_signal:
            if "predicted_RET" not in train_df.columns or "predicted_RET" not in test_df.columns:
                raise ValueError("use_predicted_signal 但数据中缺少 predicted_RET 列")
            _pred = pd.to_numeric(train_df["predicted_RET"], errors="coerce")
            _y = pd.to_numeric(y_train, errors="coerce")
            valid_dir = _pred.notna() & _y.notna() & np.isfinite(_pred.values) & np.isfinite(_y.values)
            invert_dir = False
            if valid_dir.sum() > 20:
                corr_dir = float(np.corrcoef(_pred.loc[valid_dir].values, _y.loc[valid_dir].values)[0, 1])
                invert_dir = corr_dir < 0
            feature_col = "predicted_RET"
            if invert_dir:
                train_df = train_df.copy()
                test_df = test_df.copy()
                train_df["predicted_RET"] = -pd.to_numeric(train_df["predicted_RET"], errors="coerce")
                test_df["predicted_RET"] = -pd.to_numeric(test_df["predicted_RET"], errors="coerce")

            X_train = train_df[[feature_col]].replace([np.inf, -np.inf], np.nan).fillna(0)
            valid = y_train.notna()
            X_train, y_train = X_train.loc[valid], y_train.loc[valid]
            if len(X_train) == 0:
                continue
            X_test = test_df[[feature_col]].replace([np.inf, -np.inf], np.nan).fillna(0)
            test_df = test_df.copy()
            if model_type == "ensemble":
                fitted = train_ensemble(X_train, y_train, use_xgb=False)
                test_df["prediction"] = predict_ensemble(fitted, X_test)
            else:
                fitted = train_predictor(X_train, y_train, model_type=model_type)
                test_df["prediction"] = predict(fitted, X_test)
        else:
            X_train = train_df[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0)
            valid = y_train.notna()
            X_train, y_train = X_train.loc[valid], y_train.loc[valid]
            if len(X_train) == 0:
                continue
            X_test = (
                test_df[FEATURE_NAMES]
                .reindex(columns=FEATURE_NAMES)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            test_df = test_df.copy()
            if model_type == "ensemble":
                fitted = train_ensemble(X_train, y_train, use_xgb=False)
                test_df["prediction"] = predict_ensemble(fitted, X_test)
            else:
                fitted = train_predictor(X_train, y_train, model_type=model_type)
                test_df["prediction"] = predict(fitted, X_test)
        # 综合打分：prediction*0.7 + 低波动*0.3
        if "volatility_20" in test_df.columns:
            vol_rank = test_df.groupby("date")["volatility_20"].rank(pct=True, ascending=True).fillna(0.5)
            test_df["score"] = 0.7 * test_df["prediction"] + 0.3 * (1.0 - vol_rank)
        else:
            test_df["score"] = test_df["prediction"]
        rank_col = "score"
        panel_result = run_portfolio_backtest(
            test_df,
            id_col="ticker",
            top_n=top_n,
            bottom_n=0,
            initial_cash=INITIAL_CAPITAL if global_eq is None else float(global_eq.iloc[-1]),
            rank_col=rank_col,
            max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
            rebalance_days=rebalance_days if rebalance_days is not None else REBALANCE_DAYS,
            market_exposure=market_exposure_series,
        )
        eq_seg: pd.Series = panel_result["equity_curve"]
        if eq_seg is None or len(eq_seg) == 0:
            continue
        if global_eq is None:
            global_eq = eq_seg
        else:
            # 将片段曲线与上一段衔接（按起点归一后接上）
            start_val = eq_seg.iloc[0]
            if start_val <= 0:
                continue
            scale = float(global_eq.iloc[-1]) / float(start_val)
            eq_scaled = eq_seg * scale
            global_eq = pd.concat([global_eq, eq_scaled.iloc[1:]], ignore_index=True)

        # 同期大盘收益（该测试段内 sprtrn 复利）
        if has_sprtrn:
            sub = df.loc[df["date"].isin(test_range), ["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
            if not sub.empty:
                seg_bench = float((1 + sub["sprtrn"].fillna(0)).prod() - 1)
                benchmark_compound *= 1 + seg_bench

    if global_eq is None or len(global_eq) == 0:
        return {
            "equity_curve": pd.Series([INITIAL_CAPITAL]),
            "total_return": 0.0,
            "final_value": INITIAL_CAPITAL,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "benchmark_return": None,
            "benchmark_universe_return": None,
        }

    # 同池等权基准：所有测试日上 496 只每日等权复利
    benchmark_universe = None
    if all_test_dates and "ret" in df.columns:
        sub = df.loc[df["date"].isin(all_test_dates), ["date", "ret"]].copy()
        sub["ret"] = pd.to_numeric(sub["ret"], errors="coerce").fillna(0)
        daily = sub.groupby("date")["ret"].mean()
        if len(daily) > 0:
            benchmark_universe = float((1 + daily).prod() - 1)

    final_value = float(global_eq.iloc[-1])
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_ret = global_eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    cummax = global_eq.cummax()
    max_dd = ((global_eq - cummax) / cummax.replace(0, np.nan)).min()
    benchmark_return = float(benchmark_compound - 1) if has_sprtrn and benchmark_compound != 1.0 else None
    return {
        "equity_curve": global_eq,
        "total_return": float(total_return),
        "final_value": final_value,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "benchmark_return": benchmark_return,
        "benchmark_universe_return": benchmark_universe,
    }

