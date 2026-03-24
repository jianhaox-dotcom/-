# -*- coding: utf-8 -*-
"""
量化研究主流程（无未来函数）：
  数据加载 → 特征工程 → 预测模型(t 时刻预测 t+1 收益) → 交易策略 → 回测 → 风险指标 → 可视化
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TEST_RATIO,
    MAX_WEIGHT_PER_STOCK,
    REBALANCE_DAYS,
    TARGET_FORWARD_DAYS,
    INDEX_SHORT_HEDGE_RATIO,
    INDEX_SHORT_MA_DAYS,
    INDEX_SHORT_SUM_DAYS,
)
from data import load_dataset, train_test_split
from features import build_features
from features.build import FEATURE_NAMES
from models import train_predictor, predict, train_ensemble, predict_ensemble
from portfolio import run_portfolio_backtest
from evaluation import walk_forward_panel
from viz import plot_backtest_report
from analysis import report_ic_and_groups


def benchmark_return_over_dates(df: pd.DataFrame, dates) -> float | None:
    """计算同期大盘（sprtrn）累计收益。df 需含 date、sprtrn。"""
    if "sprtrn" not in df.columns:
        return None
    sub = df.loc[df["date"].isin(dates), ["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    if sub.empty:
        return None
    return float((1 + sub["sprtrn"].fillna(0)).prod() - 1)


def market_exposure_series(df: pd.DataFrame, ma_days: int = 20) -> pd.Series | None:
    """按大盘(sprtrn)20日均线择时：指数在均线上方满仓(1.0)，下方半仓(0.5)。返回 date -> 暴露度。"""
    if "sprtrn" not in df.columns:
        return None
    sub = df[["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    if sub.empty:
        return None
    sub = sub.set_index("date")
    sub["ret"] = pd.to_numeric(sub["sprtrn"], errors="coerce").fillna(0)
    index_level = (1 + sub["ret"]).cumprod()
    ma = index_level.rolling(ma_days, min_periods=1).mean()
    exposure = np.where(index_level >= ma, 1.0, 0.5)
    return pd.Series(exposure, index=sub.index)


def apply_conditional_index_short(
    long_equity: pd.Series,
    test_dates: np.ndarray | list,
    df: pd.DataFrame,
    hedge_ratio: float = INDEX_SHORT_HEDGE_RATIO,
    ma_days: int = INDEX_SHORT_MA_DAYS,
    sum_days: int = INDEX_SHORT_SUM_DAYS,
) -> tuple[pd.Series, dict]:
    """
    条件指数空头（不压满多头）：
    仅在「昨日指数 < 昨日 MA」且「昨日为止近 sum_days 日 sprtrn 累计 < 0」时，
    当日用 hedge_ratio × 昨日多头市值 的名义做空指数（日收益 ≈ -sprtrn × 名义）。
    返回 (总权益曲线, 统计信息)。
    """
    if "sprtrn" not in df.columns or len(long_equity) != len(test_dates):
        return long_equity.copy(), {"short_days": 0, "hedge_pnl": 0.0}
    daily = df[["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    daily = daily.set_index("date")
    daily["r"] = pd.to_numeric(daily["sprtrn"], errors="coerce").fillna(0)
    idx_lvl = (1 + daily["r"]).cumprod()
    ma = idx_lvl.rolling(ma_days, min_periods=1).mean()
    sum_n = daily["r"].rolling(sum_days, min_periods=1).sum()

    dates_all = daily.index.tolist()
    pos = {d: i for i, d in enumerate(dates_all)}

    L = long_equity.astype(float).values
    H = 0.0
    total = []
    short_days = 0
    for i, d in enumerate(test_dates):
        d = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        if i == 0:
            total.append(float(L[0]))
            continue
        prev = test_dates[i - 1]
        prev = pd.Timestamp(prev) if not isinstance(prev, pd.Timestamp) else prev
        if prev not in pos:
            total.append(float(L[i] + H))
            continue
        j = pos[prev]
        idx_p = float(idx_lvl.iloc[j])
        ma_p = float(ma.iloc[j])
        s5 = float(sum_n.iloc[j])
        short_on = (idx_p < ma_p) and (s5 < 0)
        r_t = float(daily["r"].get(d, 0)) if d in daily.index else 0.0
        if short_on:
            notional = hedge_ratio * float(L[i - 1])
            H += notional * (-r_t)
            short_days += 1
        total.append(float(L[i] + H))

    stats = {"short_days": short_days, "hedge_pnl": float(H), "total_days": len(test_dates)}
    return pd.Series(total, dtype=float), stats


def universe_equal_weight_return(df: pd.DataFrame, dates) -> float | None:
    """计算同期「股票池等权」累计收益：每日所有股票收益的均值再复利。与策略同池、可比。"""
    sub = df.loc[df["date"].isin(dates), ["date", "ret"]].copy()
    if sub.empty or "ret" not in sub.columns:
        return None
    sub["ret"] = pd.to_numeric(sub["ret"], errors="coerce").fillna(0)
    daily = sub.groupby("date")["ret"].mean()
    if len(daily) == 0:
        return None
    return float((1 + daily).prod() - 1)


def main():
    parser = argparse.ArgumentParser(description="量化回测流程：多股票特征→模型→组合→回测→风险→图表")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="A.csv",
        help="数据路径：单个 CSV 或包含多只股票 CSV 的目录",
    )
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="简单切分测试集比例（非 walk-forward 时使用）")
    parser.add_argument(
        "--test-last-trading-days",
        type=int,
        default=None,
        help="在按 test-ratio 划出的测试段内，只保留最后 N 个交易日再回测与算基准（约三个月用 63）",
    )
    parser.add_argument("--model", default="rf", choices=["ridge", "rf", "xgb", "ensemble"], help="预测模型（ensemble=Ridge+RF+可选XGB 取平均）")
    parser.add_argument("--walk-forward", action="store_true", help="使用 walk-forward 回测")
    parser.add_argument("--top-n", type=int, default=20, help="组合做多股票数量（单票最大权重 5% 时建议≥20）")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    parser.add_argument("--out-dir", default=".", help="图表与报告输出目录")
    parser.add_argument("--timing", action="store_true", help="大盘择时：20日均线下方半仓、上方满仓")
    parser.add_argument("--rebalance-days", type=int, default=None, help="覆盖默认再平衡周期（如目标为5天建议改为5）")
    parser.add_argument("--target-forward-days", type=int, default=TARGET_FORWARD_DAYS, help="预测目标的未来天数（1=下一日，5=未来5日累计收益）")
    parser.add_argument("--use-predicted-signal", action="store_true", help="用数据集里的 predicted_RET 作为信号：训练一个校准模型，再用于选股排序")
    parser.add_argument(
        "--index-short",
        action="store_true",
        help="条件指数空头：仅弱市+近5日大盘跌时，用部分多头市值做空指数（不全对冲）",
    )
    parser.add_argument(
        "--predicted-direct-score",
        action="store_true",
        help="predicted_RET 用作排序分数（仅用反转方向，不训练校准模型）",
    )
    parser.add_argument(
        "--short-ratio",
        type=float,
        default=0.0,
        help="做空名义比例：short 名义=多头名义*short-ratio（>0 才启用做空，适用于 predicted-direct-score）",
    )
    parser.add_argument(
        "--use-predicted-as-feature",
        action="store_true",
        help="把 predicted_RET 当作额外输入特征（仍使用真实 target 训练评估）",
    )
    parser.add_argument(
        "--rank-long-always",
        action="store_true",
        help="与 --predicted-direct-score 联用：按预测排序选股；若当日无正分，则对 top_n 等权做多，避免空仓在上涨市大幅跑输 sprtrn",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"未找到数据文件或目录: {data_path}")
        return

    # ---------- 1. 数据加载：单文件或多股票目录（递归找所有 CSV） ----------
    if data_path.is_dir():
        # 你的数据是多层嵌套目录，最后才是 csv，这里直接递归搜全部 *.csv
        csv_files = sorted([p for p in data_path.rglob("*.csv") if p.is_file() and not p.name.startswith(".")])
        dfs = []
        for csv in csv_files:
            try:
                df_i = load_dataset(csv, close_col="PRC", prediction_col="RET")
                dfs.append(df_i)
            except Exception as e:
                print(f"跳过 {csv.name}: {e}")
        if not dfs:
            print("目录及子目录中没有可用的 CSV 数据。")
            return
        df = pd.concat(dfs, ignore_index=True)
        print(f"  已加载 {len(dfs)} 个 CSV，共 {len(df)} 条记录")
    else:
        close_col = "PRC" if data_path.name.endswith(".csv") else None
        pred_col = "RET" if data_path.name.endswith(".csv") else None
        df = load_dataset(data_path, close_col=close_col, prediction_col=pred_col)
    print("1. 数据加载完成（支持多股票面板，ret 仅作目标/评估，不作交易信号）")

    # 使用 ret 重建 close（保证跨 split 的 close 连续）
    if "ticker" in df.columns and "ret" in df.columns and "date" in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
        df["close"] = df.groupby("ticker")["ret"].apply(lambda s: (1.0 + s).cumprod()).reset_index(level=0, drop=True)

    # ---------- 2. 特征工程（仅用历史信息，按 ticker 分组滚动） ----------
    df = build_features(df, target_forward_days=args.target_forward_days)
    print(f"2. 特征工程完成 | 总样本 {len(df)} | 特征数: {len(FEATURE_NAMES)} | target_forward_days={args.target_forward_days}")

    # 择时：大盘 20 日均线向上满仓、向下半仓（仅当 --timing 且存在 sprtrn）
    market_exposure = market_exposure_series(df) if args.timing else None
    if args.timing and market_exposure is not None:
        print("  已开启大盘择时（20 日均线下方半仓、上方满仓）")

    rebal_days = args.rebalance_days if args.rebalance_days is not None else REBALANCE_DAYS

    # Walk-forward 回测
    if args.walk_forward:
        print("3. 使用 walk-forward 回测（多窗口滚动训练/测试 + 组合回测）")
        wf_result = walk_forward_panel(
            df,
            model_type=args.model,
            top_n=args.top_n,
            market_exposure_series=market_exposure,
            rebalance_days=rebal_days,
            use_predicted_signal=args.use_predicted_signal,
        )
        eq = wf_result["equity_curve"]
        print("\n" + "=" * 72)
        print("Walk-forward 组合回测结果")
        print("=" * 72)
        print(f"  总收益: {wf_result['total_return']:.2%}")
        print(f"  最终资产: {wf_result['final_value']:,.0f}")
        print(f"  夏普: {wf_result['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {wf_result['max_drawdown']:.2%}")
        if wf_result.get("benchmark_return") is not None:
            b = wf_result["benchmark_return"]
            excess = wf_result["total_return"] - b
            print(f"  同期大盘(sprtrn)收益: {b:.2%}")
            print(f"  超额收益(vs大盘): {excess:.2%}")
            print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")
        if wf_result.get("benchmark_universe_return") is not None:
            bu = wf_result["benchmark_universe_return"]
            ex_u = wf_result["total_return"] - bu
            print(f"  同期股票池等权收益: {bu:.2%}")
            print(f"  超额收益(vs等权池): {ex_u:.2%}")
            print("  → 跑赢同池等权" if ex_u > 0 else "  → 未跑赢同池等权")

        if not args.no_plot and eq is not None and len(eq) > 0:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_backtest_report(
                eq,
                save_path=out_dir / "backtest_report.png",
                title="Walk-forward Portfolio Backtest",
            )
            print(f"\n图表已保存: {out_dir / 'backtest_report.png'}")
        return

    # ---------- 3. 简单 train/test 切分 + 组合回测 ----------
    # 按日期划分训练/测试（对所有股票统一切分）
    dates = pd.to_datetime(df["date"]).sort_values().unique()
    cut_idx = int(len(dates) * (1 - args.test_ratio))
    train_dates = dates[:cut_idx]
    test_dates = dates[cut_idx:]
    if args.test_last_trading_days is not None and int(args.test_last_trading_days) > 0:
        n_keep = int(args.test_last_trading_days)
        td_sorted = np.sort(pd.to_datetime(test_dates))
        if len(td_sorted) > n_keep:
            test_dates = td_sorted[-n_keep:]
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()
        print(
            f"3. 划分训练/测试 | 训练 {len(train_df)} | 测试 {len(test_df)} "
            f"（测试段已截断为最后 {len(np.unique(test_dates))} 个交易日）"
        )
    else:
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"].isin(test_dates)].copy()
        print(f"3. 划分训练/测试 | 训练 {len(train_df)} | 测试 {len(test_df)}")

    # ---------- 4. 用于选股排序的“预测分数” ----------
    # - 默认：用历史特征训练模型，预测 target（未来收益）
    # - 开启 --use-predicted-signal：用 predicted_RET 做校准训练，再输出校准后的 prediction 用于排序
    if args.use_predicted_signal:
        if "predicted_RET" not in train_df.columns or "predicted_RET" not in test_df.columns:
            raise ValueError("use-predicted-signal 但数据中缺少 predicted_RET 列")
        # 自动判定方向：若 predicted_RET 与目标的相关为负，则反转特征（避免“高分买入却亏钱”）
        _pred = pd.to_numeric(train_df["predicted_RET"], errors="coerce")
        _y = pd.to_numeric(train_df["target"], errors="coerce")
        valid_dir = _pred.notna() & _y.notna() & np.isfinite(_pred.values) & np.isfinite(_y.values)
        invert_dir = False
        if valid_dir.sum() > 20:
            corr_dir = float(np.corrcoef(_pred.loc[valid_dir].values, _y.loc[valid_dir].values)[0, 1])
            invert_dir = corr_dir < 0
            print(f"  predicted_RET vs target 相关系数: {corr_dir:.6f} | invert_dir={invert_dir}")
        feature_col = "predicted_RET"
        if invert_dir:
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["predicted_RET"] = -pd.to_numeric(train_df["predicted_RET"], errors="coerce")
            test_df["predicted_RET"] = -pd.to_numeric(test_df["predicted_RET"], errors="coerce")

        if args.predicted_direct_score:
            test_df = test_df.copy()
            test_df["prediction"] = pd.to_numeric(test_df["predicted_RET"], errors="coerce")
            print("4. predicted-direct-score：直接使用（已可能反转的）predicted_RET 作为排序分数")
        else:
            X_train = train_df[[feature_col]].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
            y_train = train_df["target"]
            valid = y_train.notna()
            X_train, y_train = X_train.loc[valid], y_train.loc[valid]
            X_test = test_df[[feature_col]].reindex(columns=[feature_col]).replace(
                [float("inf"), float("-inf")], float("nan")
            ).fillna(0)
            if args.model == "ensemble":
                fitted = train_ensemble(X_train, y_train, use_xgb=False)
                test_df["prediction"] = predict_ensemble(fitted, X_test)
            else:
                fitted = train_predictor(X_train, y_train, model_type=args.model)
                test_df["prediction"] = predict(fitted, X_test)
            print(f"4. 用 predicted_RET 校准后的分数完成（目标为未来 {args.target_forward_days} 日收益）")
    else:
        feature_cols = FEATURE_NAMES.copy()
        if (
            args.use_predicted_as_feature
            and "predicted_RET" in train_df.columns
            and "predicted_RET" in test_df.columns
            and "predicted_RET" not in feature_cols
        ):
            feature_cols.append("predicted_RET")
        X_train = train_df[feature_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
        y_train = train_df["target"]
        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]
        X_test = test_df[feature_cols].reindex(columns=feature_cols).replace(
            [float("inf"), float("-inf")], float("nan")
        ).fillna(0)

        if args.model == "ensemble":
            fitted = train_ensemble(X_train, y_train, use_xgb=False)
            test_df["prediction"] = predict_ensemble(fitted, X_test)
        else:
            fitted = train_predictor(X_train, y_train, model_type=args.model)
            test_df["prediction"] = predict(fitted, X_test)
        print(f"4. 预测模型完成（目标为未来 {args.target_forward_days} 日收益，无未来信息）")

    # 若为 XGBoost/模型可返回重要性，输出前若干重要因子
    if "fitted" in locals():
        fi = fitted.get("feature_importances")
        if fi is not None:
            imp = pd.Series(fi, index=fitted["feature_names"]).sort_values(ascending=False)
            top_imp = imp.head(10)
            print("\n前 10 个因子重要性：")
            for name, val in top_imp.items():
                print(f"  {name:20s}: {val:.4f}")

    # ---------- 5. 综合打分选股：prediction*0.7 + 低波动*0.3，单票最大 5%，20 日再平衡 ----------
    if args.predicted_direct_score:
        pred_score = pd.to_numeric(test_df["prediction"], errors="coerce")
        if float(args.short_ratio) > 0 or args.rank_long_always:
            # 做空或「无正分仍按排名做多」需要保留分数符号用于排序/选空
            test_df["score"] = pred_score
        else:
            # 纯多头：负分不建仓
            test_df["score"] = pred_score.clip(lower=0.0)
    elif "volatility_20" in test_df.columns:
        # 日内波动率升序排名：低波动 rank 小，low_vol_score=1-rank 则低波动得高分
        vol_rank = test_df.groupby("date")["volatility_20"].rank(pct=True, ascending=True).fillna(0.5)
        low_vol_score = 1.0 - vol_rank
        test_df["score"] = 0.7 * test_df["prediction"] + 0.3 * low_vol_score
    else:
        test_df["score"] = test_df["prediction"]
    rank_col = "score"

    if args.rank_long_always and not args.predicted_direct_score:
        print("  警告: --rank-long-always 仅在与 --predicted-direct-score 联用时生效，已忽略。")
    if args.rank_long_always and args.predicted_direct_score:
        print("  已开启 rank-long-always：无正分时对预测排序 top_n 等权做多")

    panel_result = run_portfolio_backtest(
        test_df,
        id_col="ticker",
        top_n=args.top_n,
        bottom_n=args.top_n if float(args.short_ratio) > 0 else 0,
        initial_cash=INITIAL_CAPITAL,
        rank_col=rank_col,
        max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
        rebalance_days=rebal_days,
        market_exposure=market_exposure,
        short_notional_ratio=float(args.short_ratio),
        equal_weight_long_if_no_positive=bool(args.rank_long_always and args.predicted_direct_score),
    )
    eq = panel_result["equity_curve"]
    test_dates_sorted = np.sort(test_df["date"].unique())
    eq_for_plot = eq
    is_long_short = float(args.short_ratio) > 0
    print("\n" + "=" * 72)
    print("测试集组合回测结果（" + ("多头+空头" if is_long_short else "纯多头") + "）")
    print("=" * 72)
    print(f"  总收益: {panel_result['total_return']:.2%}")
    print(f"  最终资产: {panel_result['final_value']:,.0f}")
    print(f"  夏普: {panel_result['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {panel_result['max_drawdown']:.2%}")

    if args.index_short and len(eq) == len(test_dates_sorted):
        eq_total, st = apply_conditional_index_short(eq, test_dates_sorted, df)
        eq_for_plot = eq_total
        ret_total = (float(eq_total.iloc[-1]) - INITIAL_CAPITAL) / INITIAL_CAPITAL
        dr = eq_total.pct_change().dropna()
        sh = (float(dr.mean()) / float(dr.std()) * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
        cm = eq_total.cummax()
        mdd = float(((eq_total - cm) / cm.replace(0, np.nan)).min())
        print("\n" + "-" * 72)
        print(
            f"叠加条件指数空头（仅弱市+近{INDEX_SHORT_SUM_DAYS}日大盘累计为负；"
            f"空头名义≤多头×{INDEX_SHORT_HEDGE_RATIO:.0%}）"
        )
        print("-" * 72)
        print(f"  开空交易日数: {st['short_days']} / {st['total_days']}")
        print(f"  空头累计贡献(近似): {st['hedge_pnl']:,.0f}")
        print(f"  总收益(多头+条件空头): {ret_total:.2%}")
        print(f"  最终资产: {float(eq_total.iloc[-1]):,.0f}")
        print(f"  夏普: {sh:.3f}")
        print(f"  最大回撤: {mdd:.2%}")
        if ret_total > panel_result["total_return"]:
            print("  → 相对纯多头，叠加空头后总收益更高")
        else:
            print("  → 本段样本下叠加空头未提升总收益（属正常，空头只在部分下跌日盈利）")
    bench = benchmark_return_over_dates(df, test_dates)
    if bench is not None:
        excess = panel_result["total_return"] - bench
        print(f"  同期大盘(sprtrn)收益: {bench:.2%}")
        print(f"  超额收益(vs大盘): {excess:.2%}")
        print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")
    # 同池等权：496 只每日等权的收益，比「满仓指数」更公平
    univ_ret = universe_equal_weight_return(df, test_dates)
    if univ_ret is not None:
        ex_univ = panel_result["total_return"] - univ_ret
        print(f"  同期股票池等权收益: {univ_ret:.2%}")
        print(f"  超额收益(vs等权池): {ex_univ:.2%}")
        print("  → 跑赢同池等权" if ex_univ > 0 else "  → 未跑赢同池等权")

    # ---------- 5.1 回测分析：IC、分组收益 ----------
    print("\n预测与未来收益分析（IC / 分组）:")
    report_ic_and_groups(test_df, pred_col="prediction", target_col="target", n_groups=5)

    # ---------- 6. 可视化 ----------
    if not args.no_plot and eq is not None and len(eq) > 0:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_base = "多头+空头回测" if float(args.short_ratio) > 0 else "单纯多头回测"
        plot_backtest_report(
            eq_for_plot,
            save_path=out_dir / "backtest_report.png",
            title="Portfolio + conditional index short" if args.index_short else title_base,
        )
        print(f"\n图表已保存: {out_dir / 'backtest_report.png'}")


if __name__ == "__main__":
    main()
