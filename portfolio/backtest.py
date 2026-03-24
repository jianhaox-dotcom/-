# -*- coding: utf-8 -*-
"""
组合回测：按打分排序选股，支持单票最大权重、每 N 日再平衡。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from config import INITIAL_CAPITAL, MAX_WEIGHT_PER_STOCK, REBALANCE_DAYS


def run_portfolio_backtest(
    df: pd.DataFrame,
    id_col: str = "ticker",
    top_n: int = 5,
    bottom_n: int = 0,
    initial_cash: float = INITIAL_CAPITAL,
    rebalance_freq: str = "D",
    rank_col: str = "prediction",
    max_weight_per_stock: Optional[float] = None,
    rebalance_days: Optional[int] = None,
    market_exposure: Optional[pd.Series] = None,
    short_notional_ratio: float = 0.0,
    short_when_no_long_ratio: float = 0.0,
) -> dict:
    """
    df 需含: date, id_col, close, 以及 rank_col（用于排序选股，如 prediction 或 score）。
    max_weight_per_stock: 单票最大权重（如 0.05=5%），则至少需 1/0.05=20 只分散。
    rebalance_days: 每 N 个交易日再平衡一次，其余日持仓不变，降低换手。
    market_exposure: 择时序列，index=date，值 0.5=半仓/1.0=满仓；None 表示始终满仓。
    short_notional_ratio: 做空名义（以多头名义的比例计）。例如 0.25 表示做空名义=多头名义*0.25（不会做完全对冲）。
    short_when_no_long_ratio: 当 short_notional_ratio==0 且本日没有任何正分可做多时，对预测最差且分数<0 的股票做空，
        空头名义 = 当日权益目标 equity_target × 该比例（与「无多时补空」配合）。
    """
    if id_col not in df.columns or rank_col not in df.columns:
        return {"error": f"缺少列 {id_col} 或 {rank_col}", "total_return": 0.0, "final_value": initial_cash}
    max_w = max_weight_per_stock if max_weight_per_stock is not None else MAX_WEIGHT_PER_STOCK
    rebal_days = rebalance_days if rebalance_days is not None else REBALANCE_DAYS

    df = df.sort_values(["date", id_col])
    dates = df["date"].unique()
    if len(dates) < 2:
        return {"total_return": 0.0, "final_value": initial_cash, "equity_curve": pd.Series([initial_cash])}

    n_long = top_n
    n_short = bottom_n
    cash = initial_cash
    positions = {}
    equity_curve = []
    # 维护最近一次出现的价格：避免某些日期缺失该 ticker 行时把价格当 0
    last_prices: dict[str, float] = {}

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d].dropna(subset=["close", rank_col])
        if len(day_df) == 0:
            equity_curve.append(equity_curve[-1] if equity_curve else initial_cash)
            continue
        # 同一天同一 ticker 只保留一行，避免 prices 出现重复索引导致标量变 Series
        day_df = day_df.drop_duplicates(subset=[id_col], keep="first")
        prices = day_df.set_index(id_col)["close"]

        def _scalar(v):
            if v is None:
                return 0.0
            if hasattr(v, "ndim") and getattr(v, "ndim", 0) > 0:
                return float(v.iloc[0]) if hasattr(v, "iloc") else float(v.flat[0])
            return float(v)

        # 更新当日可得价格（用于缺失时的 last_price）
        for t in prices.index:
            try:
                last_prices[t] = _scalar(prices.loc[t])
            except Exception:
                continue

        # 择时：大盘均线向下半仓，向上满仓
        exposure = 1.0
        if market_exposure is not None and len(market_exposure) > 0:
            v = market_exposure.get(d)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                exposure = float(v)
            else:
                exposure = 1.0
        exposure = max(0.0, min(1.0, exposure))

        # 仅在第 0 日或每 rebalance_days 日再平衡
        do_rebalance = (i % rebal_days == 0) or (i == 0)

        if do_rebalance:
            rank = day_df.sort_values(rank_col, ascending=False).reset_index(drop=True)
            long_tickers = rank.head(n_long)[id_col].tolist() if n_long else []
            short_tickers = rank.tail(n_short)[id_col].tolist() if n_short else []

            # 根据打分做权重：long/short 分别只用正值部分分配名义
            weights_long: dict[str, float] = {}
            weights_short: dict[str, float] = {}
            if rank_col in rank.columns:
                score_series = rank.set_index(id_col)[rank_col].astype(float)
                long_tickers = [t for t in long_tickers if float(score_series.get(t, 0.0)) > 0]
                short_tickers = [t for t in short_tickers if float(score_series.get(t, 0.0)) < 0]

                if n_long and long_tickers:
                    score_long = score_series.loc[long_tickers].clip(lower=0.0)
                    if float(score_long.sum()) > 0:
                        w_raw = score_long / float(score_long.sum())
                        w_capped = w_raw.clip(upper=max_w)
                        if float(w_capped.sum()) > 0:
                            w_capped = w_capped / float(w_capped.sum())
                            weights_long = w_capped.to_dict()

                if n_short and short_tickers and short_notional_ratio > 0:
                    score_short = (-score_series.loc[short_tickers]).clip(lower=0.0)
                    if float(score_short.sum()) > 0:
                        w_raw_s = score_short / float(score_short.sum())
                        w_capped_s = w_raw_s.clip(upper=max_w)
                        if float(w_capped_s.sum()) > 0:
                            w_capped_s = w_capped_s / float(w_capped_s.sum())
                            weights_short = w_capped_s.to_dict()

                # 无正分可做多时，可选：仅做空预测最差且 score<0 的一档（不启用常规定额 short_notional_ratio 时）
                use_short_fallback = (
                    float(short_notional_ratio) <= 0
                    and float(short_when_no_long_ratio) > 0
                    and n_long > 0
                    and not weights_long
                )
                if use_short_fallback:
                    long_tickers = []
                    cand = rank.tail(n_long)[id_col].tolist()
                    short_tickers = [t for t in cand if float(score_series.get(t, 0.0)) < 0]
                    weights_short = {}
                    if short_tickers:
                        score_short = (-score_series.loc[short_tickers]).clip(lower=0.0)
                        if float(score_short.sum()) > 0:
                            w_raw_s = score_short / float(score_short.sum())
                            w_capped_s = w_raw_s.clip(upper=max_w)
                            if float(w_capped_s.sum()) > 0:
                                w_capped_s = w_capped_s / float(w_capped_s.sum())
                                weights_short = w_capped_s.to_dict()

            to_close = [t for t in list(positions.keys()) if t not in long_tickers and t not in short_tickers]
            for t in to_close:
                if t in positions:
                    pr = last_prices.get(t)
                    if pr is None or pr <= 0:
                        continue
                    cash += positions[t] * pr
                    del positions[t]

            port_value = cash + sum(float(positions.get(t, 0.0)) * float(last_prices.get(t, 0.0)) for t in positions)
            equity_target = port_value * exposure  # 多头名义

            sr = float(short_notional_ratio)
            if sr <= 0 and float(short_when_no_long_ratio) > 0 and n_long > 0 and not weights_long and weights_short:
                long_notional = 0.0
                short_notional = equity_target * float(short_when_no_long_ratio) if equity_target > 0 else 0.0
            else:
                long_notional = equity_target if equity_target > 0 else 0.0
                short_notional = equity_target * sr if equity_target > 0 else 0.0

            target_tickers = set(long_tickers) | set(short_tickers)
            for t in target_tickers:
                if t not in prices.index:
                    continue
                pr = prices.get(t)
                pr = _scalar(pr)
                if pr is None or pr <= 0:
                    continue

                current_shares = float(positions.get(t, 0.0))
                if t in long_tickers:
                    w_t = float(weights_long.get(t, 0.0))
                    target_val = long_notional * w_t  # >0
                else:
                    w_t = float(weights_short.get(t, 0.0))
                    target_val = -short_notional * w_t  # <0 (short)

                target_shares = target_val / pr if pr > 0 else 0.0
                diff = target_shares - current_shares
                if diff != 0.0:
                    # diff>0 买入，cash减少；diff<0 做空（target为负）则现金增加（简化假设不受保证金约束）
                    cash -= diff * pr
                    positions[t] = target_shares

        mark = cash + sum(positions.get(t, 0) * float(last_prices.get(t, 0.0)) for t in positions)
        equity_curve.append(float(mark))

    eq = pd.Series(equity_curve, dtype=float)
    final_value = float(eq.iloc[-1]) if len(eq) else initial_cash
    total_return = (final_value - initial_cash) / initial_cash
    daily_ret = eq.pct_change().dropna()
    daily_ret = pd.to_numeric(daily_ret, errors="coerce").dropna()
    sharpe = (float(daily_ret.mean()) / float(daily_ret.std()) * np.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    cummax = eq.cummax()
    max_dd = ((eq - cummax) / cummax.replace(0, np.nan)).min() if len(cummax) else 0.0
    return {
        "total_return": total_return,
        "final_value": final_value,
        "equity_curve": eq,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
