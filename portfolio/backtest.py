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
) -> dict:
    """
    df 需含: date, id_col, close, 以及 rank_col（用于排序选股，如 prediction 或 score）。
    max_weight_per_stock: 单票最大权重（如 0.05=5%），则至少需 1/0.05=20 只分散。
    rebalance_days: 每 N 个交易日再平衡一次，其余日持仓不变，降低换手。
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
    # 单票最大 5% 时，等权每只 1/n_long，若 1/n_long > max_w 则用 max_w 并少买几只或接受现金
    weight_per = min(1.0 / max(n_long, 1), max_w) if max_w > 0 else 1.0 / max(n_long, 1)

    for i, d in enumerate(dates):
        day_df = df[df["date"] == d].dropna(subset=["close", rank_col])
        if len(day_df) == 0:
            equity_curve.append(equity_curve[-1] if equity_curve else initial_cash)
            continue

        prices = day_df.set_index(id_col)["close"]
        # 仅在第 0 日或每 rebalance_days 日再平衡
        do_rebalance = (i % rebal_days == 0) or (i == 0)

        if do_rebalance:
            rank = day_df.sort_values(rank_col, ascending=False).reset_index(drop=True)
            long_tickers = rank.head(n_long)[id_col].tolist() if n_long else []
            short_tickers = rank.tail(n_short)[id_col].tolist() if n_short else []

            to_close = [t for t in list(positions.keys()) if t not in long_tickers and t not in short_tickers]
            for t in to_close:
                if t in positions and t in prices.index:
                    cash += positions[t] * prices[t]
                    del positions[t]

            port_value = cash + sum(positions.get(t, 0) * prices.get(t, 0) for t in positions)
            if port_value > 0:
                for t in long_tickers:
                    pr = prices.get(t)
                    if pr is None or pr <= 0:
                        continue
                    target_val = port_value * weight_per
                    target_shares = target_val / pr
                    current_shares = positions.get(t, 0)
                    diff = target_shares - current_shares
                    if diff > 0:
                        need = diff * pr
                        if cash >= need:
                            cash -= need
                            positions[t] = positions.get(t, 0) + diff
                    else:
                        sell_shares = min(-diff, current_shares)
                        cash += sell_shares * pr
                        positions[t] = positions.get(t, 0) - sell_shares

        mark = cash + sum(positions.get(t, 0) * prices.get(t, 0) for t in positions)
        equity_curve.append(mark)

    eq = pd.Series(equity_curve)
    final_value = eq.iloc[-1] if len(eq) else initial_cash
    total_return = (final_value - initial_cash) / initial_cash
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    cummax = eq.cummax()
    max_dd = ((eq - cummax) / cummax.replace(0, np.nan)).min() if len(cummax) else 0.0
    return {
        "total_return": total_return,
        "final_value": final_value,
        "equity_curve": eq,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
