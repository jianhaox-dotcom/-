# -*- coding: utf-8 -*-
"""
回测引擎：按信号买卖，支持 A.csv 成本、仓位管理、风险指标。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from config import INITIAL_CAPITAL, COMMISSION_PCT, FIXED_COST_PER_TRADE, MAX_POSITION_PCT, RISK_FREE_RATE_ANNUAL


@dataclass
class BacktestResult:
    """回测结果（含风险指标）。"""
    total_return: float
    final_value: float
    total_trades: int
    total_commission: float
    win_trades: int
    loss_trades: int
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility_annual: float = 0.0
    win_rate: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)


def _exec_price(df: pd.DataFrame, i: int, side: str) -> float:
    """第 i 行的成交价：有 ask/bid 用其否则用 close。"""
    if side == "buy" and "ask" in df.columns:
        v = df["ask"].iloc[i]
        if pd.notna(v):
            return float(v)
    if side == "sell" and "bid" in df.columns:
        v = df["bid"].iloc[i]
        if pd.notna(v):
            return float(v)
    return float(df["close"].iloc[i])


def _tran_cost_per_share(df: pd.DataFrame, i: int) -> float:
    """第 i 行的每股交易成本（来自 TRAN_COST），缺失则 0。"""
    if "tran_cost" not in df.columns:
        return 0.0
    v = df["tran_cost"].iloc[i]
    return float(v) if pd.notna(v) else 0.0


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_cash: float = INITIAL_CAPITAL,
    commission_pct: float = COMMISSION_PCT,
    fixed_cost_per_trade: float = FIXED_COST_PER_TRADE,
    use_data_costs: bool = True,
    max_position_pct: float = MAX_POSITION_PCT,
) -> BacktestResult:
    """
    按信号买卖。max_position_pct：单次买入最多使用资金的比例（1.0=全仓）。
    成本：(1) ask/bid 成交价 (2) tran_cost 每股 (3) commission_pct + fixed_cost_per_trade。
    """
    price = df["close"].astype(float)
    cash = initial_cash
    position = 0.0
    total_commission = 0.0
    trades = []
    prev_sig = -1
    equity_curve = []

    for i in range(len(df)):
        s = signal.iloc[i]
        if pd.isna(s):
            s = prev_sig
        p_close = price.iloc[i]
        p_buy = _exec_price(df, i, "buy") if use_data_costs else p_close
        p_sell = _exec_price(df, i, "sell") if use_data_costs else p_close
        tc = _tran_cost_per_share(df, i) if use_data_costs else 0.0

        if s == 1 and prev_sig != 1:
            if position == 0 and cash > 0:
                cost = cash * max_position_pct
                commission = cost * commission_pct + fixed_cost_per_trade
                total_commission += commission
                effective_price = p_buy + tc
                pos = (cost - commission) / effective_price if effective_price > 0 else 0.0
                if pos > 0:
                    total_commission += pos * tc
                cash = cash - cost
                position = pos
                trades.append(("buy", p_buy, cost, commission + (pos * tc if pos > 0 else 0)))
        elif s == -1 and prev_sig != -1:
            if position > 0:
                value = position * p_sell
                commission = value * commission_pct + fixed_cost_per_trade
                commission += position * tc
                total_commission += commission
                cash = cash + value - commission
                trades.append(("sell", p_sell, value, commission))
                position = 0.0

        mark_to_market = cash + position * p_close
        equity_curve.append(mark_to_market)
        prev_sig = s

    final_value = equity_curve[-1] if equity_curve else initial_cash
    total_return = (final_value - initial_cash) / initial_cash

    sell_trades = [t for t in trades if t[0] == "sell"]
    buy_trades = [t for t in trades if t[0] == "buy"]
    win_trades = sum(1 for j, sell in enumerate(sell_trades) if j < len(buy_trades) and sell[1] > buy_trades[j][1])
    loss_trades = len(sell_trades) - win_trades
    win_rate = win_trades / max(win_trades + loss_trades, 1)

    eq = pd.Series(equity_curve)
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe_ratio = (daily_ret.mean() - RISK_FREE_RATE_ANNUAL / 252) / daily_ret.std() * np.sqrt(252)
        volatility_annual = daily_ret.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
        volatility_annual = 0.0
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax.replace(0, np.nan)
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    return BacktestResult(
        total_return=total_return,
        final_value=final_value,
        total_trades=len(trades),
        total_commission=total_commission,
        win_trades=win_trades,
        loss_trades=loss_trades,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        volatility_annual=volatility_annual,
        win_rate=win_rate,
        equity_curve=eq,
    )
