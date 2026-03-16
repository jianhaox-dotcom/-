# -*- coding: utf-8 -*-
"""
回测可视化：资金曲线、回撤曲线、收益分布。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def plot_backtest_report(
    equity_curve: pd.Series,
    daily_returns: Optional[pd.Series] = None,
    save_path: Optional[str | Path] = None,
    title: str = "回测报告",
) -> None:
    """
    绘制资金曲线、回撤曲线、收益分布（直方图）。
    equity_curve: 每日资产净值序列。
    daily_returns: 若为 None 则由 equity_curve.pct_change() 计算。
    """
    if not HAS_PLOT:
        return
    if daily_returns is None:
        daily_returns = equity_curve.pct_change().dropna()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    # 1. 资金曲线
    ax1 = axes[0]
    ax1.plot(equity_curve.values, color="steelblue", lw=1)
    ax1.set_ylabel("Equity")
    ax1.set_title("Equity Curve")
    ax1.grid(True, alpha=0.3)

    # 2. 回撤曲线
    ax2 = axes[1]
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax.replace(0, np.nan)
    ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="coral", alpha=0.6)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Curve")
    ax2.grid(True, alpha=0.3)

    # 3. 收益分布
    ax3 = axes[2]
    ax3.hist(daily_returns.dropna(), bins=50, color="seagreen", alpha=0.7, edgecolor="white")
    ax3.axvline(0, color="black", linestyle="--", lw=1)
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Count")
    ax3.set_title("Return Distribution")
    ax3.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(Path(save_path), dpi=150, bbox_inches="tight")
    plt.close()
