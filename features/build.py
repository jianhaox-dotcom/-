# -*- coding: utf-8 -*-
"""
特征工程：仅用历史数据构建特征，目标为未来 N 日收益率（降低噪音）。
增加：动量组合、均线距离(均值回归)、波动率、风险调整(类夏普)等。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from config import TARGET_FORWARD_DAYS

# 特征名列表（build_features 运行后填充）
FEATURE_NAMES: list[str] = []  # type: list


def build_features(
    df: pd.DataFrame,
    momentum_windows: tuple[int, ...] = (5, 10, 20, 60),
    volatility_window: int = 20,
    liquidity_window: int = 5,
    target_forward_days: int = TARGET_FORWARD_DAYS,
) -> pd.DataFrame:
    """
    支持多股票面板；目标为未来 target_forward_days 日收益（默认 5 日），减少单日噪音。
    特征含：动量、波动率、20 日均线距离(均值回归)、风险调整(收益/波动)等。
    """
    out = df.copy()
    if "ticker" not in out.columns:
        out["ticker"] = "single"

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    ret = out["ret"].astype(float)

    g = out.groupby("ticker", group_keys=False)

    # ---------- 动量类：过去 w 日累计/平均收益 ----------
    for w in momentum_windows:
        out[f"momentum_{w}"] = g["ret"].rolling(w, min_periods=1).mean().reset_index(drop=True)
    out["momentum_ratio_5_20"] = out["momentum_5"] / (out["momentum_20"].replace(0, np.nan)).fillna(0)

    # ---------- 波动率：过去窗口收益标准差 ----------
    out[f"volatility_{volatility_window}"] = (
        g["ret"].rolling(volatility_window, min_periods=2).std().reset_index(drop=True).fillna(0)
    )
    # 避免除零，用于后续风险调整
    vol = out[f"volatility_{volatility_window}"].replace(0, np.nan)

    # ---------- 均值回归：股价离 20 日均线的距离 ----------
    ma20 = g["close"].rolling(20, min_periods=1).mean().reset_index(drop=True)
    out["dist_ma20"] = (out["close"].astype(float) - ma20) / ma20.replace(0, np.nan)
    out["dist_ma20"] = out["dist_ma20"].fillna(0)

    # ---------- 价格变化（过去 1/5 日，供风险调整等使用） ----------
    out["ret_1"] = g["close"].apply(lambda s: s.pct_change(1)).reset_index(drop=True).fillna(0)
    out["ret_5"] = g["close"].apply(lambda s: s.pct_change(5)).reset_index(drop=True).fillna(0)

    # ---------- 风险调整：过去 5 日收益/波动（类夏普） ----------
    out["ret_5_vol_ratio"] = (out["ret_5"] / vol).fillna(0)

    # ---------- 流动性：买卖价差、非流动性、换手率（滚动平滑） ----------
    if "ba_spread" in out.columns:
        out["ba_spread_ma"] = g["ba_spread"].rolling(liquidity_window, min_periods=1).mean().reset_index(drop=True)
    if "illiquidity" in out.columns:
        out["illiquidity_ma"] = (
            g["illiquidity"]
            .apply(lambda s: s.abs())
            .rolling(liquidity_window, min_periods=1)
            .mean()
            .reset_index(drop=True)
        )
    if "turnover" in out.columns:
        out["turnover_ma"] = g["turnover"].rolling(liquidity_window, min_periods=1).mean().reset_index(drop=True)
    if "vol_change" in out.columns:
        out["vol_change_ma"] = g["vol_change"].rolling(liquidity_window, min_periods=1).mean().reset_index(drop=True)

    # ---------- 市场相对表现：个股相对大盘 sprtrn ----------
    if "sprtrn" in out.columns:
        sprtrn = out["sprtrn"].fillna(0)
        out["excess_ret_1"] = ret - sprtrn
        out["excess_ret_5"] = out["momentum_5"] - sprtrn

    # ---------- 目标：未来 N 日收益（减少噪音，仅用于训练） ----------
    close = out["close"].astype(float)
    if target_forward_days <= 1:
        out["target"] = g["ret"].shift(-1).reset_index(drop=True)
    else:
        # 未来 N 日累计收益 = (close_{t+N} / close_t) - 1，按 ticker 分组
        fwd_close = g["close"].shift(-target_forward_days).reset_index(drop=True)
        out["target"] = (fwd_close / close - 1.0).fillna(np.nan)

    # 特征名：排除日期/价格/目标/原始成本列，保留数值型衍生特征
    exclude = {
        "date",
        "ticker",
        "close",
        "ret",
        "target",
        "tran_cost",
        "ask",
        "bid",
        "market_cap",
        "sprtrn",
        "ba_spread",
        "illiquidity",
        "turnover",
        "vol_change",
    }
    feature_names = [
        c for c in out.columns if c not in exclude and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not feature_names:
        feature_names = [c for c in out.columns if c not in exclude]

    global FEATURE_NAMES
    FEATURE_NAMES[:] = feature_names
    return out
