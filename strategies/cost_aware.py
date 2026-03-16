# -*- coding: utf-8 -*-
"""
成本与流动性感知策略：仅在买卖成本低、流动性好时交易，其余保持。
使用 A.csv 的 tran_cost、ba_spread、illiquidity、turnover。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import BaseStrategy
from .threshold import ThresholdStrategy


class CostAwareStrategy(BaseStrategy):
    """
    在阈值信号基础上，仅当成本/流动性达标时才发出买卖；
    否则保持上一状态，减少高成本或低流动性时的交易。
    """

    def __init__(
        self,
        upper_quantile: float = 0.7,
        lower_quantile: float = 0.3,
        cost_quantile: float = 0.6,
        use_ba_spread: bool = True,
        use_tran_cost: bool = True,
        use_illiquidity: bool = True,
    ):
        """
        cost_quantile: 仅在 ba_spread/tran_cost 低于训练集该分位数时交易（成本更低）；
        use_* 为是否用对应列过滤。
        """
        self._base = ThresholdStrategy(upper_quantile=upper_quantile, lower_quantile=lower_quantile)
        self.cost_quantile = cost_quantile
        self.use_ba_spread = use_ba_spread
        self.use_tran_cost = use_tran_cost
        self.use_illiquidity = use_illiquidity
        self._ba_spread_threshold = np.nan
        self._tran_cost_threshold = np.nan
        self._illiquidity_threshold = np.nan

    def fit(self, train_df: pd.DataFrame) -> None:
        self._base.fit(train_df)
        if self.use_ba_spread and "ba_spread" in train_df.columns:
            s = train_df["ba_spread"].dropna()
            if len(s) > 0:
                self._ba_spread_threshold = float(s.quantile(self.cost_quantile))
        if self.use_tran_cost and "tran_cost" in train_df.columns:
            s = train_df["tran_cost"].dropna()
            if len(s) > 0:
                self._tran_cost_threshold = float(s.quantile(self.cost_quantile))
        if self.use_illiquidity and "illiquidity" in train_df.columns:
            s = train_df["illiquidity"].dropna().abs()
            if len(s) > 0:
                self._illiquidity_threshold = float(s.quantile(1 - self.cost_quantile))

    def _cost_ok(self, df: pd.DataFrame, i: int) -> bool:
        ok = True
        if self.use_ba_spread and "ba_spread" in df.columns and pd.notna(self._ba_spread_threshold):
            v = df["ba_spread"].iloc[i]
            if pd.notna(v) and v > self._ba_spread_threshold:
                ok = False
        if self.use_tran_cost and "tran_cost" in df.columns and pd.notna(self._tran_cost_threshold):
            v = df["tran_cost"].iloc[i]
            if pd.notna(v) and v > self._tran_cost_threshold:
                ok = False
        if self.use_illiquidity and "illiquidity" in df.columns and pd.notna(self._illiquidity_threshold):
            v = df["illiquidity"].iloc[i]
            if pd.notna(v) and abs(v) > self._illiquidity_threshold:
                ok = False
        return ok

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        base_sig = self._base.generate_signals(df)
        result = []
        state = -1
        for i in range(len(df)):
            s = base_sig.iloc[i]
            if s != 0 and self._cost_ok(df, i):
                state = s
            result.append(state)
        return pd.Series(result, index=df.index)


class MarketRelativeStrategy(BaseStrategy):
    """
    相对市场：仅当预测收益 RET 优于市场收益 sprtrn 时做多，否则空仓或保持。
    使用 A.csv 的 RET、sprtrn。
    """

    def __init__(self, min_outperform: float = 0.0):
        """min_outperform: 仅当 RET - sprtrn > min_outperform 时做多。"""
        self.min_outperform = min_outperform

    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pred = df["prediction"]
        sprtrn = df["sprtrn"].fillna(0.0) if "sprtrn" in df.columns else pd.Series(0.0, index=df.index)
        excess = pred - sprtrn
        sig = np.where(excess > self.min_outperform, 1, -1)
        state = -1
        result = []
        for s in sig:
            if s != 0:
                state = s
            result.append(state)
        return pd.Series(result, index=df.index)
