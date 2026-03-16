# -*- coding: utf-8 -*-
"""
置信度策略：仅当预测的「置信度」足够高时才交易，否则保持。
假设 prediction 数值的绝对值或与 0 的距离表示置信度；
或预测 > 某阈值做多，< 负阈值空仓，中间观望。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import BaseStrategy


class ConfidenceStrategy(BaseStrategy):
    """仅在高置信度时交易，减少频繁买卖。"""

    def __init__(self, min_abs_threshold: float = 0.0, use_quantile: bool = True, quantile: float = 0.6):
        """
        use_quantile=True 时用训练集上 prediction 绝对值的分位数作为最小置信度；
        否则用 min_abs_threshold：仅当 |prediction| >= 该值时才按符号做多/空。
        """
        self.min_abs_threshold = min_abs_threshold
        self.use_quantile = use_quantile
        self.quantile = quantile
        self._threshold: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        if not self.use_quantile or train_df is None or len(train_df) == 0:
            self._threshold = self.min_abs_threshold
            return
        abs_pred = train_df["prediction"].dropna().abs()
        if len(abs_pred) == 0:
            self._threshold = self.min_abs_threshold
            return
        self._threshold = float(abs_pred.quantile(self.quantile))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = df["prediction"]
        confident_long = (p >= self._threshold)
        confident_short = (p <= -self._threshold)
        sig = np.where(confident_long, 1, np.where(confident_short, -1, 0))
        out = pd.Series(sig, index=df.index)
        state = -1
        result = []
        for s in out:
            if s != 0:
                state = s
            result.append(state)
        return pd.Series(result, index=df.index)
