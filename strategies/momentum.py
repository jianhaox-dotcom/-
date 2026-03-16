# -*- coding: utf-8 -*-
"""
动量策略：根据预测值的变化（或预测值方向）做多/空。
预测上升且超过阈值则做多，下降且超过阈值则空仓。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """基于预测值变化的动量策略。"""

    def __init__(self, change_threshold: float = 0.0, lookback: int = 1):
        """
        lookback: 看过去几天的预测变化；
        change_threshold: 变化超过此阈值才发出信号。
        """
        self.change_threshold = change_threshold
        self.lookback = lookback

    def fit(self, train_df: pd.DataFrame) -> None:
        """可选：用训练集统计变化分布来设定阈值。此处不拟合。"""
        pass

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        pred = df["prediction"]
        change = pred.diff(self.lookback)
        sig = np.where(change > self.change_threshold, 1, np.where(change < -self.change_threshold, -1, 0))
        out = pd.Series(sig, index=df.index)
        state = -1
        result = []
        for s in out:
            if s != 0:
                state = s
            result.append(state)
        return pd.Series(result, index=df.index)
