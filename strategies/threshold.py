# -*- coding: utf-8 -*-
"""
阈值策略：预测值 > 上阈值则做多，< 下阈值则空仓，否则保持。
阈值可在训练集上按分位数或固定值设定。
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import BaseStrategy


class ThresholdStrategy(BaseStrategy):
    """基于预测值阈值的买卖策略。"""

    def __init__(
        self,
        upper_quantile: float = 0.7,
        lower_quantile: float = 0.3,
        use_quantile: bool = True,
        fixed_upper: float | None = None,
        fixed_lower: float | None = None,
    ):
        """
        use_quantile=True 时用训练集预测值的分位数作为阈值；
        否则用 fixed_upper / fixed_lower（需在 fit 或构造时给定）。
        """
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.use_quantile = use_quantile
        self.fixed_upper = fixed_upper
        self.fixed_lower = fixed_lower
        self._upper: float = 0.0
        self._lower: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        pred = train_df["prediction"].dropna()
        if len(pred) == 0:
            return
        if self.use_quantile:
            self._upper = float(pred.quantile(self.upper_quantile))
            self._lower = float(pred.quantile(self.lower_quantile))
        else:
            self._upper = self.fixed_upper if self.fixed_upper is not None else float(pred.quantile(0.7))
            self._lower = self.fixed_lower if self.fixed_lower is not None else float(pred.quantile(0.3))

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = df["prediction"]
        sig = np.where(p >= self._upper, 1, np.where(p <= self._lower, -1, 0))
        out = pd.Series(sig, index=df.index)
        # 将 0 变为“保持前一状态”，初始为 -1（空仓）
        state = -1
        result = []
        for s in out:
            if s != 0:
                state = s
            result.append(state)
        return pd.Series(result, index=df.index)
