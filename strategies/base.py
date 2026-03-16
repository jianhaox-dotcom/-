# -*- coding: utf-8 -*-
"""
策略基类：输入为带 prediction 的 DataFrame，输出为信号 Series。
信号约定：1 = 持多/买入，-1 = 空仓/卖出，0 = 保持（或与前一信号一致）。
"""
from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """基于预测数据的买卖策略基类。"""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """在训练集上拟合参数（如阈值、分位数等）。可选，默认空实现。"""
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        对 df 生成交易信号，与 df 同索引。
        1 = 做多，-1 = 空仓，0 = 保持。
        """
        pass

    def run(self, df: pd.DataFrame, train_df: pd.DataFrame | None = None) -> pd.Series:
        """若提供 train_df 则先 fit，再对 df 生成信号。"""
        if train_df is not None and len(train_df) > 0:
            self.fit(train_df)
        return self.generate_signals(df)
