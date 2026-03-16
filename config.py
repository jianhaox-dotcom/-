# -*- coding: utf-8 -*-
"""全局配置：本金、交易成本、数据列名、风险与仓位等。"""

# 资金与成本
INITIAL_CAPITAL = 10_000.0  # 美元
COMMISSION_PCT = 0.001      # 每笔交易按成交额的比例成本，如 0.1%
FIXED_COST_PER_TRADE = 0.0  # 每笔固定成本（美元）

# 仓位管理
MAX_POSITION_PCT = 1.0
MAX_WEIGHT_PER_STOCK = 0.05   # 单票最大权重 5%
REBALANCE_DAYS = 20           # 每 20 日再平衡一次，降低换手
RISK_FREE_RATE_ANNUAL = 0.0

# 预测目标：未来 N 日收益（降低噪音）
TARGET_FORWARD_DAYS = 5

# 训练集/测试集划分
TEST_RATIO = 0.2

# 数据列名
COL_DATE = "date"
COL_CLOSE = "close"
COL_PREDICTION = "prediction"
COL_RET = "ret"  # 已实现收益率（仅作目标/评估，不可作信号）
