# 框架改进说明（对应八点总结）

## 一、预测信号设计（已改进）

- **问题**：原先用 RET（真实收益率）作为交易信号，存在未来函数。
- **实现**：
  - 数据中 RET 仅改名为 `ret`，仅用于**目标变量**（target = ret.shift(-1)）和特征构造，**不参与信号**。
  - 使用 **t 时刻历史特征** 预测 **t+1 收益**，预测值作为策略的 `prediction` 列，保证无未来信息。
  - 流程：`features/build.py` 构造特征与 target → `models/predictor.py` 训练 Ridge/RF → 在测试集上得到 `prediction`。

## 二、特征变量利用（已改进）

- **实现**：`features/build.py` 中基于历史窗口构造：
  - **动量**：momentum_5/10/20、momentum_ratio_5_20
  - **波动率**：volatility_20（滚动标准差）
  - **流动性**：ba_spread_ma、illiquidity_ma、turnover_ma、vol_change_ma
  - **市场相对**：excess_ret_1、excess_ret_5（相对 sprtrn）
  - **价格变化**：ret_1、ret_5（close 的 pct_change）

## 三、策略结构（已改进）

- **实现**：引入 `models/predictor.py`，支持 **Ridge** 与 **随机森林(RF)**，用多维特征预测下一期收益；策略基于模型输出的 `prediction` 做阈值/成本感知等规则。
  - 命令行：`--model ridge` 或 `--model rf`。

## 四、组合投资设计（已预留）

- **实现**：`portfolio/backtest.py` 提供多股票组合回测接口（按预测排序、做多 top_n、可做空 bottom_n）。
  - 当前 main 流程为单标的；若数据为 panel（含 TICKER 等股票标识），可在此模块上扩展组合逻辑。

## 五、风险控制（已改进）

- **实现**：`backtest/engine.py` 的 `BacktestResult` 增加：
  - 夏普比率（sharpe_ratio）
  - 最大回撤（max_drawdown）
  - 年化波动率（volatility_annual）
  - 胜率（win_rate）
  - 并记录 `equity_curve` 供可视化与进一步分析。

## 六、资金管理（已改进）

- **实现**：`config.py` 中 `MAX_POSITION_PCT`（默认 1.0）；回测引擎支持 `max_position_pct` 参数，单次买入最多使用该比例资金，其余留作现金。

## 七、结果分析与可视化（已改进）

- **实现**：`viz/report.py` 的 `plot_backtest_report()` 生成：
  - 资金曲线（Equity Curve）
  - 回撤曲线（Drawdown Curve）
  - 日收益分布（Return Distribution）
  - 输出路径：`--out-dir` 指定目录下的 `backtest_report.png`。

## 八、研究流程系统化（已改进）

- **主流程**（`main.py`）按固定顺序执行：
  1. 数据加载（`data`）
  2. 特征工程（`features`）
  3. 训练/预测模型（`models`）
  4. 策略生成信号并回测（`strategies` + `backtest`）
  5. 风险指标输出
  6. 可视化（`viz`）
- 模块划分：`data/`、`features/`、`models/`、`strategies/`、`backtest/`、`portfolio/`、`viz/`、`config.py`，便于扩展与写报告。

---

**运行示例**  
- 默认 A.csv、Ridge、生成图表：`python main.py`  
- 随机森林、不画图：`python main.py --model rf --no-plot`  
- 指定输出目录：`python main.py --out-dir ./report`
