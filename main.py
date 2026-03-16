# -*- coding: utf-8 -*-
"""
量化研究主流程（无未来函数）：
  数据加载 → 特征工程 → 预测模型(t 时刻预测 t+1 收益) → 交易策略 → 回测 → 风险指标 → 可视化
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import INITIAL_CAPITAL, TEST_RATIO, MAX_WEIGHT_PER_STOCK, REBALANCE_DAYS
from data import load_dataset, train_test_split
from features import build_features
from features.build import FEATURE_NAMES
from models import train_predictor, predict, train_ensemble, predict_ensemble
from portfolio import run_portfolio_backtest
from evaluation import walk_forward_panel
from viz import plot_backtest_report
from analysis import report_ic_and_groups


def benchmark_return_over_dates(df: pd.DataFrame, dates) -> float | None:
    """计算同期大盘（sprtrn）累计收益。df 需含 date、sprtrn。"""
    if "sprtrn" not in df.columns:
        return None
    sub = df.loc[df["date"].isin(dates), ["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    if sub.empty:
        return None
    return float((1 + sub["sprtrn"].fillna(0)).prod() - 1)


def main():
    parser = argparse.ArgumentParser(description="量化回测流程：多股票特征→模型→组合→回测→风险→图表")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="A.csv",
        help="数据路径：单个 CSV 或包含多只股票 CSV 的目录",
    )
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="简单切分测试集比例（非 walk-forward 时使用）")
    parser.add_argument("--model", default="rf", choices=["ridge", "rf", "xgb", "ensemble"], help="预测模型（ensemble=Ridge+RF+可选XGB 取平均）")
    parser.add_argument("--walk-forward", action="store_true", help="使用 walk-forward 回测")
    parser.add_argument("--top-n", type=int, default=20, help="组合做多股票数量（单票最大权重 5% 时建议≥20）")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    parser.add_argument("--out-dir", default=".", help="图表与报告输出目录")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"未找到数据文件或目录: {data_path}")
        return

    # ---------- 1. 数据加载：单文件或多股票目录（含一级子目录） ----------
    if data_path.is_dir():
        csv_files = sorted(data_path.glob("*.csv"))
        if not csv_files:
            for sub in sorted(data_path.iterdir()):
                if sub.is_dir():
                    csv_files.extend(sorted(sub.glob("*.csv")))
        csv_files = sorted(csv_files)
        dfs = []
        for csv in csv_files:
            try:
                df_i = load_dataset(csv, close_col="PRC", prediction_col="RET")
                dfs.append(df_i)
            except Exception as e:
                print(f"跳过 {csv.name}: {e}")
        if not dfs:
            print("目录及子目录中没有可用的 CSV 数据。")
            return
        df = pd.concat(dfs, ignore_index=True)
        print(f"  已加载 {len(dfs)} 个 CSV，共 {len(df)} 条记录")
    else:
        close_col = "PRC" if data_path.name.endswith(".csv") else None
        pred_col = "RET" if data_path.name.endswith(".csv") else None
        df = load_dataset(data_path, close_col=close_col, prediction_col=pred_col)
    print("1. 数据加载完成（支持多股票面板，ret 仅作目标/评估，不作交易信号）")

    # ---------- 2. 特征工程（仅用历史信息，按 ticker 分组滚动） ----------
    df = build_features(df)
    print(f"2. 特征工程完成 | 总样本 {len(df)} | 特征数: {len(FEATURE_NAMES)}")

    # Walk-forward 回测
    if args.walk_forward:
        print("3. 使用 walk-forward 回测（多窗口滚动训练/测试 + 组合回测）")
        wf_result = walk_forward_panel(df, model_type=args.model, top_n=args.top_n)
        eq = wf_result["equity_curve"]
        print("\n" + "=" * 72)
        print("Walk-forward 组合回测结果")
        print("=" * 72)
        print(f"  总收益: {wf_result['total_return']:.2%}")
        print(f"  最终资产: {wf_result['final_value']:,.0f}")
        print(f"  夏普: {wf_result['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {wf_result['max_drawdown']:.2%}")
        if wf_result.get("benchmark_return") is not None:
            b = wf_result["benchmark_return"]
            excess = wf_result["total_return"] - b
            print(f"  同期大盘(sprtrn)收益: {b:.2%}")
            print(f"  超额收益: {excess:.2%}")
            print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")

        if not args.no_plot and eq is not None and len(eq) > 0:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_backtest_report(
                eq,
                save_path=out_dir / "backtest_report.png",
                title="Walk-forward Portfolio Backtest",
            )
            print(f"\n图表已保存: {out_dir / 'backtest_report.png'}")
        return

    # ---------- 3. 简单 train/test 切分 + 组合回测 ----------
    # 按日期划分训练/测试（对所有股票统一切分）
    dates = pd.to_datetime(df["date"]).sort_values().unique()
    cut_idx = int(len(dates) * (1 - args.test_ratio))
    train_dates = dates[:cut_idx]
    test_dates = dates[cut_idx:]
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    print(f"3. 划分训练/测试 | 训练 {len(train_df)} | 测试 {len(test_df)}")

    # ---------- 4. 预测模型：用历史特征预测未来收益（默认 5 日） ----------
    X_train = train_df[FEATURE_NAMES].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
    y_train = train_df["target"]
    valid = y_train.notna()
    X_train, y_train = X_train.loc[valid], y_train.loc[valid]
    X_test = test_df[FEATURE_NAMES].reindex(columns=FEATURE_NAMES).replace(
        [float("inf"), float("-inf")], float("nan")
    ).fillna(0)

    if args.model == "ensemble":
        fitted = train_ensemble(X_train, y_train, use_xgb=False)
        test_df["prediction"] = predict_ensemble(fitted, X_test)
    else:
        fitted = train_predictor(X_train, y_train, model_type=args.model)
        test_df["prediction"] = predict(fitted, X_test)
    print("4. 预测模型完成（目标为未来 5 日收益，无未来信息）")

    # 若为 XGBoost，输出前若干重要因子
    fi = fitted.get("feature_importances")
    if fi is not None:
        imp = pd.Series(fi, index=fitted["feature_names"]).sort_values(ascending=False)
        top_imp = imp.head(10)
        print("\n前 10 个因子重要性：")
        for name, val in top_imp.items():
            print(f"  {name:20s}: {val:.4f}")

    # ---------- 5. 综合打分选股：prediction*0.7 + 低波动*0.3，单票最大 5%，20 日再平衡 ----------
    if "volatility_20" in test_df.columns:
        # 日内波动率升序排名：低波动 rank 小，low_vol_score=1-rank 则低波动得高分
        vol_rank = test_df.groupby("date")["volatility_20"].rank(pct=True, ascending=True).fillna(0.5)
        low_vol_score = 1.0 - vol_rank
        test_df["score"] = 0.7 * test_df["prediction"] + 0.3 * low_vol_score
    else:
        test_df["score"] = test_df["prediction"]
    rank_col = "score"

    panel_result = run_portfolio_backtest(
        test_df,
        id_col="ticker",
        top_n=args.top_n,
        bottom_n=0,
        initial_cash=INITIAL_CAPITAL,
        rank_col=rank_col,
        max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
        rebalance_days=REBALANCE_DAYS,
    )
    eq = panel_result["equity_curve"]
    print("\n" + "=" * 72)
    print("测试集组合回测结果")
    print("=" * 72)
    print(f"  总收益: {panel_result['total_return']:.2%}")
    print(f"  最终资产: {panel_result['final_value']:,.0f}")
    print(f"  夏普: {panel_result['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {panel_result['max_drawdown']:.2%}")
    bench = benchmark_return_over_dates(df, test_dates)
    if bench is not None:
        excess = panel_result["total_return"] - bench
        print(f"  同期大盘(sprtrn)收益: {bench:.2%}")
        print(f"  超额收益: {excess:.2%}")
        print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")

    # ---------- 5.1 回测分析：IC、分组收益 ----------
    print("\n预测与未来收益分析（IC / 分组）:")
    report_ic_and_groups(test_df, pred_col="prediction", target_col="target", n_groups=5)

    # ---------- 6. 可视化 ----------
    if not args.no_plot and eq is not None and len(eq) > 0:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_backtest_report(
            eq,
            save_path=out_dir / "backtest_report.png",
            title="Single-split Portfolio Backtest",
        )
        print(f"\n图表已保存: {out_dir / 'backtest_report.png'}")


if __name__ == "__main__":
    main()
