# -*- coding: utf-8 -*-
"""
量化研究主流程（无未来函数）：
  数据加载 → 特征工程 → 预测模型(t 时刻预测 t+1 收益) → 交易策略 → 回测 → 风险指标 → 可视化
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    TEST_RATIO,
    MAX_WEIGHT_PER_STOCK,
    REBALANCE_DAYS,
    TARGET_FORWARD_DAYS,
    INDEX_SHORT_HEDGE_RATIO,
    INDEX_SHORT_MA_DAYS,
    INDEX_SHORT_SUM_DAYS,
)
from data import apply_return_percent_scale, load_dataset, train_test_split
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


def market_exposure_series(df: pd.DataFrame, ma_days: int = 20) -> pd.Series | None:
    """按大盘(sprtrn)20日均线择时：指数在均线上方满仓(1.0)，下方半仓(0.5)。返回 date -> 暴露度。"""
    if "sprtrn" not in df.columns:
        return None
    sub = df[["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    if sub.empty:
        return None
    sub = sub.set_index("date")
    sub["ret"] = pd.to_numeric(sub["sprtrn"], errors="coerce").fillna(0)
    index_level = (1 + sub["ret"]).cumprod()
    ma = index_level.rolling(ma_days, min_periods=1).mean()
    exposure = np.where(index_level >= ma, 1.0, 0.5)
    return pd.Series(exposure, index=sub.index)


def sprtrn_daily_return_series(df: pd.DataFrame) -> pd.Series | None:
    """按交易日去重后的 sprtrn 日收益序列，index 为 date。"""
    if "sprtrn" not in df.columns:
        return None
    sub = df[["date", "sprtrn"]].drop_duplicates("date", keep="first").sort_values("date")
    r = pd.to_numeric(sub["sprtrn"], errors="coerce").fillna(0.0)
    idx = pd.DatetimeIndex(pd.to_datetime(sub["date"].values))
    return pd.Series(r.values, index=idx)


def merged_market_exposure(
    df: pd.DataFrame,
    *,
    timing: bool,
    vol_target_ann: float | None,
    vol_lookback: int = 21,
    vol_floor: float = 0.45,
    vol_cap: float = 1.0,
) -> tuple[pd.Series | None, list[str]]:
    """择时与「大盘波动率目标」乘在一条暴露序列上；无任一功能时返回 (None, [])。"""
    msgs: list[str] = []
    if not timing and (vol_target_ann is None or float(vol_target_ann) <= 0):
        return None, msgs

    dates = pd.DatetimeIndex(pd.to_datetime(df["date"].drop_duplicates().sort_values().unique()))
    tser = market_exposure_series(df) if timing else None
    if timing:
        if tser is not None:
            msgs.append("  已开启大盘择时（20 日均线下方半仓、上方满仓）")
        else:
            msgs.append("  警告: --timing 已开启但数据无 sprtrn，择时未生效")

    vser = None
    if vol_target_ann is not None and float(vol_target_ann) > 0:
        if "sprtrn" not in df.columns:
            msgs.append("  警告: --vol-target-ann 需要 sprtrn 列，未启用波动缩放")
        else:
            sub = df[["date", "sprtrn"]].drop_duplicates("date", keep="first").sort_values("date")
            r = pd.to_numeric(sub["sprtrn"], errors="coerce").fillna(0.0)
            lb = max(5, int(vol_lookback))
            minp = max(3, lb // 3)
            rv = r.rolling(lb, min_periods=minp).std(ddof=0) * np.sqrt(252.0)
            tgt = float(vol_target_ann)
            mult = tgt / rv.replace(0.0, np.nan)
            mult = mult.clip(lower=float(vol_floor), upper=float(vol_cap)).fillna(1.0)
            vser = pd.Series(mult.values, index=pd.DatetimeIndex(pd.to_datetime(sub["date"].values)))
            msgs.append(
                f"  已按 sprtrn 滚动波动做暴露缩放：目标年化≈{tgt:.0%}，窗口 {lb} 日，乘子∈[{vol_floor},{vol_cap}]（可与择时叠乘）"
            )

    if tser is None and vser is None:
        return None, msgs

    if tser is not None:
        t_part = tser.reindex(dates).ffill().bfill()
        if bool(t_part.isna().all()):
            t_part = pd.Series(1.0, index=dates)
    else:
        t_part = pd.Series(1.0, index=dates)
    if vser is not None:
        v_part = vser.reindex(dates).ffill().bfill()
        if bool(v_part.isna().all()):
            v_part = pd.Series(1.0, index=dates)
    else:
        v_part = pd.Series(1.0, index=dates)
    merged = (t_part.astype(float) * v_part.astype(float)).clip(0.0, 1.0)
    return merged, msgs


def apply_conditional_index_short(
    long_equity: pd.Series,
    test_dates: np.ndarray | list,
    df: pd.DataFrame,
    hedge_ratio: float = INDEX_SHORT_HEDGE_RATIO,
    ma_days: int = INDEX_SHORT_MA_DAYS,
    sum_days: int = INDEX_SHORT_SUM_DAYS,
) -> tuple[pd.Series, dict]:
    """
    条件指数空头（不压满多头）：
    仅在「昨日指数 < 昨日 MA」且「昨日为止近 sum_days 日 sprtrn 累计 < 0」时，
    当日用 hedge_ratio × 昨日多头市值 的名义做空指数（日收益 ≈ -sprtrn × 名义）。
    返回 (总权益曲线, 统计信息)。
    """
    if "sprtrn" not in df.columns or len(long_equity) != len(test_dates):
        return long_equity.copy(), {"short_days": 0, "hedge_pnl": 0.0}
    daily = df[["date", "sprtrn"]].drop_duplicates("date").sort_values("date")
    daily = daily.set_index("date")
    daily["r"] = pd.to_numeric(daily["sprtrn"], errors="coerce").fillna(0)
    idx_lvl = (1 + daily["r"]).cumprod()
    ma = idx_lvl.rolling(ma_days, min_periods=1).mean()
    sum_n = daily["r"].rolling(sum_days, min_periods=1).sum()

    dates_all = daily.index.tolist()
    pos = {d: i for i, d in enumerate(dates_all)}

    L = long_equity.astype(float).values
    H = 0.0
    total = []
    short_days = 0
    for i, d in enumerate(test_dates):
        d = pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d
        if i == 0:
            total.append(float(L[0]))
            continue
        prev = test_dates[i - 1]
        prev = pd.Timestamp(prev) if not isinstance(prev, pd.Timestamp) else prev
        if prev not in pos:
            total.append(float(L[i] + H))
            continue
        j = pos[prev]
        idx_p = float(idx_lvl.iloc[j])
        ma_p = float(ma.iloc[j])
        s5 = float(sum_n.iloc[j])
        short_on = (idx_p < ma_p) and (s5 < 0)
        r_t = float(daily["r"].get(d, 0)) if d in daily.index else 0.0
        if short_on:
            notional = hedge_ratio * float(L[i - 1])
            H += notional * (-r_t)
            short_days += 1
        total.append(float(L[i] + H))

    stats = {"short_days": short_days, "hedge_pnl": float(H), "total_days": len(test_dates)}
    return pd.Series(total, dtype=float), stats


def universe_equal_weight_return(df: pd.DataFrame, dates) -> float | None:
    """计算同期「股票池等权」累计收益：每日所有股票收益的均值再复利。与策略同池、可比。"""
    sub = df.loc[df["date"].isin(dates), ["date", "ret"]].copy()
    if sub.empty or "ret" not in sub.columns:
        return None
    sub["ret"] = pd.to_numeric(sub["ret"], errors="coerce").fillna(0)
    daily = sub.groupby("date")["ret"].mean()
    if len(daily) == 0:
        return None
    return float((1 + daily).prod() - 1)


def run_rolling_test_windows(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    window: int,
    step: int,
    top_n: int,
    bottom_n: int,
    rebalance_days: int,
    market_exposure,
    short_notional_ratio: float,
    rank_col: str,
    equal_weight_long_if_no_positive: bool,
    max_weight_per_stock: float,
    benchmark_daily_ret: pd.Series | None = None,
    benchmark_leg_fraction: float = 0.0,
) -> list[dict]:
    """在完整测试段上按交易日滑动 window 日，每窗跑一次组合回测并算 sprtrn/等权。"""
    dates_u = np.sort(test_df["date"].unique())
    n = len(dates_u)
    if n < window:
        return []
    rows: list[dict] = []
    for start in range(0, n - window + 1, step):
        wdates = dates_u[start : start + window]
        sub = test_df[test_df["date"].isin(wdates)]
        if sub.empty:
            continue
        pr = run_portfolio_backtest(
            sub,
            id_col="ticker",
            top_n=top_n,
            bottom_n=bottom_n,
            initial_cash=INITIAL_CAPITAL,
            rank_col=rank_col,
            max_weight_per_stock=max_weight_per_stock,
            rebalance_days=rebalance_days,
            market_exposure=market_exposure,
            short_notional_ratio=short_notional_ratio,
            equal_weight_long_if_no_positive=equal_weight_long_if_no_positive,
            benchmark_daily_ret=benchmark_daily_ret,
            benchmark_leg_fraction=benchmark_leg_fraction,
        )
        st = float(pr["total_return"])
        bench = benchmark_return_over_dates(full_df, wdates)
        univ = universe_equal_weight_return(full_df, wdates)
        rows.append(
            {
                "start_date": wdates[0],
                "end_date": wdates[-1],
                "strat": st,
                "bench": bench,
                "univ": univ,
                "ex_bench": (st - bench) if bench is not None else None,
                "ex_univ": (st - univ) if univ is not None else None,
            }
        )
    return rows


def _summarize_rolling_excess(name: str, excesses: list[float]) -> None:
    if not excesses:
        return
    xs = np.asarray(excesses, dtype=float)
    win = float(np.mean(xs > 0))
    print(f"  [{name}] 窗口数 {len(xs)} | 跑赢比例 {win:.2%}")
    print(f"    超额均值 {float(np.mean(xs)):.2%} | 中位数 {float(np.median(xs)):.2%}")
    print(f"    分位 p25/p75 {float(np.percentile(xs, 25)):.2%} / {float(np.percentile(xs, 75)):.2%}")
    print(f"    最小/最大超额 {float(np.min(xs)):.2%} / {float(np.max(xs)):.2%}")


def run_nonoverlapping_bucket_backtests(
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    bucket_trading_days: int,
    top_n: int,
    bottom_n: int,
    rebalance_days: int,
    market_exposure,
    short_notional_ratio: float,
    rank_col: str,
    equal_weight_long_if_no_positive: bool,
    max_weight_per_stock: float,
    benchmark_daily_ret: pd.Series | None,
    benchmark_leg_fraction: float,
) -> list[dict]:
    """按连续 N 个交易日无重叠切块，每段独立初始资金回测，并算同期 sprtrn / 等权。"""
    dates_u = np.sort(test_df["date"].unique())
    n = len(dates_u)
    bd = max(1, int(bucket_trading_days))
    min_days = 5
    rows: list[dict] = []
    i = 0
    seg = 0
    while i < n:
        j = min(i + bd, n)
        chunk_dates = dates_u[i:j]
        if len(chunk_dates) < min_days:
            break
        sub = test_df[test_df["date"].isin(chunk_dates)]
        if sub.empty:
            i = j
            continue
        pr = run_portfolio_backtest(
            sub,
            id_col="ticker",
            top_n=top_n,
            bottom_n=bottom_n,
            initial_cash=INITIAL_CAPITAL,
            rank_col=rank_col,
            max_weight_per_stock=max_weight_per_stock,
            rebalance_days=rebalance_days,
            market_exposure=market_exposure,
            short_notional_ratio=short_notional_ratio,
            equal_weight_long_if_no_positive=equal_weight_long_if_no_positive,
            benchmark_daily_ret=benchmark_daily_ret,
            benchmark_leg_fraction=benchmark_leg_fraction,
        )
        st = float(pr["total_return"])
        bench = benchmark_return_over_dates(full_df, chunk_dates)
        univ = universe_equal_weight_return(full_df, chunk_dates)
        ex_b = (st - bench) if bench is not None else None
        ex_u = (st - univ) if univ is not None else None
        seg += 1
        rows.append(
            {
                "segment": seg,
                "n_days": len(chunk_dates),
                "start_date": chunk_dates[0],
                "end_date": chunk_dates[-1],
                "strat": st,
                "bench": bench,
                "univ": univ,
                "ex_bench": ex_b,
                "ex_univ": ex_u,
                "beat_bench": (ex_b > 0) if ex_b is not None else None,
                "beat_univ": (ex_u > 0) if ex_u is not None else None,
            }
        )
        i = j
    return rows


def print_bucket_feedback_report(rows: list[dict]) -> None:
    if not rows:
        print("  无有效分段（数据过短或 bucket 过大）。")
        return
    print(f"  {'段':>4} {'交易日':>6} {'起止日期':^24} {'策略':>10} {'sprtrn':>10} {'超额':>8} {'跑赢大盘':>8} | {'等权':>8} {'超额':>8} {'跑赢等权':>8}")
    for r in rows:
        d0 = pd.Timestamp(r["start_date"]).strftime("%Y-%m-%d")
        d1 = pd.Timestamp(r["end_date"]).strftime("%Y-%m-%d")
        sb = f"{r['strat']:.2%}"
        bb = f"{r['bench']:.2%}" if r["bench"] is not None else "  —"
        eb = f"{r['ex_bench']:.2%}" if r["ex_bench"] is not None else "   —"
        yb = "是" if r["beat_bench"] is True else ("否" if r["beat_bench"] is False else "—")
        ub = f"{r['univ']:.2%}" if r["univ"] is not None else "    —"
        eu = f"{r['ex_univ']:.2%}" if r["ex_univ"] is not None else "   —"
        yu = "是" if r["beat_univ"] is True else ("否" if r["beat_univ"] is False else "—")
        print(
            f"  {r['segment']:4d} {r['n_days']:6d} {d0}~{d1} {sb:>10} {bb:>10} {eb:>8} {yb:>8} | {ub:>8} {eu:>8} {yu:>8}"
        )
    beats_b = [r for r in rows if r["beat_bench"] is True]
    misses_b = [r for r in rows if r["beat_bench"] is False]
    nb = len(beats_b)
    mb = len(misses_b)
    tot_b = nb + mb
    if tot_b > 0:
        print(
            f"\n  跑赢大盘(sprtrn)段数: {nb} / {tot_b} = {nb / tot_b:.2%}（未跑赢 {mb} 段）"
        )
    beats_u = [r for r in rows if r["beat_univ"] is True]
    misses_u = [r for r in rows if r["beat_univ"] is False]
    nu = len(beats_u)
    mu = len(misses_u)
    tot_u = nu + mu
    if tot_u > 0:
        print(
            f"  跑赢股票池等权段数: {nu} / {tot_u} = {nu / tot_u:.2%}（未跑赢 {mu} 段）"
        )
    ex_bs = [float(r["ex_bench"]) for r in rows if r["ex_bench"] is not None]
    if ex_bs:
        xb = np.asarray(ex_bs, dtype=float)
        print(
            f"  各段相对大盘超额: 均值 {float(np.mean(xb)):.2%} | 中位数 {float(np.median(xb)):.2%}"
        )


def main():
    parser = argparse.ArgumentParser(description="量化回测流程：多股票特征→模型→组合→回测→风险→图表")
    parser.add_argument(
        "data_path",
        nargs="?",
        default="A.csv",
        help="数据路径：单个 CSV 或包含多只股票 CSV 的目录",
    )
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="简单切分测试集比例（非 walk-forward 时使用）")
    parser.add_argument(
        "--test-first-trading-days",
        type=int,
        default=None,
        help="在 test-ratio 测试段内只保留最初 N 个交易日（与 --test-last-trading-days 二选一，约三个月用 63）",
    )
    parser.add_argument(
        "--test-last-trading-days",
        type=int,
        default=None,
        help="在 test-ratio 测试段内只保留最后 N 个交易日（与 --test-first-trading-days 二选一，约三个月用 63）",
    )
    parser.add_argument("--model", default="rf", choices=["ridge", "rf", "xgb", "ensemble"], help="预测模型（ensemble=Ridge+RF+可选XGB 取平均）")
    parser.add_argument("--walk-forward", action="store_true", help="使用 walk-forward 回测")
    parser.add_argument("--top-n", type=int, default=20, help="组合做多股票数量（单票最大权重 5%% 时建议≥20）")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    parser.add_argument("--out-dir", default=".", help="图表与报告输出目录")
    parser.add_argument("--timing", action="store_true", help="大盘择时：20日均线下方半仓、上方满仓")
    parser.add_argument("--rebalance-days", type=int, default=None, help="覆盖默认再平衡周期（如目标为5天建议改为5）")
    parser.add_argument("--target-forward-days", type=int, default=TARGET_FORWARD_DAYS, help="预测目标的未来天数（1=下一日，5=未来5日累计收益）")
    parser.add_argument("--use-predicted-signal", action="store_true", help="用数据集里的 predicted_RET 作为信号：训练一个校准模型，再用于选股排序")
    parser.add_argument(
        "--index-short",
        action="store_true",
        help="条件指数空头：仅弱市+近5日大盘跌时，用部分多头市值做空指数（不全对冲）",
    )
    parser.add_argument(
        "--predicted-direct-score",
        action="store_true",
        help="predicted_RET 用作排序分数（仅用反转方向，不训练校准模型）",
    )
    parser.add_argument(
        "--short-ratio",
        type=float,
        default=0.0,
        help="做空名义比例：short 名义=多头名义*short-ratio（>0 才启用做空，适用于 predicted-direct-score）",
    )
    parser.add_argument(
        "--use-predicted-as-feature",
        action="store_true",
        help="把 predicted_RET 当作额外输入特征（仍使用真实 target 训练评估）",
    )
    parser.add_argument(
        "--rank-long-always",
        action="store_true",
        help="与 --predicted-direct-score 联用：按预测排序选股；若当日无正分，则对 top_n 等权做多，避免空仓在上涨市大幅跑输 sprtrn",
    )
    parser.add_argument(
        "--rolling-test-days",
        type=int,
        default=None,
        help="在完整测试段上滑动 N 个交易日，逐窗回测并与 sprtrn/等权对比（如 63）；与「最后 N 天单次回测」不同，用于检验策略是否只在特定区间有效",
    )
    parser.add_argument(
        "--rolling-step",
        type=int,
        default=1,
        help="滑动步长（默认 1=每个可能起点一个窗口；可改为 5/21/63 加快速度）",
    )
    parser.add_argument(
        "--bucket-feedback-days",
        type=int,
        default=None,
        help="将当前分析用测试面板按每段连续 N 个交易日无重叠切块（如 63≈季），每段独立回测并对比 sprtrn/等权，打印各段是否跑赢及总体占比",
    )
    parser.add_argument(
        "--blend-sprtrn",
        type=float,
        default=0.0,
        help="0~1：目标多头名义中该比例按 sprtrn 日复利复制大盘，其余做选股多头，降低相对大盘的跟踪误差、超额更稳（需 sprtrn；勿与 --short-ratio>0 同用）",
    )
    parser.add_argument(
        "--vol-target-ann",
        type=float,
        default=None,
        help="若设置（如 0.12）：用 sprtrn 滚动日波动估计年化波动，将总暴露乘子夹在 [floor,cap]，高波动日自动减仓（可与 --timing 叠乘，需 sprtrn）",
    )
    parser.add_argument("--vol-target-lookback", type=int, default=21, help="--vol-target-ann 的滚动天数")
    parser.add_argument(
        "--vol-target-floor",
        type=float,
        default=0.45,
        help="波动缩放乘子下限（避免满仓过狠）",
    )
    parser.add_argument(
        "--vol-target-cap",
        type=float,
        default=1.0,
        help="波动缩放乘子上限",
    )
    parser.add_argument(
        "--ret-scale",
        type=str,
        default="auto",
        choices=["auto", "decimal", "percent"],
        help="收益列量纲：percent=CSV 为百分数(如 1.5 表示 1.5%%)对 ret/sprtrn/expected_RET/predicted_RET 除以 100；decimal=已是小数；auto=按分布自动判定",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"未找到数据文件或目录: {data_path}")
        return

    # ---------- 1. 数据加载：单文件或多股票目录（递归找所有 CSV） ----------
    if data_path.is_dir():
        # 你的数据是多层嵌套目录，最后才是 csv，这里直接递归搜全部 *.csv
        csv_files = sorted([p for p in data_path.rglob("*.csv") if p.is_file() and not p.name.startswith(".")])
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

    df, _ret_note = apply_return_percent_scale(df, args.ret_scale)
    print(f"  收益口径: {_ret_note}")

    # 使用 ret 重建 close（保证跨 split 的 close 连续）
    if "ticker" in df.columns and "ret" in df.columns and "date" in df.columns:
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)
        df["close"] = df.groupby("ticker")["ret"].apply(lambda s: (1.0 + s).cumprod()).reset_index(level=0, drop=True)

    # ---------- 2. 特征工程（仅用历史信息，按 ticker 分组滚动） ----------
    df = build_features(df, target_forward_days=args.target_forward_days)
    print(f"2. 特征工程完成 | 总样本 {len(df)} | 特征数: {len(FEATURE_NAMES)} | target_forward_days={args.target_forward_days}")

    market_exposure, _exp_msgs = merged_market_exposure(
        df,
        timing=args.timing,
        vol_target_ann=args.vol_target_ann,
        vol_lookback=args.vol_target_lookback,
        vol_floor=args.vol_target_floor,
        vol_cap=args.vol_target_cap,
    )
    for _m in _exp_msgs:
        print(_m)

    bench_daily = sprtrn_daily_return_series(df)
    blend_f = 0.0
    if float(args.blend_sprtrn) > 0 and float(args.short_ratio) <= 0:
        if bench_daily is None:
            print("  警告: --blend-sprtrn 需要 sprtrn 列，已忽略")
        else:
            blend_f = min(1.0, max(0.0, float(args.blend_sprtrn)))
            print(
                f"  已开启 sprtrn 复制腿：目标多头名义的 {blend_f:.1%} 按大盘日收益复利，{1.0 - blend_f:.1%} 为选股多头"
            )
    elif float(args.blend_sprtrn) > 0:
        print("  警告: --blend-sprtrn 与 --short-ratio>0 不并用，已忽略")

    rebal_days = args.rebalance_days if args.rebalance_days is not None else REBALANCE_DAYS

    # Walk-forward 回测
    if args.walk_forward:
        print("3. 使用 walk-forward 回测（多窗口滚动训练/测试 + 组合回测）")
        wf_result = walk_forward_panel(
            df,
            model_type=args.model,
            top_n=args.top_n,
            market_exposure_series=market_exposure,
            rebalance_days=rebal_days,
            use_predicted_signal=args.use_predicted_signal,
            benchmark_daily_ret=bench_daily if blend_f > 0 else None,
            benchmark_leg_fraction=blend_f,
        )
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
            print(f"  超额收益(vs大盘): {excess:.2%}")
            print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")
        if wf_result.get("benchmark_universe_return") is not None:
            bu = wf_result["benchmark_universe_return"]
            ex_u = wf_result["total_return"] - bu
            print(f"  同期股票池等权收益: {bu:.2%}")
            print(f"  超额收益(vs等权池): {ex_u:.2%}")
            print("  → 跑赢同池等权" if ex_u > 0 else "  → 未跑赢同池等权")

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
    # 先保留完整测试段；打分后再截断做「单次回测」；滑动窗口在完整测试段上跑
    train_df = df[df["date"].isin(train_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    n_test_days = len(np.unique(test_df["date"].values))
    print(f"3. 划分训练/测试 | 训练 {len(train_df)} | 测试 {len(test_df)}（完整测试段 {n_test_days} 个交易日）")

    # ---------- 4. 用于选股排序的“预测分数” ----------
    # - 默认：用历史特征训练模型，预测 target（未来收益）
    # - 开启 --use-predicted-signal：用 predicted_RET 做校准训练，再输出校准后的 prediction 用于排序
    if args.use_predicted_signal:
        if "predicted_RET" not in train_df.columns or "predicted_RET" not in test_df.columns:
            raise ValueError("use-predicted-signal 但数据中缺少 predicted_RET 列")
        # 自动判定方向：若 predicted_RET 与目标的相关为负，则反转特征（避免“高分买入却亏钱”）
        _pred = pd.to_numeric(train_df["predicted_RET"], errors="coerce")
        _y = pd.to_numeric(train_df["target"], errors="coerce")
        valid_dir = _pred.notna() & _y.notna() & np.isfinite(_pred.values) & np.isfinite(_y.values)
        invert_dir = False
        if valid_dir.sum() > 20:
            corr_dir = float(np.corrcoef(_pred.loc[valid_dir].values, _y.loc[valid_dir].values)[0, 1])
            invert_dir = corr_dir < 0
            print(f"  predicted_RET vs target 相关系数: {corr_dir:.6f} | invert_dir={invert_dir}")
        feature_col = "predicted_RET"
        if invert_dir:
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df["predicted_RET"] = -pd.to_numeric(train_df["predicted_RET"], errors="coerce")
            test_df["predicted_RET"] = -pd.to_numeric(test_df["predicted_RET"], errors="coerce")

        if args.predicted_direct_score:
            test_df = test_df.copy()
            test_df["prediction"] = pd.to_numeric(test_df["predicted_RET"], errors="coerce")
            print("4. predicted-direct-score：直接使用（已可能反转的）predicted_RET 作为排序分数")
        else:
            X_train = train_df[[feature_col]].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
            y_train = train_df["target"]
            valid = y_train.notna()
            X_train, y_train = X_train.loc[valid], y_train.loc[valid]
            X_test = test_df[[feature_col]].reindex(columns=[feature_col]).replace(
                [float("inf"), float("-inf")], float("nan")
            ).fillna(0)
            if args.model == "ensemble":
                fitted = train_ensemble(X_train, y_train, use_xgb=False)
                test_df["prediction"] = predict_ensemble(fitted, X_test)
            else:
                fitted = train_predictor(X_train, y_train, model_type=args.model)
                test_df["prediction"] = predict(fitted, X_test)
            print(f"4. 用 predicted_RET 校准后的分数完成（目标为未来 {args.target_forward_days} 日收益）")
    else:
        feature_cols = FEATURE_NAMES.copy()
        if (
            args.use_predicted_as_feature
            and "predicted_RET" in train_df.columns
            and "predicted_RET" in test_df.columns
            and "predicted_RET" not in feature_cols
        ):
            feature_cols.append("predicted_RET")
        X_train = train_df[feature_cols].replace([float("inf"), float("-inf")], float("nan")).fillna(0)
        y_train = train_df["target"]
        valid = y_train.notna()
        X_train, y_train = X_train.loc[valid], y_train.loc[valid]
        X_test = test_df[feature_cols].reindex(columns=feature_cols).replace(
            [float("inf"), float("-inf")], float("nan")
        ).fillna(0)

        if args.model == "ensemble":
            fitted = train_ensemble(X_train, y_train, use_xgb=False)
            test_df["prediction"] = predict_ensemble(fitted, X_test)
        else:
            fitted = train_predictor(X_train, y_train, model_type=args.model)
            test_df["prediction"] = predict(fitted, X_test)
        print(f"4. 预测模型完成（目标为未来 {args.target_forward_days} 日收益，无未来信息）")

    # 若为 XGBoost/模型可返回重要性，输出前若干重要因子
    if "fitted" in locals():
        fi = fitted.get("feature_importances")
        if fi is not None:
            imp = pd.Series(fi, index=fitted["feature_names"]).sort_values(ascending=False)
            top_imp = imp.head(10)
            print("\n前 10 个因子重要性：")
            for name, val in top_imp.items():
                print(f"  {name:20s}: {val:.4f}")

    # ---------- 5. 综合打分选股：prediction*0.7 + 低波动*0.3，单票最大 5%，20 日再平衡 ----------
    if args.predicted_direct_score:
        pred_score = pd.to_numeric(test_df["prediction"], errors="coerce")
        if float(args.short_ratio) > 0 or args.rank_long_always:
            # 做空或「无正分仍按排名做多」需要保留分数符号用于排序/选空
            test_df["score"] = pred_score
        else:
            # 纯多头：负分不建仓
            test_df["score"] = pred_score.clip(lower=0.0)
    elif "volatility_20" in test_df.columns:
        # 日内波动率升序排名：低波动 rank 小，low_vol_score=1-rank 则低波动得高分
        vol_rank = test_df.groupby("date")["volatility_20"].rank(pct=True, ascending=True).fillna(0.5)
        low_vol_score = 1.0 - vol_rank
        test_df["score"] = 0.7 * test_df["prediction"] + 0.3 * low_vol_score
    else:
        test_df["score"] = test_df["prediction"]
    rank_col = "score"

    if args.rank_long_always and not args.predicted_direct_score:
        print("  警告: --rank-long-always 仅在与 --predicted-direct-score 联用时生效，已忽略。")
    if args.rank_long_always and args.predicted_direct_score:
        print("  已开启 rank-long-always：无正分时对预测排序 top_n 等权做多")

    eq_long = bool(args.rank_long_always and args.predicted_direct_score)
    bottom_n_bt = args.top_n if float(args.short_ratio) > 0 else 0

    _tf = args.test_first_trading_days
    _tl = args.test_last_trading_days
    if _tf is not None and int(_tf) > 0 and _tl is not None and int(_tl) > 0:
        print("错误: 请勿同时指定 --test-first-trading-days 与 --test-last-trading-days")
        return

    # 可选：全流程仅用测试段最初或最后 N 日（含 rolling / 单次回测 / IC / 基准）
    if _tf is not None and int(_tf) > 0:
        n_keep = int(_tf)
        td_sorted = np.sort(test_df["date"].unique())
        if len(td_sorted) > n_keep:
            first_n = td_sorted[:n_keep]
            test_df = test_df[test_df["date"].isin(first_n)].copy()
            n_days = len(np.unique(test_df["date"].values))
            print(
                f"\n  已限定分析区间：测试段最初 {n_days} 个交易日（滑动回测、净值、IC、基准均只在此区间）"
            )
        else:
            print(
                f"\n  提示: --test-first-trading-days={n_keep} ≥ 当前测试段交易日数，仍使用全部测试日。"
            )
    elif _tl is not None and int(_tl) > 0:
        n_keep = int(_tl)
        td_sorted = np.sort(test_df["date"].unique())
        if len(td_sorted) > n_keep:
            last_n = td_sorted[-n_keep:]
            test_df = test_df[test_df["date"].isin(last_n)].copy()
            n_days = len(np.unique(test_df["date"].values))
            print(
                f"\n  已限定分析区间：测试段最后 {n_days} 个交易日（滑动回测、净值、IC、基准均只在此区间）"
            )
        else:
            print(
                f"\n  提示: --test-last-trading-days={n_keep} ≥ 当前测试段交易日数，仍使用全部测试日。"
            )

    if args.bucket_feedback_days is not None and int(args.bucket_feedback_days) > 0:
        bfd = max(1, int(args.bucket_feedback_days))
        bucket_rows = run_nonoverlapping_bucket_backtests(
            test_df,
            df,
            bucket_trading_days=bfd,
            top_n=args.top_n,
            bottom_n=bottom_n_bt,
            rebalance_days=rebal_days,
            market_exposure=market_exposure,
            short_notional_ratio=float(args.short_ratio),
            rank_col=rank_col,
            equal_weight_long_if_no_positive=eq_long,
            max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
            benchmark_daily_ret=bench_daily if blend_f > 0 else None,
            benchmark_leg_fraction=blend_f,
        )
        _bd_n = len(np.sort(test_df["date"].unique()))
        print("\n" + "=" * 72)
        print(f"按段反馈（无重叠；每段 {bfd} 个交易日，末段可不足；当前区间共 {_bd_n} 个交易日）")
        print("=" * 72)
        print_bucket_feedback_report(bucket_rows)
        print("  （每段从 INITIAL_CAPITAL 独立回测，段与段不复利衔接；跑赢=该段策略累计收益高于同期基准）")

    if args.rolling_test_days is not None and int(args.rolling_test_days) > 0:
        rw = int(args.rolling_test_days)
        rs = max(1, int(args.rolling_step))
        _du = len(np.sort(test_df["date"].unique()))
        if _du >= rw:
            _nw = (_du - rw) // rs + 1
            if _nw > 400:
                print(
                    f"  提示: 当前约 {_nw} 个滑动窗口，计算较久；可加 --rolling-step 21 或 63 加快速度"
                )
        roll_rows = run_rolling_test_windows(
            test_df,
            df,
            window=rw,
            step=rs,
            top_n=args.top_n,
            bottom_n=bottom_n_bt,
            rebalance_days=rebal_days,
            market_exposure=market_exposure,
            short_notional_ratio=float(args.short_ratio),
            rank_col=rank_col,
            equal_weight_long_if_no_positive=eq_long,
            max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
            benchmark_daily_ret=bench_daily if blend_f > 0 else None,
            benchmark_leg_fraction=blend_f,
        )
        _an_days = len(np.sort(test_df["date"].unique()))
        print("\n" + "=" * 72)
        print(f"滑动窗口回测（当前分析区间共 {_an_days} 个交易日；每窗 {rw} 日，步长 {rs}）")
        print("=" * 72)
        if not roll_rows:
            print("  测试段过短，无法形成窗口。")
        else:
            ex_b = [r["ex_bench"] for r in roll_rows if r["ex_bench"] is not None]
            ex_u = [r["ex_univ"] for r in roll_rows if r["ex_univ"] is not None]
            _summarize_rolling_excess("vs 大盘(sprtrn)", ex_b)
            _summarize_rolling_excess("vs 股票池等权", ex_u)
            print("  （每个窗口为当前分析区间内连续交易日，起点按步长滑动，非随机抽取）")

    test_df_single = test_df
    test_dates_for_bench = np.sort(test_df["date"].unique())

    panel_result = run_portfolio_backtest(
        test_df_single,
        id_col="ticker",
        top_n=args.top_n,
        bottom_n=bottom_n_bt,
        initial_cash=INITIAL_CAPITAL,
        rank_col=rank_col,
        max_weight_per_stock=MAX_WEIGHT_PER_STOCK,
        rebalance_days=rebal_days,
        market_exposure=market_exposure,
        short_notional_ratio=float(args.short_ratio),
        equal_weight_long_if_no_positive=eq_long,
        benchmark_daily_ret=bench_daily if blend_f > 0 else None,
        benchmark_leg_fraction=blend_f,
    )
    eq = panel_result["equity_curve"]
    test_dates_sorted = np.sort(test_df_single["date"].unique())
    eq_for_plot = eq
    is_long_short = float(args.short_ratio) > 0
    print("\n" + "=" * 72)
    print("测试集组合回测结果（" + ("多头+空头" if is_long_short else "纯多头") + "）")
    print("=" * 72)
    print(f"  总收益: {panel_result['total_return']:.2%}")
    print(f"  最终资产: {panel_result['final_value']:,.0f}")
    print(f"  夏普: {panel_result['sharpe_ratio']:.3f}")
    print(f"  最大回撤: {panel_result['max_drawdown']:.2%}")

    if args.index_short and len(eq) == len(test_dates_sorted):
        eq_total, st = apply_conditional_index_short(eq, test_dates_sorted, df)
        eq_for_plot = eq_total
        ret_total = (float(eq_total.iloc[-1]) - INITIAL_CAPITAL) / INITIAL_CAPITAL
        dr = eq_total.pct_change().dropna()
        sh = (float(dr.mean()) / float(dr.std()) * np.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0
        cm = eq_total.cummax()
        mdd = float(((eq_total - cm) / cm.replace(0, np.nan)).min())
        print("\n" + "-" * 72)
        print(
            f"叠加条件指数空头（仅弱市+近{INDEX_SHORT_SUM_DAYS}日大盘累计为负；"
            f"空头名义≤多头×{INDEX_SHORT_HEDGE_RATIO:.0%}）"
        )
        print("-" * 72)
        print(f"  开空交易日数: {st['short_days']} / {st['total_days']}")
        print(f"  空头累计贡献(近似): {st['hedge_pnl']:,.0f}")
        print(f"  总收益(多头+条件空头): {ret_total:.2%}")
        print(f"  最终资产: {float(eq_total.iloc[-1]):,.0f}")
        print(f"  夏普: {sh:.3f}")
        print(f"  最大回撤: {mdd:.2%}")
        if ret_total > panel_result["total_return"]:
            print("  → 相对纯多头，叠加空头后总收益更高")
        else:
            print("  → 本段样本下叠加空头未提升总收益（属正常，空头只在部分下跌日盈利）")
    bench = benchmark_return_over_dates(df, test_dates_for_bench)
    if bench is not None:
        excess = panel_result["total_return"] - bench
        print(f"  同期大盘(sprtrn)收益: {bench:.2%}")
        print(f"  超额收益(vs大盘): {excess:.2%}")
        print("  结论: 跑赢大盘" if excess > 0 else "  结论: 未跑赢大盘")
    # 同池等权：496 只每日等权的收益，比「满仓指数」更公平
    univ_ret = universe_equal_weight_return(df, test_dates_for_bench)
    if univ_ret is not None:
        ex_univ = panel_result["total_return"] - univ_ret
        print(f"  同期股票池等权收益: {univ_ret:.2%}")
        print(f"  超额收益(vs等权池): {ex_univ:.2%}")
        print("  → 跑赢同池等权" if ex_univ > 0 else "  → 未跑赢同池等权")

    # ---------- 5.1 回测分析：IC、分组收益 ----------
    print("\n预测与未来收益分析（IC / 分组）:")
    report_ic_and_groups(test_df_single, pred_col="prediction", target_col="target", n_groups=5)

    # ---------- 6. 可视化 ----------
    if not args.no_plot and eq is not None and len(eq) > 0:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        title_base = "多头+空头回测" if float(args.short_ratio) > 0 else "单纯多头回测"
        plot_backtest_report(
            eq_for_plot,
            save_path=out_dir / "backtest_report.png",
            title="Portfolio + conditional index short" if args.index_short else title_base,
        )
        print(f"\n图表已保存: {out_dir / 'backtest_report.png'}")


if __name__ == "__main__":
    main()
