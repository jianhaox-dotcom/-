# -*- coding: utf-8 -*-
"""
数据集加载与训练/测试划分。
假设数据为「预测完」的数据，至少包含：日期、收盘价、预测值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from config import COL_DATE, COL_CLOSE, COL_PREDICTION, TEST_RATIO


def apply_return_percent_scale(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str]:
    """
    统一 ret / sprtrn / expected_RET / predicted_RET 的量纲为「小数收益」
    （如 0.01 = 1%%）。若 CSV 存的是百分数（1.5 表示 1.5%%），应除以 100。

    mode:
      - decimal: 不缩放（已为小数）
      - percent: 强制上述列 ÷100
      - auto: 根据 ret 的分布推断（看中位、高分位与最大值，避免把正常小数误判成百分数）
    """
    m = (mode or "auto").strip().lower()
    scale_cols = [c for c in ("ret", "sprtrn", "expected_RET", "predicted_RET") if c in df.columns]
    if not scale_cols:
        return df, "无 ret/sprtrn 等列，跳过收益口径处理"

    def _scaled_copy(d: pd.DataFrame) -> pd.DataFrame:
        out = d.copy()
        for c in scale_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce") / 100.0
        return out

    if m == "decimal":
        return df, "decimal：列已为小数收益，未 ÷100"

    if m == "percent":
        return _scaled_copy(df), "percent：已按百分数解读并对 ret/sprtrn/expected_RET/predicted_RET 除以 100"

    # ---------- auto ----------
    r = pd.to_numeric(df["ret"], errors="coerce")
    r = r[np.isfinite(r.values)]
    r = r[r != 0]
    if len(r) < 200:
        return df, "auto：有效 ret 过少，保持不缩放"

    a = np.abs(r.to_numpy(dtype=float))
    med = float(np.median(a))
    p90 = float(np.percentile(a, 90))
    p99 = float(np.percentile(a, 99))
    mx = float(np.max(a))

    # 小数日收益：成熟股票池 |ret| 中位多在 0.002～0.015（0.2%～1.5%），p90 多在 0.02～0.05
    # 百分数写法（列里写 1.5 表示 1.5%%）：数值比「已是小数的同日收益」大约大 100 倍，中位常 ≥0.08
    # 边界：若 CSV 把「百分数」写成 0.01 表示 1%%（未写 1.0），中位会在 0.008～0.02，仍须 ÷100，
    #       否则按小数复利会爆出天文收益；故 med≥0.006 或 p90≥0.018 优先按百分数处理。
    treat_as_percent = False
    if med >= 0.006:
        treat_as_percent = True
    elif p90 >= 0.018:
        treat_as_percent = True
    elif med >= 0.12:
        treat_as_percent = True
    elif med >= 0.04 and p90 >= 0.25:
        treat_as_percent = True
    elif p90 >= 0.45 and mx <= 30.0:
        treat_as_percent = True

    if treat_as_percent:
        return (
            _scaled_copy(df),
            f"auto：按百分数列处理（|ret| 中位={med:.4f}, p90={p90:.4f}, p99={p99:.4f}）已 ÷100",
        )
    return (
        df,
        f"auto：按小数收益处理（|ret| 中位={med:.6f}, p90={p90:.6f}）未 ÷100",
    )


def load_dataset(
    path: str | Path,
    date_col: Optional[str] = None,
    close_col: Optional[str] = None,
    prediction_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    从 CSV 加载数据集。要求至少包含：日期、收盘价、预测列。
    列名默认使用 config 中的配置，也可在此传入覆盖。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    df = pd.read_csv(path)
    date_col = date_col or COL_DATE
    close_col = close_col or COL_CLOSE
    prediction_col = prediction_col or COL_PREDICTION

    # ---------- 兼容「仅预测/回测特征」类 CSV ----------
    # 这类文件通常只有：
    #   #RET, VOL_CHANGE, BA_SPREAD, ILLIQUIDITY, sprtrn, TURNOVER, expected_RET, predicted_RET
    # 没有 date/close/ret/ticker。我们用：
    #   ticker = 文件名（去掉 _test_predictions 等后缀）
    #   ret = #RET
    #   close = (1+ret).cumprod()（合成价格，仅影响份额计算的标度，不影响回报）
    #   date = 按「行号」对齐的交易日：第 i 行 = 全样本第 i 个交易日（横截面上多股同一天），
    #         不把不同 CSV 用 split_id*n 错开到不同年。各行数可不同，短样本在后段日期缺失。
    if "#RET" in df.columns and "date" not in df.columns:
        ticker = path.stem
        for suffix in ["_test_predictions", "_predictions", "_test_prediction", "_prediction", "_test"]:
            if ticker.endswith(suffix):
                ticker = ticker[: -len(suffix)]
        if not ticker:
            ticker = "single"

        df = df.copy()
        df["ticker"] = ticker
        df = df.rename(columns={"#RET": "ret"})
        df["ret"] = pd.to_numeric(df["ret"], errors="coerce").fillna(0.0)

        base_date = pd.Timestamp("2020-01-01")
        n = len(df)
        # 与日历日不同：用连续交易日，使「约 n 行 ≈ n 个交易日」
        df["date"] = pd.bdate_range(start=base_date, periods=n, freq="B")
        # 合成 close：从 1 开始累计回报
        df["close"] = (1.0 + df["ret"]).cumprod()

        # 可选列统一命名（和后续 pipeline 保持一致）
        rename_map = {
            "VOL_CHANGE": "vol_change",
            "BA_SPREAD": "ba_spread",
            "ILLIQUIDITY": "illiquidity",
            "TURNOVER": "turnover",
            "sprtrn": "sprtrn",
        }
        for old_name, new_name in rename_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        keep = ["date", "ticker", "close", "ret"]
        for c in ["vol_change", "ba_spread", "illiquidity", "turnover", "sprtrn"]:
            if c in df.columns:
                keep.append(c)
        # 若存在真实标签/预测标签列，保留给 features 直接用作 target
        for c in ["expected_RET", "predicted_RET"]:
            if c in df.columns:
                keep.append(c)
        return df[keep].sort_values(["ticker", "date"]).reset_index(drop=True).dropna(subset=["close", "ret"])

    # 兼容常见列名
    def _pick(col: str, aliases: list[str]) -> str:
        if col in df.columns:
            return col
        for a in aliases:
            if a in df.columns:
                return a
        return col

    date_col = _pick(date_col, ["Date", "datetime", "time"])
    close_col = _pick(close_col, ["Close", "price", "PRC"])
    prediction_col = _pick(prediction_col, ["pred", "Prediction", "pred_return", "RET"])

    # RET 作为已实现收益，仅用于构建目标/特征，不可作交易信号（避免未来函数）
    df = df.rename(
        columns={
            date_col: "date",
            close_col: "close",
            prediction_col: "ret",
        }
    )

    # 保留股票代码列，统一命名为 ticker（若存在 TICKER）
    if "TICKER" in df.columns and "ticker" not in df.columns:
        df = df.rename(columns={"TICKER": "ticker"})

    need = {"date", "close", "ret"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少列: {missing}。可用列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])

    # A.csv 等扩展列：成本与流动性，统一为小写列名供回测与策略使用
    optional_columns = {
        "TRAN_COST": "tran_cost",   # 买卖成本（美元/股或单笔）
        "BA_SPREAD": "ba_spread",   # 买卖价差
        "ILLIQUIDITY": "illiquidity",
        "TURNOVER": "turnover",
        "VOL_CHANGE": "vol_change",
        "sprtrn": "sprtrn",        # 市场收益
        "ASK": "ask",
        "BID": "bid",
        "MARKET_CAP": "market_cap",
    }
    for old_name, new_name in optional_columns.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    # 保留用于后续特征与回测的列（含 ticker）
    keep = ["date", "ticker", "close", "ret"] + [
        c for c in ["tran_cost", "ba_spread", "illiquidity", "turnover", "vol_change", "sprtrn", "ask", "bid", "market_cap"]
        if c in df.columns
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    # 多股票面板：按 ticker, date 排序
    sort_cols = ["date"]
    if "ticker" in df.columns:
        sort_cols = ["ticker", "date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df = df.dropna(subset=["close", "ret"])
    return df


def train_test_split(
    df: pd.DataFrame,
    test_ratio: float = TEST_RATIO,
    by_date: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分训练集与测试集（避免未来信息泄露）。
    by_date=True 时按最后 test_ratio 比例的天数切分；
    否则按行数比例切分。
    """
    n = len(df)
    if n == 0:
        return df.copy(), df.copy()
    test_size = max(1, int(n * test_ratio))
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    return train_df, test_df
