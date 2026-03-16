# -*- coding: utf-8 -*-
"""
回测分析：IC（预测与真实收益相关性）、分组收益（预测最高组是否真的涨得最好）。
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def report_ic_and_groups(
    df: pd.DataFrame,
    pred_col: str = "prediction",
    target_col: str = "target",
    n_groups: int = 5,
) -> None:
    """
    打印 IC（pred 与 target 的相关系数）以及按 pred 分组的平均 target 收益。
    df 需含 pred_col、target_col（如未来 5 日收益）。
    """
    sub = df[[pred_col, target_col]].dropna()
    if len(sub) < 10:
        return
    ic = sub[pred_col].corr(sub[target_col])
    print(f"\n  IC（预测与未来收益相关系数）: {ic:.4f}")

    sub = sub.copy()
    sub["group"] = pd.qcut(sub[pred_col].rank(method="first"), n_groups, labels=range(1, n_groups + 1))
    grp = sub.groupby("group", observed=True)[target_col].agg(["mean", "count"])
    grp.columns = ["平均收益", "样本数"]
    print("  按预测分组平均收益（组 1=预测最低，组 5=预测最高）:")
    for g, row in grp.iterrows():
        print(f"    组 {int(g)}: {row['平均收益']:.4f}  (n={int(row['样本数'])})")
    if len(grp) >= 2:
        spread = float(grp["平均收益"].iloc[-1] - grp["平均收益"].iloc[0])
        print(f"  多空差（最高组-最低组）: {spread:.4f}")
