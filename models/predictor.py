# -*- coding: utf-8 -*-
"""
预测模型：用历史特征预测 t+1 收益率，仅使用 t 时刻及之前信息。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Dict

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# XGBoost 仅在选用 --model xgb 时再导入，避免未装 libomp 时连 ridge/rf 都无法运行


def train_predictor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "ridge",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    在训练集上训练预测器。X_train 为特征，y_train 为目标（下一期收益）。
    model_type: "ridge" | "rf"（随机森林）| "xgb"（XGBoost）
    """
    if model_type in {"ridge", "rf"} and not HAS_SKLEARN:
        raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
    if model_type == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError(
                "使用 XGBoost 需先安装: pip install xgboost；Mac 上还需安装 OpenMP: brew install libomp"
            ) from e

    X = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y_train
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    scaler: Any = None
    X_model = X

    if model_type in {"ridge", "rf"}:
        scaler = StandardScaler()
        X_model = scaler.fit_transform(X)

    if model_type == "ridge":
        model = Ridge(alpha=1.0, **kwargs)
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            n_jobs=-1,
            **kwargs,
        )
    elif model_type == "xgb":
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            objective="reg:squarederror",
            **kwargs,
        )
    else:
        model = Ridge(alpha=1.0, **kwargs)

    model.fit(X_model, y)
    importances = getattr(model, "feature_importances_", None)
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "feature_importances": importances,
    }


def predict(fitted: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """用已拟合的模型对 X 预测，返回一维数组。"""
    names = fitted["feature_names"]
    X = (
        X[[c for c in names if c in X.columns]]
        .reindex(columns=names)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    scaler = fitted.get("scaler")
    if scaler is not None:
        X_model = scaler.transform(X)
    else:
        X_model = X
    return fitted["model"].predict(X_model)


def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_xgb: bool = False,
) -> Dict[str, Any]:
    """
    模型集成：Ridge + RF（+ 可选 XGB）取平均，提高稳定性。
    返回的 fitted 带 "ensemble" 标记，predict_ensemble 使用。
    """
    if not HAS_SKLEARN:
        raise ImportError("需要 scikit-learn")
    X = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    y = y_train
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = []
    # Ridge
    m1 = Ridge(alpha=1.0)
    m1.fit(X_scaled, y)
    models.append(("ridge", m1, True))
    # RF
    m2 = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=20, n_jobs=-1)
    m2.fit(X_scaled, y)
    models.append(("rf", m2, True))
    # XGB (optional)
    if use_xgb:
        try:
            from xgboost import XGBRegressor
            m3 = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, objective="reg:squarederror")
            m3.fit(X_scaled, y)
            models.append(("xgb", m3, True))
        except Exception:
            pass
    imp = getattr(models[1][1], "feature_importances_", None)  # RF importance
    return {
        "ensemble": True,
        "models": models,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "feature_importances": imp,
    }


def predict_ensemble(fitted: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """集成模型预测：多个模型预测取平均。"""
    names = fitted["feature_names"]
    X = (
        X[[c for c in names if c in X.columns]]
        .reindex(columns=names)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    X_scaled = fitted["scaler"].transform(X)
    preds = [m.predict(X_scaled) for _, m, _ in fitted["models"]]
    return np.mean(preds, axis=0)
