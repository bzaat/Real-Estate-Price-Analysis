from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score


def _to_numpy(x):
    """Avoid pandas indexing"""
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.asarray(x)


def _rmse(y_true, y_pred) -> float:
    y_true = _to_numpy(y_true).ravel()
    y_pred = _to_numpy(y_pred).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_kfold(model, X, y, n_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    K-Fold CV evaluation.
    IMPORTANT: pass a leakage-safe model (ideally an sklearn Pipeline with scaler inside).
    """
    X_np = _to_numpy(X)
    y_np = _to_numpy(y).ravel()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    mae_list: List[float] = []
    rmse_list: List[float] = []
    r2_list: List[float] = []

    for train_idx, val_idx in kf.split(X_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        mae_list.append(float(mean_absolute_error(y_val, pred)))
        rmse_list.append(_rmse(y_val, pred))
        r2_list.append(float(r2_score(y_val, pred)))

    return {
        "MAE_mean": float(np.mean(mae_list)),
        "MAE_std": float(np.std(mae_list)),
        "RMSE_mean": float(np.mean(rmse_list)),
        "RMSE_std": float(np.std(rmse_list)),
        "R2_mean": float(np.mean(r2_list)),
        "R2_std": float(np.std(r2_list)),
    }


def time_based_split(df, date_column: str, train_ratio: float = 0.7):
    """Sort by date_column then split into past(train) and future(test)."""
    df_sorted = df.sort_values(date_column)
    split_idx = int(len(df_sorted) * train_ratio)
    return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]


def evaluate_time_split(model, train_df, test_df, feature_cols: Sequence[str], target_col: str) -> Dict[str, float]:
    """
    Time-based split evaluation.
    IMPORTANT: pass a leakage-safe model (Pipeline recommended).
    """
    X_train = _to_numpy(train_df[feature_cols])
    y_train = _to_numpy(train_df[target_col]).ravel()
    X_test = _to_numpy(test_df[feature_cols])
    y_test = _to_numpy(test_df[target_col]).ravel()

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": _rmse(y_test, pred),
        "R2": float(r2_score(y_test, pred)),
    }
