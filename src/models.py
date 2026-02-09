# models.py
from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def make_mean_baseline(strategy: str = "mean") -> DummyRegressor:
    """
    Baseline: Predict-Mean (naive).
    For regression, strategy typically "mean" (default) or "median".
    """
    return DummyRegressor(strategy=strategy)


def make_ols() -> Pipeline:
    """
    OLS Linear Regression with scaling (scale helps coefficient comparability).
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_ridge(alpha: float = 1.0, random_state: Optional[int] = None) -> Pipeline:
    """
    Ridge Regression (L2).
    alpha: regularization strength.
    """
    # Ridge supports random_state only for some solvers; keep generic.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_lasso(alpha: float = 0.01, random_state: int = 42, max_iter: int = 10000) -> Pipeline:
    """
    Lasso Regression (L1).
    alpha: regularization strength. Often needs tuning.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, random_state=random_state, max_iter=max_iter)),
        ]
    )


def make_svr_linear(C: float = 1.0, epsilon: float = 0.1) -> Pipeline:
    """
    SVR with linear kernel.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="linear", C=C, epsilon=epsilon)),
        ]
    )


def make_svr_rbf(C: float = 10.0, epsilon: float = 0.1, gamma: str | float = "scale") -> Pipeline:
    """
    SVR with RBF kernel (nonlinear).
    gamma: "scale" (default) or "auto" or numeric.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
        ]
    )


def get_models(
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.01,
    svr_linear_C: float = 1.0,
    svr_linear_epsilon: float = 0.1,
    svr_rbf_C: float = 10.0,
    svr_rbf_epsilon: float = 0.1,
    svr_rbf_gamma: str | float = "scale",
) -> Dict[str, Any]:
    """
    Convenience registry for benchmarking.
    Returns a dict name -> sklearn estimator (Pipeline or model).
    """
    return {
        "Predict-Mean": make_mean_baseline("mean"),
        "OLS": make_ols(),
        "Ridge": make_ridge(alpha=ridge_alpha),
        "Lasso": make_lasso(alpha=lasso_alpha),
        "SVR-Linear": make_svr_linear(C=svr_linear_C, epsilon=svr_linear_epsilon),
        "SVR-RBF": make_svr_rbf(C=svr_rbf_C, epsilon=svr_rbf_epsilon, gamma=svr_rbf_gamma),
    }
