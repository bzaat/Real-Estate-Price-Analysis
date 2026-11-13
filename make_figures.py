"""Regenerate the README figures from melb_data_cleaned.csv.

Settings mirror notebook "6. Regression Analysis.ipynb":
log_price target, 80/20 split (random_state=42), 5-fold CV on the training
split (KFold shuffle, random_state=42), Suburb target encoding
(smoothing=10, min_samples_leaf=20) fitted inside the pipeline, and the same
model hyperparameters. The shuffled-suburb control uses a fixed seed here
(the notebook's np.random.permutation call is unseeded).

All numbers produced here are RE-COMPUTED values. They can differ slightly
from the values reported in the paper (see README).

Run:  uv run python make_figures.py
Output: figures/*.png and figures/figures_data.json
"""

from __future__ import annotations

import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from category_encoders import TargetEncoder  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Settings (copied from notebook 6)
# ---------------------------------------------------------------------------
SEED = 42
DATA_PATH = Path("melb_data_cleaned.csv")
OUT_DIR = Path("figures")
DPI = 150

TARGET = "log_price"
BASE_FEATURES = [
    "Rooms", "Bathroom", "Car", "Landsize", "Distance",
    "Type_h", "Type_t", "Type_u",
    "Region_NM", "Region_WM", "Region_SM", "Region_EM", "Region_SEM",
]
SUBURB_FEATURES = [
    "Rooms", "Bathroom", "Car", "Landsize", "Distance",
    "Suburb",
    "Type_h", "Type_t", "Type_u",
    "Region_NM", "Region_WM", "Region_SM", "Region_EM", "Region_SEM",
]
LINEAR_MODELS = {"Linear Regression", "Ridge", "Lasso"}


def make_encoder() -> TargetEncoder:
    return TargetEncoder(cols=["Suburb"], smoothing=10, min_samples_leaf=20)


def rf_tuned() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=8,
        min_samples_split=20, random_state=SEED, n_jobs=-1,
    )


def xgb_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
    )


def all_models() -> dict:
    """The 7 models of the paper. Linear/Ridge/Lasso as in notebook 6 cell 1;
    RF/ExtraTrees/XGBoost/LightGBM as in notebook 6 cell 7 (tuned RF)."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": rf_tuned(),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=8,
            min_samples_split=20, random_state=SEED, n_jobs=-1,
        ),
        "XGBoost": xgb_model(),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            n_jobs=-1, verbose=-1,
        ),
    }


def build_pipeline(model, linear: bool = False, with_suburb: bool = True) -> Pipeline:
    steps = []
    if with_suburb:
        steps.append(("encoder", make_encoder()))
    if linear:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save(fig, name: str) -> str:
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")
    return str(path)


def r3(x: float) -> float:
    return round(float(x), 4)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_price_distribution(df: pd.DataFrame, data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(df["Price"] / 1e6, bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Raw sale price (right-skewed)")
    axes[0].set_xlabel("Price (million AUD)")
    axes[0].set_ylabel("Number of sales")
    sns.histplot(df["log_price"], bins=60, ax=axes[1], color="#55A868")
    axes[1].set_title("log(Price) (model target)")
    axes[1].set_xlabel("log(Price)")
    axes[1].set_ylabel("Number of sales")
    fig.suptitle(f"Distribution of sale prices (n = {len(df):,})", fontsize=15)
    fig.tight_layout()
    save(fig, "01_price_distribution.png")
    data["01_price_distribution"] = {
        "n": int(len(df)),
        "price_median": float(df["Price"].median()),
        "price_skew": r3(df["Price"].skew()),
        "log_price_skew": r3(df["log_price"].skew()),
    }


def fig_median_by_region(df: pd.DataFrame, data: dict) -> None:
    g = (df.groupby("Regionname")["Price"]
           .agg(median="median", n="size")
           .sort_values("median"))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.barh(g.index, g["median"] / 1e6, color="#4C72B0")
    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{row['median'] / 1e6:.2f}M  (n={int(row['n']):,})",
                va="center", fontsize=11)
    ax.set_xlabel("Median sale price (million AUD)")
    ax.set_ylabel("")
    ax.set_xlim(0, g["median"].max() / 1e6 * 1.35)
    ax.set_title("Median sale price by region")
    fig.tight_layout()
    save(fig, "02_median_price_by_region.png")
    data["02_median_price_by_region"] = {
        k: {"median_price": float(v["median"]), "n": int(v["n"])}
        for k, v in g.iterrows()
    }


def fig_price_vs_distance(df: pd.DataFrame, data: dict) -> None:
    labels = {"h": "House (h)", "t": "Townhouse (t)", "u": "Unit (u)"}
    plot_df = df.assign(Property=df["Type"].map(labels))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df, x="Distance", y="log_price", hue="Property",
        hue_order=list(labels.values()), alpha=0.35, s=14, linewidth=0, ax=ax,
    )
    ax.set_xlabel("Distance to Melbourne CBD (km)")
    ax.set_ylabel("log(Price)")
    ax.set_title("log(Price) vs distance to CBD, by property type")
    leg = ax.legend(title="Property type", markerscale=2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.tight_layout()
    save(fig, "03_logprice_vs_distance.png")
    data["03_logprice_vs_distance"] = {
        "corr_distance_logprice_by_type": {
            labels[t]: r3(g["Distance"].corr(g["log_price"]))
            for t, g in df.groupby("Type")
        },
        "n_by_type": {labels[t]: int(n) for t, n in df["Type"].value_counts().items()},
    }


def fig_correlation(df: pd.DataFrame, data: dict) -> None:
    cols = ["Price", "log_price", "Rooms", "Bedroom2", "Bathroom", "Car",
            "Distance", "Landsize"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation of numeric features")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save(fig, "04_correlation_heatmap.png")
    data["04_correlation_heatmap"] = corr.round(4).to_dict()


def fig_model_comparison(X_train, y_train, data: dict) -> None:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    rows = []
    for name, model in all_models().items():
        pipe = build_pipeline(model, linear=name in LINEAR_MODELS)
        s = cross_validate(pipe, X_train, y_train, cv=kf, scoring=scoring)
        rows.append({
            "model": name,
            "r2_mean": s["test_r2"].mean(), "r2_std": s["test_r2"].std(),
            "mae_mean": -s["test_mae"].mean(), "mae_std": s["test_mae"].std(),
            "rmse_mean": -s["test_rmse"].mean(), "rmse_std": s["test_rmse"].std(),
            "r2_folds": s["test_r2"].tolist(),
        })
        print(f"  CV {name:<18} R2 {rows[-1]['r2_mean']:.3f} ± {rows[-1]['r2_std']:.3f}")
    res = pd.DataFrame(rows)

    colors = ["#8C8C8C"] * 3 + ["#4C72B0"] * 2 + ["#DD8452"] * 2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(res["model"], res["r2_mean"], yerr=res["r2_std"], capsize=5,
           color=colors, edgecolor="black", linewidth=0.6)
    for i, r in res.iterrows():
        ax.text(i, r["r2_mean"] + r["r2_std"] + 0.004, f"{r['r2_mean']:.3f}",
                ha="center", fontsize=11)
    ax.set_ylim(0.65, 0.88)
    ax.set_ylabel("5-fold CV R² (log-price)\n(y-axis starts at 0.65)")
    ax.set_title("Model comparison: 5-fold CV R² on the training split (mean ± std, re-computed)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#8C8C8C", label="Linear"),
                       Patch(color="#4C72B0", label="Bagging"),
                       Patch(color="#DD8452", label="Boosting")],
              loc="upper left")
    fig.tight_layout()
    save(fig, "05_model_comparison_cv_r2.png")
    data["05_model_comparison_cv"] = {
        "note": "Random Forest = tuned config (300 trees, max_depth=12, "
                "min_samples_leaf=8, min_samples_split=20) from notebook 6 cell 7",
        "results": [{k: (r3(v) if isinstance(v, float) else
                         [r3(x) for x in v] if isinstance(v, list) else v)
                     for k, v in row.items()} for row in rows],
    }


def fig_suburb_ablation(df: pd.DataFrame, data: dict) -> None:
    y = df[TARGET]
    settings = {}

    # Setting A: without suburb
    X = df[BASE_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    p = build_pipeline(rf_tuned(), with_suburb=False).fit(Xtr, ytr)
    settings["Without suburb"] = {"train_r2": r2_score(ytr, p.predict(Xtr)),
                                  **metrics(yte, p.predict(Xte))}

    # Setting B: with suburb (target encoded)
    X = df[SUBURB_FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    p = build_pipeline(rf_tuned()).fit(Xtr, ytr)
    settings["With suburb"] = {"train_r2": r2_score(ytr, p.predict(Xtr)),
                               **metrics(yte, p.predict(Xte))}

    # Setting C: suburb labels shuffled (seeded) before the split
    Xs = df[SUBURB_FEATURES].copy()
    Xs["Suburb"] = np.random.RandomState(SEED).permutation(Xs["Suburb"].values)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=SEED)
    p = build_pipeline(rf_tuned()).fit(Xtr, ytr)
    settings["Shuffled suburb"] = {"train_r2": r2_score(ytr, p.predict(Xtr)),
                                   **metrics(yte, p.predict(Xte))}

    for k, v in settings.items():
        print(f"  RF {k:<16} train R2 {v['train_r2']:.3f}  test R2 {v['r2']:.3f}"
              f"  MAE {v['mae']:.3f}  RMSE {v['rmse']:.3f}")

    names = list(settings)
    test_r2 = [settings[n]["r2"] for n in names]
    train_r2 = [settings[n]["train_r2"] for n in names]
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(13, 5),
                                  gridspec_kw={"width_ratios": [1.2, 1]})
    colors = ["#8C8C8C", "#4C72B0", "#C44E52"]
    ax.bar(names, test_r2, color=colors, edgecolor="black", linewidth=0.6,
           label="Test R²")
    ax.scatter(names, train_r2, color="black", marker="D", zorder=3,
               label="Train R²")
    for i, v in enumerate(test_r2):
        ax.text(i, v - 0.012, f"{v:.3f}", ha="center", va="top",
                color="white", fontsize=12, fontweight="bold")
    ax.set_ylim(0.78, 0.89)
    ax.set_ylabel("R² (log-price)\n(y-axis starts at 0.78)")
    ax.set_title("Random Forest hold-out R²")
    ax.legend(loc="upper left")

    axt.axis("off")
    cell_text = [[f"{settings[n]['train_r2']:.3f}", f"{settings[n]['r2']:.3f}",
                  f"{settings[n]['mae']:.3f}", f"{settings[n]['rmse']:.3f}"]
                 for n in names]
    tbl = axt.table(cellText=cell_text, rowLabels=names,
                    colLabels=["Train R²", "Test R²", "MAE", "RMSE"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.2)
    axt.set_title("Hold-out metrics (log-price scale)")
    fig.suptitle("Suburb ablation and shuffled-suburb control (re-computed, 80/20 split)",
                 fontsize=15)
    fig.tight_layout()
    save(fig, "06_suburb_ablation.png")
    data["06_suburb_ablation"] = {
        "shuffle": f"np.random.RandomState({SEED}).permutation(Suburb) before the split",
        "results": {k: {m: r3(v) for m, v in d.items()} for k, d in settings.items()},
    }


def fig_pred_vs_actual(X_train, X_test, y_train, y_test, data: dict) -> None:
    p = build_pipeline(xgb_model()).fit(X_train, y_train)
    y_pred = p.predict(X_test)
    m = metrics(y_test, y_pred)
    print(f"  XGBoost hold-out R2 {m['r2']:.3f} MAE {m['mae']:.3f} RMSE {m['rmse']:.3f}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, s=10, alpha=0.4, color="#4C72B0", linewidths=0)
    lo = min(y_test.min(), y_pred.min()) - 0.1
    hi = max(y_test.max(), y_pred.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual log(Price)")
    ax.set_ylabel("Predicted log(Price)")
    ax.set_title(f"XGBoost: predicted vs actual on the hold-out set (n = {len(y_test):,})")
    ax.text(0.04, 0.96,
            f"R² = {m['r2']:.3f}\nMAE = {m['mae']:.3f}\nRMSE = {m['rmse']:.3f}",
            transform=ax.transAxes, va="top", fontsize=12,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "07_predicted_vs_actual.png")
    data["07_predicted_vs_actual"] = {
        "model": "XGBoost, fitted on the 80% training split (real Suburb labels)",
        "n_test": int(len(y_test)),
        **{k: r3(v) for k, v in m.items()},
    }


def fig_shap(df: pd.DataFrame, data: dict) -> None:
    # As in notebook 6 cell 12: XGBoost pipeline fitted on the full data.
    X = df[SUBURB_FEATURES]
    y = df[TARGET]
    p = build_pipeline(xgb_model()).fit(X, y)
    X_enc = p.named_steps["encoder"].transform(X).astype(float)
    explainer = shap.TreeExplainer(p.named_steps["model"])
    sv = explainer.shap_values(X_enc)

    shap.summary_plot(sv, X_enc, show=False, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP summary (XGBoost, all 10,480 sales)", fontsize=14)
    fig.axes[0].set_xlabel("SHAP value (impact on predicted log(Price))")
    save(fig, "08a_shap_beeswarm.png")

    shap.summary_plot(sv, X_enc, plot_type="bar", show=False, plot_size=(9, 6.5),
                      color="#4C72B0")
    fig = plt.gcf()
    fig.axes[0].set_title("Mean |SHAP| feature importance (XGBoost)", fontsize=14)
    fig.axes[0].set_xlabel("mean(|SHAP value|) on log(Price)")
    save(fig, "08b_shap_bar.png")

    imp = (pd.Series(np.abs(sv).mean(axis=0), index=X_enc.columns)
             .sort_values(ascending=False))
    print("  SHAP top 4:", list(imp.index[:4]))
    data["08_shap"] = {
        "model": "XGBoost pipeline fitted on the full dataset (as in notebook 6)",
        "mean_abs_shap": {k: r3(v) for k, v in imp.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                         "axes.labelsize": 12})
    sns.set_theme(style="whitegrid", font_scale=1.05)
    np.random.seed(SEED)

    df = pd.read_csv(DATA_PATH)
    data: dict = {
        "source": "re-computed by make_figures.py (not the paper-reported values)",
        "seed": SEED,
        "data": str(DATA_PATH),
        "n_rows": int(len(df)),
        "python": platform.python_version(),
        "versions": {pkg: version(pkg) for pkg in [
            "pandas", "numpy", "scikit-learn", "category_encoders", "xgboost",
            "lightgbm", "shap", "matplotlib", "seaborn"]},
    }

    fig_price_distribution(df, data)
    fig_median_by_region(df, data)
    fig_price_vs_distance(df, data)
    fig_correlation(df, data)

    X = df[SUBURB_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    print("5-fold CV model comparison ...")
    fig_model_comparison(X_train, y_train, data)
    print("Suburb ablation ...")
    fig_suburb_ablation(df, data)
    print("Predicted vs actual ...")
    fig_pred_vs_actual(X_train, X_test, y_train, y_test, data)
    print("SHAP ...")
    fig_shap(df, data)

    with open(OUT_DIR / "figures_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"saved {OUT_DIR / 'figures_data.json'}")


if __name__ == "__main__":
    main()
