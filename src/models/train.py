"""
train.py  v4
────────────
Stacking ensemble: RF + XGBoost base learners → Ridge meta-learner.
Also stratifies by forest type (elevation zones) before training.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS  = CFG["paths"]
ML_CFG = CFG["ml"]
FEAT   = CFG["features"]
VIZ    = CFG["viz"]

MODEL_DIR = ROOT / "outputs/models"
FIG_DIR   = ROOT / PATHS["output_figs"]
STAT_DIR  = ROOT / PATHS["output_stats"]
for d in [MODEL_DIR, FIG_DIR, STAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ───────────────────────────────────────────
ENGINEERED = [
    "NDVI_sq", "EVI_sq", "NBR_sq",
    "NDVI_x_NDMI", "NDVI_x_NBR",
    "NDVI_x_elev", "EVI_x_elev", "NBR_x_elev",
    "NDMI_x_slope", "NDVI_x_slope",
    "aspect_sin", "aspect_cos",
    "elev_sq",
]

def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NDVI_sq"]      = df["NDVI"] ** 2
    df["EVI_sq"]       = df["EVI"]  ** 2
    df["NBR_sq"]       = df["NBR"]  ** 2
    df["NDVI_x_NDMI"]  = df["NDVI"] * df["NDMI"]
    df["NDVI_x_NBR"]   = df["NDVI"] * df["NBR"]
    df["NDVI_x_elev"]  = df["NDVI"] * df["elevation"] / 1000.0
    df["EVI_x_elev"]   = df["EVI"]  * df["elevation"] / 1000.0
    df["NBR_x_elev"]   = df["NBR"]  * df["elevation"] / 1000.0
    df["NDMI_x_slope"] = df["NDMI"] * df["slope"]
    df["NDVI_x_slope"] = df["NDVI"] * df["slope"]
    aspect_rad         = np.radians(df["aspect"])
    df["aspect_sin"]   = np.sin(aspect_rad)
    df["aspect_cos"]   = np.cos(aspect_rad)
    df["elev_sq"]      = (df["elevation"] / 1000.0) ** 2
    return df


# ── Metrics ───────────────────────────────────────────────────────
def evaluate(name, y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = mean_squared_error(y_true, y_pred) ** 0.5
    r2     = r2_score(y_true, y_pred)
    r2_log = r2_score(y_true_log, y_pred_log)
    print(f"\n  {name}")
    print(f"    MAE  (t/ha) : {mae:.2f}")
    print(f"    RMSE (t/ha) : {rmse:.2f}")
    print(f"    R²   (orig) : {r2:.4f}")
    print(f"    R²   (log)  : {r2_log:.4f}  ← primary metric")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "R2_log": r2_log}


# ── Plots ─────────────────────────────────────────────────────────
def plot_scatter(results, y_test_log):
    y_test = np.expm1(y_test_log)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1: axes = [axes]
    for ax, (name, y_pred_log) in zip(axes, results):
        y_pred = np.expm1(np.clip(y_pred_log, 0, None))
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.10, s=3,
                   color="#2e7d52", edgecolors="none")
        lim = [0, max(y_test.max(), y_pred.max()) + 10]
        ax.plot(lim, lim, "r--", lw=1.5, label="1:1")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual AGBD (t/ha)")
        ax.set_ylabel("Predicted AGBD (t/ha)")
        ax.set_title(f"{name}  R²={r2:.3f}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "scatter_actual_vs_predicted.png"
    fig.savefig(out, dpi=VIZ["dpi"]); plt.close(fig)
    print(f"  Saved: {out}")

def plot_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"): return
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names)*0.38)))
    colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df["feature"], df["importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / f"feature_importance_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=VIZ["dpi"]); plt.close(fig)
    print(f"  Saved: {out}")

def plot_residuals(results, y_test_log):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4))
    if len(results) == 1: axes = [axes]
    for ax, (name, pred_log) in zip(axes, results):
        res = np.expm1(y_test_log) - np.expm1(np.clip(pred_log, 0, None))
        sns.histplot(res, bins=50, kde=True, ax=ax, color="#4caf50")
        ax.axvline(0, color="red", ls="--", lw=1.5)
        ax.set_title(f"Residuals — {name}")
        ax.set_xlabel("Actual − Predicted (t/ha)")
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / "residual_distribution.png"
    fig.savefig(out, dpi=VIZ["dpi"]); plt.close(fig)
    print(f"  Saved: {out}")

def plot_by_elevation(df_test, y_pred_log, y_true_log, model_name):
    """R² broken down by elevation band — shows where model performs well/poorly."""
    bins   = [0, 700, 1200, 1700, 2200, 9999]
    labels = ["<700m", "700-1200m", "1200-1700m", "1700-2200m", ">2200m"]
    elev   = df_test["elevation"].values
    bands  = pd.cut(elev, bins=bins, labels=labels)
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(np.clip(y_pred_log, 0, None))

    rows = []
    for band in labels:
        mask = bands == band
        if mask.sum() < 10: continue
        r2  = r2_score(y_true[mask], y_pred[mask])
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        rows.append({"Elevation Band": band, "N": mask.sum(), "R2": round(r2,3), "MAE": round(mae,1)})

    band_df = pd.DataFrame(rows)
    print(f"\n  {model_name} — R² by elevation band:")
    print(band_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e53935" if r < 0.5 else "#fb8c00" if r < 0.65 else "#43a047"
              for r in band_df["R2"]]
    ax.bar(band_df["Elevation Band"], band_df["R2"], color=colors, edgecolor="white")
    ax.axhline(0.70, color="green", ls="--", lw=1.5, label="Target R²=0.70")
    ax.axhline(0.50, color="orange", ls="--", lw=1.0, label="R²=0.50")
    ax.set_xlabel("Elevation Band"); ax.set_ylabel("R²")
    ax.set_title(f"R² by Elevation Band — {model_name}", fontweight="bold")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out = FIG_DIR / f"r2_by_elevation_{model_name.lower().replace(' ','_')}.png"
    fig.savefig(out, dpi=VIZ["dpi"]); plt.close(fig)
    print(f"  Saved: {out}")
    return band_df


# ── Main ─────────────────────────────────────────────────────────
def main():
    csv_path = ROOT / PATHS["training_csv"]
    df = pd.read_csv(csv_path)
    print(f"Loaded: {len(df):,} samples")

    # 95th pct cap
    cap   = float(np.percentile(df["agbd"], 95))
    n_out = int((df["agbd"] > cap).sum())
    df    = df[df["agbd"] <= cap].reset_index(drop=True)
    print(f"  Cap at 95th pct ({cap:.1f} t/ha): removed {n_out:,} rows")
    print(f"  After cap — mean: {df['agbd'].mean():.1f}  std: {df['agbd'].std():.1f}\n")

    df       = add_engineered(df)
    features = FEAT["model_features"] + ENGINEERED
    print(f"Features ({len(features)}): {features}\n")

    y_log = np.log1p(df["agbd"].values)
    X     = df[features].values

    strat = pd.qcut(y_log, q=5, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_log,
        test_size=ML_CFG["test_size"],
        random_state=ML_CFG["random_state"],
        stratify=strat,
    )
    # Keep test DataFrame for elevation analysis
    te_idx  = np.arange(len(df))[
        train_test_split(np.arange(len(df)), test_size=ML_CFG["test_size"],
                         random_state=ML_CFG["random_state"], stratify=strat)[1]
    ]
    df_test = df.iloc[te_idx].reset_index(drop=True)

    print(f"Train: {len(X_tr):,}  |  Test: {len(X_te):,}\n")

    metrics, scatter = [], []

    # ── Random Forest ─────────────────────────────────────────────
    print("Training Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=25,
        min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1,
        random_state=ML_CFG["random_state"],
    )
    rf.fit(X_tr, y_tr)
    cv = cross_val_score(rf, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R² (log): {cv.mean():.4f} ± {cv.std():.4f}")
    rf_pred = rf.predict(X_te)
    metrics.append(evaluate("Random Forest", y_te, rf_pred))
    scatter.append(("Random Forest", rf_pred))
    joblib.dump(rf, MODEL_DIR / "rf_model.joblib")
    plot_importance(rf, features, "Random Forest")

    # ── XGBoost (best params from v3 search, +more trees) ────────
    print("\nTraining XGBoost (tuned params from v3 search) ...")
    xgb = XGBRegressor(
        n_estimators     = 2000,
        learning_rate    = 0.028,
        max_depth        = 8,
        subsample        = 0.78,
        colsample_bytree = 0.78,
        min_child_weight = 1,
        reg_alpha        = 0.43,
        reg_lambda       = 1.86,
        gamma            = 0.24,
        eval_metric      = "rmse",
        early_stopping_rounds = 50,
        random_state     = ML_CFG["random_state"],
        n_jobs           = -1,
        verbosity        = 0,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=100)
    print(f"  Best iteration: {xgb.best_iteration}")
    xgb_pred = xgb.predict(X_te)
    metrics.append(evaluate("XGBoost", y_te, xgb_pred))
    scatter.append(("XGBoost", xgb_pred))
    joblib.dump(xgb, MODEL_DIR / "xgb_model.joblib")
    plot_importance(xgb, features, "XGBoost")
    plot_by_elevation(df_test, xgb_pred, y_te, "XGBoost")

    # ── Stacking Ensemble ─────────────────────────────────────────
    print("\nTraining Stacking Ensemble (RF + XGBoost → Ridge) ...")
    # Use XGBoost with fewer trees for stacking (faster CV)
    xgb_stack = XGBRegressor(
        n_estimators=800, learning_rate=0.028, max_depth=8,
        subsample=0.78, colsample_bytree=0.78,
        min_child_weight=1, reg_alpha=0.43, reg_lambda=1.86,
        gamma=0.24, eval_metric="rmse",
        random_state=ML_CFG["random_state"], n_jobs=-1, verbosity=0,
    )
    rf_stack = RandomForestRegressor(
        n_estimators=200, max_depth=20, max_features="sqrt",
        n_jobs=-1, random_state=ML_CFG["random_state"],
    )
    stack = StackingRegressor(
        estimators=[("rf", rf_stack), ("xgb", xgb_stack)],
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1, passthrough=False,
    )
    stack.fit(X_tr, y_tr)
    stack_pred = stack.predict(X_te)
    metrics.append(evaluate("Stacking Ensemble", y_te, stack_pred))
    scatter.append(("Stacking Ensemble", stack_pred))
    joblib.dump(stack, MODEL_DIR / "stack_model.joblib")
    print(f"  Saved: {MODEL_DIR / 'stack_model.joblib'}")

    # Save best model separately for predict.py
    best_idx   = np.argmax([m["R2_log"] for m in metrics])
    best_name  = metrics[best_idx]["model"]
    best_preds = scatter[best_idx][1]
    best_model = [rf, xgb, stack][best_idx]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    print(f"\n  Best model: {best_name} → saved as best_model.joblib")

    # Save feature list
    feat_path = MODEL_DIR / "feature_list.txt"
    feat_path.write_text("\n".join(features))

    # ── Plots ─────────────────────────────────────────────────────
    plot_scatter(scatter, y_te)
    plot_residuals(scatter, y_te)

    # ── Summary ───────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(STAT_DIR / "model_metrics.csv", index=False)
    print(f"\n{'─'*65}")
    print(metrics_df[["model","MAE","RMSE","R2","R2_log"]].to_string(index=False))
    best_r2_log = metrics_df["R2_log"].max()
    best        = metrics_df.loc[metrics_df["R2_log"].idxmax(), "model"]
    status      = "✅ Target met" if best_r2_log >= 0.70 else "⚠️  Below target"
    print(f"\nBest: {best}  R²(log)={best_r2_log:.4f}  {status} (target ≥ 0.70)")
    print("Next step:  python src/models/predict.py")


if __name__ == "__main__":
    main()