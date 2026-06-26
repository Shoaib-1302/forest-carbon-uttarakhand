"""
predict.py  v3
──────────────
Wall-to-wall biomass prediction. Reads feature_list.txt from train.py
to ensure exact feature match regardless of which version was trained.

Run
───
    python src/models/predict.py
"""

import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
import yaml
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS = CFG["paths"]
FEAT  = CFG["features"]
IPCC  = CFG["ipcc"]
SEQ   = CFG["seq_potential"]

MODEL_DIR = ROOT / "outputs/models"
MAP_DIR   = ROOT / PATHS["output_maps"]
STAT_DIR  = ROOT / PATHS["output_stats"]
for d in [MAP_DIR, STAT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Feature engineering — must match train.py exactly ────────────
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


def classify_seq_potential(biomass, ndvi, slope, elevation):
    thresh = SEQ["degraded_biomass_threshold"]
    zones  = np.zeros_like(biomass, dtype=np.uint8)
    c      = (biomass < thresh) | (ndvi < 0.4)
    z4 = c & (slope < 15)  & (elevation < 2000) & (elevation > 500) & (ndvi >= 0.25)
    z3 = c & ~z4 & (slope < 30) & (ndvi >= 0.15)
    z2 = c & ~z4 & ~z3 & (slope < 45) & (ndvi >= 0.05)
    z1 = c & ~z4 & ~z3 & ~z2
    zones[z1] = 1; zones[z2] = 2; zones[z3] = 3; zones[z4] = 4
    return zones


def write_raster(ref_tif, lons, lats, values, out_path,
                 nodata=-9999.0, dtype="float32"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(ref_tif) as src:
        profile      = src.profile.copy()
        transform    = src.transform
        nrows, ncols = src.height, src.width
    out_arr = np.full((nrows, ncols), nodata, dtype=np.float32)
    rows, cols = rowcol(transform, lons, lats)
    rows = np.array(rows); cols = np.array(cols)
    valid = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    out_arr[rows[valid], cols[valid]] = values[valid].astype(np.float32)
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)
    print(f"  Saved: {out_path}")


def main():
    # ── Load model ────────────────────────────────────────────────
    # Prefer best_model.joblib (saved by train.py), fall back to xgb
    for fname in ["best_model.joblib", "xgb_model.joblib"]:
        model_path = MODEL_DIR / fname
        if model_path.exists():
            model = joblib.load(model_path)
            print(f"Model loaded: {fname}")
            break
    else:
        raise FileNotFoundError(f"No model found in {MODEL_DIR}. Run train.py first.")

    # ── Load feature list saved by train.py ──────────────────────
    feat_path = MODEL_DIR / "feature_list.txt"
    if feat_path.exists():
        features = feat_path.read_text().splitlines()
        print(f"Feature list: {len(features)} features from {feat_path.name}")
    else:
        features = FEAT["model_features"]
        print("feature_list.txt not found — using config features only")

    # ── Load and engineer prediction grid ────────────────────────
    grid_path = ROOT / PATHS["prediction_grid"]
    if not grid_path.exists():
        raise FileNotFoundError(f"Prediction grid not found: {grid_path}")

    print(f"Loading prediction grid ...")
    df = pd.read_csv(grid_path)
    print(f"  {len(df):,} pixels")

    df = add_engineered(df)

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Features missing from prediction grid after engineering: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df[features].values

    # ── Predict in chunks (18M rows — avoid OOM) ─────────────────
    print("Predicting AGBD in chunks ...")
    CHUNK = 500_000
    biomass_log = np.empty(len(df), dtype=np.float32)
    for i in range(0, len(df), CHUNK):
        biomass_log[i:i+CHUNK] = model.predict(X[i:i+CHUNK])
        if (i // CHUNK) % 5 == 0:
            print(f"  {min(i+CHUNK, len(df)):,} / {len(df):,}")

    biomass = np.expm1(np.clip(biomass_log, 0, None)).astype(np.float32)

    df["biomass_tpha"] = biomass
    df["carbon_tcha"]  = biomass * IPCC["biomass_to_carbon"]
    df["co2e_tco2eha"] = df["carbon_tcha"] * IPCC["carbon_to_co2e"]
    df["seq_zone"]     = classify_seq_potential(
        biomass,
        df["NDVI"].values,
        df["slope"].values,
        df["elevation"].values,
    )

    print(f"\nPrediction summary:")
    print(f"  Biomass — mean: {df['biomass_tpha'].mean():.1f}  "
          f"max: {df['biomass_tpha'].max():.1f} t/ha")
    print(f"  Carbon  — mean: {df['carbon_tcha'].mean():.1f} tC/ha")
    print(f"  CO2e    — mean: {df['co2e_tco2eha'].mean():.1f} tCO2e/ha")

    # ── Write rasters ─────────────────────────────────────────────
    ref_tif = ROOT / PATHS["sentinel_stack"]
    if ref_tif.exists():
        print("\nWriting output rasters ...")
        lons = df["longitude"].values
        lats = df["latitude"].values
        write_raster(ref_tif, lons, lats, df["biomass_tpha"].values,
                     ROOT / PATHS["biomass_tif"])
        write_raster(ref_tif, lons, lats, df["carbon_tcha"].values,
                     ROOT / PATHS["carbon_tif"])
        write_raster(ref_tif, lons, lats, df["co2e_tco2eha"].values,
                     ROOT / PATHS["co2e_tif"])
        write_raster(ref_tif, lons, lats, df["seq_zone"].values.astype(np.float32),
                     ROOT / PATHS["seq_tif"], nodata=0, dtype="uint8")
    else:
        print("\nSentinel-2 GeoTIFF not found — saving prediction CSV ...")
        pred_out = MAP_DIR / "predictions.csv"
        df[["latitude","longitude","biomass_tpha","carbon_tcha",
            "co2e_tco2eha","seq_zone"]].to_csv(pred_out, index=False)
        print(f"  Saved: {pred_out}")

    # ── District summary ──────────────────────────────────────────
    PIXEL_HA    = (20 * 20) / 10_000
    forest_ha   = len(df[df["biomass_tpha"] > 0]) * PIXEL_HA
    total_c_mtc = (df["carbon_tcha"].sum()  * PIXEL_HA) / 1e6
    total_co2_m = (df["co2e_tco2eha"].sum() * PIXEL_HA) / 1e6

    summary = {
        "District":                   "Nainital",
        "State":                      "Uttarakhand",
        "Forest Area (km2)":          round(forest_ha / 100, 1),
        "Forest Area (ha)":           round(forest_ha, 0),
        "Mean Biomass (t/ha)":        round(float(df["biomass_tpha"].mean()), 1),
        "Median Biomass (t/ha)":      round(float(df["biomass_tpha"].median()), 1),
        "Mean Carbon Stock (tC/ha)":  round(float(df["carbon_tcha"].mean()), 1),
        "Total Carbon Stock (MtC)":   round(total_c_mtc, 2),
        "Total CO2e Stored (MtCO2e)": round(total_co2_m, 2),
        "Model":                      type(model).__name__,
        "R2_log":                     0.649,
        "MAE_tpha":                   56.3,
    }

    print("\n── District Summary ─────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")

    pd.DataFrame([summary]).to_csv(STAT_DIR / "district_summary.csv", index=False)
    print(f"\nSaved: {STAT_DIR / 'district_summary.csv'}")
    print("Next step:  python src/visualization/maps.py")


if __name__ == "__main__":
    main()