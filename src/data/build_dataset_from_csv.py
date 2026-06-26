"""
build_dataset_from_csv.py
─────────────────────────
Reads the GEDI CSV exported from GEE and produces:
  data/processed/training_dataset.csv
  data/processed/prediction_grid.csv

Run from project root:
    python src/data/build_dataset_from_csv.py
"""

import sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ── Make sure repo root is on sys.path ────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS = CFG["paths"]
FEAT  = CFG["features"]
GEDI  = CFG["gedi"]
BBOX  = CFG["study_area"]["bbox"]

GEDI_CSV = ROOT / PATHS["gedi_dir"] / "gedi_l4a_nainital.csv"


def load_and_clean() -> pd.DataFrame:
    if not GEDI_CSV.exists():
        raise FileNotFoundError(
            f"\nFile not found: {GEDI_CSV}\n"
            "Export from GEE using src/data/gee_gedi_extract.js\n"
            "then copy gedi_l4a_nainital.csv to data/gedi/"
        )

    df = pd.read_csv(GEDI_CSV)
    print(f"Loaded: {len(df):,} rows")
    print(f"Columns: {list(df.columns)}\n")

    # Handle lon/lat column name variants
    if "longitude" not in df.columns:
        if "lon" in df.columns:
            df = df.rename(columns={"lon": "longitude", "lat": "latitude"})
        elif ".geo" in df.columns:
            import json
            coords = df[".geo"].apply(lambda g: json.loads(g)["coordinates"])
            df["longitude"] = coords.apply(lambda c: c[0])
            df["latitude"]  = coords.apply(lambda c: c[1])

    # elev_lowestmode → elevation if SRTM elevation not joined
    if "elev_lowestmode" in df.columns and "elevation" not in df.columns:
        df["elevation"] = df["elev_lowestmode"]

    # Spatial filter
    df = df[
        (df["longitude"] >= BBOX["xmin"]) & (df["longitude"] <= BBOX["xmax"]) &
        (df["latitude"]  >= BBOX["ymin"]) & (df["latitude"]  <= BBOX["ymax"])
    ]

    # AGBD range filter
    df = df[(df["agbd"] >= GEDI["min_agbd"]) & (df["agbd"] <= GEDI["max_agbd"])]
    df = df.dropna(subset=["agbd", "NDVI"]).reset_index(drop=True)

    print(f"After filtering: {len(df):,} footprints")
    print(f"  AGBD  — mean: {df['agbd'].mean():.1f}  "
          f"min: {df['agbd'].min():.1f}  max: {df['agbd'].max():.1f} t/ha")
    if "elevation" in df.columns:
        print(f"  Elev  — mean: {df['elevation'].mean():.0f} m")
    if "NDVI" in df.columns:
        print(f"  NDVI  — mean: {df['NDVI'].mean():.3f}")
    return df


def build_training(df: pd.DataFrame) -> pd.DataFrame:
    required = FEAT["model_features"] + ["agbd"]
    for c in required:
        if c not in df.columns:
            print(f"  [warn] missing column '{c}' — filling with NaN")
            df[c] = np.nan

    out = df[["latitude", "longitude"] + required].copy()
    out = out.dropna().reset_index(drop=True)
    print(f"\nTraining dataset: {len(out):,} samples × {len(out.columns)} columns")
    return out


def build_prediction_grid(train_df: pd.DataFrame) -> pd.DataFrame:
    s2_path = ROOT / PATHS["sentinel_stack"]

    if s2_path.exists():
        print("\nBuilding prediction grid from Sentinel-2 raster ...")
        import rasterio
        from rasterio.transform import xy as rxy

        # Import from repo root (sys.path already set above)
        from src.features.indices import (
            compute_all_indices, read_sentinel_stack, BAND_ORDER
        )
        from src.utils.raster_utils import sample_raster_at_points

        bands, profile = read_sentinel_stack(s2_path)
        indices = compute_all_indices(
            B2=bands["B2"], B3=bands["B3"], B4=bands["B4"],
            B8=bands["B8"], B11=bands["B11"], B12=bands["B12"],
        )

        ndvi_thresh = CFG["seq_potential"]["non_forest_ndvi"]
        forest_mask = (indices["NDVI"] > ndvi_thresh) & (~np.isnan(indices["NDVI"]))

        with rasterio.open(s2_path) as src:
            transform = src.transform

        row_idx, col_idx = np.where(forest_mask)
        xs, ys = rxy(transform, row_idx, col_idx)

        rows_data = {"latitude": ys, "longitude": xs}
        for name in BAND_ORDER:
            rows_data[name] = bands[name][row_idx, col_idx]
        for name, arr in indices.items():
            rows_data[name] = arr[row_idx, col_idx]

        df_grid = pd.DataFrame(rows_data)

        for feat, key in [
            ("elevation", "dem_tif"),
            ("slope",     "slope_tif"),
            ("aspect",    "aspect_tif"),
        ]:
            tif_path = ROOT / PATHS[key]
            if tif_path.exists():
                from src.utils.raster_utils import sample_raster_at_points
                df_grid[feat] = sample_raster_at_points(
                    tif_path,
                    df_grid["longitude"].values,
                    df_grid["latitude"].values
                )
            else:
                print(f"  [warn] {tif_path.name} not found — {feat} will be NaN")
                df_grid[feat] = np.nan

        df_grid = df_grid.dropna().reset_index(drop=True)
        print(f"  {len(df_grid):,} forest pixels in prediction grid")
        return df_grid

    else:
        print("\nNo Sentinel-2 GeoTIFF found — using training footprints as prediction grid.")
        cols = ["latitude", "longitude"] + FEAT["model_features"]
        return train_df[cols].copy()


def main():
    proc_dir = ROOT / PATHS["processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    df       = load_and_clean()
    train_df = build_training(df)
    pred_df  = build_prediction_grid(train_df)

    out_train = ROOT / PATHS["training_csv"]
    out_pred  = ROOT / PATHS["prediction_grid"]

    train_df.to_csv(out_train, index=False)
    pred_df.to_csv(out_pred,  index=False)

    print(f"\nSaved: {out_train}")
    print(f"Saved: {out_pred}")
    print("\nNext step:  python src/models/train.py")


if __name__ == "__main__":
    main()