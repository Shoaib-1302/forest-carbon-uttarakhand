import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GEDI_CSV = ROOT / "data/gedi/gedi_l4a_nainital.csv"

S2_FILE = ROOT / "data/sentinel/sentinel2_stack.tif"
DEM_FILE = ROOT / "data/dem/srtm_nainital.tif"
SLOPE_FILE = ROOT / "data/dem/slope.tif"
ASPECT_FILE = ROOT / "data/dem/aspect.tif"

OUTPUT_FILE = ROOT / "data/processed/training_dataset.csv"


# ============================================================
# CONFIGURATION
# ============================================================

MIN_AGBD = 0
MAX_AGBD = 600

# Nainital project bounding box
MIN_LON = 79.0
MAX_LON = 79.9
MIN_LAT = 29.0
MAX_LAT = 29.7


# ============================================================
# RASTER SAMPLING
# ============================================================

def sample_raster(path, lons, lats):
    """
    Sample a single-band raster at GEDI coordinates.
    """

    print(f"Sampling: {path.name}")

    with rasterio.open(path) as src:

        coords = list(zip(lons, lats))

        values = []

        for value in src.sample(coords):

            if len(value) == 0:
                values.append(np.nan)
            else:
                values.append(value[0])

        values = np.asarray(values, dtype=float)

        nodata = src.nodata

        if nodata is not None:
            values[values == nodata] = np.nan

        return values


def sample_sentinel(path, lons, lats):
    """
    Sample all Sentinel-2 bands at GEDI locations.

    Expected band order:
        B2, B3, B4, B8, B11, B12
    """

    print(f"Sampling Sentinel-2: {path.name}")

    with rasterio.open(path) as src:

        print("  Bands:", src.count)
        print("  CRS:", src.crs)
        print("  Resolution:", src.res)

        coords = list(zip(lons, lats))

        sampled = list(src.sample(coords))

        arr = np.asarray(sampled, dtype=float)

        if arr.ndim != 2:
            raise RuntimeError(
                "Unexpected Sentinel-2 raster shape."
            )

        if arr.shape[1] < 6:
            raise RuntimeError(
                f"Expected at least 6 bands, found {arr.shape[1]}."
            )

        # Expected GEE export order
        bands = {
            "B2": arr[:, 0],
            "B3": arr[:, 1],
            "B4": arr[:, 2],
            "B8": arr[:, 3],
            "B11": arr[:, 4],
            "B12": arr[:, 5],
        }

        nodata = src.nodata

        if nodata is not None:

            for key in bands:

                bands[key] = bands[key].astype(float)

                bands[key][
                    bands[key] == nodata
                ] = np.nan

        return bands


# ============================================================
# SPECTRAL INDICES
# ============================================================

def calculate_indices(df):

    print("\nCalculating spectral indices...")

    B2 = df["B2"]
    B3 = df["B3"]
    B4 = df["B4"]
    B8 = df["B8"]
    B11 = df["B11"]
    B12 = df["B12"]

    # Avoid divide-by-zero
    eps = 1e-10

    # NDVI
    df["NDVI"] = (
        (B8 - B4) /
        (B8 + B4 + eps)
    )

    # EVI
    df["EVI"] = (
        2.5 *
        (B8 - B4) /
        (
            B8
            + 6 * B4
            - 7.5 * B2
            + 1
            + eps
        )
    )

    # NDMI
    df["NDMI"] = (
        (B8 - B11) /
        (B8 + B11 + eps)
    )

    # NBR
    df["NBR"] = (
        (B8 - B12) /
        (B8 + B12 + eps)
    )

    # SAVI
    L = 0.5

    df["SAVI"] = (
        ((B8 - B4) / (B8 + B4 + L + eps))
        * (1 + L)
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GEDI + SENTINEL-2 + TERRAIN FEATURE BUILDER")
    print("Nainital, Uttarakhand")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        GEDI_CSV,
        S2_FILE,
        DEM_FILE,
        SLOPE_FILE,
        ASPECT_FILE,
    ]

    print("\nChecking required files...")

    for path in required_files:

        if not path.exists():

            print(f"\nERROR: Missing file:")
            print(path)

            return

        print(f"  OK: {path}")

    # --------------------------------------------------------
    # Load GEDI
    # --------------------------------------------------------

    print("\nLoading GEDI CSV...")

    df = pd.read_csv(GEDI_CSV)

    print(f"GEDI rows: {len(df):,}")

    # --------------------------------------------------------
    # Spatial filtering
    # --------------------------------------------------------

    df = df[
        (df["longitude"] >= MIN_LON) &
        (df["longitude"] <= MAX_LON) &
        (df["latitude"] >= MIN_LAT) &
        (df["latitude"] <= MAX_LAT)
    ].copy()

    print(
        f"After spatial filtering: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # AGBD filtering
    # --------------------------------------------------------

    df = df[
        (df["agbd"] >= MIN_AGBD) &
        (df["agbd"] <= MAX_AGBD)
    ].copy()

    print(
        f"After AGBD filtering (0-600): "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lons = df["longitude"].values
    lats = df["latitude"].values

    # --------------------------------------------------------
    # Sentinel-2
    # --------------------------------------------------------

    s2 = sample_sentinel(
        S2_FILE,
        lons,
        lats
    )

    for name, values in s2.items():

        df[name] = values

    # --------------------------------------------------------
    # Spectral indices
    # --------------------------------------------------------

    df = calculate_indices(df)

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------

    df["elevation"] = sample_raster(
        DEM_FILE,
        lons,
        lats
    )

    df["slope"] = sample_raster(
        SLOPE_FILE,
        lons,
        lats
    )

    df["aspect"] = sample_raster(
        ASPECT_FILE,
        lons,
        lats
    )

    # --------------------------------------------------------
    # Required model features
    # --------------------------------------------------------

    model_features = [
        "NDVI",
        "EVI",
        "NDMI",
        "NBR",
        "SAVI",
        "B8",
        "B11",
        "B12",
        "elevation",
        "slope",
        "aspect",
        "latitude",
        "longitude",
    ]

    required = model_features + ["agbd"]

    print("\nChecking required columns...")

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing required column: {column}"
            )

        print(f"  OK: {column}")

    # --------------------------------------------------------
    # Remove invalid samples
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=required
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"\nRemoved {removed:,} rows "
        f"with missing raster/features."
    )

    # --------------------------------------------------------
    # NDVI sanity filter
    # --------------------------------------------------------

    before = len(df)

    df = df[
        (df["NDVI"] >= -1) &
        (df["NDVI"] <= 1)
    ].copy()

    print(
        f"Removed {before - len(df):,} "
        f"invalid NDVI rows."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE BUILD COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    print(
        f"\nTraining samples: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    print("\nFinal columns:")

    for column in df.columns:

        print(
            f"  - {column}"
        )

    print("\nAGBD statistics:")

    print(
        df["agbd"].describe()
    )

    print("\nNDVI statistics:")

    print(
        df["NDVI"].describe()
    )

    print("\nFirst 5 rows:")

    print(
        df[
            [
                "latitude",
                "longitude",
                "agbd",
                "NDVI",
                "EVI",
                "NDMI",
                "NBR",
                "SAVI",
                "B8",
                "B11",
                "B12",
                "elevation",
                "slope",
                "aspect",
            ]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()