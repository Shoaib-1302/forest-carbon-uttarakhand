"""
build_prediction_grid.py

Creates a wall-to-wall prediction grid from:
- Sentinel-2 stack
- SRTM elevation
- slope
- aspect

Output:
    data/processed/prediction_grid.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS = CFG["paths"]


SENTINEL = ROOT / PATHS["sentinel_stack"]
DEM = ROOT / PATHS["dem_tif"]
SLOPE = ROOT / PATHS["slope_tif"]
ASPECT = ROOT / PATHS["aspect_tif"]

OUTPUT = ROOT / PATHS["prediction_grid"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Check inputs
# ---------------------------------------------------------

for path in [SENTINEL, DEM, SLOPE, ASPECT]:
    if not path.exists():
        raise FileNotFoundError(f"Missing raster: {path}")

print("Input rasters:")
print(f"  Sentinel: {SENTINEL}")
print(f"  DEM:      {DEM}")
print(f"  Slope:    {SLOPE}")
print(f"  Aspect:   {ASPECT}")


# ---------------------------------------------------------
# Read Sentinel-2
# ---------------------------------------------------------

with rasterio.open(SENTINEL) as src:

    print("\nSentinel-2 information:")
    print(f"  CRS:      {src.crs}")
    print(f"  Size:     {src.width} x {src.height}")
    print(f"  Bands:    {src.count}")
    print(f"  Res:      {src.res}")

    sentinel = src.read().astype(np.float32)

    transform = src.transform
    height = src.height
    width = src.width

    nodata = src.nodata


if sentinel.shape[0] < 6:
    raise ValueError(
        f"Sentinel raster has {sentinel.shape[0]} bands. "
        "Expected at least 6 bands."
    )


# ---------------------------------------------------------
# Sentinel bands
#
# Export order from GEE:
# B2 B3 B4 B8 B11 B12
# ---------------------------------------------------------

B2 = sentinel[0]
B3 = sentinel[1]
B4 = sentinel[2]
B8 = sentinel[3]
B11 = sentinel[4]
B12 = sentinel[5]


# ---------------------------------------------------------
# Read terrain rasters
# ---------------------------------------------------------

def read_resampled(path, reference):
    """
    Read a raster using the Sentinel grid as the reference.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(path) as src:

        output = np.empty(
            reference.shape,
            dtype=np.float32
        )

        reproject(
            source=rasterio.band(src, 1),
            destination=output,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=reference_crs,
            resampling=Resampling.bilinear
        )

    return output


reference_crs = None

with rasterio.open(SENTINEL) as src:
    reference_crs = src.crs


print("\nReading terrain layers...")

elevation = read_resampled(
    DEM,
    B8
)

slope = read_resampled(
    SLOPE,
    B8
)

aspect = read_resampled(
    ASPECT,
    B8
)


# ---------------------------------------------------------
# Calculate spectral indices
# ---------------------------------------------------------

print("Calculating spectral indices...")


def safe_divide(numerator, denominator):
    result = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.isfinite(numerator) & np.isfinite(denominator)

    valid &= np.abs(denominator) > 1e-10

    result[valid] = (
        numerator[valid] /
        denominator[valid]
    )

    return result


# NDVI
NDVI = safe_divide(
    B8 - B4,
    B8 + B4
)


# EVI
EVI = np.full(
    B8.shape,
    np.nan,
    dtype=np.float32
)

denom = B8 + 6 * B4 - 7.5 * B2 + 1

valid = (
    np.isfinite(B8) &
    np.isfinite(B4) &
    np.isfinite(B2) &
    (np.abs(denom) > 1e-10)
)

EVI[valid] = (
    2.5 *
    (B8[valid] - B4[valid]) /
    denom[valid]
)


# NDMI
NDMI = safe_divide(
    B8 - B11,
    B8 + B11
)


# NBR
NBR = safe_divide(
    B8 - B12,
    B8 + B12
)


# SAVI
SAVI = safe_divide(
    1.5 * (B8 - B4),
    B8 + B4 + 0.5
)


# ---------------------------------------------------------
# Generate coordinates
# ---------------------------------------------------------

print("Generating coordinates...")

rows, cols = np.indices(
    (height, width)
)

longitudes, latitudes = rasterio.transform.xy(
    transform,
    rows,
    cols,
    offset="center"
)

longitude = np.asarray(
    longitudes,
    dtype=np.float64
).ravel()

latitude = np.asarray(
    latitudes,
    dtype=np.float64
).ravel()


# ---------------------------------------------------------
# Flatten all features
# ---------------------------------------------------------

data = {
    "latitude": latitude,
    "longitude": longitude,

    "B2": B2.ravel(),
    "B3": B3.ravel(),
    "B4": B4.ravel(),
    "B8": B8.ravel(),
    "B11": B11.ravel(),
    "B12": B12.ravel(),

    "NDVI": NDVI.ravel(),
    "EVI": EVI.ravel(),
    "NDMI": NDMI.ravel(),
    "NBR": NBR.ravel(),
    "SAVI": SAVI.ravel(),

    "elevation": elevation.ravel(),
    "slope": slope.ravel(),
    "aspect": aspect.ravel(),
}


df = pd.DataFrame(data)


print(f"\nInitial pixels: {len(df):,}")


# ---------------------------------------------------------
# Remove invalid pixels
# ---------------------------------------------------------

required = [
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

before = len(df)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=required
)

removed = before - len(df)

print(f"Removed invalid pixels: {removed:,}")
print(f"Valid prediction pixels: {len(df):,}")


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

print("\nSaving prediction grid...")

df.to_csv(
    OUTPUT,
    index=False
)

print(f"Saved: {OUTPUT}")

print("\nColumns:")
print(list(df.columns))

print("\nPreview:")
print(df.head())

print("\nDone.")