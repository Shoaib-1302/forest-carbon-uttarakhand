"""
raster_utils.py
───────────────
Shared helpers for raster I/O, coordinate transforms, and
point-in-raster sampling used throughout the pipeline.
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol
from pathlib import Path


def read_band(tif_path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    """Read a single band from a GeoTIFF. Returns (array, profile)."""
    with rasterio.open(tif_path) as src:
        arr     = src.read(band).astype(np.float32)
        profile = src.profile.copy()
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, profile


def sample_raster_at_points(
    tif_path: str | Path,
    lons: np.ndarray,
    lats: np.ndarray,
    band: int = 1,
) -> np.ndarray:
    """
    Sample a raster at (lon, lat) coordinate pairs using nearest-pixel.

    Parameters
    ----------
    tif_path : GeoTIFF path
    lons, lats : 1-D arrays of WGS-84 coordinates
    band : band index (1-based)

    Returns
    -------
    values : 1-D float32 array, NaN where out-of-bounds or nodata
    """
    values = np.full(len(lons), np.nan, dtype=np.float32)

    with rasterio.open(tif_path) as src:
        transform = src.transform
        nodata    = src.nodata
        data      = src.read(band).astype(np.float32)
        nrows, ncols = data.shape

        rows, cols = rowcol(transform, lons, lats)
        rows = np.array(rows)
        cols = np.array(cols)

        valid = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
        v     = data[rows[valid], cols[valid]]

        if nodata is not None:
            v[v == nodata] = np.nan

        values[valid] = v

    return values


def reproject_match(src_path: str | Path, ref_path: str | Path, out_path: str | Path) -> None:
    """
    Reproject and resample src_path to match the grid of ref_path.
    Uses bilinear resampling. Writes to out_path.
    """
    from rasterio.warp import reproject, Resampling

    with rasterio.open(ref_path) as ref:
        ref_crs       = ref.crs
        ref_transform = ref.transform
        ref_width     = ref.width
        ref_height    = ref.height

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
    print(f"  Reprojected: {out_path}")


def raster_to_dataframe(tif_paths: dict, mask_nodata: bool = True):
    """
    Stack multiple single-band GeoTIFFs (same grid) into a DataFrame.
    All rasters must share the same CRS, transform, and shape.

    Parameters
    ----------
    tif_paths : dict {column_name: tif_path}
    mask_nodata : drop rows with any NaN

    Returns
    -------
    pd.DataFrame with columns = tif_paths keys + 'latitude', 'longitude'
    """
    import pandas as pd

    arrays = {}
    profile_ref = None

    for col, path in tif_paths.items():
        arr, profile = read_band(path)
        arrays[col]  = arr
        if profile_ref is None:
            profile_ref = profile

    # Build coordinate grids
    transform = profile_ref["transform"]
    nrows     = profile_ref["height"]
    ncols     = profile_ref["width"]

    row_idx, col_idx = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    xs, ys = rasterio.transform.xy(transform, row_idx.ravel(), col_idx.ravel())

    df = pd.DataFrame({"longitude": xs, "latitude": ys})
    for col, arr in arrays.items():
        df[col] = arr.ravel()

    if mask_nodata:
        df = df.dropna().reset_index(drop=True)

    return df


def get_raster_bounds(tif_path: str | Path) -> tuple:
    """Return (xmin, ymin, xmax, ymax) bounds of a GeoTIFF."""
    with rasterio.open(tif_path) as src:
        b = src.bounds
    return b.left, b.bottom, b.right, b.top
