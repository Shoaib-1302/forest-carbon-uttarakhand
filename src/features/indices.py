"""
indices.py
──────────
Spectral index computation from Sentinel-2 band arrays and
terrain feature derivation from a DEM GeoTIFF.

richdem is NOT used — slope and aspect are computed with pure numpy
(Horn 1981 finite-difference method), so this runs on Windows without
any C compiler or binary wheel.

Indices computed
────────────────
  NDVI  — Normalised Difference Vegetation Index
  EVI   — Enhanced Vegetation Index
  NDMI  — Normalised Difference Moisture Index
  NBR   — Normalised Burn Ratio
  SAVI  — Soil-Adjusted Vegetation Index

Terrain derived
───────────────
  Elevation, Slope (degrees), Aspect (degrees)  — Horn 1981 via numpy
"""

import numpy as np
import rasterio
from pathlib import Path


# ── Safe division ─────────────────────────────────────────────────
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(den == 0, np.nan, num / den)
    return result.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Spectral indices
# ═══════════════════════════════════════════════════════════════════

def ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red).  S2: B8, B4."""
    return _safe_div(nir - red, nir + red)


def evi(nir, red, blue):
    """EVI = 2.5 * (NIR-Red) / (NIR + 6*Red - 7.5*Blue + 1).  S2: B8, B4, B2."""
    return _safe_div(2.5 * (nir - red),
                     nir + 6.0 * red - 7.5 * blue + 1.0)


def ndmi(nir, swir1):
    """NDMI = (NIR - SWIR1) / (NIR + SWIR1).  S2: B8, B11."""
    return _safe_div(nir - swir1, nir + swir1)


def nbr(nir, swir2):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2).  S2: B8, B12."""
    return _safe_div(nir - swir2, nir + swir2)


def savi(nir, red, L=0.5):
    """SAVI = 1.5 * (NIR-Red) / (NIR + Red + L).  S2: B8, B4."""
    return _safe_div((1.0 + L) * (nir - red), nir + red + L)


def compute_all_indices(B2, B3, B4, B8, B11, B12):   # noqa: B3 unused
    return {
        "NDVI": ndvi(B8, B4),
        "EVI":  evi(B8, B4, B2),
        "NDMI": ndmi(B8, B11),
        "NBR":  nbr(B8, B12),
        "SAVI": savi(B8, B4),
    }


# ═══════════════════════════════════════════════════════════════════
# Read Sentinel-2 GeoTIFF
# ═══════════════════════════════════════════════════════════════════

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]


def read_sentinel_stack(tif_path):
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        bands = {}
        for i, name in enumerate(BAND_ORDER, start=1):
            arr = src.read(i).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            bands[name] = arr
    return bands, profile


# ═══════════════════════════════════════════════════════════════════
# Terrain features — pure numpy, no richdem, works on Windows
# Horn (1981) finite-difference method
# ═══════════════════════════════════════════════════════════════════

def _horn_slope_aspect(elev: np.ndarray, cell_size_m: float):
    """
    Compute slope (degrees) and aspect (degrees) from a DEM array
    using Horn's (1981) 3×3 finite-difference method.

    Parameters
    ----------
    elev       : 2-D float32 elevation array (metres), NaN for nodata
    cell_size_m: pixel size in metres (e.g. 30 for SRTM)

    Returns
    -------
    slope  : 2-D float32, degrees from horizontal [0, 90]
    aspect : 2-D float32, degrees clockwise from north [0, 360)
    """
    # Pad with NaN so edges don't wrap
    e = np.pad(elev.astype(np.float64), 1, mode="edge")

    # 3×3 neighbourhood labels (Horn 1981 notation)
    a = e[:-2, :-2];  b = e[:-2, 1:-1];  c = e[:-2, 2:]
    d = e[1:-1, :-2];                     f = e[1:-1, 2:]
    g = e[2:,  :-2];  h = e[2:,  1:-1];  i = e[2:,  2:]

    # Rate of change in x and y directions
    dzdx = ((c + 2*f + i) - (a + 2*d + g)) / (8.0 * cell_size_m)
    dzdy = ((g + 2*h + i) - (a + 2*b + c)) / (8.0 * cell_size_m)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)

    # Aspect in degrees clockwise from north
    aspect_rad = np.arctan2(dzdx, dzdy)           # N=0, E=π/2
    aspect = np.degrees(aspect_rad) % 360
    aspect = aspect.astype(np.float32)

    # Propagate NaN mask from elevation
    nan_mask = np.isnan(elev)
    slope[nan_mask]  = np.nan
    aspect[nan_mask] = np.nan

    return slope, aspect


def compute_terrain_features(dem_tif):
    """
    Read a SRTM GeoTIFF and return elevation, slope, aspect arrays.

    Returns
    -------
    terrain : dict {"elevation": arr, "slope": arr, "aspect": arr}
    profile : rasterio profile of the source DEM
    """
    with rasterio.open(dem_tif) as src:
        profile  = src.profile.copy()
        elev_arr = src.read(1).astype(np.float32)
        nodata   = src.nodata
        # Cell size in metres from the transform
        # For EPSG:4326 the pixel is in degrees; convert roughly.
        # SRTM is ~30 m natively; if the raster is in degrees we approximate.
        res_x = abs(src.transform.a)   # degrees or metres
        if res_x < 0.01:               # in degrees → convert to metres
            lat_centre = (src.bounds.top + src.bounds.bottom) / 2
            cell_size_m = res_x * 111_320 * np.cos(np.radians(lat_centre))
        else:
            cell_size_m = res_x        # already metres

    if nodata is not None:
        elev_arr[elev_arr == nodata] = np.nan

    slope_arr, aspect_arr = _horn_slope_aspect(elev_arr, cell_size_m)

    terrain = {
        "elevation": elev_arr,
        "slope":     slope_arr,
        "aspect":    aspect_arr,
    }
    return terrain, profile


# ═══════════════════════════════════════════════════════════════════
# Save single-band float32 as GeoTIFF
# ═══════════════════════════════════════════════════════════════════

def save_raster(array, profile, out_path, nodata=-9999.0):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = array.copy().astype(np.float32)
    arr[np.isnan(arr)] = nodata
    profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    print(f"  Saved: {out_path}")
