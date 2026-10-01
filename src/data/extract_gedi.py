import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

GEDI_DIR = Path("data/gedi")
OUTPUT_FILE = GEDI_DIR / "gedi_l4a_nainital.csv"

# Nainital bounding box used by the project
MIN_LON = 79.0
MAX_LON = 79.9
MIN_LAT = 29.0
MAX_LAT = 29.7


# ============================================================
# EXTRACT ONE HDF5 FILE
# ============================================================

def extract_file(filepath):

    rows = []

    print(f"\nReading: {filepath.name}")

    with h5py.File(filepath, "r") as h5:

        beams = [
            key for key in h5.keys()
            if key.startswith("BEAM")
        ]

        for beam_name in beams:

            beam = h5[beam_name]

            # Required fields
            required = [
                "lat_lowestmode",
                "lon_lowestmode",
                "agbd",
                "l4_quality_flag",
                "shot_number"
            ]

            if not all(field in beam for field in required):
                print(f"  Skipping {beam_name}: missing fields")
                continue

            lat = beam["lat_lowestmode"][:]
            lon = beam["lon_lowestmode"][:]
            agbd = beam["agbd"][:]
            quality = beam["l4_quality_flag"][:]
            shot = beam["shot_number"][:]

            # Optional fields
            agbd_se = (
                beam["agbd_se"][:]
                if "agbd_se" in beam
                else np.full_like(agbd, np.nan, dtype=float)
            )

            sensitivity = (
                beam["sensitivity"][:]
                if "sensitivity" in beam
                else np.full_like(agbd, np.nan, dtype=float)
            )

            degrade_flag = (
                beam["degrade_flag"][:]
                if "degrade_flag" in beam
                else np.full_like(agbd, np.nan, dtype=float)
            )

            # Convert masked/invalid values to NaN
            lat = np.asarray(lat, dtype=float)
            lon = np.asarray(lon, dtype=float)
            agbd = np.asarray(agbd, dtype=float)
            quality = np.asarray(quality)
            shot = np.asarray(shot)

            # Geographic filter
            spatial_mask = (
                (lon >= MIN_LON) &
                (lon <= MAX_LON) &
                (lat >= MIN_LAT) &
                (lat <= MAX_LAT)
            )

            # Basic validity
            valid_mask = (
                spatial_mask &
                np.isfinite(lat) &
                np.isfinite(lon) &
                np.isfinite(agbd) &
                (agbd >= 0)
            )

            # GEDI L4 quality flag
            valid_mask &= quality == 1

            count = int(valid_mask.sum())

            if count == 0:
                continue

            beam_rows = pd.DataFrame({
                "latitude": lat[valid_mask],
                "longitude": lon[valid_mask],
                "agbd": agbd[valid_mask],
                "agbd_se": agbd_se[valid_mask],
                "sensitivity": sensitivity[valid_mask],
                "degrade_flag": degrade_flag[valid_mask],
                "shot_number": shot[valid_mask],
                "beam": beam_name,
                "source_file": filepath.name
            })

            rows.append(beam_rows)

            print(
                f"  {beam_name}: "
                f"{count:,} valid shots"
            )

    if rows:
        return pd.concat(rows, ignore_index=True)

    return pd.DataFrame()


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(GEDI_DIR.glob("*.h5"))

    print("=" * 70)
    print("GEDI L4A V2.1 EXTRACTION")
    print("Nainital, Uttarakhand")
    print("=" * 70)

    print(f"\nHDF5 files found: {len(files)}")

    if not files:
        print("ERROR: No HDF5 files found in data/gedi")
        return

    all_data = []

    for filepath in files:

        try:

            df = extract_file(filepath)

            if not df.empty:
                all_data.append(df)

        except Exception as e:

            print(
                f"ERROR reading {filepath.name}: {e}"
            )

    if not all_data:

        print("\nNo valid GEDI observations found.")
        return

    result = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove duplicate shots
    before = len(result)

    result = result.drop_duplicates(
        subset=["shot_number"]
    )

    after = len(result)

    print("\nDuplicate shots removed:", before - after)

    # Sort geographically
    result = result.sort_values(
        ["latitude", "longitude"]
    ).reset_index(drop=True)

    # Save
    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(result):,}")

    print("\nColumns:")
    for column in result.columns:
        print(" -", column)

    print("\nAGBD statistics:")
    print(result["agbd"].describe())

    print("\nFirst 5 rows:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()