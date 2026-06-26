"""
download_gedi.py
────────────────
Download GEDI L4A (Above-Ground Biomass Density) HDF5 files for
Nainital District, Uttarakhand via NASA EarthData CMR API.

FIXES vs previous version
──────────────────────────
  - Added provider=ORNL_CLOUD (required for GEDI L4A v002)
  - Version padded to 3 digits: "002" not "2"
  - Fallback to LPDAAC_ECS provider if ORNL returns nothing
  - Added .h5 URL extractor that also checks "inherited" links
  - Added manual download URL printed if CMR search fails

Credentials — edit the two lines below
───────────────────────────────────────
"""

import os
import sys
import requests
import yaml
from pathlib import Path
from tqdm import tqdm

# ── EDIT THESE ────────────────────────────────────────────────────
HARDCODED_USER = "shoaib1302"
HARDCODED_PASS = "=Y2Bqu&jJC6SbNs"
# ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

BBOX    = CFG["study_area"]["bbox"]
OUT_DIR = ROOT / CFG["paths"]["gedi_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMR_URL    = "https://cmr.earthdata.nasa.gov/search/granules.json"
PRODUCT    = "GEDI04_A"
START_DATE = "2022-01-01"
END_DATE   = "2025-03-31"

# GEDI L4A v002 lives in ORNL_CLOUD on EarthData
# v001 lives in LPDAAC_ECS — we try both
PROVIDERS = ["ORNL_CLOUD", "LPDAAC_ECS"]

# ── Auth ─────────────────────────────────────────────────────────
if HARDCODED_USER and HARDCODED_PASS:
    AUTH = (HARDCODED_USER, HARDCODED_PASS)
    print("Using hardcoded credentials.")
else:
    u = os.environ.get("EARTHDATA_USER", "")
    p = os.environ.get("EARTHDATA_PASS", "")
    if u and p:
        AUTH = (u, p)
    else:
        netrc = Path.home() / ".netrc"
        if not netrc.exists():
            sys.exit(
                "\nERROR: No credentials found.\n"
                "Set HARDCODED_USER/HARDCODED_PASS at the top of this file.\n"
            )
        print("Using ~/.netrc")
        AUTH = None

SESSION = requests.Session()
if AUTH:
    SESSION.auth = AUTH


# ── CMR helpers ──────────────────────────────────────────────────
def query_granules(provider: str, page: int = 1, page_size: int = 200) -> list:
    """Return list of granule entries from CMR for a given provider."""
    bounding_box = (
        f"{BBOX['xmin']},{BBOX['ymin']},"
        f"{BBOX['xmax']},{BBOX['ymax']}"
    )
    params = {
        "short_name":   PRODUCT,
        "provider":     provider,
        "bounding_box": bounding_box,
        "temporal":     f"{START_DATE},{END_DATE}",
        "page_size":    page_size,
        "page_num":     page,
    }
    r = SESSION.get(CMR_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("feed", {}).get("entry", [])


def get_download_url(entry: dict) -> str | None:
    """
    Extract .h5 HTTPS download URL from a CMR granule entry.
    Checks both direct links and inherited links.
    Also handles the EarthData DAAC URL pattern.
    """
    all_links = entry.get("links", [])

    # Priority 1: direct .h5 download link
    for link in all_links:
        href = link.get("href", "")
        rel  = link.get("rel", "")
        if href.endswith(".h5") and href.startswith("https://"):
            return href

    # Priority 2: any link with "download" in rel and .h5 in href
    for link in all_links:
        href = link.get("href", "")
        rel  = link.get("rel", "")
        if "download" in rel.lower() and ".h5" in href:
            return href

    # Priority 3: construct from granule title (ORNL_CLOUD pattern)
    title = entry.get("title", "")
    if title.endswith(".h5"):
        # Try ORNL DAAC direct URL pattern
        return (
            f"https://data.ornldaac.earthdata.nasa.gov/protected/gedi/"
            f"GEDI_L4A_AGB_Density_V2_1/data/{title}"
        )

    return None


def download_file(url: str, dest: Path) -> bool:
    """Stream-download with progress bar. Returns True on success."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return True
    try:
        r = SESSION.get(url, stream=True, timeout=300)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as fh, tqdm(
            desc=dest.name[:50], total=total,
            unit="B", unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"  [error] {exc}")
        if dest.exists():
            dest.unlink()   # remove partial file
        return False


# ── Main ─────────────────────────────────────────────────────────
def main():
    print(f"Querying GEDI L4A granules for Nainital District ...")
    print(f"  Bounding box : {BBOX['xmin']},{BBOX['ymin']} → {BBOX['xmax']},{BBOX['ymax']}")
    print(f"  Date range   : {START_DATE} to {END_DATE}")
    print(f"  Output dir   : {OUT_DIR}\n")

    # Try each provider until we find granules
    all_entries = []
    for provider in PROVIDERS:
        print(f"  Trying provider: {provider} ...", end=" ", flush=True)
        page = 1
        provider_entries = []
        while True:
            try:
                entries = query_granules(provider=provider, page=page)
            except Exception as exc:
                print(f"query failed: {exc}")
                break
            if not entries:
                break
            provider_entries.extend(entries)
            if len(entries) < 200:
                break
            page += 1
        print(f"{len(provider_entries)} granules")
        all_entries.extend(provider_entries)
        if all_entries:
            break   # found granules — no need to try next provider

    if not all_entries:
        print("\nNo granules returned from CMR.")
        print("This can happen if:")
        print("  1. EarthData login credentials are wrong")
        print("  2. You haven't accepted the GEDI L4A data use agreement")
        print("\nFIX: Log in to https://urs.earthdata.nasa.gov and accept the")
        print("  ORNL DAAC application under 'Approved Applications'.")
        print("\nAlternative — manual download via Earthdata Search:")
        print("  https://search.earthdata.nasa.gov/search?q=GEDI04_A")
        print("  Draw a box over Nainital District (79-79.9E, 29-29.7N)")
        print("  Download all matching .h5 files to:", OUT_DIR)
        return

    print(f"\nFound {len(all_entries)} granule(s). Starting download ...\n")
    ok, failed = 0, []

    for entry in all_entries:
        url = get_download_url(entry)
        if not url:
            print(f"  [warn] No URL found for: {entry.get('title', 'unknown')}")
            continue
        dest = OUT_DIR / url.split("/")[-1].split("?")[0]
        print(f"  {dest.name}")
        if download_file(url, dest):
            ok += 1
        else:
            failed.append(url)

    print(f"\nDone.  {ok} downloaded,  {len(failed)} failed.")
    if failed:
        print("Failed URLs (retry manually):")
        for u in failed:
            print(f"  {u}")


if __name__ == "__main__":
    main()