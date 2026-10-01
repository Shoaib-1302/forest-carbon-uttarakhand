import requests
import os
from pathlib import Path

# ============================================================
# GEDI L4A V2.1 downloader for Nainital
# ============================================================

COLLECTION_ID = "C2237824918-ORNL_CLOUD"

BBOX = "79.0,29.0,79.9,29.7"

START_DATE = "2024-01-01"
END_DATE = "2025-03-31"

OUT_DIR = Path("data/gedi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

print("=" * 70)
print("GEDI L4A V2.1 - Nainital downloader")
print("=" * 70)

params = {
    "collection_concept_id": COLLECTION_ID,
    "bounding_box": BBOX,
    "temporal": f"{START_DATE},{END_DATE}",
    "page_size": 2000
}

print("\nQuerying CMR...")
print("Collection:", COLLECTION_ID)
print("Bounding box:", BBOX)
print("Dates:", START_DATE, "to", END_DATE)

response = requests.get(CMR_URL, params=params, timeout=60)

print("\nHTTP status:", response.status_code)

response.raise_for_status()

data = response.json()

entries = data.get("feed", {}).get("entry", [])

print("Granules found:", len(entries))

if not entries:
    print("\nNo GEDI granules found.")
    raise SystemExit(1)

print("\nFiles returned:")
for i, entry in enumerate(entries, 1):
    print(f"{i:02d}. {entry.get('title')}")

print("\n" + "=" * 70)
print("Finding downloadable HDF5 URLs")
print("=" * 70)

download_list = []

for entry in entries:
    title = entry.get("title", "")

    links = entry.get("links", [])

    h5_url = None

    for link in links:
        href = link.get("href", "")
        if href.lower().endswith(".h5"):
            h5_url = href
            break

    if h5_url:
        download_list.append((title, h5_url))

print("Downloadable files:", len(download_list))

if not download_list:
    print("\nNo HDF5 download links were found.")
    print("First entry:")
    print(entries[0])
    raise SystemExit(1)

print("\n" + "=" * 70)
print("Downloading GEDI granules")
print("=" * 70)

for i, (title, url) in enumerate(download_list, 1):

    output_file = OUT_DIR / title

    if output_file.exists():
        print(f"\n[{i}/{len(download_list)}] Already exists:")
        print(output_file.name)
        continue

    print(f"\n[{i}/{len(download_list)}] Downloading:")
    print(title)
    print("URL:", url)

    try:
        with requests.get(
            url,
            stream=True,
            timeout=120
        ) as r:

            print("HTTP:", r.status_code)

            r.raise_for_status()

            with open(output_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"Saved: {output_file}")
        print(f"Size: {size_mb:.2f} MB")

    except Exception as e:
        print("DOWNLOAD FAILED:", e)

print("\n" + "=" * 70)
print("GEDI download process finished")
print("=" * 70)

files = list(OUT_DIR.glob("*.h5"))

print("HDF5 files currently in data/gedi:", len(files))

for f in files:
    print(f" - {f.name}")

