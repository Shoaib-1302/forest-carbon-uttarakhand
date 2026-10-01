import requests
from pathlib import Path

COLLECTION_ID = "C2237824918-ORNL_CLOUD"
BBOX = "79.0,29.0,79.9,29.7"

# Match the Sentinel-2 period we are using
START_DATE = "2024-11-01"
END_DATE = "2025-03-31"

OUT_DIR = Path("data/gedi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

TOKEN = __import__("os").environ.get("EARTHDATA_TOKEN")

if not TOKEN:
    print("ERROR: EARTHDATA_TOKEN is not set.")
    print('Run:')
    print('$env:EARTHDATA_TOKEN="YOUR_TOKEN"')
    raise SystemExit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}"
}

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

response = requests.get(
    CMR_URL,
    params=params,
    timeout=60
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()
entries = data.get("feed", {}).get("entry", [])

print("Granules returned:", len(entries))

if not entries:
    print("No GEDI granules found.")
    raise SystemExit(1)

# Filter using the actual CMR time_start field.
filtered_entries = []

for entry in entries:
    time_start = entry.get("time_start", "")

    if not time_start:
        continue

    date_part = time_start[:10]

    if START_DATE <= date_part <= END_DATE:
        filtered_entries.append(entry)

print("Granules inside requested date range:", len(filtered_entries))

if not filtered_entries:
    print("No granules remain after date filtering.")
    raise SystemExit(1)

print("\nFinding HDF5 download URLs...")

download_list = []

for entry in filtered_entries:

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

print("Downloadable HDF5 files:", len(download_list))

if not download_list:
    print("No HDF5 URLs found.")
    raise SystemExit(1)

print("\n" + "=" * 70)
print("DOWNLOADING GEDI")
print("=" * 70)

successful = 0
failed = 0

for i, (title, url) in enumerate(download_list, 1):

    output_file = OUT_DIR / title

    print(f"\n[{i}/{len(download_list)}]")
    print(title)

    if output_file.exists() and output_file.stat().st_size > 0:

        print("Already exists.")
        successful += 1
        continue

    try:

        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=180
        ) as r:

            print("HTTP:", r.status_code)

            if r.status_code != 200:

                print("Download failed.")
                print("Response:", r.text[:500])

                failed += 1
                continue

            with open(output_file, "wb") as f:

                for chunk in r.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        f.write(chunk)

        size_mb = output_file.stat().st_size / (1024 * 1024)

        print(f"Saved: {output_file}")
        print(f"Size: {size_mb:.2f} MB")

        successful += 1

    except Exception as e:

        print("DOWNLOAD FAILED:", e)
        failed += 1

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

print("Successful:", successful)
print("Failed:", failed)

files = list(OUT_DIR.glob("*.h5"))

print("HDF5 files in data/gedi:", len(files))

for f in files:

    size_mb = f.stat().st_size / (1024 * 1024)

    print(f" - {f.name} ({size_mb:.2f} MB)")
