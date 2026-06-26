// ═══════════════════════════════════════════════════════════════════
// Google Earth Engine Export Script
// Forest Carbon — Nainital District, Uttarakhand
//
// USAGE:
//   1. Open https://code.earthengine.google.com
//   2. Paste this entire file
//   3. Click Run
//   4. In the Tasks panel, click RUN for each export task
//   5. Files land in your Google Drive under the folder below
//
// Exports:
//   A) sentinel2_stack.tif  — 6-band S2 median composite (20 m)
//   B) srtm_nainital.tif    — SRTM elevation (30 m)
//   C) slope.tif            — slope in degrees
//   D) aspect.tif           — aspect in degrees
// ═══════════════════════════════════════════════════════════════════

// ── 1. Study area ─────────────────────────────────────────────────
var DISTRICT = ee.FeatureCollection("FAO/GAUL/2015/level2")
  .filter(ee.Filter.and(
    ee.Filter.eq("ADM1_NAME", "Uttarakhand"),
    ee.Filter.eq("ADM2_NAME", "Nainital")
  ));

var ROI = DISTRICT.geometry();
Map.centerObject(ROI, 10);

// ── 2. Sentinel-2 cloud masking ───────────────────────────────────
function maskS2clouds(image) {
  var qa = image.select("QA60");
  // Bits 10 and 11 are clouds and cirrus
  var cloudBitMask  = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
               .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000)  // scale to [0,1]
              .copyProperties(image, ["system:time_start"]);
}

// ── 3. Sentinel-2 median composite ───────────────────────────────
var S2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(ROI)
  .filterDate("2024-11-01", "2025-03-31")   // dry season — fewer clouds
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  .map(maskS2clouds)
  .median()
  .select(["B2", "B3", "B4", "B8", "B11", "B12"])
  .clip(ROI);

print("Sentinel-2 composite bands:", S2.bandNames());

// Quick vis check
Map.addLayer(S2, {
  bands: ["B8", "B4", "B3"],
  min: 0, max: 0.35,
  gamma: 1.4
}, "S2 False Colour (NIR-R-G)");

// ── 4. DEM — SRTM ────────────────────────────────────────────────
var DEM = ee.Image("USGS/SRTMGL1_003")
  .select("elevation")
  .clip(ROI);

var TERRAIN = ee.Terrain.products(DEM);
var SLOPE   = TERRAIN.select("slope");
var ASPECT  = TERRAIN.select("aspect");

Map.addLayer(DEM,    {min: 500, max: 2800, palette: ["#f5f5f0","#4caf50","#1a5c38"]}, "Elevation");
Map.addLayer(SLOPE,  {min: 0, max: 60,    palette: ["white","orange","red"]},          "Slope");

// ── 5. Export tasks ───────────────────────────────────────────────
var DRIVE_FOLDER = "forest_carbon_nainital";
var SCALE_S2     = 20;   // metres — S2 native 10 m resampled to 20 m
var SCALE_DEM    = 30;   // metres — SRTM native

// A) Sentinel-2 stack
Export.image.toDrive({
  image:           S2,
  description:     "sentinel2_stack",
  folder:          DRIVE_FOLDER,
  fileNamePrefix:  "sentinel2_stack",
  region:          ROI,
  scale:           SCALE_S2,
  crs:             "EPSG:4326",
  maxPixels:       1e10,
  fileFormat:      "GeoTIFF"
});

// B) SRTM elevation
Export.image.toDrive({
  image:           DEM,
  description:     "srtm_nainital",
  folder:          DRIVE_FOLDER,
  fileNamePrefix:  "srtm_nainital",
  region:          ROI,
  scale:           SCALE_DEM,
  crs:             "EPSG:4326",
  maxPixels:       1e10,
  fileFormat:      "GeoTIFF"
});

// C) Slope
Export.image.toDrive({
  image:           SLOPE,
  description:     "slope",
  folder:          DRIVE_FOLDER,
  fileNamePrefix:  "slope",
  region:          ROI,
  scale:           SCALE_DEM,
  crs:             "EPSG:4326",
  maxPixels:       1e10,
  fileFormat:      "GeoTIFF"
});

// D) Aspect
Export.image.toDrive({
  image:           ASPECT,
  description:     "aspect",
  folder:          DRIVE_FOLDER,
  fileNamePrefix:  "aspect",
  region:          ROI,
  scale:           SCALE_DEM,
  crs:             "EPSG:4326",
  maxPixels:       1e10,
  fileFormat:      "GeoTIFF"
});

print("✅ Four export tasks queued. Open the Tasks panel and click RUN on each.");
