// ═══════════════════════════════════════════════════════════════════
// Google Earth Engine Export Script
// Forest Carbon — Nainital District, Uttarakhand
// CORRECTED VERSION
// ═══════════════════════════════════════════════════════════════════


// ─────────────────────────────────────────────────────────────────
// 1. STUDY AREA
// ─────────────────────────────────────────────────────────────────

// Use current GAUL 2025 administrative boundaries.
var DISTRICTS = ee.FeatureCollection("FAO/GAUL/2025/level2");

// Inspect available properties
print("GAUL sample:", DISTRICTS.first());

// Filter Nainital district
var DISTRICT = DISTRICTS
  .filter(ee.Filter.eq("GAUL1_NAME", "Uttarakhand"))
  .filter(ee.Filter.eq("GAUL2_NAME", "Nainital"));

print("Number of matching districts:", DISTRICT.size());
print("Selected district:", DISTRICT);

// Geometry
var ROI = DISTRICT.geometry();

print("ROI:", ROI);

// Display boundary
Map.centerObject(DISTRICT, 9);

Map.addLayer(
  DISTRICT.style({
    color: "red",
    fillColor: "00000000",
    width: 2
  }),
  {},
  "Nainital District Boundary"
);


// ─────────────────────────────────────────────────────────────────
// 2. SENTINEL-2 CLOUD MASK
// ─────────────────────────────────────────────────────────────────

function maskS2clouds(image) {

  var qa = image.select("QA60");

  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(
      qa.bitwiseAnd(cirrusBitMask).eq(0)
    );

  return image
    .updateMask(mask)
    .divide(10000)
    .copyProperties(image, ["system:time_start"]);
}


// ─────────────────────────────────────────────────────────────────
// 3. SENTINEL-2 COLLECTION
// ─────────────────────────────────────────────────────────────────

var S2_COLLECTION = ee.ImageCollection(
  "COPERNICUS/S2_SR_HARMONIZED"
)
  .filterBounds(ROI)
  .filterDate(
    "2024-11-01",
    "2025-03-31"
  )
  .filter(
    ee.Filter.lt(
      "CLOUDY_PIXEL_PERCENTAGE",
      40
    )
  );


// IMPORTANT DEBUG INFORMATION

print(
  "Sentinel-2 image count:",
  S2_COLLECTION.size()
);

print(
  "First Sentinel-2 image:",
  S2_COLLECTION.first()
);


// ─────────────────────────────────────────────────────────────────
// 4. CREATE COMPOSITE
// ─────────────────────────────────────────────────────────────────

var S2 = S2_COLLECTION
  .map(maskS2clouds)
  .median()
  .select([
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12"
  ])
  .clip(ROI);

print(
  "Sentinel-2 composite bands:",
  S2.bandNames()
);


// ─────────────────────────────────────────────────────────────────
// 5. SENTINEL-2 VISUALIZATION
// ─────────────────────────────────────────────────────────────────

Map.addLayer(
  S2,
  {
    bands: [
      "B8",
      "B4",
      "B3"
    ],
    min: 0,
    max: 0.35,
    gamma: 1.4
  },
  "S2 False Colour (NIR-R-G)"
);


// ─────────────────────────────────────────────────────────────────
// 6. SRTM DEM
// ─────────────────────────────────────────────────────────────────

var DEM = ee.Image(
  "USGS/SRTMGL1_003"
)
  .select("elevation")
  .clip(ROI);

print(
  "DEM:",
  DEM
);


// ─────────────────────────────────────────────────────────────────
// 7. TERRAIN VARIABLES
// ─────────────────────────────────────────────────────────────────

var TERRAIN = ee.Terrain.products(DEM);

var SLOPE = TERRAIN.select("slope");

var ASPECT = TERRAIN.select("aspect");


// ─────────────────────────────────────────────────────────────────
// 8. TERRAIN VISUALIZATION
// ─────────────────────────────────────────────────────────────────

Map.addLayer(
  DEM,
  {
    min: 500,
    max: 2800,
    palette: [
      "#f5f5f0",
      "#4caf50",
      "#1a5c38"
    ]
  },
  "Elevation"
);

Map.addLayer(
  SLOPE,
  {
    min: 0,
    max: 60,
    palette: [
      "white",
      "orange",
      "red"
    ]
  },
  "Slope"
);

Map.addLayer(
  ASPECT,
  {
    min: 0,
    max: 360
  },
  "Aspect"
);


// ─────────────────────────────────────────────────────────────────
// 9. EXPORT SETTINGS
// ─────────────────────────────────────────────────────────────────

var DRIVE_FOLDER =
  "forest_carbon_nainital";

var SCALE_S2 = 20;

var SCALE_DEM = 30;


// ─────────────────────────────────────────────────────────────────
// 10. EXPORT SENTINEL-2
// ─────────────────────────────────────────────────────────────────

Export.image.toDrive({

  image: S2,

  description:
    "sentinel2_stack",

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    "sentinel2_stack",

  region:
    ROI,

  scale:
    SCALE_S2,

  crs:
    "EPSG:4326",

  maxPixels:
    1e10,

  fileFormat:
    "GeoTIFF"
});


// ─────────────────────────────────────────────────────────────────
// 11. EXPORT DEM
// ─────────────────────────────────────────────────────────────────

Export.image.toDrive({

  image: DEM,

  description:
    "srtm_nainital",

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    "srtm_nainital",

  region:
    ROI,

  scale:
    SCALE_DEM,

  crs:
    "EPSG:4326",

  maxPixels:
    1e10,

  fileFormat:
    "GeoTIFF"
});


// ─────────────────────────────────────────────────────────────────
// 12. EXPORT SLOPE
// ─────────────────────────────────────────────────────────────────

Export.image.toDrive({

  image: SLOPE,

  description:
    "slope",

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    "slope",

  region:
    ROI,

  scale:
    SCALE_DEM,

  crs:
    "EPSG:4326",

  maxPixels:
    1e10,

  fileFormat:
    "GeoTIFF"
});


// ─────────────────────────────────────────────────────────────────
// 13. EXPORT ASPECT
// ─────────────────────────────────────────────────────────────────

Export.image.toDrive({

  image: ASPECT,

  description:
    "aspect",

  folder:
    DRIVE_FOLDER,

  fileNamePrefix:
    "aspect",

  region:
    ROI,

  scale:
    SCALE_DEM,

  crs:
    "EPSG:4326",

  maxPixels:
    1e10,

  fileFormat:
    "GeoTIFF"
});


print(
  "Four export tasks created successfully."
);

print(
  "Open the Tasks panel and click RUN."
);