# Forest Carbon Stock Estimation — Nainital District, Uttarakhand

**Sentinel-2 · GEDI LiDAR · SRTM DEM · Random Forest · XGBoost**

Estimates above-ground biomass density (AGBD) across forested pixels using
GEDI L4A footprints as ground truth, Sentinel-2 spectral indices and SRTM
terrain features as predictors, and converts predictions to carbon stock
(tC/ha) and CO₂ equivalent (tCO₂e/ha) using IPCC (2006) factors.

---

## Repository Structure

```
forest_carbon/
├── config/
│   └── config.yaml              # All paths, constants, GEE params
├── data/
│   ├── raw/                     # Original downloads (git-ignored)
│   ├── sentinel/                # GEE-exported Sentinel-2 GeoTIFFs
│   ├── gedi/                    # GEDI L4A .h5 files
│   ├── dem/                     # SRTM DEM + slope + aspect GeoTIFFs
│   └── processed/               # Merged training CSV + prediction grid
├── src/
│   ├── data/
│   │   ├── gee_export.js        # GEE script — Sentinel-2 + DEM export
│   │   ├── download_gedi.py     # GEDI L4A download via NASA EarthData
│   │   └── build_dataset.py     # Merge GEDI footprints + raster features
│   ├── features/
│   │   └── indices.py           # Spectral indices + terrain derivation
│   ├── models/
│   │   ├── train.py             # Train RF + XGBoost, save artifacts
│   │   └── predict.py           # Wall-to-wall biomass prediction
│   ├── visualization/
│   │   └── maps.py              # Biomass / carbon / CO₂e map plots
│   └── utils/
│       └── raster_utils.py      # Common raster I/O helpers
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_carbon_mapping.ipynb
├── outputs/
│   ├── maps/                    # Output GeoTIFFs
│   ├── stats/                   # District summary CSVs
│   └── figures/                 # PNG plots
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run GEE export (paste gee_export.js into code.earthengine.google.com)

# 3. Download GEDI
python src/data/download_gedi.py

# 4. Build training dataset
python src/data/build_dataset.py

# 5. Train models
python src/models/train.py

# 6. Predict + generate maps
python src/models/predict.py

# 7. Plot outputs
python src/visualization/maps.py
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/maps/biomass.tif` | Predicted AGBD (t/ha) |
| `outputs/maps/carbon_stock.tif` | Carbon stock (tC/ha) |
| `outputs/maps/co2e.tif` | CO₂ equivalent (tCO₂e/ha) |
| `outputs/maps/seq_potential.tif` | Sequestration potential zones (1–4) |
| `outputs/stats/district_summary.csv` | Total area, mean biomass, total carbon |

---

## IPCC Conversion Factors

```
Carbon Stock (tC/ha)   = AGBD (t/ha) × 0.47
CO₂e       (tCO₂e/ha) = Carbon Stock × 3.67
```

---

## Target Performance

| Metric | Target |
|--------|--------|
| R²     | ≥ 0.70 |
| MAE    | < 30 t/ha |
| RMSE   | < 45 t/ha |
