# Forest Carbon Stock Estimation

An end-to-end biomass and carbon mapping workflow for Nainital District,
Uttarakhand. The project combines Sentinel-2 surface reflectance, GEDI L4A
above-ground biomass density (AGBD), and SRTM terrain data to estimate biomass,
carbon stock, CO2-equivalent storage, and sequestration-potential zones.

The repository includes a Streamlit dashboard for exploring the generated maps,
model results, GEDI samples, live predictions, and methodology.

## Study Area and Method

- **Area:** Nainital District, Uttarakhand
- **Approximate extent:** 79.0-79.9 E, 29.0-29.7 N
- **Coordinate system:** WGS 84 for source data; UTM Zone 44N for area calculations
- **Sentinel-2 period:** November 2024 to March 2025
- **Sentinel-2 bands:** B2, B3, B4, B8, B11, and B12
- **Derived indices:** NDVI, EVI, NDMI, NBR, and SAVI
- **Terrain features:** elevation, slope, and aspect from SRTM DEM
- **Target:** GEDI L4A `agbd`
- **Model inputs:** 26 engineered features, including spectral, terrain, spatial,
	squared, interaction, and aspect features

Biomass is converted using the configured IPCC factors:

```text
Carbon stock (tC/ha) = Biomass (t/ha) * 0.47
CO2e (tCO2e/ha)      = Carbon stock (tC/ha) * 3.67
```

## Generated Results

The values below come from `outputs/stats/district_summary.csv`.

| Measure | Result |
|---|---:|
| Forest area | 4,596.8 km2 |
| Forest area | 459,680 ha |
| Mean biomass | 119.1 t/ha |
| Median biomass | 116.1 t/ha |
| Mean carbon stock | 56.0 tC/ha |
| Total carbon stock | 25.73 MtC |
| Total CO2e stored | 94.42 MtCO2e |
| Recorded model | StackingRegressor |
| Recorded log-scale R2 | 0.649 |
| Recorded MAE | 56.3 t/ha |

The model comparison table in `outputs/stats/model_metrics.csv` records the
following held-out evaluation results:

| Model | MAE | RMSE | R2 | R2 log |
|---|---:|---:|---:|---:|
| Random Forest | 62.56 | 81.93 | 0.346 | 0.477 |
| XGBoost | 61.84 | 80.76 | 0.364 | 0.476 |
| Stacking Ensemble | 61.69 | 80.61 | 0.366 | 0.482 |

These are observed project results, not performance targets.

## Repository Layout

```text
config/config.yaml                 Central paths, factors, thresholds, and parameters
data/gedi/                         GEDI L4A source files
data/processed/                    Training dataset and prediction grid
data/sentinel/                     Sentinel-2 raster inputs
notebooks/                         EDA, feature engineering, training, and mapping notebooks
outputs/models/                    Trained joblib models and feature list
outputs/maps/                      Biomass, carbon, CO2e, and sequestration rasters
outputs/stats/                     District summary and model metrics CSVs
outputs/figures/                   Static map figures
src/app.py                         Streamlit dashboard
src/data/                          Data download and dataset construction code
src/features/                      Spectral and terrain feature engineering
src/models/                        Model training and wall-to-wall prediction
src/utils/                         Raster and project utilities
src/visualization/                 Output map generation
```

## Quick Start

Create or activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard from the repository root:

```bash
streamlit run src/app.py
```

To reproduce the processing workflow, run the stages in order:

```bash
python src/data/download_gedi.py
python src/data/build_dataset.py
python src/models/train.py
python src/models/predict.py
python src/visualization/maps.py
```

The GEDI download stage requires NASA Earthdata access. The Sentinel-2 and DEM
inputs are configured for Google Earth Engine exports in `config/config.yaml`.

## Output Files

| File | Description |
|---|---|
| `outputs/maps/biomass.tif` | Predicted biomass density in t/ha |
| `outputs/maps/carbon_stock.tif` | Carbon stock in tC/ha |
| `outputs/maps/co2e.tif` | Stored CO2 equivalent in tCO2e/ha |
| `outputs/maps/seq_potential.tif` | Sequestration-potential zones from 1 to 4 |
| `outputs/stats/district_summary.csv` | District-level area and carbon summary |
| `outputs/stats/model_metrics.csv` | Validation metrics for each model |
| `outputs/models/best_model.joblib` | Selected production model |
| `outputs/models/feature_list.txt` | Feature order used during prediction |

Sequestration zones are configured as Low, Medium, High, and Very High. They
are assigned using biomass, NDVI, slope, and elevation thresholds in
`config/config.yaml`.

## Notes

- `src/models/predict.py` processes the prediction grid in chunks to limit memory use.
- The dashboard can fall back to embedded demo data when large rasters, CSVs, or
	model files are unavailable.
- Generated rasters and statistics should be treated as estimates for the
	configured study area and input period.
