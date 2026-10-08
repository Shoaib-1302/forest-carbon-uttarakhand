# Machine Learning-Based Forest Carbon Stock Estimation

A reproducible machine-learning workflow for spatially extrapolating
GEDI-derived above-ground biomass density (AGBD) across a heterogeneous
Himalayan forest landscape in Uttarakhand, India.

The project integrates **GEDI L4A v002**, **Sentinel-2 multispectral
imagery**, and **SRTM terrain data** to estimate above-ground biomass density,
carbon density, and CO2-equivalent storage over a spatially continuous
prediction grid.

The workflow combines Google Earth Engine preprocessing with Python-based
feature engineering, machine learning, model evaluation, and wall-to-wall
spatial prediction.

> **Important:** This study is presented as a reproducible spatial-extrapolation
> baseline. GEDI L4A AGBD is itself a model-derived biomass product rather than
> an independent field-measured ground-truth dataset. The resulting maps should
> therefore be interpreted as landscape-scale estimates rather than a validated
> forest inventory.

---

## Overview

Forests in mountainous regions are spatially heterogeneous, while direct
biomass observations from field inventories and spaceborne LiDAR are spatially
limited.

This project addresses that gap by learning the relationship between
**GEDI-derived biomass observations** and spatially continuous
**Sentinel-2 spectral** and **SRTM terrain** information.

The trained model is subsequently applied across the valid Sentinel-2/SRTM
prediction grid to produce spatially continuous estimates of:

- Above-ground biomass density (AGBD)
- Carbon density
- CO2-equivalent storage
- Restoration-suitability screening zones

The primary objective is **spatial extrapolation**, rather than claiming
precise biomass measurements at every individual prediction pixel.

---

## Study Area

The study area comprises the union of the:

- **Nainital forest division**
- **Almora forest division**
- **Champawat forest division**

in the Kumaon Himalayan region of Uttarakhand, India.

### Study Area Characteristics

| Parameter | Description |
|---|---|
| Region | Kumaon Himalaya, Uttarakhand, India |
| Forest divisions | Nainital, Almora, Champawat |
| Approximate longitude | 79.0°--79.9°E |
| Approximate latitude | 29.0°--29.7°N |
| Elevation range | Approximately 350--2700 m a.s.l. |
| Source CRS | WGS 84 |
| Area calculation CRS | UTM Zone 44N |
| Prediction resolution | 20 m |

The gross study area and the valid prediction area are not equivalent.

The final valid prediction domain contains **11,492,006 pixels**, corresponding
to **4,596.8 km² (459,680 ha)**.

This valid prediction area represents the portion of the prediction grid with
usable predictor coverage and vegetation screening. It should **not** be
interpreted as a verified forest-cover extent or a complete forest inventory.

---

# Data Sources

## 1. GEDI L4A v002

NASA's Global Ecosystem Dynamics Investigation (GEDI) L4A v002 product was used
as the above-ground biomass density target.

GEDI provides footprint-level estimates of above-ground biomass density (AGBD).

### GEDI Processing

The workflow extracted:

```text
20,518
```

GEDI L4A footprints from the study area.

After quality filtering:

```text
20,345
```

footprints satisfied the AGBD-range and quality requirements.

After spatial co-location with the Sentinel-2/SRTM predictor stack:

```text
14,690
```

footprints retained complete raster predictor coverage.

After the final feature-completeness check:

```text
13,955
```

samples remained for model development.

The final dataset was divided into:

```text
Training samples: 11,164
Testing samples:   2,791
Split:             80:20
```

### AGBD Quality Filter

The target was restricted to:

```text
0 <= AGBD <= 600 t ha^-1
```

along with the required GEDI quality information.

---

## Important GEDI Target Interpretation

GEDI L4A AGBD is itself a **model-derived biomass estimate**.

It is generated through statistical modelling calibrated against field inventory
information and is therefore not equivalent to an independently measured
ground-truth biomass observation.

Consequently, the modelling problem in this project is best described as:

```text
Sentinel-2 + SRTM
       |
       v
Spatial relationship
       |
       v
GEDI-derived AGBD
       |
       v
Landscape-scale spatial extrapolation
```

rather than direct prediction of independently measured field biomass.

This distinction is important when interpreting model accuracy and the resulting
spatial prediction maps.

---

# 2. Sentinel-2

Sentinel-2 Level-2A Surface Reflectance data from:

```text
COPERNICUS/S2_SR_HARMONIZED
```

were processed using a cloud-filtered seasonal median composite.

The following bands were used:

```text
B2
B3
B4
B8
B11
B12
```

The working prediction grid was produced at a **20 m spatial resolution**.

---

# 3. SRTM

SRTM 30 m elevation data were used to derive terrain variables:

```text
Elevation
Slope
Aspect
```

Terrain information was integrated with the Sentinel-2 spectral predictors to
represent the strong topographic variability of the Himalayan landscape.

---

# Feature Engineering

The final model uses **26 features** covering spectral, terrain, spatial, and
engineered relationships.

## Spectral Indices

```text
NDVI
EVI
NDMI
NBR
SAVI
```

## Raw Spectral Bands

```text
B8
B11
B12
```

## Terrain Variables

```text
Elevation
Slope
Aspect
```

## Spatial Variables

```text
Latitude
Longitude
```

## Engineered Features

The workflow additionally generates nonlinear and interaction features:

```text
NDVI²
EVI²
NBR²

NDVI × NDMI
NDVI × NBR

NDVI × Elevation
EVI × Elevation
NBR × Elevation

NDMI × Slope
NDVI × Slope

(Elevation / 1000)²

sin(Aspect)
cos(Aspect)
```

These features were designed to capture relationships between vegetation
condition, spectral response, terrain, and spatial location.

---

# Target Transformation

The AGBD target is positively skewed.

To reduce the influence of extreme values during model fitting, the target was
transformed using:

```text
y = ln(1 + AGBD)
```

After prediction, values were transformed back to the original biomass scale:

```text
AGBD = exp(predicted_y) - 1
```

Both original-scale and log-space performance metrics are reported.

---

# Machine Learning Models

Three modelling approaches were evaluated.

## Random Forest

Random Forest was implemented using Scikit-learn.

It provides a nonlinear tree-based baseline for modelling the relationship between
the remote-sensing predictors and GEDI-derived AGBD.

---

## XGBoost

XGBoost was used as a gradient-boosted tree model.

Hyperparameters were optimized using `RandomizedSearchCV`.

The search space included:

```text
n_estimators
learning_rate
max_depth
subsample
colsample_bytree
reg_alpha
reg_lambda
gamma
min_child_weight
```

---

## Stacking Ensemble

A stacking ensemble was constructed using:

```text
                 Random Forest
                      |
                      |
                      v
                    OOF
                 Predictions
                      |
                      |
XGBoost -----------> Ridge
                      |
                      v
                Final Prediction
```

The ensemble combines Random Forest and XGBoost predictions through a
ridge-regression meta-learner.

The stacking model uses internal 5-fold cross-validation to generate
out-of-fold base-model predictions.

---

# Model Evaluation

The models were evaluated using the held-out test set:

```text
n = 2,791
```

The following metrics were calculated:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Log-space R²

---

## Model Performance

| Model | MAE (t/ha) | RMSE (t/ha) | R² | Log-space R² |
|---|---:|---:|---:|---:|
| Random Forest | 62.56 | 81.93 | 0.346 | 0.477 |
| XGBoost | 61.84 | 80.76 | 0.364 | 0.476 |
| **Stacking Ensemble** | **61.69** | **80.61** | **0.367** | **0.482** |

The stacking ensemble achieved the best held-out performance.

### Best Model

```text
Model:              Stacking Ensemble
Original-scale R²:  0.367
Log-space R²:       0.482
MAE:                61.69 t/ha
RMSE:               80.61 t/ha
```

These are observed project results and are not performance targets.

The results indicate **moderate predictive agreement** between the
Sentinel-2/SRTM predictors and the GEDI-derived AGBD target.

The project is therefore positioned as a **reproducible spatial-extrapolation
baseline**, rather than a high-accuracy biomass measurement system.

---

# Feature Importance

Feature importance was examined for the Random Forest and XGBoost models.

## XGBoost

The strongest predictors included:

```text
B12
NDVI × NBR
NDVI × Slope
Elevation-related variables
Spectral bands
```

The importance of the B12 shortwave-infrared band and engineered
spectral-terrain interactions indicates that both vegetation and terrain
information contribute to the modelled biomass relationship.

## Random Forest

Latitude and longitude ranked highly among the Random Forest predictors.

This is an important diagnostic because strong dependence on spatial coordinates
can indicate that the model is partially learning regional geographic
structure rather than a fully transferable spectral/terrain relationship.

This also means that the random train/test split may not completely represent
performance on geographically independent locations.

---

# Spatial Prediction

After model training and evaluation, the selected stacking model was applied
across the valid Sentinel-2/SRTM prediction grid.

## Prediction Grid

| Parameter | Result |
|---|---:|
| Prediction pixels | 11,492,006 |
| Spatial resolution | 20 m |
| Valid prediction area | 4,596.8 km² |
| Valid prediction area | 459,680 ha |
| Mean predicted AGBD | 119.1 t/ha |
| Mean carbon density | 56.0 tC/ha |

The resulting prediction surface provides a spatially continuous representation
of broad biomass patterns across areas where GEDI footprints are not uniformly
available.

The individual 20 m predictions should not be interpreted as precise local
biomass measurements.

---

# Carbon Conversion

Predicted AGBD was converted into carbon density and CO2-equivalent storage
using IPCC Tier-1 conversion factors.

```text
Carbon density (tC/ha)
    =
AGBD (t/ha) × 0.47
```

and:

```text
CO2e (tCO2e/ha)
    =
Carbon density (tC/ha) × 3.67
```

---

# Carbon Stock Results

The resulting values over the valid prediction grid are:

| Measure | Result |
|---|---:|
| Valid prediction area | 4,596.8 km² |
| Valid prediction area | 459,680 ha |
| Prediction pixels | 11,492,006 |
| Prediction resolution | 20 m |
| Mean predicted AGBD | 119.1 t/ha |
| Mean carbon density | 56.0 tC/ha |
| **Total carbon stock** | **25.73 MtC** |
| **Total CO2e** | **94.42 MtCO2e** |

### Interpretation

These values represent carbon estimated over the **valid prediction grid**.

They should **not** be interpreted as a validated forest-inventory total for:

- Nainital District
- the three forest divisions
- Uttarakhand as a whole

The prediction area has not been independently intersected with a verified
forest-cover classification.

---

# Restoration-Suitability Screening

The project includes a rule-based spatial screening framework for identifying
areas that may warrant further restoration assessment.

The screening uses:

```text
Predicted AGBD
NDVI
Slope
Elevation
```

Pixels satisfying the degradation-screening conditions are assigned to one of
four screening classes.

| Zone | Label | Criteria |
|---|---|---|
| 4 | Very High Suitability | Slope < 15°, elevation 500--2000 m, NDVI >= 0.25 |
| 3 | High Suitability | Slope < 30°, NDVI >= 0.15 |
| 2 | Medium Suitability | Slope < 45°, NDVI >= 0.05 |
| 1 | Low Suitability | Remaining degraded pixels |

## Important Interpretation

These classes represent:

> **Restoration-Suitability Screening Zones**

They do **not** represent predicted future carbon-sequestration rates.

A pixel classified as "Very High Suitability" does not imply a specific
quantity or rate of future carbon uptake.

Actual restoration outcomes would also depend on factors such as:

- land tenure
- disturbance history
- soil conditions
- precipitation
- accessibility
- species suitability
- competing land use

These factors are not modelled in the current implementation.

---

# End-to-End Workflow

```text
                    DATA SOURCES
                         |
          +--------------+--------------+
          |                             |
          v                             v
    GEDI L4A v002                 Sentinel-2
    AGBD footprints              Surface Reflectance
          |                             |
          |                             |
          |                       Cloud Filtering
          |                       Seasonal Composite
          |                             |
          |                             v
          |                       Spectral Features
          |                             |
          |                             |
          |                       +-------------+
          |                       |    SRTM     |
          |                       | Elevation    |
          |                       | Slope        |
          |                       | Aspect       |
          |                       +-------------+
          |                             |
          +-------------+---------------+
                        |
                        v
               Spatial Co-location
                        |
                        v
              Feature Completeness
                        |
                        v
                13,955 Samples
                        |
                        v
                  80:20 Split
                  /           \
                 /             \
                v               v
        11,164 Training     2,791 Testing
                |
                v
        Feature Engineering
          26 Predictors
                |
        +-------+-------+
        |               |
        v               v
 Random Forest       XGBoost
        |               |
        +-------+-------+
                |
                v
        Stacking Ensemble
        Ridge Meta-Learner
                |
                v
        Held-Out Evaluation
                |
                v
      Selected Production Model
                |
                v
       Wall-to-Wall Prediction
                |
       +--------+---------+
       |        |         |
       v        v         v
     AGBD    Carbon     CO2e
     Map      Map        Map
                |
                v
 Restoration-Suitability
      Screening Zones
```

---

# Repository Structure

```text
forest-carbon-uttarakhand/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── gedi/
│   ├── processed/
│   └── sentinel/
│
├── notebooks/
│   ├── EDA/
│   ├── feature_engineering/
│   ├── modelling/
│   └── mapping/
│
├── outputs/
│   ├── models/
│   ├── maps/
│   ├── stats/
│   └── figures/
│
├── src/
│   ├── app.py
│   │
│   ├── data/
│   │   ├── download_gedi.py
│   │   └── build_dataset.py
│   │
│   ├── features/
│   │   └── ...
│   │
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── utils/
│   │   └── ...
│   │
│   └── visualization/
│       └── maps.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Shoaib-1302/forest-carbon-uttarakhand.git

cd forest-carbon-uttarakhand
```

---

## 2. Create a Python Environment

Using Conda:

```bash
conda create -n forest-carbon python=3.11

conda activate forest-carbon
```

Or using Python virtual environments:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Dashboard

The repository includes a Streamlit dashboard for exploring the generated
outputs.

Run:

```bash
streamlit run src/app.py
```

The dashboard can be used to explore available:

- biomass maps
- carbon maps
- CO2-equivalent maps
- restoration-suitability screening zones
- model performance
- GEDI samples
- spatial predictions
- methodology

When large output files are unavailable, the application may use embedded
demonstration data depending on the implementation.

---

# Reproduce the Workflow

The main processing stages can be executed in sequence:

```bash
python src/data/download_gedi.py

python src/data/build_dataset.py

python src/models/train.py

python src/models/predict.py

python src/visualization/maps.py
```

---

# Configuration

Project configuration is centralized in:

```text
config/config.yaml
```

The configuration controls items such as:

- input/output paths
- study-area parameters
- carbon conversion factors
- CO2 conversion factor
- vegetation thresholds
- restoration screening thresholds
- model parameters
- prediction settings

---

# Output Files

| File | Description |
|---|---|
| `outputs/maps/biomass.tif` | Predicted above-ground biomass density in t/ha |
| `outputs/maps/carbon_stock.tif` | Predicted carbon density in tC/ha |
| `outputs/maps/co2e.tif` | Predicted CO2-equivalent storage in tCO2e/ha |
| `outputs/maps/seq_potential.tif` | Restoration-suitability screening zones |
| `outputs/stats/district_summary.csv` | Prediction-area and carbon summary |
| `outputs/stats/model_metrics.csv` | Held-out model evaluation metrics |
| `outputs/models/best_model.joblib` | Selected production model |
| `outputs/models/feature_list.txt` | Feature ordering used during prediction |

> The `seq_potential.tif` filename is retained for compatibility with the
> repository structure. The output represents **restoration-suitability
> screening**, not predicted sequestration rates.

---

# Dashboard Features

The Streamlit dashboard is designed to provide an interactive view of the
project outputs.

Potential dashboard sections include:

```text
Overview
   |
   +-- Study Area
   |
   +-- GEDI Samples
   |
   +-- Model Performance
   |
   +-- Biomass Map
   |
   +-- Carbon Map
   |
   +-- CO2e Map
   |
   +-- Restoration Screening
   |
   +-- Methodology
```

The dashboard allows users to inspect both the spatial outputs and the
underlying modelling results.

---

# Key Findings

The current modelling run demonstrates that sparse GEDI-derived biomass
observations can be combined with spatially continuous Sentinel-2 and terrain
predictors to produce a landscape-scale biomass prediction surface.

### Dataset

```text
Raw GEDI footprints:             20,518
Quality-filtered footprints:     20,345
Spatially co-located footprints: 14,690
Final modelling samples:         13,955

Training samples:                11,164
Test samples:                     2,791
```

### Model

```text
Best model:                      Stacking Ensemble

R²:                              0.367
Log-space R²:                    0.482
MAE:                             61.69 t/ha
RMSE:                            80.61 t/ha
```

### Spatial Prediction

```text
Prediction pixels:               11,492,006
Resolution:                      20 m
Valid prediction area:           4,596.8 km²
Mean predicted AGBD:             119.1 t/ha
Mean carbon density:             56.0 tC/ha
Total carbon stock:              25.73 MtC
Total CO2e:                      94.42 MtCO2e
```

---

# Why the Spatial Prediction Matters

GEDI provides valuable biomass information but does not provide uniform
wall-to-wall observations across the landscape.

Sentinel-2 provides substantially more spatially continuous optical coverage,
while SRTM provides terrain information.

The machine-learning framework provides a mechanism for connecting these
datasets:

```text
Sparse GEDI observations
          +
Continuous Sentinel-2
          +
Terrain information
          |
          v
Machine Learning
          |
          v
Spatially continuous
biomass estimates
```

The primary value of the resulting map is therefore its ability to represent
**broad spatial gradients and relative differences in estimated biomass** over
areas where direct GEDI observations are unavailable.

---

# Limitations

## 1. GEDI Target Uncertainty

GEDI L4A AGBD is itself model-derived.

The model therefore learns a relationship with an existing biomass product
rather than independently measured field biomass.

---

## 2. Moderate Predictive Performance

The best model achieves:

```text
R² = 0.367
```

on the original AGBD scale.

The model should therefore not be interpreted as a highly accurate local
biomass measurement system.

---

## 3. Random-Split Validation

The current evaluation uses a random train/test split.

Because latitude and longitude are included as predictors, geographically
nearby observations may appear in both the training and test datasets.

Future work should include spatially independent validation to better assess
geographic generalization.

---

## 4. Forest-Mask Uncertainty

The current valid prediction area has not been independently intersected with
a verified forest-cover classification.

Therefore:

```text
4,596.8 km²
```

should be described as the **valid prediction area**, not the confirmed forest
area.

---

## 5. Temporal Mismatch

GEDI observations and the Sentinel-2 composite may not correspond to exactly
the same acquisition dates.

This can introduce temporal differences between the biomass target and optical
predictors.

---

## 6. Terrain Complexity

The Himalayan landscape contains strong variation in:

- elevation
- slope
- canopy structure
- forest composition
- environmental conditions

These factors make biomass modelling challenging.

---

## 7. No Independent Field Validation

The current modelling run does not include independent field validation plots.

---

## 8. Additional Sensors

Sentinel-1 SAR was not included in the current modelling run.

Future versions could investigate whether SAR backscatter and multi-temporal
observations improve biomass estimation.

---

# Future Work

Potential improvements include:

- Spatially independent cross-validation
- Independent field-plot validation
- Improved forest-cover classification
- GEDI beam and slope corrections
- Sentinel-1 SAR integration
- Multi-temporal Sentinel-2 observations
- Additional terrain predictors
- Climate and environmental variables
- Uncertainty estimation
- Larger and more spatially balanced GEDI training datasets
- Improved restoration-suitability modelling
- Temporal biomass monitoring
- Transferability testing across Himalayan regions

---

# Reproducibility

The project follows a reproducible pipeline:

```text
Data Acquisition
       ↓
Quality Filtering
       ↓
Remote-Sensing Preprocessing
       ↓
Spatial Co-location
       ↓
Feature Engineering
       ↓
Target Transformation
       ↓
Model Training
       ↓
Hyperparameter Search
       ↓
Held-Out Evaluation
       ↓
Wall-to-Wall Prediction
       ↓
Carbon Conversion
       ↓
Restoration Screening
       ↓
Visualization
```

The project configuration and source code are intended to make the main
processing stages reproducible.

---

# Technology Stack

### Programming

```text
Python
SQL
```

### Machine Learning

```text
Scikit-learn
XGBoost
Random Forest
StackingRegressor
```

### Geospatial

```text
GeoPandas
Rasterio
QGIS
ArcGIS
PostGIS
```

### Remote Sensing

```text
Google Earth Engine
Sentinel-2
GEDI LiDAR
SRTM
```

### Data Processing

```text
Pandas
NumPy
```

### Visualization

```text
Matplotlib
Folium
Kepler.gl
Streamlit
```

---

# Code Availability

The complete project repository is available at:

```text
https://github.com/Shoaib-1302/forest-carbon-uttarakhand
```

The paper also identifies the repository as the location of the project code.

---

# Data Availability

The project uses publicly available Earth observation datasets including:

- NASA GEDI L4A v002
- Sentinel-2 Harmonized Surface Reflectance
- SRTM elevation data

The datasets can be accessed through Google Earth Engine subject to the
relevant platform requirements.

field observations, improved forest masking, and
additional remote-sensing predictors.
