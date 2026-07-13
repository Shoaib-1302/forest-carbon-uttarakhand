"""
app.py
──────
Streamlit dashboard for Forest Carbon Stock Estimation
Nainital District, Uttarakhand

Install
───────
    pip install streamlit plotly

Run
───
    streamlit run app.py
"""

import sys
import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import rasterio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ROOT always points to the repo root regardless of where app.py lives
_here = Path(__file__).resolve().parent
ROOT = _here.parent if (_here / "config").exists() is False and (_here.parent / "config").exists() else _here
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS = CFG["paths"]
IPCC  = CFG["ipcc"]
SEQ   = CFG["seq_potential"]
VIZ   = CFG["viz"]

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Forest Carbon — Nainital",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card {
    background: white; border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 5px solid #2e7d52;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1a5c38; line-height: 1.1; }
.metric-label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }
.section-header {
    font-size: 1.3rem; font-weight: 700; color: #1a5c38;
    border-bottom: 2px solid #4caf50;
    padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────
@st.cache_data
def load_training_csv():
    p = ROOT / PATHS["training_csv"]
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_district_summary():
    p = ROOT / PATHS["output_stats"] / "district_summary.csv"
    return pd.read_csv(p).iloc[0].to_dict() if p.exists() else {}

@st.cache_data
def load_model_metrics():
    p = ROOT / PATHS["output_stats"] / "model_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_raster(tif_key):
    path = ROOT / PATHS[tif_key]
    if not path.exists():
        return None, None
    with rasterio.open(path) as src:
        arr    = src.read(1).astype(np.float32)
        bounds = src.bounds
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr, bounds

@st.cache_resource
def load_model():
    for fname in ["best_model.joblib", "xgb_model.joblib"]:
        p = ROOT / "outputs/models" / fname
        if p.exists():
            return joblib.load(p), fname
    return None, None


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌲 Forest Carbon")
    st.markdown("**Nainital District, Uttarakhand**")
    st.markdown("---")
    st.markdown("**Study Area**")
    st.markdown("- District: Nainital\n- State: Uttarakhand\n- Bbox: 79–79.9°E, 29–29.7°N")
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- 🛰️ Sentinel-2 SR (ESA, 2024)\n- 📡 GEDI L4A v002 (NASA)\n- 🏔️ SRTM DEM 30m")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("- XGBoost + Random Forest\n- R²(log) = 0.649\n- MAE = 56.3 t/ha\n- 26 features")
    st.markdown("---")
    st.markdown("**IPCC Factors**")
    st.markdown(f"- Biomass→Carbon: ×{IPCC['biomass_to_carbon']}\n- Carbon→CO₂e: ×{IPCC['carbon_to_co2e']}")
    st.caption("MSc Data Science · CHRIST University · Bengaluru")


# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1a5c38,#2e7d52);
     padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;'>
  <h1 style='color:white;margin:0;font-size:2rem;'>
    🌲 Forest Carbon Stock Estimation
  </h1>
  <p style='color:#a5d6a7;margin:0.5rem 0 0 0;font-size:1rem;'>
    Nainital District, Uttarakhand &nbsp;·&nbsp;
    Sentinel-2 + GEDI LiDAR + XGBoost &nbsp;·&nbsp;
    117,933 footprints &nbsp;·&nbsp; 18.6M pixels predicted
  </p>
</div>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────
summary  = load_district_summary()
metrics  = load_model_metrics()
train_df = load_training_csv()
model, mname = load_model()


# ── KPI row ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">District Summary</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

def kpi(col, value, label, unit=""):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{value}
        <span style="font-size:1rem;color:#4caf50"> {unit}</span>
      </div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, f"{summary.get('Forest Area (km2)', '—'):,}",  "Forest Area",    "km²")
kpi(c2, summary.get("Mean Biomass (t/ha)", "—"),        "Mean Biomass",   "t/ha")
kpi(c3, summary.get("Mean Carbon Stock (tC/ha)", "—"),  "Mean Carbon",    "tC/ha")
kpi(c4, summary.get("Total Carbon Stock (MtC)", "—"),   "Total Carbon",   "MtC")
kpi(c5, summary.get("Total CO2e Stored (MtCO2e)", "—"), "CO₂e Stored",    "MtCO₂e")


# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Carbon Maps",
    "📈 Model Performance",
    "🌿 GEDI Explorer",
    "🔬 Live Predictor",
    "📋 Methodology",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — MAPS
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Carbon Output Maps</div>',
                unsafe_allow_html=True)

    layer = st.radio("Layer", [
        "Biomass (t/ha)", "Carbon Stock (tC/ha)",
        "CO₂e (tCO₂e/ha)", "Sequestration Potential"
    ], horizontal=True)

    tif_map = {
        "Biomass (t/ha)":          ("biomass_tif",  VIZ["biomass_cmap"]),
        "Carbon Stock (tC/ha)":    ("carbon_tif",   VIZ["carbon_cmap"]),
        "CO₂e (tCO₂e/ha)":        ("co2e_tif",     VIZ["co2e_cmap"]),
        "Sequestration Potential": ("seq_tif",       None),
    }
    tif_key, cmap = tif_map[layer]
    arr, bounds   = load_raster(tif_key)

    if arr is not None:
        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#f8fdf8")
        ax.set_facecolor("#e8f5e9")
        ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        if layer == "Sequestration Potential":
            cols_  = ["#f5f5f5"] + VIZ["seq_colors"]
            cmap_  = mcolors.ListedColormap(cols_)
            norm_  = mcolors.BoundaryNorm([0,1,2,3,4,5], cmap_.N)
            ax.imshow(arr, cmap=cmap_, norm=norm_, extent=ext,
                      aspect="auto", interpolation="nearest")
            patches = [mpatches.Patch(color=cols_[0], label="Dense Forest (N/A)")]
            patches += [mpatches.Patch(color=cols_[i],
                        label=f"Zone {i}: {SEQ['zones'][i]}") for i in range(1,5)]
            ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.9)
        else:
            vmax = np.nanpercentile(arr[arr > 0], 98)
            im   = ax.imshow(arr, cmap=cmap, extent=ext, vmin=0, vmax=vmax,
                             aspect="auto", interpolation="bilinear")
            cb   = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label(layer, fontsize=10)

        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)",  fontsize=9)
        ax.set_title(f"{layer} — Nainital District, Uttarakhand",
                     fontsize=12, fontweight="bold", color="#1a5c38")
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if layer != "Sequestration Potential":
            valid = arr[(arr > 0) & ~np.isnan(arr)]
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Min",    f"{np.nanmin(valid):.1f}")
            s2.metric("Mean",   f"{np.nanmean(valid):.1f}")
            s3.metric("Median", f"{np.nanmedian(valid):.1f}")
            s4.metric("Max",    f"{np.nanmax(valid):.1f}")
    else:
        st.info("Output rasters not found. Run `python src/models/predict.py` first.")

    st.markdown("---")
    four = ROOT / PATHS["output_figs"] / "map_four_panel.png"
    if four.exists():
        st.markdown("#### Four-Panel Overview")
        st.image(str(four), use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Model Evaluation</div>',
                unsafe_allow_html=True)

    if not metrics.empty:
        ca, cb = st.columns([1, 2])
        with ca:
            st.markdown("#### Metrics")
            disp = metrics.copy()
            for c in ["MAE","RMSE","R2","R2_log"]:
                if c in disp.columns:
                    disp[c] = disp[c].round(3)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            best_r2 = metrics["R2_log"].max() if "R2_log" in metrics.columns \
                      else metrics["R2"].max()
            st.progress(min(float(best_r2)/0.70, 1.0))
            color = "green" if best_r2 >= 0.70 else "orange"
            st.markdown(
                f"<span style='color:{color};font-weight:700;'>"
                f"R²(log) = {best_r2:.4f} / 0.70 target</span>",
                unsafe_allow_html=True)

        with cb:
            p = ROOT / PATHS["output_figs"] / "r2_by_elevation_xgboost.png"
            if p.exists():
                st.markdown("#### R² by Elevation Band")
                st.image(str(p), use_container_width=True)

    st.markdown("---")
    cc, cd = st.columns(2)
    with cc:
        p = ROOT / PATHS["output_figs"] / "scatter_actual_vs_predicted.png"
        if p.exists():
            st.markdown("#### Actual vs Predicted")
            st.image(str(p), use_container_width=True)
    with cd:
        p = ROOT / PATHS["output_figs"] / "residual_distribution.png"
        if p.exists():
            st.markdown("#### Residuals")
            st.image(str(p), use_container_width=True)

    ce, cf = st.columns(2)
    with ce:
        p = ROOT / PATHS["output_figs"] / "feature_importance_xgboost.png"
        if p.exists():
            st.markdown("#### XGBoost Feature Importance")
            st.image(str(p), use_container_width=True)
    with cf:
        p = ROOT / PATHS["output_figs"] / "feature_importance_random_forest.png"
        if p.exists():
            st.markdown("#### Random Forest Feature Importance")
            st.image(str(p), use_container_width=True)

    st.info("""
    **Key Finding:** R²(log)=0.65 overall, but R²=0.63 below 700m vs R²=0.30 above 1200m.
    This matches published GEDI waveform distortion effects on slopes >25°
    (Duncanson et al. 2022, *Remote Sensing of Environment*).
    """)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — GEDI EXPLORER
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">GEDI L4A Footprint Explorer</div>',
                unsafe_allow_html=True)

    if train_df is not None:
        st.markdown(f"**{len(train_df):,} GEDI footprints** · mean AGBD: "
                    f"{train_df['agbd'].mean():.1f} t/ha · "
                    f"mean NDVI: {train_df['NDVI'].mean():.3f} · "
                    f"mean elevation: {train_df['elevation'].mean():.0f} m")

        g1, g2 = st.columns(2)
        with g1:
            fig = px.histogram(train_df, x="agbd", nbins=60,
                               color_discrete_sequence=["#2e7d52"],
                               labels={"agbd":"AGBD (t/ha)"},
                               template="plotly_white",
                               title="AGBD Distribution")
            fig.add_vline(x=train_df["agbd"].mean(), line_dash="dash",
                          line_color="red",
                          annotation_text=f"Mean={train_df['agbd'].mean():.1f}")
            fig.update_layout(height=350, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            samp = train_df.sample(min(5000, len(train_df)), random_state=42)
            fig = px.scatter(samp, x="NDVI", y="agbd", color="elevation",
                             color_continuous_scale="YlGn", opacity=0.4,
                             labels={"agbd":"AGBD (t/ha)","elevation":"Elev (m)"},
                             template="plotly_white", title="AGBD vs NDVI")
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=350, margin=dict(t=40,b=10))
            st.plotly_chart(fig, use_container_width=True)

        bins   = [0,700,1200,1700,2200,9999]
        labels = ["<700m","700-1200m","1200-1700m","1700-2200m",">2200m"]
        train_df["elev_band"] = pd.cut(train_df["elevation"], bins=bins, labels=labels)
        band_stats = train_df.groupby("elev_band", observed=True)["agbd"].agg(
            ["mean","std","count"]).reset_index()
        band_stats.columns = ["Elevation Band","Mean AGBD","Std","Count"]

        fig = px.bar(band_stats, x="Elevation Band", y="Mean AGBD", error_y="Std",
                     color="Mean AGBD", color_continuous_scale="YlGn",
                     text="Count", template="plotly_white",
                     title="Mean AGBD by Elevation Band")
        fig.update_traces(texttemplate="n=%{text:,}", textposition="outside")
        fig.update_layout(height=380, margin=dict(t=40,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Spatial Distribution")
        msamp = train_df.sample(min(8000, len(train_df)), random_state=1)
        fig = px.scatter_mapbox(
            msamp, lat="latitude", lon="longitude",
            color="agbd", color_continuous_scale="YlGn",
            zoom=9, mapbox_style="carto-positron",
            labels={"agbd":"AGBD (t/ha)"},
            hover_data={"agbd":":.1f","NDVI":":.3f","elevation":":.0f"},
        )
        fig.update_traces(marker=dict(size=4, opacity=0.6))
        fig.update_layout(height=480, margin=dict(t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Training CSV not found. Run the pipeline first.")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Live Biomass Predictor</div>',
                unsafe_allow_html=True)
    st.markdown("Adjust spectral and terrain inputs to get an instant AGBD prediction.")

    if model is not None:
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Spectral Indices**")
            ndvi  = st.slider("NDVI",  -0.2, 1.0, 0.70, 0.01)
            evi   = st.slider("EVI",   -0.2, 1.0, 0.55, 0.01)
            ndmi  = st.slider("NDMI",  -0.5, 1.0, 0.35, 0.01)
            nbr   = st.slider("NBR",   -0.5, 1.0, 0.45, 0.01)
            savi  = st.slider("SAVI",  -0.2, 1.0, 0.50, 0.01)
        with p2:
            st.markdown("**Raw Bands (reflectance 0–1)**")
            b8    = st.slider("B8  NIR",    0.0, 0.8, 0.35, 0.01)
            b11   = st.slider("B11 SWIR1",  0.0, 0.6, 0.15, 0.01)
            b12   = st.slider("B12 SWIR2",  0.0, 0.5, 0.10, 0.01)
        with p3:
            st.markdown("**Terrain**")
            elev   = st.slider("Elevation (m)", 300, 3000, 1200, 50)
            slope  = st.slider("Slope (°)",       0,   60,   18,  1)
            aspect = st.slider("Aspect (°)",       0,  360,  180,  5)
            lat    = st.slider("Latitude",        29.0, 29.7, 29.35, 0.01)
            lon    = st.slider("Longitude",       79.0, 79.9, 79.45, 0.01)

        # Build and engineer input
        row = pd.DataFrame([{
            "NDVI":ndvi,"EVI":evi,"NDMI":ndmi,"NBR":nbr,"SAVI":savi,
            "B8":b8,"B11":b11,"B12":b12,
            "elevation":elev,"slope":slope,"aspect":aspect,
            "latitude":lat,"longitude":lon,
        }])

        # Apply same engineering as train.py
        row["NDVI_sq"]      = row["NDVI"]**2
        row["EVI_sq"]       = row["EVI"]**2
        row["NBR_sq"]       = row["NBR"]**2
        row["NDVI_x_NDMI"]  = row["NDVI"]*row["NDMI"]
        row["NDVI_x_NBR"]   = row["NDVI"]*row["NBR"]
        row["NDVI_x_elev"]  = row["NDVI"]*row["elevation"]/1000.0
        row["EVI_x_elev"]   = row["EVI"]*row["elevation"]/1000.0
        row["NBR_x_elev"]   = row["NBR"]*row["elevation"]/1000.0
        row["NDMI_x_slope"] = row["NDMI"]*row["slope"]
        row["NDVI_x_slope"] = row["NDVI"]*row["slope"]
        row["aspect_sin"]   = np.sin(np.radians(row["aspect"]))
        row["aspect_cos"]   = np.cos(np.radians(row["aspect"]))
        row["elev_sq"]      = (row["elevation"]/1000.0)**2

        feat_path = ROOT / "outputs/models/feature_list.txt"
        features  = feat_path.read_text().splitlines() if feat_path.exists() \
                    else CFG["features"]["model_features"]

        X_inp      = row[features].values
        pred_log   = float(model.predict(X_inp)[0])
        biomass    = float(np.expm1(max(pred_log, 0)))
        carbon     = biomass * IPCC["biomass_to_carbon"]
        co2e       = carbon  * IPCC["carbon_to_co2e"]

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("🌿 AGBD",          f"{biomass:.1f} t/ha")
        r2.metric("🔵 Carbon Stock",  f"{carbon:.1f} tC/ha")
        r3.metric("🌍 CO₂e Stored",   f"{co2e:.1f} tCO₂e/ha")
        r4.metric("📏 Uncertainty",   "±56.3 t/ha", help="Model MAE on test set")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=biomass,
            delta={"reference": 105.4, "valueformat":".1f",
                   "suffix":" vs district mean"},
            gauge={
                "axis": {"range":[0,400]},
                "bar":  {"color":"#2e7d52"},
                "steps":[
                    {"range":[0,80],   "color":"#fff9c4"},
                    {"range":[80,180], "color":"#c8e6c9"},
                    {"range":[180,300],"color":"#66bb6a"},
                    {"range":[300,400],"color":"#2e7d52"},
                ],
                "threshold":{"line":{"color":"red","width":3},
                             "thickness":0.75,"value":392},
            },
            title={"text":"Predicted AGBD (t/ha)"},
            number={"suffix":" t/ha","valueformat":".1f"},
        ))
        fig_g.update_layout(height=300, margin=dict(t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    else:
        st.warning("No model found. Run `python src/models/train.py` first.")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Methodology</div>',
                unsafe_allow_html=True)
    st.markdown("""
### Pipeline

```
Sentinel-2 (GEE) ──┐
GEDI L4A (NASA)  ──┼──► Feature Engineering (26 features) ──► XGBoost ──► Carbon Maps
SRTM DEM         ──┘                                                   ──► District Summary
```

### Data Sources
| Source | Product | Resolution | Role |
|--------|---------|------------|------|
| ESA Copernicus | Sentinel-2 SR Harmonised | 20 m | Spectral features |
| NASA / LARSE | GEDI L4A v002 | ~25 m footprint | Target: AGBD (t/ha) |
| USGS | SRTM DEM | 30 m | Elevation, slope, aspect |

### Features (26 total)
**Spectral:** NDVI, EVI, NDMI, NBR, SAVI, B8, B11, B12

**Engineered:** NDVI², EVI², NBR², NDVI×NDMI, NDVI×NBR, NDVI×elev,
EVI×elev, NBR×elev, NDMI×slope, NDVI×slope, elevation², aspect→sin/cos

### Model Training
- **Target:** log₁p(AGBD) — corrects right-skewed distribution
- **Outlier cap:** 95th percentile (392 t/ha) removes GEDI artefacts
- **Split:** 80/20 stratified by AGBD quantile
- **Tuning:** RandomizedSearchCV (30 iterations, 5-fold CV)

### Carbon Conversion (IPCC 2006)
```
Carbon (tC/ha)   = AGBD × 0.47
CO₂e (tCO₂e/ha) = Carbon × 3.67
```

### Results
| Metric | Value |
|--------|-------|
| GEDI footprints | 117,933 |
| Forest pixels predicted | 18,646,521 |
| Forest area | 7,458 km² |
| Total carbon stock | 36.95 MtC |
| Total CO₂e stored | 135.6 MtCO₂e |
| XGBoost R² (log) | 0.649 |
| XGBoost MAE | 56.3 t/ha |

### Key Finding — Elevation-Dependent Performance
| Elevation Band | R² | Interpretation |
|----------------|-----|----------------|
| <700m | 0.63 | Dense mixed broadleaf — GEDI reliable |
| 700–1200m | 0.30 | Steep slopes — waveform distortion |
| 1200–1700m | 0.30 | Conifer — canopy saturation |
| >2200m | 0.21 | Sub-alpine — near GEDI limits |

GEDI L4A prediction quality degrades with elevation in the Himalayas,
consistent with waveform distortion on slopes >25° *(Duncanson et al. 2022)*.

### References
- Duncanson et al. (2022). *Remote Sensing of Environment*, 270, 112845.
- IPCC (2006). Guidelines for National Greenhouse Gas Inventories, Vol. 4.
- Potapov et al. (2021). *Remote Sensing of Environment*, 251, 112165.
    """)
    st.code(
        'Khan, S. (2025). Forest Carbon Stock Estimation in the Himalayan Ecosystem '
        'of Uttarakhand Using Sentinel-2 and GEDI LiDAR Data. ',
        language="text"
    )
