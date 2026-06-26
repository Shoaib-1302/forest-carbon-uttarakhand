"""
maps.py
───────
Phase 5 (cont.): Generate publication-quality maps and plots.

Outputs
───────
    outputs/figures/map_biomass.png
    outputs/figures/map_carbon_stock.png
    outputs/figures/map_co2e.png
    outputs/figures/map_seq_potential.png
    outputs/figures/map_four_panel.png   ← all four in one figure
    outputs/figures/agbd_distribution.png
    Interactive HTML:
    outputs/figures/interactive_biomass.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
import rasterio
import yaml
import folium
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config/config.yaml") as f:
    CFG = yaml.safe_load(f)

PATHS  = CFG["paths"]
VIZ    = CFG["viz"]
IPCC   = CFG["ipcc"]
SEQ    = CFG["seq_potential"]

FIG_DIR = ROOT / PATHS["output_figs"]
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr     = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        bounds  = src.bounds
        nodata  = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, bounds


def _extent(bounds) -> list:
    """Return [xmin, xmax, ymin, ymax] for matplotlib imshow extent."""
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def add_map_furniture(ax, title: str, unit: str) -> None:
    """Add title, axis labels, and a minimal clean frame."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#1a1a1a")
    ax.set_xlabel("Longitude (°E)", fontsize=9, color="#555")
    ax.set_ylabel("Latitude (°N)", fontsize=9, color="#555")
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


# ═══════════════════════════════════════════════════════════════════
# Individual maps
# ═══════════════════════════════════════════════════════════════════

def map_biomass(save=True) -> plt.Figure:
    path = ROOT / PATHS["biomass_tif"]
    arr, bounds = read_raster(path)

    fig, ax = plt.subplots(figsize=VIZ["figsize"])
    vmax = np.nanpercentile(arr[arr > 0], 98)
    im = ax.imshow(arr, cmap=VIZ["biomass_cmap"], extent=_extent(bounds),
                   vmin=0, vmax=vmax, aspect="auto", interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AGBD (t/ha)", fontsize=10)
    add_map_furniture(ax, "Forest Above-Ground Biomass Density — Nainital District", "t/ha")

    # Annotate mean
    mean_val = np.nanmean(arr[arr > 0])
    ax.text(0.97, 0.03, f"Mean: {mean_val:.1f} t/ha",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1a5c38", alpha=0.85))

    plt.tight_layout()
    if save:
        out = FIG_DIR / "map_biomass.png"
        fig.savefig(out, dpi=VIZ["dpi"])
        print(f"  Saved: {out}")
    return fig


def map_carbon(save=True) -> plt.Figure:
    path = ROOT / PATHS["carbon_tif"]
    arr, bounds = read_raster(path)

    fig, ax = plt.subplots(figsize=VIZ["figsize"])
    vmax = np.nanpercentile(arr[arr > 0], 98)
    im = ax.imshow(arr, cmap=VIZ["carbon_cmap"], extent=_extent(bounds),
                   vmin=0, vmax=vmax, aspect="auto", interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Carbon Stock (tC/ha)", fontsize=10)
    add_map_furniture(ax, "Forest Carbon Stock — Nainital District", "tC/ha")
    plt.tight_layout()
    if save:
        out = FIG_DIR / "map_carbon_stock.png"
        fig.savefig(out, dpi=VIZ["dpi"])
        print(f"  Saved: {out}")
    return fig


def map_co2e(save=True) -> plt.Figure:
    path = ROOT / PATHS["co2e_tif"]
    arr, bounds = read_raster(path)

    fig, ax = plt.subplots(figsize=VIZ["figsize"])
    vmax = np.nanpercentile(arr[arr > 0], 98)
    im = ax.imshow(arr, cmap=VIZ["co2e_cmap"], extent=_extent(bounds),
                   vmin=0, vmax=vmax, aspect="auto", interpolation="bilinear")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("CO₂e Stored (tCO₂e/ha)", fontsize=10)
    add_map_furniture(ax, "CO₂ Equivalent Storage — Nainital District", "tCO₂e/ha")
    plt.tight_layout()
    if save:
        out = FIG_DIR / "map_co2e.png"
        fig.savefig(out, dpi=VIZ["dpi"])
        print(f"  Saved: {out}")
    return fig


def map_seq_potential(save=True) -> plt.Figure:
    path = ROOT / PATHS["seq_tif"]
    arr, bounds = read_raster(path)

    colors  = ["#f5f5f5"] + VIZ["seq_colors"]   # 0 = no data / high forest (grey)
    cmap    = mcolors.ListedColormap(colors)
    norm    = mcolors.BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

    fig, ax = plt.subplots(figsize=VIZ["figsize"])
    ax.imshow(arr, cmap=cmap, norm=norm, extent=_extent(bounds),
              aspect="auto", interpolation="nearest")

    zone_labels = SEQ["zones"]
    patches = [
        mpatches.Patch(color=colors[0], label="Dense Forest (N/A)"),
    ] + [
        mpatches.Patch(color=colors[i], label=f"Zone {i}: {zone_labels[i]} Potential")
        for i in range(1, 5)
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")
    add_map_furniture(ax, "Carbon Sequestration Potential — Nainital District", "zone")
    plt.tight_layout()
    if save:
        out = FIG_DIR / "map_seq_potential.png"
        fig.savefig(out, dpi=VIZ["dpi"])
        print(f"  Saved: {out}")
    return fig


def map_four_panel(save=True) -> plt.Figure:
    """All four outputs in a 2×2 panel."""
    tifs = [
        (PATHS["biomass_tif"], VIZ["biomass_cmap"],  "Biomass (t/ha)",     "AGBD (t/ha)"),
        (PATHS["carbon_tif"],  VIZ["carbon_cmap"],   "Carbon Stock",       "tC/ha"),
        (PATHS["co2e_tif"],    VIZ["co2e_cmap"],     "CO₂e Stored",        "tCO₂e/ha"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.ravel()

    for i, (tif_key, cmap, title, unit) in enumerate(tifs):
        path = ROOT / tif_key
        if not path.exists():
            axes[i].set_visible(False)
            continue
        arr, bounds = read_raster(path)
        vmax = np.nanpercentile(arr[arr > 0], 98)
        im = axes[i].imshow(arr, cmap=cmap, extent=_extent(bounds),
                            vmin=0, vmax=vmax, aspect="auto", interpolation="bilinear")
        cb = plt.colorbar(im, ax=axes[i], fraction=0.035, pad=0.02)
        cb.set_label(unit, fontsize=8)
        axes[i].set_title(title, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Lon (°E)", fontsize=8)
        axes[i].set_ylabel("Lat (°N)", fontsize=8)
        axes[i].tick_params(labelsize=7)

    # Sequestration potential in position 4
    seq_path = ROOT / PATHS["seq_tif"]
    if seq_path.exists():
        arr, bounds = read_raster(seq_path)
        colors = ["#f5f5f5"] + VIZ["seq_colors"]
        cmap4  = mcolors.ListedColormap(colors)
        norm4  = mcolors.BoundaryNorm([0,1,2,3,4,5], cmap4.N)
        axes[3].imshow(arr, cmap=cmap4, norm=norm4, extent=_extent(bounds),
                       aspect="auto", interpolation="nearest")
        patches = [mpatches.Patch(color=colors[j], label=f"Zone {j}")
                   for j in range(1, 5)]
        axes[3].legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.85)
        axes[3].set_title("Sequestration Potential", fontsize=11, fontweight="bold")
        axes[3].set_xlabel("Lon (°E)", fontsize=8)
        axes[3].set_ylabel("Lat (°N)", fontsize=8)
        axes[3].tick_params(labelsize=7)

    fig.suptitle(
        "Forest Carbon Assessment — Nainital District, Uttarakhand",
        fontsize=14, fontweight="bold", y=1.01, color="#1a5c38"
    )
    plt.tight_layout()
    if save:
        out = FIG_DIR / "map_four_panel.png"
        fig.savefig(out, dpi=VIZ["dpi"], bbox_inches="tight")
        print(f"  Saved: {out}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# AGBD distribution plot
# ═══════════════════════════════════════════════════════════════════

def plot_agbd_distribution(save=True) -> plt.Figure:
    import pandas as pd
    csv_path = ROOT / PATHS["training_csv"]
    if not csv_path.exists():
        print("Training CSV not found — skipping distribution plot.")
        return None

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(df["agbd"], bins=50, color="#2e7d52", edgecolor="white", linewidth=0.5)
    axes[0].axvline(df["agbd"].mean(), color="#e53935", linestyle="--", lw=1.5,
                    label=f"Mean = {df['agbd'].mean():.1f} t/ha")
    axes[0].axvline(df["agbd"].median(), color="#fb8c00", linestyle="--", lw=1.5,
                    label=f"Median = {df['agbd'].median():.1f} t/ha")
    axes[0].set_xlabel("AGBD (t/ha)", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("AGBD Distribution (GEDI Footprints)", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].spines[["top","right"]].set_visible(False)

    # AGBD vs NDVI scatter
    axes[1].scatter(df["NDVI"], df["agbd"], alpha=0.25, s=8,
                    color="#4caf50", edgecolors="none")
    axes[1].set_xlabel("NDVI", fontsize=11)
    axes[1].set_ylabel("AGBD (t/ha)", fontsize=11)
    axes[1].set_title("AGBD vs NDVI (GEDI footprints)", fontsize=12)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    if save:
        out = FIG_DIR / "agbd_distribution.png"
        fig.savefig(out, dpi=VIZ["dpi"])
        print(f"  Saved: {out}")
    return fig


# ═══════════════════════════════════════════════════════════════════
# Interactive Folium map
# ═══════════════════════════════════════════════════════════════════

def make_interactive_map(save=True) -> folium.Map:
    """
    Lightweight interactive map: GEDI sample points coloured by AGBD.
    Full raster overlay would require tile service; this uses a subset.
    """
    import pandas as pd

    csv_path = ROOT / PATHS["training_csv"]
    if not csv_path.exists():
        print("Training CSV not found — skipping interactive map.")
        return None

    df = pd.read_csv(csv_path).sample(min(2000, len(df := pd.read_csv(csv_path))),
                                       random_state=42)

    centre_lat = (CFG["study_area"]["bbox"]["ymin"] + CFG["study_area"]["bbox"]["ymax"]) / 2
    centre_lon = (CFG["study_area"]["bbox"]["xmin"] + CFG["study_area"]["bbox"]["xmax"]) / 2

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=10,
                   tiles="CartoDB positron")

    # Colour scale: green gradient
    def agbd_color(agbd: float) -> str:
        norm = min(max(agbd / 400, 0), 1)
        r = int(255 * (1 - norm))
        g = int(150 + 105 * norm)
        b = int(50 * (1 - norm))
        return f"#{r:02x}{g:02x}{b:02x}"

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color=agbd_color(row["agbd"]),
            fill=True, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>AGBD:</b> {row['agbd']:.1f} t/ha<br>"
                f"<b>NDVI:</b> {row['NDVI']:.3f}<br>"
                f"<b>Elev:</b> {row['elevation']:.0f} m",
                max_width=200
            )
        ).add_to(m)

    folium.LayerControl().add_to(m)

    if save:
        out = FIG_DIR / "interactive_biomass.html"
        m.save(str(out))
        print(f"  Saved: {out}")
    return m


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Generating maps and figures …\n")

    missing_tifs = []
    for key in ["biomass_tif", "carbon_tif", "co2e_tif", "seq_tif"]:
        if not (ROOT / PATHS[key]).exists():
            missing_tifs.append(PATHS[key])

    if missing_tifs:
        print(f"⚠️  Output rasters not found: {missing_tifs}")
        print("   Run src/models/predict.py first to generate maps.")
        print("   Only distribution and training-data plots will be generated.\n")
    else:
        map_biomass()
        map_carbon()
        map_co2e()
        map_seq_potential()
        map_four_panel()

    plot_agbd_distribution()
    make_interactive_map()
    print("\n✅ All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
