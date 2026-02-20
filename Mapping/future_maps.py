"""Generate future prediction figures: 3-panel maps per SSP scenario.

Each figure shows: (a) Baseline (2010-2024 avg), (b) 2029 prediction, (c) 2034 prediction.
Produces two figures: one for SSP2-4.5, one for SSP5-8.5.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
GADM_SHP = os.path.join(SCRIPT_DIR, "gadm_data", "gadm41_PHL_1.shp")

YIELD_BINS = [0, 15, 30, 46, 61, 100]
YIELD_COLORS = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]
NO_DATA_COLOR = "#D3D3D3"

PROVINCE_MAPPING = {
    "North Cotabato": "Cotabato",
    "Davao de Oro": "Davao De Oro",
    "Davao del Norte": "Davao Del Norte",
    "Davao del Sur": "Davao Del Sur",
    "Davao Occidental": "Davao Del Sur",
    "Agusan del Norte": "Agusan Del Norte",
    "Agusan del Sur": "Agusan Del Sur",
    "Lanao del Norte": "Lanao Del Norte",
    "Lanao del Sur": "Lanao Del Sur",
    "Zamboanga del Norte": "Zamboanga Del Norte",
    "Zamboanga del Sur": "Zamboanga Del Sur",
    "Surigao del Norte": "Surigao Del Norte",
    "Surigao del Sur": "Surigao Del Sur",
    "Maguindanao del Norte": "Maguindanao",
    "Maguindanao del Sur": "Maguindanao",
    "Dinagat Islands": "Dinagat Islands",
    "Compostela Valley": "Davao De Oro",
}

# Historical data also uses slightly different name conventions
HIST_PROVINCE_MAPPING = {
    "North Cotabato": "Cotabato",
    "Compostela Valley": "Davao De Oro",
    "Davao de Oro": "Davao De Oro",
    "Davao del Norte": "Davao Del Norte",
    "Davao del Sur": "Davao Del Sur",
    "Davao Occidental": "Davao Del Sur",
    "Agusan del Norte": "Agusan Del Norte",
    "Agusan del Sur": "Agusan Del Sur",
    "Lanao del Norte": "Lanao Del Norte",
    "Lanao del Sur": "Lanao Del Sur",
    "Zamboanga del Norte": "Zamboanga Del Norte",
    "Zamboanga del Sur": "Zamboanga Del Sur",
    "Surigao del Norte": "Surigao Del Norte",
    "Surigao del Sur": "Surigao Del Sur",
    "Maguindanao del Norte": "Maguindanao",
    "Maguindanao del Sur": "Maguindanao",
    "Dinagat Islands": "Dinagat Islands",
}

BOUNDS = {"x": [117, 127], "y": [4.5, 21]}


def load_gadm():
    if not os.path.exists(GADM_SHP):
        print(f"GADM shapefile not found: {GADM_SHP}")
        print("Run Mapping/map.py first to download the shapefiles.")
        sys.exit(1)
    gdf = gpd.read_file(GADM_SHP)
    gdf["NAME_1"] = gdf["NAME_1"].str.strip().str.title().replace({
        "Compostela Valley": "Davao De Oro",
        "Davao Occidental": "Davao Del Sur",
    })
    return gdf


def load_baseline():
    """Load historical data and compute average yield per province (2010-2024)."""
    hist_path = os.path.join(PROJECT_ROOT, "Training", "data", "banana_yield_2010-2024.xlsx")
    df = pd.read_excel(hist_path)
    df["province"] = df["province"].str.strip()
    df["province"] = df["province"].replace(HIST_PROVINCE_MAPPING)
    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    avg = df.groupby("province")["yield"].mean().reset_index()
    avg.columns = ["province", "yield"]
    return avg


def load_ssp_year(ssp, year):
    """Load predicted yield for a specific SSP scenario and year."""
    pred_path = os.path.join(
        PROJECT_ROOT, "banana-yield-api", "ssp", ssp,
        "banana_yield_predictions_2025-2034.xlsx"
    )
    df = pd.read_excel(pred_path)
    df = df[df["year"] == year][["province", "yield"]].copy()
    df["province"] = df["province"].str.strip()
    df["province"] = df["province"].replace(PROVINCE_MAPPING)
    # Average duplicates (e.g. Maguindanao del Norte/Sur → Maguindanao)
    df = df.groupby("province")["yield"].mean().reset_index()
    return df


def merge(gdf, yield_df):
    merged = gdf.merge(yield_df, left_on="NAME_1", right_on="province", how="left")
    merged["yield"] = merged["yield"].fillna(0)
    return merged


def _plot_panel(ax, merged, title):
    """Plot a single choropleth panel."""
    # No data
    no_data = merged[merged["yield"] == 0]
    if not no_data.empty:
        no_data.plot(color=NO_DATA_COLOR, linewidth=0.8, edgecolor="black", ax=ax, aspect=None)

    # Yield bins
    for i in range(len(YIELD_BINS) - 1):
        lower, upper = YIELD_BINS[i], YIELD_BINS[i + 1]
        mask = (merged["yield"] > lower) & (merged["yield"] <= upper)
        subset = merged[mask]
        if not subset.empty:
            subset.plot(color=YIELD_COLORS[i], linewidth=0.3, edgecolor="black", ax=ax, aspect=None)

    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.3, color="gray", zorder=0)
    ax.set_xlim(BOUNDS["x"])
    ax.set_ylim(BOUNDS["y"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}° E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}° N"))
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_title(title, fontsize=11, loc="center", pad=8)


def create_ssp_figure(gdf, baseline_df, ssp, output_dir):
    """Create a 3-panel figure: baseline, 2029, 2034 for one SSP."""
    ssp_short = ssp.replace("-", "").replace(".", "")  # SSP245 or SSP585

    pred_2029 = load_ssp_year(ssp, 2029)
    pred_2034 = load_ssp_year(ssp, 2034)

    merged_base = merge(gdf, baseline_df)
    merged_2029 = merge(gdf, pred_2029)
    merged_2034 = merge(gdf, pred_2034)

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))

    _plot_panel(axes[0], merged_base, "(a) Baseline (2010\u20132024 Avg)")
    _plot_panel(axes[1], merged_2029, f"(b) {ssp} Projection (2029)")
    _plot_panel(axes[2], merged_2034, f"(c) {ssp} Projection (2034)")

    # Single shared colorbar
    norm = BoundaryNorm(YIELD_BINS, len(YIELD_COLORS))
    cmap = ListedColormap(YIELD_COLORS)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
    cbar = plt.colorbar(sm, cax=cbar_ax, ticks=YIELD_BINS, orientation="vertical")
    cbar.set_ticklabels(["0", "15", "30", "46", "61", "100"])
    cbar.set_label("Yield (tons/ha)", fontsize=10)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.08, right=0.90)

    filename = f"future_yield_{ssp_short}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"{ssp} figure saved: {output_path}")


if __name__ == "__main__":
    print("Loading GADM shapefile...")
    gdf = load_gadm()

    print("Loading baseline (historical average)...")
    baseline_df = load_baseline()

    for ssp in ["SSP2-4.5", "SSP5-8.5"]:
        print(f"\nGenerating {ssp} figure...")
        create_ssp_figure(gdf, baseline_df, ssp, SCRIPT_DIR)

    print("\nDone! Two future prediction figures created.")
