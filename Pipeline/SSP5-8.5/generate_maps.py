"""Generate static choropleth maps for SSP5-8.5 banana yield projections.

Produces two map types per region:
  1. Projected yield (absolute t/ha) — same color scale as historical
  2. Percent change from historical — diverging red/green scale
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
GADM_SHP = os.path.join(PROJECT_ROOT, "Mapping", "gadm_data", "gadm41_PHL_1.shp")
SCENARIO_LABEL = "SSP5-8.5"
SCENARIO_PERIOD = "2025–2034"

# Absolute yield bins (same as historical)
YIELD_BINS = [0, 15, 30, 46, 61, 100]
YIELD_COLORS = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]

# Percent-change bins & diverging colors (red → white → green)
PCT_BINS = [-50, -30, -15, 0, 15, 50, 220]
PCT_COLORS = ["#b2182b", "#ef8a62", "#fddbc7", "#d9f0d3", "#7fbf7b", "#1b7837"]
PCT_LABELS = ["< −30%", "−30 to −15%", "−15 to 0%", "0 to 15%", "15 to 50%", "> 50%"]

# Map CSV province names → GADM NAME_1 (title-cased)
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

REGION_BOUNDS = {
    "philippines": {"x": [117, 127], "y": [4.5, 21], "scale_km": 300, "figsize": (7, 9)},
    "luzon": {"x": [117, 124], "y": [12, 19], "scale_km": 100, "figsize": (10, 8)},
    "visayas": {"x": [121.5, 126], "y": [9, 13], "scale_km": 50, "figsize": (10, 8)},
    "mindanao": {"x": [121, 127], "y": [5, 10], "scale_km": 100, "figsize": (10, 8)},
}


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


def load_ssp_data():
    csv_path = os.path.join(SCRIPT_DIR, "province_summary.csv")
    df = pd.read_csv(csv_path)
    df["province"] = df["province"].str.strip()
    df["province"] = df["province"].replace(PROVINCE_MAPPING)

    yield_col = "Future Avg (2025–2034)"
    pct_col = "% Change"
    df[yield_col] = pd.to_numeric(df[yield_col], errors="coerce")
    df[pct_col] = pd.to_numeric(df[pct_col], errors="coerce")

    grouped = df.groupby("province").agg({yield_col: "mean", pct_col: "mean"}).reset_index()
    grouped.columns = ["province", "future_yield", "pct_change"]
    return grouped


def merge_data(gdf, yield_df):
    merged = gdf.merge(yield_df, left_on="NAME_1", right_on="province", how="left")
    merged["future_yield"] = merged["future_yield"].fillna(0)
    merged["pct_change"] = merged["pct_change"].fillna(np.nan)
    return merged


def _add_scale_bar(ax, bounds, region):
    scale_km = bounds["scale_km"]
    x_pos = bounds["x"][0] + (bounds["x"][1] - bounds["x"][0]) * 0.1
    y_pos = bounds["y"][0] + (bounds["y"][1] - bounds["y"][0]) * 0.1
    y_offset = 0.25 if region == "philippines" else 0.1
    ax.plot([x_pos, x_pos + scale_km / 111], [y_pos, y_pos], "k-", linewidth=1)
    ax.text(x_pos, y_pos - y_offset, "0", ha="center", fontsize=7)
    ax.text(x_pos + scale_km / 222, y_pos - y_offset, f"{scale_km // 2}", ha="center", fontsize=7)
    ax.text(x_pos + scale_km / 111, y_pos - y_offset, f"{scale_km}", ha="center", fontsize=7)
    ax.text(x_pos + scale_km / 222, y_pos - y_offset * 2, "km", ha="center", fontsize=7)


def _format_axes(ax, bounds):
    ax.set_facecolor("white")
    ax.grid(True, linestyle="-", alpha=0.3, color="gray", zorder=0)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}° E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}° N"))
    ax.tick_params(axis="both", which="major", labelsize=8)


def create_yield_map(merged_data, region, output_dir):
    """Absolute projected yield map."""
    bounds = REGION_BOUNDS[region]
    fig, ax = plt.subplots(1, 1, figsize=bounds["figsize"])

    no_data = merged_data[merged_data["future_yield"] == 0]
    if not no_data.empty:
        no_data.plot(color="#D3D3D3", linewidth=0.8, edgecolor="black", ax=ax, aspect=None)

    for i in range(len(YIELD_BINS) - 1):
        lower, upper = YIELD_BINS[i], YIELD_BINS[i + 1]
        mask = (merged_data["future_yield"] > lower) & (merged_data["future_yield"] <= upper)
        subset = merged_data[mask]
        if not subset.empty:
            subset.plot(color=YIELD_COLORS[i], linewidth=0.3, edgecolor="black", ax=ax, aspect=None)

    _format_axes(ax, bounds)
    _add_scale_bar(ax, bounds, region)

    if region == "philippines":
        title = f"Projected Banana Yield under {SCENARIO_LABEL} ({SCENARIO_PERIOD})"
    else:
        title = f"Projected Banana Yield in {region.title()} under {SCENARIO_LABEL} ({SCENARIO_PERIOD})"
    ax.set_title(title, fontsize=12 if region == "philippines" else 14, loc="center", pad=10)

    ax.text(
        0.98, 0.98, "Projected\nYield\n(tons/ha)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    cax = fig.add_axes([0.7, 0.6, 0.02, 0.2])
    norm = BoundaryNorm(YIELD_BINS, len(YIELD_COLORS))
    sm = plt.cm.ScalarMappable(cmap=ListedColormap(YIELD_COLORS), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, ticks=YIELD_BINS, orientation="vertical")
    cbar.set_ticklabels(["0", "15", "30", "46", "61", "100"])
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    filename = "banana_yield_map_ssp585.png" if region == "philippines" else f"banana_yield_map_{region}_ssp585.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Yield map ({region.title()}) saved")


def create_pct_change_map(merged_data, region, output_dir):
    """Percent-change map with diverging red/green colors."""
    bounds = REGION_BOUNDS[region]
    fig, ax = plt.subplots(1, 1, figsize=bounds["figsize"])

    no_data = merged_data[merged_data["pct_change"].isna()]
    if not no_data.empty:
        no_data.plot(color="#D3D3D3", linewidth=0.8, edgecolor="black", ax=ax, aspect=None)

    has_data = merged_data[merged_data["pct_change"].notna()]
    for i in range(len(PCT_BINS) - 1):
        lower, upper = PCT_BINS[i], PCT_BINS[i + 1]
        mask = (has_data["pct_change"] > lower) & (has_data["pct_change"] <= upper)
        subset = has_data[mask]
        if not subset.empty:
            subset.plot(color=PCT_COLORS[i], linewidth=0.3, edgecolor="black", ax=ax, aspect=None)
    # Anything <= first bin
    extreme_low = has_data[has_data["pct_change"] <= PCT_BINS[0]]
    if not extreme_low.empty:
        extreme_low.plot(color=PCT_COLORS[0], linewidth=0.3, edgecolor="black", ax=ax, aspect=None)

    _format_axes(ax, bounds)
    _add_scale_bar(ax, bounds, region)

    if region == "philippines":
        title = f"Projected Yield Change (%) under {SCENARIO_LABEL} ({SCENARIO_PERIOD})"
    else:
        title = f"Projected Yield Change (%) in {region.title()} under {SCENARIO_LABEL} ({SCENARIO_PERIOD})"
    ax.set_title(title, fontsize=12 if region == "philippines" else 14, loc="center", pad=10)

    ax.text(
        0.98, 0.98, "Change from\nHistorical\n(2010–2024)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=3),
    )

    # Legend
    cax = fig.add_axes([0.7, 0.55, 0.02, 0.25])
    norm = BoundaryNorm(PCT_BINS, len(PCT_COLORS))
    sm = plt.cm.ScalarMappable(cmap=ListedColormap(PCT_COLORS), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, ticks=PCT_BINS, orientation="vertical")
    cbar.set_ticklabels([f"{b}%" for b in PCT_BINS])
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)

    # No data legend entry
    ax.plot([], [], "s", color="#D3D3D3", markersize=8, label="No data")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    filename = "banana_yield_pct_change_ssp585.png" if region == "philippines" else f"banana_yield_pct_change_{region}_ssp585.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Pct-change map ({region.title()}) saved")


if __name__ == "__main__":
    print(f"Generating {SCENARIO_LABEL} choropleth maps...")
    gdf = load_gadm()
    yield_df = load_ssp_data()
    merged = merge_data(gdf, yield_df)

    for region in REGION_BOUNDS:
        create_yield_map(merged, region, SCRIPT_DIR)
        create_pct_change_map(merged, region, SCRIPT_DIR)

    print("All SSP5-8.5 maps created successfully!")
