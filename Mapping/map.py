import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
from matplotlib.colors import ListedColormap, BoundaryNorm
import requests
import zipfile
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COLOR_BINS = [0, 15, 30, 46, 61, 100]
COLORS = ["#D3D3D3", "#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]

PROVINCE_MAPPING = {
    "North Cotabato": "Cotabato",
    "Compostela Valley": "Davao De Oro",
    "Davao De Oro": "Davao De Oro",
    "Maguindanao Del Norte": "Maguindanao",
    "Maguindanao Del Sur": "Maguindanao",
    "City Of Davao": "Davao Del Sur",
    "City Of Zamboanga": "Zamboanga Del Sur",
    "Metropolitan Manila": "Ncr",
    "Cotabato": "North Cotabato",
    "Davao Occidental": "Davao Del Sur",
}

REGION_BOUNDS = {
    "philippines": {"x": [117, 127], "y": [4.5, 21], "scale_km": 300, "figsize": (7, 9)},
    "luzon": {"x": [117, 124], "y": [12, 19], "scale_km": 100, "figsize": (10, 8)},
    "visayas": {"x": [121.5, 126], "y": [9, 13], "scale_km": 50, "figsize": (10, 8)},
    "mindanao": {"x": [121, 127], "y": [5, 10], "scale_km": 100, "figsize": (10, 8)},
}


def download_and_load_gadm_data():
    """Download Philippines provincial boundaries from GADM"""
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_PHL_shp.zip"

    data_dir = os.path.join(SCRIPT_DIR, "gadm_data")
    shp_path = os.path.join(data_dir, "gadm41_PHL_1.shp")

    if not os.path.exists(shp_path):
        print("Downloading Philippines provincial boundaries...")
        os.makedirs(data_dir, exist_ok=True)
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        print("Download complete.")

    provinces = gpd.read_file(shp_path)
    provinces["NAME_1"] = provinces["NAME_1"].str.strip().str.title().replace({
        "Compostela Valley": "Davao De Oro",
        "Davao Occidental": "Davao Del Sur"
    })
    return provinces


def process_data(data_file, provinces_gdf):
    """Load yield data, clean province names, merge with GADM boundaries."""
    df = pd.read_excel(data_file)

    df.rename(columns={"Geolocation": "province"}, inplace=True)
    df["province"] = df["province"].apply(lambda x: re.sub(r"^\.*\s*", "", str(x))).str.strip().str.title()
    df["province"] = df["province"].replace(PROVINCE_MAPPING)

    region_keywords = [
        "Region", "Autonomous Region", "Caraga", "Calabarzon", "Mimaropa", "Barmm", "Cordillera"
    ]
    df = df[~df["province"].str.contains("|".join(region_keywords), case=False, na=False)]
    df = df[df["province"] != "Philippines"]

    # Combine Maguindanao if both exist
    if sum(df["province"] == "Maguindanao") > 1:
        year_columns = [col for col in df.columns if "Annual" in col]
        avg_data = {"province": "Maguindanao"}
        for col in year_columns:
            avg_data[col] = df[df["province"] == "Maguindanao"][col].mean()
        df = df[df["province"] != "Maguindanao"]
        df = pd.concat([df, pd.DataFrame([avg_data])], ignore_index=True)

    year_columns = [col for col in df.columns if "Annual" in col]
    df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')
    df["average_yield"] = df[year_columns].mean(axis=1)
    df["cv_yield"] = (df[year_columns].std(axis=1) / df[year_columns].mean(axis=1)) * 100

    merged = provinces_gdf.merge(df, left_on="NAME_1", right_on="province", how="left")
    merged["average_yield"] = merged["average_yield"].fillna(0)
    merged["cv_yield"] = merged["cv_yield"].fillna(0)

    return merged


def create_map(merged_data, region, output_dir):
    """Create a choropleth map for a given region."""
    bounds = REGION_BOUNDS[region]

    fig, ax = plt.subplots(1, 1, figsize=bounds["figsize"])

    # Plot provinces with no data
    merged_data[merged_data["average_yield"] == 0].plot(
        color="#D3D3D3", linewidth=0.8, edgecolor='black', ax=ax
    )

    # Plot each color bin
    for i in range(len(COLOR_BINS) - 1):
        lower = COLOR_BINS[i]
        upper = COLOR_BINS[i + 1]
        color = COLORS[i + 1] if i < len(COLORS) - 1 else COLORS[-1]
        mask = (merged_data["average_yield"] > lower) & (merged_data["average_yield"] <= upper)
        merged_data[mask].plot(color=color, linewidth=0.3, edgecolor='black', ax=ax)

    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.set_xlim(bounds['x'])
    ax.set_ylim(bounds['y'])

    # Scale bar
    scale_bar_length = bounds["scale_km"]
    x_pos = bounds['x'][0] + (bounds['x'][1] - bounds['x'][0]) * 0.1
    y_pos = bounds['y'][0] + (bounds['y'][1] - bounds['y'][0]) * 0.1
    y_offset = 0.25 if region == "philippines" else 0.1

    ax.plot([x_pos, x_pos + scale_bar_length / 111], [y_pos, y_pos], 'k-', linewidth=1)
    ax.text(x_pos, y_pos - y_offset, '0', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 222, y_pos - y_offset, f'{scale_bar_length / 2}', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 111, y_pos - y_offset, f'{scale_bar_length}', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 222, y_pos - y_offset * 2, 'km', ha='center', fontsize=7)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}° E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}° N"))
    ax.tick_params(axis='both', which='major', labelsize=8)

    title = "Average Banana Yield (2010-2024)" if region == "philippines" else f"Average Banana Yield in {region.title()} (2010-2024)"
    ax.set_title(title, fontsize=12 if region == "philippines" else 14, loc='center', pad=10)

    ax.text(0.98, 0.98, "Average\nYield\n(tons/ha)",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    cax = fig.add_axes([0.7, 0.6, 0.02, 0.2])
    norm = BoundaryNorm(COLOR_BINS, len(COLORS) - 1)
    legend_cmap = ListedColormap(COLORS[1:])

    sm = plt.cm.ScalarMappable(cmap=legend_cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax, ticks=COLOR_BINS, orientation='vertical')
    cbar.set_ticks(COLOR_BINS)
    cbar.set_ticklabels(['0', '15', '30', '46', '61', '100'])
    cax.axhline(y=0, color='lightgray', linewidth=3.0)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)

    filename = "banana_yield_map_gadm.png" if region == "philippines" else f"banana_yield_map_{region}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"{region.title()} map saved as {output_path}")


CV_BINS = [0, 10, 20, 30, 50, 100]
CV_COLORS = ["#D3D3D3", "#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]


def create_combined_map(merged_data, output_dir):
    """Create a two-panel Figure 2: (a) Mean Yield and (b) Coefficient of Variation."""
    bounds = REGION_BOUNDS["philippines"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9))

    # --- Panel (a): Mean Yield ---
    merged_data[merged_data["average_yield"] == 0].plot(
        color="#D3D3D3", linewidth=0.8, edgecolor='black', ax=ax1
    )
    for i in range(len(COLOR_BINS) - 1):
        lower = COLOR_BINS[i]
        upper = COLOR_BINS[i + 1]
        color = COLORS[i + 1] if i < len(COLORS) - 1 else COLORS[-1]
        mask = (merged_data["average_yield"] > lower) & (merged_data["average_yield"] <= upper)
        merged_data[mask].plot(color=color, linewidth=0.3, edgecolor='black', ax=ax1)

    _style_axis(ax1, bounds, "(a) Mean Yield (tons/ha)")

    # Mean Yield colorbar (inside panel a, upper-right)
    norm1 = BoundaryNorm(COLOR_BINS, len(COLORS) - 1)
    cmap1 = ListedColormap(COLORS[1:])
    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    sm1.set_array([])
    pos1 = ax1.get_position()
    cax1 = fig.add_axes([pos1.x1 - 0.03, pos1.y0 + pos1.height * 0.6, 0.012, pos1.height * 0.3])
    cbar1 = plt.colorbar(sm1, cax=cax1, ticks=COLOR_BINS, orientation='vertical')
    cbar1.set_ticklabels(['0', '15', '30', '46', '61', '100'])
    cbar1.outline.set_edgecolor('black')
    cbar1.outline.set_linewidth(0.5)

    # --- Panel (b): Coefficient of Variation ---
    merged_data[merged_data["cv_yield"] == 0].plot(
        color="#D3D3D3", linewidth=0.8, edgecolor='black', ax=ax2
    )
    for i in range(len(CV_BINS) - 1):
        lower = CV_BINS[i]
        upper = CV_BINS[i + 1]
        color = CV_COLORS[i + 1] if i < len(CV_COLORS) - 1 else CV_COLORS[-1]
        mask = (merged_data["cv_yield"] > lower) & (merged_data["cv_yield"] <= upper)
        merged_data[mask].plot(color=color, linewidth=0.3, edgecolor='black', ax=ax2)

    _style_axis(ax2, bounds, "(b) Coefficient of Variation (%)")

    # CV colorbar (inside panel b, upper-right)
    norm2 = BoundaryNorm(CV_BINS, len(CV_COLORS) - 1)
    cmap2 = ListedColormap(CV_COLORS[1:])
    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    pos2 = ax2.get_position()
    cax2 = fig.add_axes([pos2.x1 - 0.03, pos2.y0 + pos2.height * 0.6, 0.012, pos2.height * 0.3])
    cbar2 = plt.colorbar(sm2, cax=cax2, ticks=CV_BINS, orientation='vertical')
    cbar2.set_ticklabels(['0', '10', '20', '30', '50', '100'])
    cbar2.outline.set_edgecolor('black')
    cbar2.outline.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.05)
    output_path = os.path.join(output_dir, "figure2_mean_yield_and_cv.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 (combined) saved as {output_path}")


def _style_axis(ax, bounds, title):
    """Apply common styling to a map axis."""
    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.set_xlim(bounds['x'])
    ax.set_ylim(bounds['y'])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}° E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}° N"))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_title(title, fontsize=12, loc='center', pad=10)


if __name__ == "__main__":
    provinces_gdf = download_and_load_gadm_data()
    data_file = os.path.join(SCRIPT_DIR, "banana_yield_2010-2024.xlsx")

    if os.path.exists(data_file):
        merged_data = process_data(data_file, provinces_gdf)

        for region in REGION_BOUNDS:
            create_map(merged_data, region, SCRIPT_DIR)

        create_combined_map(merged_data, SCRIPT_DIR)

        print("All maps created successfully!")
    else:
        print("Data file not found! Make sure banana_yield_2010-2024.xlsx is in the script directory.")
