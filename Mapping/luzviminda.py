import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import requests
import zipfile
import io

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def process_data(csv_file, provinces_gdf):
    df = pd.read_excel(csv_file)

    df.rename(columns={"Geolocation": "province"}, inplace=True)
    df["province"] = df["province"].apply(lambda x: re.sub(r"^\.*\s*", "", str(x))).str.strip().str.title()

    province_mapping = {
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

    df["province"] = df["province"].replace(province_mapping)

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
    df["average_harvested_area"] = df[year_columns].mean(axis=1)

    merged = provinces_gdf.merge(df, left_on="NAME_1", right_on="province", how="left")
    merged["average_harvested_area"] = merged["average_harvested_area"].fillna(0)

    return merged


def create_region_map(merged_data, region, output_dir):
    """Create a map for a specific region (Luzon, Visayas, or Mindanao)"""

    region_bounds = {
        'luzon': {'x': [117, 124], 'y': [12, 19]},
        'visayas': {'x': [121.5, 126], 'y': [9, 13]},
        'mindanao': {'x': [121, 127], 'y': [5, 10]}
    }

    bounds = region_bounds[region.lower()]

    color_bins = [0, 15, 30, 46, 61, 100]
    distinct_colors = ["#D3D3D3", "#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    merged_data[merged_data["average_harvested_area"] == 0].plot(
        color="#D3D3D3", linewidth=0.8, edgecolor='black', ax=ax
    )

    for i in range(len(color_bins) - 1):
        lower = color_bins[i]
        upper = color_bins[i + 1]
        color = distinct_colors[i + 1] if i < len(distinct_colors) - 1 else distinct_colors[-1]
        mask = (merged_data["average_harvested_area"] > lower) & (merged_data["average_harvested_area"] <= upper)
        merged_data[mask].plot(color=color, linewidth=0.3, edgecolor='black', ax=ax)

    ax.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.set_xlim(bounds['x'])
    ax.set_ylim(bounds['y'])

    # Scale bar
    scale_lengths = {'luzon': 100, 'visayas': 50, 'mindanao': 100}
    scale_bar_length = scale_lengths[region.lower()]

    x_pos = bounds['x'][0] + (bounds['x'][1] - bounds['x'][0]) * 0.1
    y_pos = bounds['y'][0] + (bounds['y'][1] - bounds['y'][0]) * 0.1

    ax.plot([x_pos, x_pos + scale_bar_length / 111], [y_pos, y_pos], 'k-', linewidth=1)
    ax.text(x_pos, y_pos - 0.1, '0', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 222, y_pos - 0.1, f'{scale_bar_length / 2}', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 111, y_pos - 0.1, f'{scale_bar_length}', ha='center', fontsize=7)
    ax.text(x_pos + scale_bar_length / 222, y_pos - 0.2, 'km', ha='center', fontsize=7)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}° E"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}° N"))
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.set_title(f"Average Banana Yield in {region.title()} (2010-2024)",
                 fontsize=14, loc='center', pad=10)

    ax.text(0.98, 0.98, "Average\nYield\n(tons/ha)",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    cax = fig.add_axes([0.7, 0.6, 0.02, 0.2])
    norm = BoundaryNorm(color_bins, len(distinct_colors) - 1)
    legend_cmap = ListedColormap(distinct_colors[1:])

    sm = plt.cm.ScalarMappable(cmap=legend_cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cax, ticks=color_bins, orientation='vertical')
    cbar.set_ticks(color_bins)
    cbar.set_ticklabels(['0', '15', '30', '46', '61', '100'])
    cax.axhline(y=0, color='lightgray', linewidth=3.0)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)

    output_path = os.path.join(output_dir, f"banana_yield_map_{region.lower()}.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"{region.title()} map saved as {output_path}")


if __name__ == "__main__":
    provinces_gdf = download_and_load_gadm_data()
    csv_file = os.path.join(SCRIPT_DIR, "banana_yield_2010-2024.xlsx")

    if os.path.exists(csv_file):
        merged_data = process_data(csv_file, provinces_gdf)

        create_region_map(merged_data, 'luzon', SCRIPT_DIR)
        create_region_map(merged_data, 'visayas', SCRIPT_DIR)
        create_region_map(merged_data, 'mindanao', SCRIPT_DIR)

        print("All regional maps created successfully!")
    else:
        print("CSV file not found! Make sure it's in the script directory.")
