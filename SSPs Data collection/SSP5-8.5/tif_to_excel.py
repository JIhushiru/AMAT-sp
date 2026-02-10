import warnings
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import os
import numpy as np
from shapely.geometry import Point
import time

warnings.filterwarnings("ignore", message=".*Can't decode floating point timedelta to 's'.*")

start_time = time.time()

# === CONFIG ===
features = ['pr', 'tas']
START_YEAR = 2025
END_YEAR = 2034

script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_path = os.path.join(script_dir, "philippines_provinces.geojson")

# === Load GeoJSON ===
gdf = gpd.read_file(geojson_path)
gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].to_crs("EPSG:4326")

if 'name' in gdf.columns:
    gdf['province_name'] = gdf['name']
else:
    name_cols = [c for c in gdf.columns if 'name' in c.lower()]
    gdf['province_name'] = gdf[name_cols[0]] if name_cols else [f"Province_{i}" for i in range(len(gdf))]

# === Process Climate Features ===
feature_results = {}
polygon_used = 0
centroid_used = 0
skipped_provinces = 0

centroid_log_path = os.path.join(script_dir, "centroid_fallback_log.txt")
with open(centroid_log_path, "w") as f:
    f.write("Centroid fallback log:\n")

skipped_log_path = os.path.join(script_dir, "skipped_provinces_log.txt")
with open(skipped_log_path, "w") as f:
    f.write("Skipped provinces (no data found):\n")

for feature in features:
    print(f"\n=== Processing feature: {feature} ===")
    file_path = os.path.join(script_dir, f"gfdl_{feature}_{START_YEAR}_{END_YEAR}_monthly.tif")

    var = rioxarray.open_rasterio(file_path, masked=True)

    num_bands = var.sizes['band']
    var = var.assign_coords(band=pd.date_range(f"{START_YEAR}-01-01", periods=num_bands, freq='MS'))
    var = var.rename({'band': 'time'})

    if 'latitude' in var.dims and 'longitude' in var.dims:
        var = var.rename({'latitude': 'y', 'longitude': 'x'})

    var = var.rio.set_spatial_dims(x_dim='x', y_dim='y')
    var = var.rio.write_crs("EPSG:4326")

    var = var.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    print(f"Dimensions before clipping: {var.dims}")
    print(f"Coordinates: {var.coords}")
    records = []

    for i, row in gdf.iterrows():
        province_name = row['province_name']
        province_geom = gpd.GeoDataFrame([row], crs=gdf.crs)
        fallback_used = False
        province_geom = province_geom.to_crs(var.rio.crs)
        mean_series = None

        try:
            var.rio.clip(province_geom.geometry, province_geom.crs, drop=True)

            valid_points = []
            province_polygon = row.geometry
            for lat in var.coords['y'].values:
                for lon in var.coords['x'].values:
                    point = Point(lon, lat)
                    if point.within(province_polygon):
                        point_data = var.sel(y=lat, x=lon, method="nearest")
                        if not np.isnan(point_data.values).all():
                            valid_points.append(point_data)

            if valid_points:
                mean_series = xr.concat(valid_points, dim='x').mean(dim='x')
                polygon_used += 1
                print(f"  Valid data found using polygon method for {province_name}")
            else:
                raise ValueError("No valid points inside polygon.")

        except Exception as e:
            with open(centroid_log_path, "a") as f:
                f.write(f"{province_name} used centroid fallback for feature: {feature}\n")
            print(f"  Polygon method failed for {province_name}: {e}")
            print(f"  Switching to centroid fallback for {province_name}")
            fallback_used = True
            centroid = row.geometry.centroid
            mean_series = None
            search_radius = 1.0
            step = 0.1

            lons = np.arange(centroid.x - search_radius, centroid.x + search_radius + step, step)
            lats = np.arange(centroid.y - search_radius, centroid.y + search_radius + step, step)

            found = False
            for lat in lats:
                for lon in lons:
                    try:
                        point_data = var.sel(x=lon, y=lat, method="nearest")
                        if not np.isnan(point_data.values).all():
                            print(f"  Valid data found near centroid at ({lat:.4f}, {lon:.4f})")
                            mean_series = point_data
                            centroid_used += 1
                            found = True
                            break
                    except Exception:
                        continue
                if found:
                    break

            if not found:
                print(f"  No valid fallback data found near centroid for {province_name}")
                with open(skipped_log_path, "a") as f:
                    f.write(f"{province_name} - no valid data found\n")
                skipped_provinces += 1
                continue

        if mean_series is None:
            print(f"  Skipping {province_name} - no valid data found")
            with open(skipped_log_path, "a") as f:
                f.write(f"{province_name} - no valid data found\n")
            skipped_provinces += 1
            continue

        df = mean_series.to_dataframe(name=feature).reset_index()
        df['province'] = province_name
        df['method'] = 'centroid' if fallback_used else 'polygon'
        records.append(df)

    if records:
        combined_df = pd.concat(records)
        combined_df['year'] = combined_df['time'].dt.year
        combined_df = combined_df[(combined_df['year'] >= START_YEAR) & (combined_df['year'] <= END_YEAR)]
        yearly_stat = combined_df.groupby(['province', 'year'])[feature].mean().reset_index()
        feature_results[feature] = yearly_stat
    else:
        print("No valid data found for any province!")
        feature_results[feature] = pd.DataFrame(columns=['province', 'year', feature])

# === Save Results ===
final_df = None
for feature, df in feature_results.items():
    if final_df is None:
        final_df = df
    else:
        final_df = pd.merge(final_df, df, on=['province', 'year'], how='outer')

if final_df is not None and not final_df.empty:
    rename_map = {'pr': 'pre', 'tas': 'tmp'}
    final_df = final_df.rename(columns=rename_map)
    output_path = os.path.join(script_dir, f"province_year_all_features_{START_YEAR}_{END_YEAR}.xlsx")
    final_df.to_excel(output_path, index=False)
    print(f"\nSaved final dataset to: {output_path}")
else:
    print("\nNo data to save as no valid climate data was found")

print(f"\nPolygon method used: {polygon_used} times")
print(f"Centroid method used: {centroid_used} times")
print(f"Skipped provinces: {skipped_provinces}")
print(f"Execution time: {time.time() - start_time:.2f} seconds")
