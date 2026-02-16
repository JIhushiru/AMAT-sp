import warnings
import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import os
import numpy as np
from shapely.geometry import Point
import warnings

warnings.filterwarnings(
    "ignore", message=".*Can't decode floating point timedelta to 's'.*"
)


# === CONFIG ===
features = ["cld", "wet", "vap", "tmx", "tmp", "tmn", "pre", "pet", "dtr"]

START_YEAR = 2010
END_YEAR = 2024

script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_path = os.path.join(script_dir, "philippines_provinces.geojson")
yield_path = os.path.join(script_dir, "actual_banana_yield_2010-2024.csv")

# === Load GeoJSON ===
gdf = gpd.read_file(geojson_path)
gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs("EPSG:4326")

# Extract province names
if "name" in gdf.columns:
    gdf["province_name"] = gdf["name"]
else:
    name_cols = [c for c in gdf.columns if "name" in c.lower()]
    gdf["province_name"] = (
        gdf[name_cols[0]] if name_cols else [f"Province_{i}" for i in range(len(gdf))]
    )

# === Process Climate Features ===
feature_results = {}
polygon_used = 0
centroid_used = 0

centroid_log_path = os.path.join(script_dir, "centroid_fallback_log.txt")
with open(centroid_log_path, "w") as f:
    f.write("Centroid fallback log:\n")
for feature in features:
    print(f"\n=== Processing feature: {feature} ===")
    file_name = f"cru_ts4.08.1901.2023.{feature}.dat.nc"
    file_path = os.path.join(script_dir, file_name)

    ds = xr.open_dataset(file_path, decode_timedelta="ns")

    for var in ds.data_vars:
        ds[var].encoding.clear()
    if feature not in ds:
        print(f"  Feature '{feature}' not found in {file_name}")
        continue

    var = ds[feature]
    # Rename lat and lon to y and x
    var = var.rename({"lat": "y", "lon": "x"})

    # Now perform the clipping
    var = var.rio.set_spatial_dims(x_dim="x", y_dim="y")
    var = var.rio.write_crs("EPSG:4326")

    # Subset time to 2010-01-01 to 2024-12-31
    var = var.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    print(f"Dimensions before clipping: {var.dims}")
    print(f"Coordinates: {var.coords}")
    records = []

    for i, row in gdf.iterrows():
        province_name = row["province_name"]
        province_geom = gpd.GeoDataFrame([row], crs=gdf.crs)
        fallback_used = False
        province_geom = province_geom.to_crs(var.rio.crs)

        try:
            # Try polygon-based clipping
            clipped = var.rio.clip(province_geom.geometry, province_geom.crs, drop=True)

            valid_points = []
            province_polygon = row.geometry
            for lat in var.coords["y"].values:
                for lon in var.coords["x"].values:
                    point = Point(lon, lat)
                    if point.within(province_polygon):
                        point_data = var.sel(y=lat, x=lon, method="nearest")
                        if not np.isnan(point_data.values).all():
                            valid_points.append(point_data)

            if valid_points:
                mean_series = xr.concat(valid_points, dim="x").mean(dim="x")
                polygon_used += 1
                print(f"  Valid data found using polygon method for {province_name}")
            else:
                raise ValueError("No valid points inside polygon.")

        except Exception as e:
            with open(centroid_log_path, "a") as f:
                f.write(
                    f"{province_name} used centroid fallback for feature: {feature}\n"
                )
            print(f"  Polygon method failed for {province_name}: {e}")
            print(f"  Switching to centroid fallback for {province_name}")
            fallback_used = True
            centroid = row.geometry.centroid
            mean_series = None
            search_radius = 1.0  # degrees (~100km)
            step = 0.1

            lons = np.arange(
                centroid.x - search_radius, centroid.x + search_radius + step, step
            )
            lats = np.arange(
                centroid.y - search_radius, centroid.y + search_radius + step, step
            )

            found = False
            for lat in lats:
                for lon in lons:
                    try:
                        point_data = var.sel(x=lon, y=lat, method="nearest")
                        if not np.isnan(point_data.values).all():
                            print(
                                f"  Valid data found near centroid at ({lat:.4f}, {lon:.4f})"
                            )
                            mean_series = point_data
                            centroid_used += 1
                            found = True
                            break
                    except:
                        continue
                if found:
                    break

            if not found:
                print(
                    f"  No valid fallback data found near centroid for {province_name}"
                )

        # Convert DataArray to DataFrame
        df = mean_series.to_dataframe().reset_index()

        # If 'wet', convert timedelta to days as float
        if feature == "wet":
            df[feature] = (df[feature].dt.total_seconds() / 86400) * 12

        df["province"] = province_name
        df["method"] = "centroid" if fallback_used else "polygon"
        records.append(df)
        # print(f"  {i + 1}/{len(gdf)} - {province_name}")
    if records:
        combined_df = pd.concat(records)
        combined_df["year"] = combined_df["time"].dt.year
        combined_df = combined_df[
            (combined_df["year"] >= START_YEAR) & (combined_df["year"] <= END_YEAR)
        ]
        yearly_avg = (
            combined_df.groupby(["province", "year"])[feature].mean().reset_index()
        )
        feature_results[feature] = yearly_avg

# === Merge All Features ===
merged_df = feature_results[features[0]]
for feat in features[1:]:
    merged_df = pd.merge(
        merged_df, feature_results[feat], on=["province", "year"], how="outer"
    )

# === Load & Clean Yield Data ===
raw_yield = pd.read_csv(yield_path)

# Filter province-level rows (start with exactly 4 dots)
prov_yield = raw_yield[raw_yield["Geolocation"].str.startswith("....")].copy()
prov_yield["province"] = (
    prov_yield["Geolocation"].str.replace("....", "", regex=False).str.strip()
)

# Melt wide to long
value_vars = [col for col in raw_yield.columns if "Annual" in col]
yield_long = prov_yield.melt(
    id_vars="province", value_vars=value_vars, var_name="year", value_name="yield"
)

# Clean year column (e.g., "2010 Annual" -> 2010)
yield_long["year"] = yield_long["year"].str.extract(r"(\d{4})").astype(int)

# Drop NaNs and keep only 2010–2024
yield_long = yield_long.dropna(subset=["yield"])
yield_long = yield_long[
    (yield_long["year"] >= START_YEAR) & (yield_long["year"] <= END_YEAR)
]

# === Merge All Together ===
final_df = pd.merge(merged_df, yield_long, on=["province", "year"], how="left")

# === Save Final CSV ===
output_path = os.path.join(script_dir, "province_year_climate_features_and_yield.csv")
final_df.to_csv(output_path, index=False)

# === Final Report ===
print(f"\nSaved final dataset to: {output_path}")
print(f"\nPolygon method used: {polygon_used} times")
print(f"Centroid method used: {centroid_used} times")
