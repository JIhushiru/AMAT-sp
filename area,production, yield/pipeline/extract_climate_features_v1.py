import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import os
import numpy as np
from shapely.geometry import Point

# === CONFIG ===
features = ["cld", "wet", "vap", "tmx", "tmp", "tmn", "pre", "pet", "frs", "dtr"]

START_YEAR = 2010
END_YEAR = 2024

script_dir = os.path.dirname(os.path.abspath(__file__))
geojson_path = os.path.join(script_dir, "philippines_provinces.geojson")
yield_path = os.path.join(script_dir, "banana_area_harvested_2010-2024.csv")

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

for feature in features:
    print(f"\n=== Processing feature: {feature} ===")
    file_name = f"cru_ts4.08.1901.2023.{feature}.dat.nc"
    file_path = os.path.join(script_dir, file_name)

    ds = xr.open_dataset(file_path, decode_timedelta=True)

    if feature not in ds:
        print(f"  Feature '{feature}' not found in {file_name}")
        continue

    var = ds[feature]
    if "latitude" in var.dims and "longitude" in var.dims:
        var = var.rename({"latitude": "lat", "longitude": "lon"})

    var = var.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    var = var.rio.write_crs("EPSG:4326")  # Ensure CRS is set to EPSG:4326

    # Subset time to 2010-01-01 to 2024-12-31
    var = var.sel(time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31"))
    print(ds.copy())
    # Ensure that the province geometry has the correct CRS (EPSG:4326)
    gdf = gdf.to_crs("EPSG:4326")
    # Ensure correct spatial dimensions (lon, lat)
    var = var.rename({"latitude": "lat", "longitude": "lon"})  # Rename if necessary
    var = var.rio.set_spatial_dims(
        x_dim="lon", y_dim="lat"
    )  # Set spatial dimensions for rioxarray
    var = var.rio.write_crs("EPSG:4326")  # Ensure CRS is set to EPSG:4326

    records = []
    for i, row in gdf.iterrows():
        province_name = row["province_name"]
        province_geom = gpd.GeoDataFrame([row], crs=gdf.crs)
        fallback_used = False
        if i == 1:
            break
        clipped = var.rio.clip(province_geom.geometry, province_geom.crs, drop=True)
        try:
            # Clip the data for the polygon region

            # Debugging to check clipped data
            if clipped.isnull().all():
                print(f"  No valid data found in clipped region for {province_name}.")
            else:
                print(f"  Valid data found for {province_name} after clipping.")

            if np.isnan(clipped.values).all():
                print(f"  No data found in polygon, switching to centroid.")
                raise ValueError("No data found in polygon, switching to centroid.")

            # Check if clipping resulted in all NaNs
            if np.isnan(clipped.values).all():
                raise ValueError(
                    f"Clipped data is all NaN for {province_name}, switching to centroid."
                )

            mean_series = clipped.mean(dim=["lat", "lon"])
            polygon_used += 1
        except Exception as e:
            # Fallback to centroid if clipping fails or polygon data is all NaN
            fallback_used = True
            centroid = row.geometry.centroid
            print(
                f"  No data found in polygon for {province_name}, switching to centroid. ({e})"
            )

            # Iterate through nearby grid points to find the nearest valid data point
            for delta in np.linspace(-0.1, 0.1, 5):
                for dx in [-delta, 0, delta]:
                    for dy in [-delta, 0, delta]:
                        point_data = var.sel(
                            lat=centroid.y + dy, lon=centroid.x + dx, method="nearest"
                        )
                        # Check if valid data is found (i.e., not all NaN)
                        if not np.isnan(point_data.values).all():
                            print(
                                f"  Valid data found near centroid for {province_name} at ({centroid.y + dy}, {centroid.x + dx})"
                            )
                            mean_series = point_data
                            centroid_used += 1
                            break
                    if not np.isnan(mean_series.values).all():
                        break
                if not np.isnan(mean_series.values).all():
                    break

        # Convert DataArray to DataFrame
        df = mean_series.to_dataframe().reset_index()

        # If 'wet', convert timedelta to days as float
        if feature == "wet":
            df[feature] = (df[feature].dt.total_seconds() / 86400) * 12

        df["province"] = province_name
        df["method"] = "centroid" if fallback_used else "polygon"
        records.append(df)

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
print(f"\n✅ Saved final dataset to: {output_path}")
print(f"\nPolygon method used: {polygon_used} times")
print(f"Centroid method used: {centroid_used} times")
