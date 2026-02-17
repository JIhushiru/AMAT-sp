"""
Extract CMIP6 climate data for the historical overlap period (2015-2024).

Uses the same centroid point sampling approach as extract_cmip6.py but for
the historical period (SSP scenarios start at 2015 in NASA/GDDP-CMIP6).

Extracts all 5 GCMs in the ensemble so each model gets its own bias
correction factors computed against TerraClimate.
"""
import os
import json
import time
import ee
import numpy as np
import pandas as pd
from shapely.geometry import shape

ee.Initialize(project='sp-final-project-459621')

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
geojson_path = os.path.join(base_dir, '..', '..', 'SSPs Data collection', 'SSP2-4.5', 'philippines_provinces.geojson')

# ── Load province boundaries ──────────────────────────────────────────────
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_provinces = set(hist_df['province'].unique())

with open(geojson_path, encoding='utf-8') as f:
    geojson_data = json.load(f)

# Compute centroids client-side using shapely
province_info = []
for feature in geojson_data['features']:
    props = feature.get('properties', {})
    name = props.get('name', props.get('NAME', 'Unknown'))
    if name in hist_provinces:
        geom = shape(feature['geometry'])
        centroid = geom.centroid
        province_info.append((name, centroid.x, centroid.y))

print(f"Matched {len(province_info)} provinces")

# ── Config ────────────────────────────────────────────────────────────────
cmip6_vars = ['pr', 'tas', 'tasmax', 'tasmin', 'rsds', 'sfcWind', 'hurs']
rename_map = {
    'pr': 'pre', 'tas': 'tmp', 'tasmax': 'tmx', 'tasmin': 'tmn',
    'rsds': 'srad', 'sfcWind': 'ws', 'hurs': 'hurs',
}

MODELS = ["GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "CanESM5"]
# NASA/GDDP-CMIP6: "historical" scenario covers up to 2014, SSP starts 2015
HIST_YEARS = list(range(2015, 2025))  # overlap with training data
SCENARIO = "ssp245"  # SSP scenarios barely diverge this early
SCALE = 27830
BATCH_SIZE = 20


def postprocess(df):
    """Derive additional climate features from raw CMIP6 variables."""
    df = df.rename(columns=rename_map)
    df['dtr'] = df['tmx'] - df['tmn']
    es = 0.6108 * np.exp(17.27 * df['tmp'] / (df['tmp'] + 237.3))
    df['vpd'] = es * (1 - df['hurs'] / 100)
    df['vap'] = es * (df['hurs'] / 100)
    df['pet'] = 0.0023 * (df['tmp'] + 17.8) * \
        np.sqrt(df['dtr'].clip(lower=0)) * (df['srad'] * 0.0864)
    df['cld'] = ((1 - df['srad'] / 250.0) * 100).clip(lower=0, upper=100)
    df = df.drop(columns=['hurs'], errors='ignore')
    return df


def extract_historical(gcm_model):
    """Extract CMIP6 historical data for one GCM over the overlap period."""
    print(f"\nExtracting CMIP6 historical overlap ({HIST_YEARS[0]}-{HIST_YEARS[-1]})")
    print(f"Model: {gcm_model}, Scenario: {SCENARIO}")

    # Build EE point features
    ee_points = []
    for name, lon, lat in province_info:
        ee_points.append(ee.Feature(ee.Geometry.Point([lon, lat]), {'name': name}))

    batches = [ee_points[i:i + BATCH_SIZE] for i in range(0, len(ee_points), BATCH_SIZE)]
    print(f"  {len(ee_points)} provinces in {len(batches)} batches")

    all_rows = []
    start_time = time.time()

    for year in HIST_YEARS:
        year_start = time.time()
        print(f"  {year}: ", end="", flush=True)

        year_data = {}

        for var in cmip6_vars:
            var_start = time.time()

            collection = ee.ImageCollection("NASA/GDDP-CMIP6") \
                .filter(ee.Filter.eq("model", gcm_model)) \
                .filter(ee.Filter.eq("scenario", SCENARIO)) \
                .filterDate(f"{year}-01-01", f"{year}-12-31") \
                .select(var)

            annual_mean = collection.mean()

            if var in ('tas', 'tasmax', 'tasmin'):
                annual_mean = annual_mean.subtract(273.15)
            elif var == 'pr':
                annual_mean = annual_mean.multiply(30 * 24 * 60 * 60)

            for batch in batches:
                batch_fc = ee.FeatureCollection(batch)
                try:
                    sampled = annual_mean.sampleRegions(
                        collection=batch_fc, scale=SCALE, geometries=False
                    )
                    data = sampled.getInfo()

                    for feat in data['features']:
                        props = feat['properties']
                        prov = props.get('name', 'Unknown')
                        if prov not in year_data:
                            year_data[prov] = {}
                        year_data[prov][var] = props.get(var, np.nan)

                except Exception as e:
                    print(f"ERR({var}):{e} ", end="", flush=True)

            var_elapsed = time.time() - var_start
            print(f"{var}({var_elapsed:.0f}s) ", end="", flush=True)

        for prov, vals in year_data.items():
            row = {'province': prov, 'year': year}
            row.update(vals)
            all_rows.append(row)

        elapsed = time.time() - year_start
        print(f" [{elapsed:.0f}s]")

    total_time = time.time() - start_time
    print(f"  Total: {total_time:.0f}s")

    df = pd.DataFrame(all_rows)
    df = postprocess(df)

    output_path = os.path.join(base_dir, f'cmip6_historical_{gcm_model}_2015-2024.csv')
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  Shape: {df.shape}, Provinces: {df['province'].nunique()}")

    return df


if __name__ == '__main__':
    for gcm in MODELS:
        extract_historical(gcm)
    print(f"\nDone! Historical data extracted for {len(MODELS)} models")
