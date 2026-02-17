"""
Merge historical climate features with SSP2-4.5 projections for 2025-2034.

Uses the full CMIP6 extraction (7 variables + derived features) from
extract_cmip6.py. Only features without CMIP6 projections (aet, def,
PDSI, q, soil, wet) use historical province averages.

See ../feature_methods.py for detailed documentation and references.
"""
import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# Load historical training data and full CMIP6 projections
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))

# Use full CMIP6 extraction if available, fall back to old 2-variable CSV
full_path = os.path.join(data_dir, 'ssp245_projections_full.csv')
old_path = os.path.join(data_dir, 'ssp245_projections.csv')

if os.path.exists(full_path):
    ssp_df = pd.read_csv(full_path)
    print(f"Using full CMIP6 projections: {ssp_df.columns.tolist()}")
    use_full = True
else:
    ssp_df = pd.read_csv(old_path)
    print(f"Using old 2-variable projections (tmp, pre only)")
    use_full = False

# All 17 climate features
all_features = [
    'cld', 'wet', 'vap', 'tmx', 'tmp', 'tmn', 'pre', 'pet', 'dtr',
    'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws'
]

assert all(col in hist_df.columns for col in all_features), "Missing columns in historical data"

years = list(range(2025, 2035))

# Features that come directly from CMIP6 (available in full extraction)
cmip6_features = ['tmp', 'pre', 'tmx', 'tmn', 'srad', 'ws', 'vpd', 'vap', 'dtr', 'pet', 'cld']

# Features that remain as historical province averages (no CMIP6 source)
frozen_features = ['aet', 'def', 'PDSI', 'q', 'soil', 'wet']

# Step 1: Compute historical averages per province (for frozen features)
avg_by_province = hist_df.groupby('province')[frozen_features].mean().reset_index()

# Step 2: Build future data
if use_full:
    # Match provinces between SSP and historical data
    ssp_provinces = set(ssp_df['province'].unique())
    hist_provinces = set(hist_df['province'].unique())
    common_provinces = ssp_provinces & hist_provinces
    print(f"SSP provinces: {len(ssp_provinces)}, Historical: {len(hist_provinces)}, Matched: {len(common_provinces)}")

    # Start with CMIP6 features
    available_cmip6 = [f for f in cmip6_features if f in ssp_df.columns]
    print(f"CMIP6 features available: {available_cmip6}")
    print(f"Frozen features (historical avg): {frozen_features}")

    merged = ssp_df[ssp_df['province'].isin(common_provinces)][['province', 'year'] + available_cmip6].copy()

    # Add frozen features from historical averages
    merged = merged.merge(avg_by_province, on='province', how='left')

    # Fill any missing CMIP6 features with historical averages
    for feat in cmip6_features:
        if feat not in merged.columns:
            hist_avg = hist_df.groupby('province')[feat].mean().reset_index()
            merged = merged.merge(hist_avg, on='province', how='left', suffixes=('', '_fill'))
            if f'{feat}_fill' in merged.columns:
                merged[feat] = merged[feat].fillna(merged[f'{feat}_fill'])
                merged = merged.drop(columns=[f'{feat}_fill'])

else:
    # Fallback: old behavior (only tmp and pre from CMIP6, derive the rest)
    avg_all = hist_df.groupby('province')[all_features].mean().reset_index()
    rows = []
    for _, row in avg_all.iterrows():
        for year in years:
            r = {feat: row[feat] for feat in all_features}
            r['province'] = row['province']
            r['year'] = year
            rows.append(r)

    merged = pd.DataFrame(rows)
    merged = merged.merge(
        ssp_df[['province', 'year', 'pre', 'tmp']],
        on=['province', 'year'], how='left', suffixes=('_hist', '_ssp')
    )

    tmp_delta = merged['tmp_ssp'] - merged['tmp_hist']
    merged['tmp'] = merged['tmp_ssp']
    merged['pre'] = merged['pre_ssp']
    merged['tmx'] = merged['tmx'] + tmp_delta
    merged['tmn'] = merged['tmn'] + tmp_delta
    merged['dtr'] = merged['tmx'] - merged['tmn']
    es_new = 0.6108 * np.exp(17.27 * merged['tmp'] / (merged['tmp'] + 237.3))
    es_hist = 0.6108 * np.exp(17.27 * merged['tmp_hist'] / (merged['tmp_hist'] + 237.3))
    merged['vpd'] = merged['vpd'] * (es_new / es_hist)
    merged['pet'] = merged['pet'] * (es_new / es_hist)
    merged = merged.drop(columns=['tmp_hist', 'tmp_ssp', 'pre_hist', 'pre_ssp'])

# Reorder columns
merged = merged[['province', 'year'] + all_features]

# Save
output_path = os.path.join(base_dir, 'merged_future_data.csv')
merged.to_csv(output_path, index=False)

print(f"\nMerged data saved to: {output_path}")
print(f"Shape: {merged.shape}")
print(f"\nFeature summary (SSP2-4.5):")
for feat in all_features:
    status = "CMIP6" if (use_full and feat in cmip6_features) else "frozen"
    print(f"  {feat:5s} [{status:6s}]: {merged[feat].min():.2f} - {merged[feat].max():.2f} (mean {merged[feat].mean():.2f})")
