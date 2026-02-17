"""
Merge historical climate features with SSP2-4.5 projections for 2025-2034.

Processes all 5 GCMs in the ensemble independently. For each model, CMIP6-sourced
features come from bias-corrected projections while features without CMIP6
equivalents (aet, def, PDSI, q, soil, wet) use historical province averages.

See ../feature_methods.py for detailed documentation and references.
"""
import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

MODELS = ["GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "CanESM5"]
SCENARIO = 'ssp245'

# Load historical training data
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))

# All 17 climate features
all_features = [
    'cld', 'wet', 'vap', 'tmx', 'tmp', 'tmn', 'pre', 'pet', 'dtr',
    'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws'
]

assert all(col in hist_df.columns for col in all_features), "Missing columns in historical data"

# Features from CMIP6 vs frozen at historical averages
cmip6_features = ['tmp', 'pre', 'tmx', 'tmn', 'srad', 'ws', 'vpd', 'vap', 'dtr', 'pet', 'cld']
frozen_features = ['aet', 'def', 'PDSI', 'q', 'soil', 'wet']

# Historical averages for frozen features
avg_by_province = hist_df.groupby('province')[frozen_features].mean().reset_index()
hist_provinces = set(hist_df['province'].unique())

for gcm in MODELS:
    print(f"\n{'='*60}")
    print(f"Merging {SCENARIO} - {gcm}")
    print(f"{'='*60}")

    # Prefer bias-corrected > raw
    corrected_path = os.path.join(base_dir, '..', 'bias_correction', f'{SCENARIO}_{gcm}_corrected.csv')
    raw_path = os.path.join(data_dir, f'{SCENARIO}_{gcm}_projections.csv')

    if os.path.exists(corrected_path):
        ssp_df = pd.read_csv(corrected_path)
        print(f"  Using bias-corrected: {corrected_path}")
    elif os.path.exists(raw_path):
        ssp_df = pd.read_csv(raw_path)
        print(f"  Using raw (no bias correction): {raw_path}")
    else:
        print(f"  SKIP: no data found for {gcm}")
        continue

    # Match provinces
    common_provinces = set(ssp_df['province'].unique()) & hist_provinces
    print(f"  Matched provinces: {len(common_provinces)}")

    available_cmip6 = [f for f in cmip6_features if f in ssp_df.columns]
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

    # Reorder columns
    merged = merged[['province', 'year'] + all_features]

    output_path = os.path.join(base_dir, f'merged_future_data_{gcm}.csv')
    merged.to_csv(output_path, index=False)
    print(f"  Saved: {output_path} ({merged.shape})")

    for feat in all_features:
        status = "CMIP6" if feat in cmip6_features else "frozen"
        print(f"    {feat:5s} [{status:6s}]: {merged[feat].min():.2f} - {merged[feat].max():.2f} (mean {merged[feat].mean():.2f})")

print(f"\nDone! Merged data for {len(MODELS)} models under {SCENARIO}")
