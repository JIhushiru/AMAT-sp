"""
Merge historical climate features with SSP5-8.5 projections for 2025-2034.

This script constructs future climate feature sets by combining CMIP6 SSP
projections (tmp, pre) with historical province-level baselines using the
delta change method. Temperature-dependent features (tmx, tmn, dtr, vpd, pet)
are adjusted for physical consistency.

See ../feature_methods.py for detailed documentation of each feature's
computation method and supporting references.
"""
import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# Load historical training data and SSP climate projections
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
ssp_df = pd.read_csv(os.path.join(data_dir, 'ssp585_projections.csv'))

# All 17 climate features
all_features = [
    'cld', 'wet', 'vap', 'tmx', 'tmp', 'tmn', 'pre', 'pet', 'dtr',
    'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws'
]

assert all(col in hist_df.columns for col in all_features), "Missing columns in historical data"
assert all(col in ssp_df.columns for col in ['province', 'year', 'pre', 'tmp']), "Missing columns in SSP data"

years = list(range(2025, 2035))

# Step 1: Compute historical averages per province
avg_by_province = hist_df.groupby('province')[all_features].mean().reset_index()

# Step 2: Replicate historical averages for each future year
rows = []
for _, row in avg_by_province.iterrows():
    for year in years:
        r = {feat: row[feat] for feat in all_features}
        r['province'] = row['province']
        r['year'] = year
        rows.append(r)

merged = pd.DataFrame(rows)

# Step 3: Merge SSP projections (pre and tmp)
merged = merged.merge(
    ssp_df[['province', 'year', 'pre', 'tmp']],
    on=['province', 'year'],
    how='left',
    suffixes=('_hist', '_ssp')
)

# Step 4: Apply SSP values with physical consistency adjustments
# See ../feature_methods.py for detailed methodology and references.

# --- tmp, pre: direct from CMIP6 SSP projections ---
# See feature_methods.compute_tmp(), feature_methods.compute_pre()
tmp_delta = merged['tmp_ssp'] - merged['tmp_hist']
merged['tmp'] = merged['tmp_ssp']
merged['pre'] = merged['pre_ssp']

# --- tmx, tmn: delta change method (Hay et al., 2000; Anandhi et al., 2011) ---
# See feature_methods.compute_tmx(), feature_methods.compute_tmn()
merged['tmx'] = merged['tmx'] + tmp_delta
merged['tmn'] = merged['tmn'] + tmp_delta

# --- dtr: derived from adjusted tmx and tmn ---
# See feature_methods.compute_dtr()
merged['dtr'] = merged['tmx'] - merged['tmn']

# --- vpd: scaled using Tetens equation (Tetens, 1930; Allen et al., 1998) ---
# Rising VPD drives crop water stress (Grossiord et al., 2020; Yuan et al., 2019)
# See feature_methods.compute_vpd()
es_new = 0.6108 * np.exp(17.27 * merged['tmp'] / (merged['tmp'] + 237.3))
es_hist = 0.6108 * np.exp(17.27 * merged['tmp_hist'] / (merged['tmp_hist'] + 237.3))
merged['vpd'] = merged['vpd'] * (es_new / es_hist)

# --- pet: scaled with saturation vapor pressure ratio (Allen et al., 1998) ---
# See feature_methods.compute_pet()
merged['pet'] = merged['pet'] * (es_new / es_hist)

# --- cld, wet, vap, aet, def, PDSI, q, soil, srad, ws: historical averages ---
# These features are held at province-level historical means (2010-2024).
# CMIP6 SSP projections were not available for these variables.
# See feature_methods.py for individual feature documentation.

# Clean up temp columns
merged = merged.drop(columns=['tmp_hist', 'tmp_ssp', 'pre_hist', 'pre_ssp'])

# Reorder columns
merged = merged[['province', 'year'] + all_features]

# Save
output_path = os.path.join(base_dir, 'merged_future_data.csv')
merged.to_csv(output_path, index=False)

print(f"Merged data saved to: {output_path}")
print(f"Shape: {merged.shape}")
print(f"\nFeature ranges (SSP5-8.5):")
for feat in ['tmp', 'tmx', 'tmn', 'dtr', 'pre', 'vpd', 'pet']:
    print(f"  {feat}: {merged[feat].min():.2f} - {merged[feat].max():.2f} (mean {merged[feat].mean():.2f})")
