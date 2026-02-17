"""
Compute and apply bias correction to CMIP6 future projections.

Method: Delta method (additive bias correction)
  corrected = future_cmip6 + (mean_terraclimate - mean_cmip6_historical)

For each province and each CMIP6-sourced variable, we compute the mean
difference between TerraClimate (training data) and CMIP6 over the
overlap period (2015-2024), then add that offset to future projections.

Reference: Maraun & Widmann (2018), Statistical Downscaling and Bias
Correction for Climate Research.
"""
import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# ── Load data ─────────────────────────────────────────────────────────────
# TerraClimate-based historical training data
terra_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))

# CMIP6 historical overlap (2015-2024)
cmip6_hist_path = os.path.join(base_dir, 'cmip6_historical_2015-2024.csv')
if not os.path.exists(cmip6_hist_path):
    raise FileNotFoundError(
        f"Run extract_historical_cmip6.py first.\n"
        f"Expected: {cmip6_hist_path}"
    )
cmip6_hist = pd.read_csv(cmip6_hist_path)

# Future CMIP6 projections
ssp245_path = os.path.join(data_dir, 'ssp245_projections_full.csv')
ssp585_path = os.path.join(data_dir, 'ssp585_projections_full.csv')
ssp245_df = pd.read_csv(ssp245_path)
ssp585_df = pd.read_csv(ssp585_path)

# ── Variables to bias-correct ─────────────────────────────────────────────
# Only correct variables that come from CMIP6 (not frozen features)
cmip6_features = ['tmp', 'tmx', 'tmn', 'pre', 'srad', 'ws', 'vpd', 'vap', 'dtr', 'pet', 'cld']

# Filter TerraClimate to overlap period (2015-2024)
terra_overlap = terra_df[terra_df['year'].between(2015, 2024)].copy()

# Common provinces
common_provinces = set(terra_overlap['province'].unique()) & \
                   set(cmip6_hist['province'].unique())
print(f"Common provinces for bias correction: {len(common_provinces)}")

terra_overlap = terra_overlap[terra_overlap['province'].isin(common_provinces)]
cmip6_hist = cmip6_hist[cmip6_hist['province'].isin(common_provinces)]

# ── Compute bias per province per variable ────────────────────────────────
terra_means = terra_overlap.groupby('province')[cmip6_features].mean()
cmip6_means = cmip6_hist.groupby('province')[cmip6_features].mean()

# Bias = TerraClimate - CMIP6 (additive offset)
bias = terra_means - cmip6_means

print(f"\n{'='*60}")
print("Bias correction factors (TerraClimate - CMIP6, province mean):")
print(f"{'='*60}")
for feat in cmip6_features:
    b = bias[feat]
    print(f"  {feat:5s}: mean={b.mean():+.3f}, std={b.std():.3f}, "
          f"range=[{b.min():+.3f}, {b.max():+.3f}]")

# Save bias factors
bias_path = os.path.join(base_dir, 'bias_factors.csv')
bias.to_csv(bias_path)
print(f"\nBias factors saved to: {bias_path}")

# ── Apply correction ──────────────────────────────────────────────────────
def apply_bias(df, scenario_label):
    """Apply additive bias correction to future projections."""
    corrected = df.copy()
    corrected_count = 0

    for province in corrected['province'].unique():
        if province in bias.index:
            mask = corrected['province'] == province
            for feat in cmip6_features:
                if feat in corrected.columns:
                    corrected.loc[mask, feat] += bias.loc[province, feat]
                    corrected_count += mask.sum()

    # Clip non-negative variables
    for feat in ['pre', 'srad', 'pet']:
        if feat in corrected.columns:
            corrected[feat] = corrected[feat].clip(lower=0)

    # Clip percentage variables
    if 'cld' in corrected.columns:
        corrected['cld'] = corrected['cld'].clip(lower=0, upper=100)

    output_path = os.path.join(base_dir, f'{scenario_label}_corrected.csv')
    corrected.to_csv(output_path, index=False)
    print(f"\n{scenario_label} corrected: {output_path}")
    print(f"  Shape: {corrected.shape}")

    return corrected


ssp245_corrected = apply_bias(ssp245_df, 'ssp245')
ssp585_corrected = apply_bias(ssp585_df, 'ssp585')

# ── Summary comparison ────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Before vs After correction (national means):")
print(f"{'='*60}")
print(f"{'Feature':>6s} | {'Terra':>8s} | {'CMIP6 raw':>9s} | {'Corrected':>9s} | {'Bias':>8s}")
print("-" * 55)

terra_national = terra_overlap[cmip6_features].mean()
for feat in cmip6_features:
    t = terra_national[feat]
    raw = ssp245_df[feat].mean() if feat in ssp245_df.columns else float('nan')
    corr = ssp245_corrected[feat].mean() if feat in ssp245_corrected.columns else float('nan')
    b = bias[feat].mean()
    print(f"{feat:>6s} | {t:8.2f} | {raw:9.2f} | {corr:9.2f} | {b:+8.3f}")

print(f"\nDone! Corrected projections saved to Pipeline/bias_correction/")
