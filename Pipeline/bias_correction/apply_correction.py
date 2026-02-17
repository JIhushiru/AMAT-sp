"""
Compute and apply bias correction to CMIP6 future projections for each GCM.

Method: Delta method (additive bias correction)
  corrected = future_cmip6 + (mean_terraclimate - mean_cmip6_historical)

Each GCM is bias-corrected independently against TerraClimate, because each
model has its own systematic biases. Per-model corrected outputs are saved
for ensemble aggregation downstream.

Reference: Maraun & Widmann (2018), Statistical Downscaling and Bias
Correction for Climate Research.
"""
import os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

MODELS = ["GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "CanESM5"]

# ── Load TerraClimate training data ──────────────────────────────────────
terra_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))

# Variables to bias-correct (only those sourced from CMIP6)
cmip6_features = ['tmp', 'tmx', 'tmn', 'pre', 'srad', 'ws', 'vpd', 'vap', 'dtr', 'pet', 'cld']

# Filter TerraClimate to overlap period (2015-2024)
terra_overlap = terra_df[terra_df['year'].between(2015, 2024)].copy()


def compute_and_apply_bias(gcm_model):
    """Compute bias factors and apply correction for one GCM."""
    print(f"\n{'='*60}")
    print(f"Bias correction: {gcm_model}")
    print(f"{'='*60}")

    # Load this model's historical overlap data
    hist_path = os.path.join(base_dir, f'cmip6_historical_{gcm_model}_2015-2024.csv')
    if not os.path.exists(hist_path):
        print(f"  SKIP: {hist_path} not found (run extract_historical_cmip6.py)")
        return None

    cmip6_hist = pd.read_csv(hist_path)

    # Common provinces
    common_provinces = set(terra_overlap['province'].unique()) & \
                       set(cmip6_hist['province'].unique())
    print(f"  Common provinces: {len(common_provinces)}")

    terra_filt = terra_overlap[terra_overlap['province'].isin(common_provinces)]
    cmip6_filt = cmip6_hist[cmip6_hist['province'].isin(common_provinces)]

    # Compute bias per province per variable
    terra_means = terra_filt.groupby('province')[cmip6_features].mean()
    cmip6_means = cmip6_filt.groupby('province')[cmip6_features].mean()
    bias = terra_means - cmip6_means

    # Save bias factors for this model
    bias_path = os.path.join(base_dir, f'bias_factors_{gcm_model}.csv')
    bias.to_csv(bias_path)

    for feat in cmip6_features:
        b = bias[feat]
        print(f"  {feat:5s}: mean={b.mean():+.3f}, std={b.std():.3f}, "
              f"range=[{b.min():+.3f}, {b.max():+.3f}]")

    # Apply correction to both SSP scenarios
    for scenario_label in ['ssp245', 'ssp585']:
        future_path = os.path.join(data_dir, f'{scenario_label}_{gcm_model}_projections.csv')
        if not os.path.exists(future_path):
            print(f"  SKIP: {future_path} not found")
            continue

        future_df = pd.read_csv(future_path)
        corrected = future_df.copy()

        for province in corrected['province'].unique():
            if province in bias.index:
                mask = corrected['province'] == province
                for feat in cmip6_features:
                    if feat in corrected.columns:
                        corrected.loc[mask, feat] += bias.loc[province, feat]

        # Clip non-negative variables
        for feat in ['pre', 'srad', 'pet']:
            if feat in corrected.columns:
                corrected[feat] = corrected[feat].clip(lower=0)

        if 'cld' in corrected.columns:
            corrected['cld'] = corrected['cld'].clip(lower=0, upper=100)

        output_path = os.path.join(base_dir, f'{scenario_label}_{gcm_model}_corrected.csv')
        corrected.to_csv(output_path, index=False)
        print(f"  {scenario_label} corrected: {output_path} ({corrected.shape})")

    return bias


# ── Run for all models ───────────────────────────────────────────────────
all_biases = {}
for gcm in MODELS:
    bias = compute_and_apply_bias(gcm)
    if bias is not None:
        all_biases[gcm] = bias

# ── Summary comparison across models ─────────────────────────────────────
if all_biases:
    print(f"\n{'='*60}")
    print("Cross-model bias summary (mean across provinces):")
    print(f"{'='*60}")
    print(f"{'Feature':>6s}", end="")
    for gcm in all_biases:
        short = gcm[:10]
        print(f" | {short:>10s}", end="")
    print()
    print("-" * (8 + 13 * len(all_biases)))

    for feat in cmip6_features:
        print(f"{feat:>6s}", end="")
        for gcm in all_biases:
            print(f" | {all_biases[gcm][feat].mean():+10.3f}", end="")
        print()

print(f"\nDone! Bias correction applied for {len(all_biases)} models")
