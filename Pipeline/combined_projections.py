"""
Generate combined chart: Historical (2010-2024) + SSP2-4.5 + SSP5-8.5 national yield trends.

Shows ensemble mean with combined uncertainty bands (model spread + bootstrap PI).
Individual GCM traces are shown as thin lines for transparency.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

MODELS = ["GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "CanESM5"]

# ── Load data ─────────────────────────────────────────────────────────────
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df['yield'] = pd.to_numeric(hist_df['yield'].replace('#DIV/0!', pd.NA), errors='coerce')
hist_df = hist_df.dropna(subset=['yield'])

ssp245_path = os.path.join(base_dir, 'SSP2-4.5', 'banana_yield_predictions_2025-2034.xlsx')
ssp585_path = os.path.join(base_dir, 'SSP5-8.5', 'banana_yield_predictions_2025-2034.xlsx')

ssp245_df = pd.read_excel(ssp245_path)
ssp585_df = pd.read_excel(ssp585_path)

# ── National trends ───────────────────────────────────────────────────────
hist_trend = hist_df.groupby('year')['yield'].mean()
ssp245_trend = ssp245_df.groupby('year')['yield'].mean()
ssp585_trend = ssp585_df.groupby('year')['yield'].mean()

# Uncertainty bands
has_pi_245 = 'yield_lower' in ssp245_df.columns
has_pi_585 = 'yield_lower' in ssp585_df.columns

if has_pi_245:
    ssp245_lower = ssp245_df.groupby('year')['yield_lower'].mean()
    ssp245_upper = ssp245_df.groupby('year')['yield_upper'].mean()

if has_pi_585:
    ssp585_lower = ssp585_df.groupby('year')['yield_lower'].mean()
    ssp585_upper = ssp585_df.groupby('year')['yield_upper'].mean()

# Per-GCM trends for individual traces
gcm_trends_245 = {}
gcm_trends_585 = {}
for gcm in MODELS:
    col = f'yield_{gcm}'
    if col in ssp245_df.columns:
        gcm_trends_245[gcm] = ssp245_df.groupby('year')[col].mean()
    if col in ssp585_df.columns:
        gcm_trends_585[gcm] = ssp585_df.groupby('year')[col].mean()

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

# Historical
ax.plot(hist_trend.index, hist_trend.values, 'o-', color='#2ecc71', linewidth=2.5,
        markersize=6, label='Historical (2010-2024)', zorder=5)

# SSP2-4.5 ensemble mean
ax.plot(ssp245_trend.index, ssp245_trend.values, 's-', color='#3498db', linewidth=2.5,
        markersize=5, label='SSP2-4.5 Ensemble Mean', zorder=4)
if has_pi_245:
    ax.fill_between(ssp245_trend.index, ssp245_lower.values, ssp245_upper.values,
                     alpha=0.12, color='#3498db', label='SSP2-4.5 Uncertainty')

# SSP2-4.5 individual GCM traces
for gcm, trend in gcm_trends_245.items():
    ax.plot(trend.index, trend.values, '-', color='#3498db', linewidth=0.5, alpha=0.3)

# SSP5-8.5 ensemble mean
ax.plot(ssp585_trend.index, ssp585_trend.values, 'D-', color='#e74c3c', linewidth=2.5,
        markersize=5, label='SSP5-8.5 Ensemble Mean', zorder=4)
if has_pi_585:
    ax.fill_between(ssp585_trend.index, ssp585_lower.values, ssp585_upper.values,
                     alpha=0.12, color='#e74c3c', label='SSP5-8.5 Uncertainty')

# SSP5-8.5 individual GCM traces
for gcm, trend in gcm_trends_585.items():
    ax.plot(trend.index, trend.values, '-', color='#e74c3c', linewidth=0.5, alpha=0.3)

# Transition line
ax.axvline(x=2024.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.text(2024.5, ax.get_ylim()[1] * 0.98, '  Projections', fontsize=9, color='gray', va='top')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Yield (tons/ha)', fontsize=12)
ax.set_title('National Banana Yield: Historical + 5-GCM Ensemble Projections (2010-2034)', fontsize=14)
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2010, 2035))
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()

output_path = os.path.join(base_dir, 'combined_national_trend.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Combined chart saved to: {output_path}")

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\nHistorical (2010-2024):")
print(f"  Mean: {hist_trend.mean():.2f} t/ha")
print(f"  Range: {hist_trend.min():.2f} - {hist_trend.max():.2f}")

print(f"\nSSP2-4.5 Ensemble (2025-2034):")
print(f"  Mean: {ssp245_trend.mean():.2f} t/ha")
print(f"  Range: {ssp245_trend.min():.2f} - {ssp245_trend.max():.2f}")
if 'model_spread' in ssp245_df.columns:
    print(f"  Avg model spread: {ssp245_df['model_spread'].mean():.3f} t/ha")

print(f"\nSSP5-8.5 Ensemble (2025-2034):")
print(f"  Mean: {ssp585_trend.mean():.2f} t/ha")
print(f"  Range: {ssp585_trend.min():.2f} - {ssp585_trend.max():.2f}")
if 'model_spread' in ssp585_df.columns:
    print(f"  Avg model spread: {ssp585_df['model_spread'].mean():.3f} t/ha")

diff = ssp585_trend.mean() - ssp245_trend.mean()
print(f"\nSSP5-8.5 vs SSP2-4.5 difference: {diff:+.2f} t/ha")

# Per-GCM summary
for scenario_label, gcm_trends in [('SSP2-4.5', gcm_trends_245), ('SSP5-8.5', gcm_trends_585)]:
    if gcm_trends:
        print(f"\n{scenario_label} per-GCM national means:")
        for gcm, trend in gcm_trends.items():
            print(f"  {gcm:15s}: {trend.mean():.2f} t/ha")
