"""
Generate SSP5-8.5 yield predictions using a 5-model GCM ensemble.

For each GCM's bias-corrected climate data, the Cubist model predicts yields.
The ensemble mean is the central estimate. Two uncertainty sources are quantified:

  1. Model uncertainty: spread across 5 GCMs (structural uncertainty)
  2. Statistical uncertainty: 90% prediction intervals from 100 bootstrap models

Combined uncertainty bounds use the outer envelope of both sources.

References:
  - Tebaldi & Knutti (2007), Phil. Trans. R. Soc. A
  - Lobell & Burke (2010), Challinor et al. (2014) for clipping
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
model_dir = os.path.join(base_dir, '..', 'saved_model')

MODELS = ["GFDL-ESM4", "MIROC6", "MRI-ESM2-0", "IPSL-CM6A-LR", "CanESM5"]

# ── Load saved model ──────────────────────────────────────────────────────
model_path = os.path.join(model_dir, 'best_model.joblib')
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"No saved model found at {model_path}.\n"
        "Run train_models.py first: python Pipeline/train_models.py"
    )

saved = joblib.load(model_path)
model = saved['model']
scaler = saved['scaler']
features = saved['features']
model_name = saved['model_name']

print(f"Loaded model: {model_name}")
print(f"  CV R²:    {saved['cv_r2']:.4f}")
print(f"  Train R²: {saved['train_r2']:.4f}")
print(f"  Features: {features}")

# ── Load historical data ─────────────────────────────────────────────────
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)

# ── Evaluate on training data ────────────────────────────────────────────
X_train = hist_df[features]
y_train = hist_df['yield']
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)

y_train_pred = model.predict(X_train_scaled)
r2 = r2_score(y_train, y_train_pred)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

print(f"\n=== {model_name} Performance on Training Data ===")
print(f"R2 Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MAPE     : {mape:.2f}%")


def create_obs_vs_pred_plot(y_true, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, color='blue')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (1:1)')

    metrics_text = f'RMSE = {rmse:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}\nMAPE = {mape:.2f}%\nR2 = {r2:.3f}'
    plt.text(0.98, 0.98, metrics_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.5),
             fontsize=11, verticalalignment='top', horizontalalignment='right')

    plt.xlabel('Observed Yield (ton/ha)', fontsize=12)
    plt.ylabel('Predicted Yield (ton/ha)', fontsize=12)
    plt.title(f'Observed vs Predicted Yield - {model_name} (Training Set)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()

    plot_path = os.path.join(base_dir, 'observed_vs_predicted_yield.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Observed vs predicted plot saved to: {plot_path}")


create_obs_vs_pred_plot(y_train, y_train_pred)

# ── Per-model predictions ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Generating ensemble predictions (SSP5-8.5)")
print(f"{'='*60}")

gcm_predictions = {}
base_future_df = None

for gcm in MODELS:
    merged_path = os.path.join(base_dir, f'merged_future_data_{gcm}.csv')
    if not os.path.exists(merged_path):
        print(f"  {gcm}: SKIP (no merged data)")
        continue

    future_df = pd.read_csv(merged_path)
    X_future = future_df[features]
    X_future_scaled = pd.DataFrame(scaler.transform(X_future), columns=features)
    preds = model.predict(X_future_scaled)
    gcm_predictions[gcm] = preds
    print(f"  {gcm}: mean={preds.mean():.2f}, range=[{preds.min():.2f}, {preds.max():.2f}]")

    if base_future_df is None:
        base_future_df = future_df.copy()

n_models = len(gcm_predictions)
print(f"\n  Ensemble size: {n_models} models")

if n_models == 0:
    raise RuntimeError("No model data found. Run merge_data.py first.")

# ── Ensemble aggregation ─────────────────────────────────────────────────
all_preds = np.array(list(gcm_predictions.values()))
ensemble_mean = all_preds.mean(axis=0)
ensemble_std = all_preds.std(axis=0)
ensemble_min = all_preds.min(axis=0)
ensemble_max = all_preds.max(axis=0)

print(f"  Ensemble mean yield: {ensemble_mean.mean():.2f} t/ha")
print(f"  Model spread (avg std): {ensemble_std.mean():.3f} t/ha")

# ── Bootstrap prediction intervals on ensemble mean climate ──────────────
bootstrap_path = os.path.join(model_dir, 'bootstrap_models.joblib')
has_bootstrap = os.path.exists(bootstrap_path)

if has_bootstrap:
    boot_data = joblib.load(bootstrap_path)
    boot_models = boot_data['models']
    n_boot = boot_data['n_bootstrap']

    boot_ensemble = np.zeros((n_boot, len(base_future_df)))
    for gcm in gcm_predictions:
        gcm_path = os.path.join(base_dir, f'merged_future_data_{gcm}.csv')
        gcm_df = pd.read_csv(gcm_path)
        X_gcm = gcm_df[features]

        for i, bm in enumerate(boot_models):
            X_boot_scaled = pd.DataFrame(bm['scaler'].transform(X_gcm), columns=features)
            boot_ensemble[i] += bm['model'].predict(X_boot_scaled)

    boot_ensemble /= n_models

    boot_lower = np.percentile(boot_ensemble, 5, axis=0)
    boot_upper = np.percentile(boot_ensemble, 95, axis=0)
    print(f"  90% bootstrap PI computed from {n_boot} models x {n_models} GCMs")
else:
    print("  No bootstrap models found, skipping prediction intervals")

# ── Build output dataframe ───────────────────────────────────────────────
future_df = base_future_df.copy()
future_df['yield'] = ensemble_mean
future_df['model_spread'] = ensemble_std

for gcm, preds in gcm_predictions.items():
    future_df[f'yield_{gcm}'] = preds

if has_bootstrap:
    future_df['yield_lower'] = np.minimum(boot_lower, ensemble_min)
    future_df['yield_upper'] = np.maximum(boot_upper, ensemble_max)
else:
    future_df['yield_lower'] = ensemble_min
    future_df['yield_upper'] = ensemble_max

# ── Clip to historical province bounds ───────────────────────────────────
hist_province_bounds = hist_df.groupby('province')['yield'].agg(['min', 'max'])
future_df['yield_raw'] = future_df['yield'].copy()

for province in future_df['province'].unique():
    if province in hist_province_bounds.index:
        pmin = hist_province_bounds.loc[province, 'min']
        pmax = hist_province_bounds.loc[province, 'max']
        mask = future_df['province'] == province
        future_df.loc[mask, 'yield'] = future_df.loc[mask, 'yield_raw'].clip(lower=pmin, upper=pmax)
        future_df.loc[mask, 'yield_lower'] = future_df.loc[mask, 'yield_lower'].clip(lower=pmin, upper=pmax)
        future_df.loc[mask, 'yield_upper'] = future_df.loc[mask, 'yield_upper'].clip(lower=pmin, upper=pmax)
        for gcm in gcm_predictions:
            col = f'yield_{gcm}'
            future_df.loc[mask, col] = future_df.loc[mask, col].clip(lower=pmin, upper=pmax)

clipped_count = (future_df['yield'] != future_df['yield_raw']).sum()
print(f"  Clipped {clipped_count}/{len(future_df)} predictions to historical province ranges")
future_df = future_df.drop(columns=['yield_raw'])

# ── Save ─────────────────────────────────────────────────────────────────
output_path = os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx')
future_df.to_excel(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(f"  Ensemble yield range: {ensemble_mean.min():.2f} - {ensemble_mean.max():.2f} (mean {ensemble_mean.mean():.2f})")
print(f"  Avg model spread: {future_df['model_spread'].mean():.3f} t/ha")
print(f"  Avg total uncertainty width: {(future_df['yield_upper'] - future_df['yield_lower']).mean():.2f} t/ha")
