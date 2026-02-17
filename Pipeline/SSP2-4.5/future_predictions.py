"""
Load the best saved model and generate SSP2-4.5 yield predictions.

Instead of retraining, this script loads the best model selected by
train_models.py (Pipeline/saved_model/best_model.joblib).

Predictions are clipped to each province's historical observed range to
prevent extrapolation artifacts (Lobell & Burke, 2010; Challinor et al., 2014).

See ../feature_methods.py for detailed methodology and references.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
model_dir = os.path.join(base_dir, '..', 'saved_model')

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

# ── Load data ─────────────────────────────────────────────────────────────
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)

future_df = pd.read_csv(os.path.join(base_dir, 'merged_future_data.csv'))

# ── Evaluate on training data (using the saved scaler) ────────────────────
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

# ── Future predictions ────────────────────────────────────────────────────
X_future = future_df[features]
X_future_scaled = pd.DataFrame(scaler.transform(X_future), columns=features)
future_preds = model.predict(X_future_scaled)

# ── Bootstrap prediction intervals (90% PI) ─────────────────────────────
bootstrap_path = os.path.join(model_dir, 'bootstrap_models.joblib')
if os.path.exists(bootstrap_path):
    boot_data = joblib.load(bootstrap_path)
    boot_models = boot_data['models']
    n_boot = boot_data['n_bootstrap']

    boot_preds = np.zeros((n_boot, len(X_future)))
    for i, bm in enumerate(boot_models):
        X_boot_scaled = pd.DataFrame(
            bm['scaler'].transform(X_future), columns=features
        )
        boot_preds[i] = bm['model'].predict(X_boot_scaled)

    future_df['yield_lower'] = np.percentile(boot_preds, 5, axis=0)
    future_df['yield_upper'] = np.percentile(boot_preds, 95, axis=0)
    print(f"\n90% prediction intervals computed from {n_boot} bootstrap models")
else:
    print("\nNo bootstrap models found, skipping prediction intervals")

# Clip predictions per province to historical observed range (prevent extrapolation artifacts)
# Lobell & Burke (2010), Challinor et al. (2014)
hist_province_bounds = hist_df.groupby('province')['yield'].agg(['min', 'max'])
future_df['yield_raw'] = future_preds
future_df['yield'] = future_preds
for province in future_df['province'].unique():
    if province in hist_province_bounds.index:
        pmin = hist_province_bounds.loc[province, 'min']
        pmax = hist_province_bounds.loc[province, 'max']
        mask = future_df['province'] == province
        future_df.loc[mask, 'yield'] = future_df.loc[mask, 'yield_raw'].clip(lower=pmin, upper=pmax)
        if 'yield_lower' in future_df.columns:
            future_df.loc[mask, 'yield_lower'] = future_df.loc[mask, 'yield_lower'].clip(lower=pmin, upper=pmax)
            future_df.loc[mask, 'yield_upper'] = future_df.loc[mask, 'yield_upper'].clip(lower=pmin, upper=pmax)

clipped_count = (future_df['yield'] != future_df['yield_raw']).sum()
print(f"Clipped {clipped_count}/{len(future_df)} predictions to historical province ranges")
future_df = future_df.drop(columns=['yield_raw'])

output_path = os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx')
future_df.to_excel(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(f"Future yield range: {future_preds.min():.2f} - {future_preds.max():.2f} (mean {future_preds.mean():.2f})")
if 'yield_lower' in future_df.columns:
    print(f"90% PI width (avg): {(future_df['yield_upper'] - future_df['yield_lower']).mean():.2f} tons/ha")
