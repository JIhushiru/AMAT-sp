import os
import pandas as pd
import numpy as np
from cubist import Cubist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')

# Load historical data
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)

# Load future data (from merge_data.py output)
future_df = pd.read_csv(os.path.join(base_dir, 'merged_future_data.csv'))
print("Historical columns:", hist_df.columns.tolist())

# Same 10 features used by the original Cubist model
features = ['cld', 'tmp', 'pre', 'aet', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws']

X_train = hist_df[features]
y_train = hist_df['yield']
X_future = future_df[features]

# Train Cubist model
model = Cubist(n_committees=20, neighbors=2, n_rules=200)
model.fit(X_train, y_train)

# Evaluate on training data
y_train_pred = model.predict(X_train)
r2 = r2_score(y_train, y_train_pred)
mae = mean_absolute_error(y_train, y_train_pred)
mse = mean_squared_error(y_train, y_train_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

print("=== Cubist Model Performance on Training Data ===")
print(f"R2 Score : {r2:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MAPE     : {mape:.2f}%")


def create_obs_vs_pred_plot(y_true, y_pred, model_name="Cubist"):
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


create_obs_vs_pred_plot(y_train, y_train_pred, "Cubist")

# === FUTURE PREDICTIONS ===
future_preds = model.predict(X_future)

# Clip predictions per province to historical observed range (prevent extrapolation artifacts)
hist_province_bounds = hist_df.groupby('province')['yield'].agg(['min', 'max'])
future_df['yield_raw'] = future_preds
future_df['yield'] = future_preds
for province in future_df['province'].unique():
    if province in hist_province_bounds.index:
        pmin = hist_province_bounds.loc[province, 'min']
        pmax = hist_province_bounds.loc[province, 'max']
        mask = future_df['province'] == province
        future_df.loc[mask, 'yield'] = future_df.loc[mask, 'yield_raw'].clip(lower=pmin, upper=pmax)

clipped_count = (future_df['yield'] != future_df['yield_raw']).sum()
print(f"\nClipped {clipped_count}/{len(future_df)} predictions to historical province ranges")
future_df = future_df.drop(columns=['yield_raw'])

output_path = os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx')
future_df.to_excel(output_path, index=False)

print(f"\nPredictions saved to: {output_path}")
print(f"Future yield range: {future_preds.min():.2f} - {future_preds.max():.2f} (mean {future_preds.mean():.2f})")
