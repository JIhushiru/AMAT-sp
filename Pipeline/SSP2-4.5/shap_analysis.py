"""
SHAP feature importance analysis for the Cubist model under SSP2-4.5.

Uses the saved model from Pipeline/saved_model/best_model.joblib (same model
used by future_predictions.py) to ensure consistency across the pipeline.

Generates:
  - SHAP bar plots (mean |SHAP value|) for historical and future data
  - SHAP beeswarm plots (feature impact distribution) for historical and future
  - Raw SHAP values saved as .npy files

Run after future_predictions.py has generated the predictions file.
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# === Paths ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, '..', 'data')
model_dir = os.path.join(base_dir, '..', 'saved_model')
shap_output_dir = os.path.join(base_dir, 'shap_results', 'cubist')
os.makedirs(shap_output_dir, exist_ok=True)

# === Configuration ===
SHAP_BATCH_SIZE = None  # None = full dataset; set integer for subset (e.g., 100)
BATCH_SIZE = 5          # rows per KernelExplainer batch (memory management)

# === Load saved model ===
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
print(f"  Features: {features}")

# === Load historical data ===
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)
X_train = hist_df[features]
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=features)

# === Load future data (SSP2-4.5 predictions) ===
future_path = os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx')
if not os.path.exists(future_path):
    raise FileNotFoundError(
        f"No predictions found at {future_path}.\n"
        "Run future_predictions.py first."
    )

future_df = pd.read_excel(future_path)
future_df = future_df[pd.to_numeric(future_df['yield'], errors='coerce').notnull()]
future_df['yield'] = future_df['yield'].astype(float)
X_future = future_df[features]
X_future_scaled = pd.DataFrame(scaler.transform(X_future), columns=features)


# === SHAP wrapper (model expects scaled input) ===
def model_predict(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=features)
    return model.predict(X)


# === Prepare SHAP explainer ===
background_sample = X_train_scaled.sample(n=1, random_state=42)
explainer = shap.KernelExplainer(model_predict, background_sample)


def run_shap_analysis(shap_data, label):
    """Run SHAP on given scaled data, save plots and values."""
    if SHAP_BATCH_SIZE is not None and SHAP_BATCH_SIZE < len(shap_data):
        print(f"SHAP analysis on subset of {SHAP_BATCH_SIZE} {label} rows")
        shap_data = shap_data.sample(n=SHAP_BATCH_SIZE, random_state=42)
    else:
        print(f"SHAP analysis on FULL {label} dataset ({len(shap_data)} rows)")

    all_shap_values = []
    n_batches = (len(shap_data) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(shap_data), BATCH_SIZE):
        batch = shap_data.iloc[i:i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{n_batches}: rows {i}-{i + len(batch) - 1}")
        shap_vals = explainer.shap_values(batch)
        all_shap_values.append(shap_vals)

    shap_values = np.vstack(all_shap_values)

    # Bar plot (mean |SHAP value|)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_data, plot_type='bar',
                      feature_names=features, show=False)
    bar_path = os.path.join(shap_output_dir,
                            f'cubist_shap_feature_importance_{label}.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Bar plot saved: {bar_path}")

    # Beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_data,
                      feature_names=features, show=False)
    beeswarm_path = os.path.join(shap_output_dir,
                                 f'cubist_shap_beeswarm_plot_{label}.png')
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Beeswarm plot saved: {beeswarm_path}")

    # Raw SHAP values
    npy_path = os.path.join(shap_output_dir, f'shap_values_{label}.npy')
    np.save(npy_path, shap_values)
    print(f"  SHAP values saved: {npy_path}")


# === Run SHAP ===
print("\n=== Historical Data SHAP (2010-2024) ===")
run_shap_analysis(X_train_scaled, label='historical')

print("\n=== Future Data SHAP (SSP2-4.5, 2025-2034) ===")
run_shap_analysis(X_future_scaled, label='future')
