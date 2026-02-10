import os
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from cubist import Cubist

SHAP_BATCH_SIZE = None

base_dir = os.path.dirname(os.path.abspath(__file__))
shap_output_dir = os.path.join(base_dir, 'shap_results', 'cubist')
os.makedirs(shap_output_dir, exist_ok=True)

features = ['cld', 'wet', 'vap', 'tmx', 'tmp', 'tmn', 'pre', 'pet', 'dtr', 'aet', 'def', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws']

# === Load Historical Data ===
hist_df = pd.read_excel(os.path.join(base_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)
X_train = hist_df[features]
y_train = hist_df['yield']

# === Scale Historical Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)

# === Load and Scale Future Data ===
future_df = pd.read_excel(os.path.join(base_dir, 'banana_yield_predictions_2025-2034.xlsx'))
future_df = future_df[pd.to_numeric(future_df['yield'], errors='coerce').notnull()]
future_df['yield'] = future_df['yield'].astype(float)
X_future = future_df[features]
X_future_scaled = scaler.transform(X_future)
X_future_scaled_df = pd.DataFrame(X_future_scaled, columns=features)

# === Train Cubist Model ===
model = Cubist(n_committees=20, neighbors=2, n_rules=200)
model.fit(X_train, y_train)


def model_predict(X):
    """Wrapper: SHAP passes scaled data, we inverse-transform before predicting."""
    if isinstance(X, np.ndarray):
        X_df = pd.DataFrame(X, columns=features)
    else:
        X_df = X
    X_unscaled = scaler.inverse_transform(X_df)
    X_unscaled_df = pd.DataFrame(X_unscaled, columns=features)
    return model.predict(X_unscaled_df)


# === Prepare SHAP Explainer ===
background_sample = X_train_scaled_df.sample(n=1, random_state=42)
explainer = shap.KernelExplainer(model_predict, background_sample)


def run_shap_analysis(shap_data, label):
    """Run SHAP on given data and save plots and values."""
    if SHAP_BATCH_SIZE is None or SHAP_BATCH_SIZE >= len(shap_data):
        print(f"SHAP Analysis on FULL {label} dataset ({len(shap_data)} rows)")
        batch_size = 5
        all_shap_values = []

        for i in range(0, len(shap_data), batch_size):
            batch = shap_data.iloc[i:i + batch_size]
            total_batches = (len(shap_data) + batch_size - 1) // batch_size
            print(f"  Processing batch {i // batch_size + 1} / {total_batches}: rows {i}-{i + len(batch) - 1}")
            shap_vals = explainer.shap_values(batch)
            all_shap_values.append(shap_vals)

        shap_values = np.vstack(all_shap_values)
    else:
        print(f"SHAP Analysis on subset of {SHAP_BATCH_SIZE} {label} rows")
        shap_data = shap_data.sample(n=SHAP_BATCH_SIZE, random_state=42)
        shap_values = explainer.shap_values(shap_data)

    # Bar plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_data, plot_type="bar", feature_names=features, show=False)
    bar_plot_path = os.path.join(shap_output_dir, f'cubist_shap_feature_importance_{label}.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to: {bar_plot_path}")

    # Beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, shap_data, feature_names=features, show=False)
    beeswarm_plot_path = os.path.join(shap_output_dir, f'cubist_shap_beeswarm_plot_{label}.png')
    plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP beeswarm plot saved to: {beeswarm_plot_path}")

    # Save raw values
    shap_values_output = os.path.join(shap_output_dir, f'shap_values_{label}.npy')
    np.save(shap_values_output, shap_values)
    print(f"SHAP values saved to: {shap_values_output}")


run_shap_analysis(X_train_scaled_df, label='historical')
run_shap_analysis(X_future_scaled_df, label='future')
