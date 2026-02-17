"""
Train all 6 ML models, compare cross-validated performance, and save the best.

Workflow:
  1. Load historical data (2010-2024)
  2. StandardScaler + TimeSeriesSplit (5-fold) cross-validation
  3. Grid search over hyperparameters for each model
  4. Rank models by mean CV R²
  5. Retrain best model on full dataset
  6. Save best model + scaler + metadata to Pipeline/saved_model/

The saved model is then loaded by future_predictions.py for SSP projections.
"""
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from cubist import Cubist
import matplotlib.pyplot as plt

# Try to import MARS (requires R + rpy2)
try:
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri, r as R_func
    from sklearn.base import BaseEstimator, RegressorMixin

    class EarthWrapper(BaseEstimator, RegressorMixin):
        def __init__(self, max_terms=10, max_degree=1):
            self.max_terms = max_terms
            self.max_degree = max_degree
            self.model = None
            self.feature_names = None

        def fit(self, X, y):
            pandas2ri.activate()
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X_df = X
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=self.feature_names)

            earth = importr('earth')
            data_frame = pandas2ri.py2rpy(pd.DataFrame({'y': y}).join(X_df.reset_index(drop=True)))
            formula = R_func('y ~ .')
            self.model = earth.earth(formula=formula, data=data_frame,
                                     nprune=self.max_terms, degree=self.max_degree)
            return self

        def predict(self, X):
            if self.model is None:
                raise ValueError("Model has not been fitted yet.")
            pandas2ri.activate()
            if isinstance(X, pd.DataFrame):
                X_df = X
            else:
                X_df = pd.DataFrame(X, columns=self.feature_names)
            X_r = pandas2ri.py2rpy(X_df)
            r_predict = R_func('predict')
            pred = r_predict(self.model, X_r)
            result = np.array(pred)
            return result.flatten() if result.ndim > 1 else result

    MARS_AVAILABLE = True
    print("MARS (R earth package) available")
except Exception:
    MARS_AVAILABLE = False
    print("MARS unavailable (rpy2/R not found), skipping")


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')

# ── Load historical data ─────────────────────────────────────────────────
hist_df = pd.read_excel(os.path.join(data_dir, 'banana_yield_2010-2024.xlsx'))
hist_df = hist_df[pd.to_numeric(hist_df['yield'], errors='coerce').notnull()]
hist_df['yield'] = hist_df['yield'].astype(float)
hist_df = hist_df.sort_values(by='year', ascending=True).reset_index(drop=True)

# 10 features (same as original Cubist model)
features = ['cld', 'tmp', 'pre', 'aet', 'PDSI', 'q', 'soil', 'srad', 'vpd', 'ws']

X = hist_df[features]
y = hist_df['yield']

print(f"Data: {len(X)} samples, {len(features)} features")
print(f"Yield range: {y.min():.2f} - {y.max():.2f} (mean {y.mean():.2f})")

# ── Model definitions and parameter grids ─────────────────────────────────
model_configs = {
    'Cubist': {
        'class': Cubist,
        'params': {
            'n_committees': [5, 10, 20],
            'n_rules': [100, 200, 300],
            'neighbors': [1, 2, 5],
        },
        'init_kwargs': {},
    },
    'GBM': {
        'class': GradientBoostingRegressor,
        'params': {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0],
        },
        'init_kwargs': {'random_state': 42},
    },
    'RF': {
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': [100, 300, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        },
        'init_kwargs': {'random_state': 42, 'n_jobs': -1},
    },
    'SVM': {
        'class': SVR,
        'params': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf'],
            'epsilon': [0.01, 0.1, 0.2],
        },
        'init_kwargs': {},
    },
    'XGB': {
        'class': XGBRegressor,
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
        },
        'init_kwargs': {'random_state': 42},
    },
}

if MARS_AVAILABLE:
    model_configs['MARS'] = {
        'class': EarthWrapper,
        'params': {
            'max_terms': [10, 18, 30],
            'max_degree': [1, 2],
        },
        'init_kwargs': {},
    }

# ── Baseline: province historical mean ────────────────────────────────────
# "Does the ML model beat just predicting each province's historical average?"
tscv = TimeSeriesSplit(n_splits=5)
provinces = hist_df['province']

baseline_fold_r2 = []
baseline_fold_rmse = []
baseline_fold_mae = []
baseline_fold_mape = []

for train_idx, test_idx in tscv.split(X):
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]
    prov_train = provinces.iloc[train_idx]
    prov_test = provinces.iloc[test_idx]

    # compute province means from training data only
    prov_means = pd.Series(y_train_fold.values, index=prov_train.values).groupby(level=0).mean()
    global_mean = y_train_fold.mean()

    # predict: use province mean if seen in training, else global mean
    y_baseline = prov_test.map(prov_means).fillna(global_mean).values

    baseline_fold_r2.append(r2_score(y_test_fold, y_baseline))
    baseline_fold_rmse.append(np.sqrt(mean_squared_error(y_test_fold, y_baseline)))
    baseline_fold_mae.append(mean_absolute_error(y_test_fold, y_baseline))
    baseline_fold_mape.append(np.mean(np.abs((y_test_fold - y_baseline) / y_test_fold)) * 100)

baseline_cv_r2 = np.mean(baseline_fold_r2)
baseline_cv_rmse = np.mean(baseline_fold_rmse)
baseline_cv_mae = np.mean(baseline_fold_mae)
baseline_cv_mape = np.mean(baseline_fold_mape)

# full-dataset baseline (province means from all data)
prov_means_full = hist_df.groupby('province')['yield'].mean()
y_baseline_full = provinces.map(prov_means_full).values
baseline_train_r2 = r2_score(y, y_baseline_full)
baseline_train_rmse = np.sqrt(mean_squared_error(y, y_baseline_full))
baseline_train_mae = mean_absolute_error(y, y_baseline_full)
baseline_train_mape = np.mean(np.abs((y - y_baseline_full) / y)) * 100

print(f"\n{'='*55}")
print("Baseline: Province Historical Mean")
print(f"  CV R²:    {baseline_cv_r2:.4f}")
print(f"  Train R²: {baseline_train_r2:.4f}  |  RMSE: {baseline_train_rmse:.4f}  |  MAE: {baseline_train_mae:.4f}")

results = {}
overall_start = time.time()

# ── Cross-validation and grid search ──────────────────────────────────────
for name, config in model_configs.items():
    print(f"\n{'='*55}")
    print(f"Training {name}...")
    model_start = time.time()

    best_avg_r2 = -np.inf
    best_params = None

    param_grid = list(ParameterGrid(config['params']))
    print(f"  {len(param_grid)} parameter combinations x 5 folds")

    for params in param_grid:
        fold_r2_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
            X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features)

            model = config['class'](**config['init_kwargs'])
            model.set_params(**params)
            model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            fold_r2_scores.append(r2_score(y_test, y_pred))

        avg_r2 = np.mean(fold_r2_scores)
        if avg_r2 > best_avg_r2:
            best_avg_r2 = avg_r2
            best_params = params

    # Retrain on full dataset with best params
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    final_model = config['class'](**config['init_kwargs'])
    final_model.set_params(**best_params)
    final_model.fit(X_scaled, y)

    y_pred_full = final_model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred_full)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred_full))
    train_mae = mean_absolute_error(y, y_pred_full)
    train_mape = np.mean(np.abs((y - y_pred_full) / y)) * 100

    elapsed = time.time() - model_start

    results[name] = {
        'model': final_model,
        'scaler': scaler,
        'best_params': best_params,
        'cv_r2': best_avg_r2,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'time': elapsed,
    }

    print(f"  Best params: {best_params}")
    print(f"  CV R²:    {best_avg_r2:.4f}")
    print(f"  Train R²: {train_r2:.4f}  |  RMSE: {train_rmse:.4f}  |  MAE: {train_mae:.4f}")
    print(f"  Time:     {elapsed:.1f}s")

# ── Add baseline to results for unified comparison ───────────────────────
results['Baseline'] = {
    'model': None,
    'scaler': None,
    'best_params': {'method': 'province_historical_mean'},
    'cv_r2': baseline_cv_r2,
    'train_r2': baseline_train_r2,
    'train_rmse': baseline_train_rmse,
    'train_mae': baseline_train_mae,
    'train_mape': baseline_train_mape,
    'time': 0.0,
}

# ── Compare and rank ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MODEL COMPARISON (ranked by CV R²)")
print(f"{'='*60}")
print(f"  {'Rank':<5} {'Model':<10} {'CV R²':>8} {'Train R²':>10} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print(f"  {'-'*57}")

sorted_results = sorted(results.items(), key=lambda x: x[1]['cv_r2'], reverse=True)
for rank, (name, res) in enumerate(sorted_results, 1):
    marker = ' <-- baseline' if name == 'Baseline' else ''
    print(f"  {rank:<5} {name:<10} {res['cv_r2']:>8.4f} {res['train_r2']:>10.4f} {res['train_rmse']:>8.4f} {res['train_mae']:>8.4f} {res['train_mape']:>7.2f}%{marker}")

# Best ML model (exclude baseline)
sorted_ml = [(n, r) for n, r in sorted_results if n != 'Baseline']
best_name = sorted_ml[0][0]
best_result = sorted_ml[0][1]

improvement = best_result['cv_r2'] - baseline_cv_r2
print(f"\nBest model: {best_name} (CV R² = {best_result['cv_r2']:.4f})")
print(f"Improvement over baseline: {improvement:+.4f} R² ({improvement / abs(baseline_cv_r2) * 100:+.1f}%)")
print(f"Total training time: {time.time() - overall_start:.1f}s")

# ── Save best model ──────────────────────────────────────────────────────
save_dir = os.path.join(base_dir, 'saved_model')
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'best_model.joblib')
joblib.dump({
    'model': best_result['model'],
    'scaler': best_result['scaler'],
    'features': features,
    'model_name': best_name,
    'best_params': best_result['best_params'],
    'cv_r2': best_result['cv_r2'],
    'train_r2': best_result['train_r2'],
    'train_rmse': best_result['train_rmse'],
    'train_mae': best_result['train_mae'],
    'train_mape': best_result['train_mape'],
}, save_path)

print(f"\nBest model saved to: {save_path}")

# ── Bootstrap models for prediction intervals ────────────────────────────
N_BOOTSTRAP = 100
print(f"\nBootstrapping {N_BOOTSTRAP} models for prediction intervals...")
bootstrap_start = time.time()

bootstrap_models = []
rng = np.random.RandomState(42)

for i in range(N_BOOTSTRAP):
    # resample with replacement
    idx = rng.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[idx].reset_index(drop=True)
    y_boot = y.iloc[idx].reset_index(drop=True)

    scaler_boot = StandardScaler()
    X_boot_s = pd.DataFrame(scaler_boot.fit_transform(X_boot), columns=features)

    model_boot = model_configs[best_name]['class'](**model_configs[best_name]['init_kwargs'])
    model_boot.set_params(**best_result['best_params'])
    model_boot.fit(X_boot_s, y_boot)

    bootstrap_models.append({'model': model_boot, 'scaler': scaler_boot})

bootstrap_path = os.path.join(save_dir, 'bootstrap_models.joblib')
joblib.dump({
    'models': bootstrap_models,
    'features': features,
    'model_name': best_name,
    'n_bootstrap': N_BOOTSTRAP,
}, bootstrap_path)

print(f"  {N_BOOTSTRAP} bootstrap models saved to: {bootstrap_path}")
print(f"  Bootstrap time: {time.time() - bootstrap_start:.1f}s")

# Save comparison table
comparison_df = pd.DataFrame([
    {
        'Model': name,
        'CV_R2': res['cv_r2'],
        'Train_R2': res['train_r2'],
        'Train_RMSE': res['train_rmse'],
        'Train_MAE': res['train_mae'],
        'Train_MAPE': res['train_mape'],
        'Best_Params': str(res['best_params']),
        'Time_Seconds': round(res['time'], 1),
    }
    for name, res in sorted_results
])
comparison_path = os.path.join(save_dir, 'model_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"Comparison table saved to: {comparison_path}")

# ── Comparison bar chart ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

model_names = [name for name, _ in sorted_results]
cv_r2s = [res['cv_r2'] for _, res in sorted_results]
rmses = [res['train_rmse'] for _, res in sorted_results]
maes = [res['train_mae'] for _, res in sorted_results]

colors = []
for name in model_names:
    if name == best_name:
        colors.append('#2ecc71')
    elif name == 'Baseline':
        colors.append('#e74c3c')
    else:
        colors.append('#3498db')

axes[0].barh(model_names, cv_r2s, color=colors)
axes[0].set_xlabel('CV R²')
axes[0].set_title('Cross-Validated R²')
axes[0].invert_yaxis()

axes[1].barh(model_names, rmses, color=colors)
axes[1].set_xlabel('RMSE')
axes[1].set_title('Training RMSE')
axes[1].invert_yaxis()

axes[2].barh(model_names, maes, color=colors)
axes[2].set_xlabel('MAE')
axes[2].set_title('Training MAE')
axes[2].invert_yaxis()

fig.suptitle(f'Model Comparison — Best: {best_name} (CV R² = {best_result["cv_r2"]:.4f})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparison chart saved to: {os.path.join(save_dir, 'model_comparison.png')}")
