"""
Model training orchestrator.

Trains all 6 regression models (Cubist, GBM, MARS, RF, SVM, XGB) with
optional feature selection (VIF + Boruta) and TimeSeriesSplit cross-validation.
"""
import os
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from models.Cubist import train_and_evaluate as train_cubist, get_parameters as get_cubist_params
from models.GBM import train_and_evaluate as train_gbm, get_parameters as get_gbm_params
from models.MARS import train_and_evaluate as train_mars, get_parameters as get_mars_params
from models.RF import train_and_evaluate as train_rf, get_parameters as get_rf_params
from models.SVM import train_and_evaluate as train_svm, get_parameters as get_svm_params
from models.XGB import train_and_evaluate as train_xgb, get_parameters as get_xgb_params

from feature_selection import feature_selection_workflow
from diagnostics import run_ols, calculate_vif, check_multicollinearity
from visualization import visualize_model_performance
from export import save_results_to_word

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_FUNCTIONS = {
    'Cubist': train_cubist,
    'GBM': train_gbm,
    'MARS': train_mars,
    'RF': train_rf,
    'SVM': train_svm,
    'XGB': train_xgb,
}

MODEL_PARAMS = {
    'Cubist': get_cubist_params,
    'GBM': get_gbm_params,
    'MARS': get_mars_params,
    'RF': get_rf_params,
    'SVM': get_svm_params,
    'XGB': get_xgb_params,
}


def save_top_models(results, data, n_top=3):
    """Retrain and save the top N models on the full dataset with their best params."""
    X = data.drop(columns=["yield", "province", "year"])
    y = data["yield"]
    non_zero = (X != 0).any(axis=1)
    X, y = X[non_zero], y[non_zero]
    features = list(X.columns)

    # Rank all models by CV R²
    model_scores = []
    for model_type, model_data in results.items():
        for model_name, metrics in model_data.items():
            model_scores.append((model_name, metrics['best_avg_r2'], metrics['best_params']))
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = model_scores[:n_top]

    print(f"\n{'='*50}")
    print(f"Top {n_top} Models by CV R²:")
    for rank, (name, r2, _) in enumerate(top_n, 1):
        print(f"  {rank}. {name}: {r2:.4f}")
    print(f"{'='*50}")

    save_dir = os.path.join('Models', 'top3')
    os.makedirs(save_dir, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    for rank, (name, r2, params) in enumerate(top_n, 1):
        print(f"\nRetraining {name} (rank #{rank}) on full dataset...")
        train_func = MODEL_FUNCTIONS[name]
        model, _ = train_func(X_scaled, X_scaled, y, y, params)

        artifact = {
            'model': model,
            'scaler': scaler,
            'selected_features': features,
            'best_params': params,
            'cv_avg_r2': r2,
            'rank': rank,
        }
        path = os.path.join(save_dir, f'rank{rank}_{name}.joblib')
        joblib.dump(artifact, path)
        print(f"  Saved: {path}")


def load_data(file_name, sheet_name="Sheet1"):
    """Load data from an Excel file in the data/ subdirectory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'data', file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in {script_dir}")
    return pd.read_excel(file_path, sheet_name=sheet_name)


def main(model_type, fs):
    """Train and evaluate all models with optional feature selection."""
    print(f"\nStarting model training for: {model_type}\n")

    data = load_data('banana_yield_2010-2024.xlsx')
    data['yield'] = data['yield'].replace('#DIV/0!', pd.NA)
    data['yield'] = pd.to_numeric(data['yield'], errors='coerce')
    data = data.dropna(subset=['yield'])
    data = data.sort_values(by='year', ascending=True)

    model_types_to_run = [model_type]
    results = {}

    total_combinations = len(model_types_to_run) * len(MODEL_FUNCTIONS)
    current_combination = 0

    X = data.drop(columns=["yield", "province", "year"])
    y = data["yield"]

    non_zero_features = (X != 0).any(axis=1)
    X = X[non_zero_features]
    y = y[non_zero_features]

    run_ols(X, y)
    calculate_vif(X)

    tscv = TimeSeriesSplit(n_splits=5)
    check_multicollinearity(X)

    if fs == "yes":
        selected_features_per_fold = []
        feature_selection_info_per_fold = []
        for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
            print(f"Fold {fold_idx + 1}:")
            X_fold_train = X.iloc[train_index]
            y_fold_train = y.iloc[train_index]

            selected = feature_selection_workflow(X_fold_train, y_fold_train)
            selected_features_per_fold.append(selected['selected_features'])
            feature_selection_info_per_fold.append(selected)

            print(f"  Selected features: {selected}")
            print(f"  Train size: {len(train_index)}")
            print(f"  Test size: {len(test_index)}")
            print("-" * 30)
    elif fs == "no":
        selected_features = list(X.columns)
        print(f"Selected features: {selected_features}")

    for current_model_type in model_types_to_run:
        results[current_model_type] = {}
        for model_name, train_func in MODEL_FUNCTIONS.items():
            current_combination += 1
            start_time = time.time()
            print(f"\nTraining {current_combination} of {total_combinations}: {model_name} for {current_model_type}...")

            best_avg_r2 = -np.inf
            best_params = None
            best_fold_r2_scores = []
            best_fold_metrics = []
            all_avg_r2_results = []
            param_grid = MODEL_PARAMS[model_name]()

            for params in ParameterGrid(param_grid):
                fold_r2_scores = []
                fold_metrics = []

                for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
                    if fs == "yes":
                        selected_features = selected_features_per_fold[fold_idx]
                    X_fold_train = X.iloc[train_index][selected_features]
                    X_fold_test = X.iloc[test_index][selected_features]
                    y_fold_train = y.iloc[train_index]
                    y_fold_test = y.iloc[test_index]

                    scaler = StandardScaler()
                    X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                    X_fold_test_scaled = scaler.transform(X_fold_test)
                    X_fold_train_scaled = pd.DataFrame(X_fold_train_scaled, columns=selected_features)
                    X_fold_test_scaled = pd.DataFrame(X_fold_test_scaled, columns=selected_features)

                    model, metrics = train_func(
                        X_fold_train_scaled, X_fold_test_scaled, y_fold_train, y_fold_test, params
                    )

                    if model is not None:
                        test_r2 = model.score(X_fold_test_scaled, y_fold_test)
                        fold_r2_scores.append(test_r2)
                        fold_metrics.append(metrics)

                if fold_r2_scores:
                    avg_r2 = np.mean(fold_r2_scores)
                    all_avg_r2_results.append((params, avg_r2))

                    if avg_r2 > best_avg_r2:
                        best_avg_r2 = avg_r2
                        best_params = params
                        best_fold_r2_scores = fold_r2_scores[:]
                        best_fold_metrics = fold_metrics[:]

            print(f"Best Params for {model_name}: {best_params}")
            print(f"Best Avg R2: {best_avg_r2:.4f}")
            print(f"Execution Time for {model_name}: {time.time() - start_time:.2f} seconds")

            avg_rmse = np.mean([m['RMSE'] for m in best_fold_metrics])
            avg_mse = np.mean([m['MSE'] for m in best_fold_metrics])
            avg_mae = np.mean([m['MAE'] for m in best_fold_metrics])
            avg_mape = np.mean([m['MAPE'] for m in best_fold_metrics])

            results[current_model_type][model_name] = {
                'best_avg_r2': best_avg_r2,
                'best_params': best_params,
                'all_results': all_avg_r2_results,
                'avg_metrics': {
                    'R2': best_avg_r2,
                    'RMSE': avg_rmse,
                    'MSE': avg_mse,
                    'MAE': avg_mae,
                    'MAPE': avg_mape
                },
                'fold_r2_scores': best_fold_r2_scores,
                'fold_best_params': [best_params for _ in range(len(best_fold_r2_scores))]
            }

    print("\nTraining Complete. Model Performance:")
    for model_type, model_data in results.items():
        print(f"\n--- {model_type.upper()} Models ---")
        for model_name, metrics in model_data.items():
            print(f"\n{model_name} Performance:")
            metrics_to_print = {k: v for k, v in metrics.get('avg_metrics', {}).items() if k != 'residuals'}
            print(f"  Metrics: {metrics_to_print}")

    if fs == "yes":
        return results, data, feature_selection_info_per_fold
    elif fs == "no":
        return results, data


if __name__ == "__main__":
    start_time = time.time()
    fs = "yes"

    if fs == "yes":
        results, data, feature_selection_info_per_fold = main(model_type='regression', fs="yes")
        performance_df = visualize_model_performance(results, True)
        save_results_to_word(results, feature_selection_info_per_fold)
    elif fs == "no":
        results, data = main(model_type='regression', fs="no")
        performance_df = visualize_model_performance(results, False)
        save_results_to_word(results, None, "model_results_without_fs.docx")

    save_top_models(results, data, n_top=3)
    print(f"Overall Execution Time: {time.time() - start_time:.2f} seconds")
