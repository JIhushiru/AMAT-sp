import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def get_parameters():
    return {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    os.makedirs('Models', exist_ok=True)
    model = GradientBoostingRegressor()

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    selected_features = list(X_train.columns)

    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)

    if params:
        model.set_params(**params)
    model.fit(X_train_df, y_train)

    y_pred = model.predict(X_test_df)

    metrics = {
        'R2': r2_score(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, np.finfo(float).eps, y_test))) * 100,
        'Best_Parameters': model.get_params(),
        'Selected_Features': selected_features
    }

    if hasattr(model, 'feature_importances_'):
        metrics['Feature_Importance'] = dict(zip(selected_features, model.feature_importances_))

    model_filename = os.path.join('Models', 'GBM.joblib')
    model_metadata = {
        'best_model': model,
        'selected_features': selected_features
    }
    joblib.dump(model_metadata, model_filename)

    return model, metrics
