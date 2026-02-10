import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error


def get_parameters():
    return {
        'n_estimators': [i * 100 for i in range(1, 9)],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10]
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    os.makedirs('Models', exist_ok=True)

    if isinstance(X_train, pd.DataFrame):
        selected_features = list(X_train.columns)
    else:
        selected_features = [f'feature_{i}' for i in range(X_train.shape[1])]

    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)

    model = RandomForestRegressor()
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

    model_filename = os.path.join('Models', 'RF.joblib')
    model_metadata = {
        'best_model': model,
        'selected_features': selected_features
    }
    joblib.dump(model_metadata, model_filename)

    return model, metrics
