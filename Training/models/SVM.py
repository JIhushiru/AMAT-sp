import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def get_parameters():
    return {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly'],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    os.makedirs('Models', exist_ok=True)

    if isinstance(X_train, pd.DataFrame):
        selected_features = list(X_train.columns)
    else:
        selected_features = [f'feature_{i}' for i in range(X_train.shape[1])]

    X_train_df = pd.DataFrame(X_train, columns=selected_features)
    X_test_df = pd.DataFrame(X_test, columns=selected_features)

    model = SVR(kernel='rbf')
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

    model_filename = os.path.join('Models', 'SVM.joblib')
    model_metadata = {
        'best_model': model,
        'selected_features': selected_features
    }
    joblib.dump(model_metadata, model_filename)

    return model, metrics
