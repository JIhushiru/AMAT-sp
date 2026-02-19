import joblib
import numpy as np
import pandas as pd
import os
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

Earth = importr('earth')
pandas2ri.activate()


class EarthWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, max_terms=10, max_degree=1):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_df = X
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names)

        from rpy2.robjects.packages import importr
        from rpy2.robjects import pandas2ri, r
        pandas2ri.activate()

        earth = importr('earth')

        data_frame = pandas2ri.py2rpy(pd.DataFrame({'y': y}).join(X_df))
        formula = r('y ~ .')

        self.model = earth.earth(
            formula=formula,
            data=data_frame,
            nprune=self.max_terms,
            degree=self.max_degree
        )

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        from rpy2.robjects import pandas2ri, r
        pandas2ri.activate()

        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names)

        X_r = pandas2ri.py2rpy(X_df)

        r_predict = r('predict')
        pred = r_predict(self.model, X_r)

        try:
            result = np.array(pred)
        except Exception:
            try:
                as_vector = r('as.vector')
                result = np.array(as_vector(pred))
            except Exception:
                try:
                    pred_df = pandas2ri.rpy2py_dataframe(pred)
                    result = pred_df.values
                except Exception:
                    try:
                        result = np.array([x for x in pred])
                    except Exception as e:
                        raise ValueError(f"Failed to convert prediction results: {e}")

        return result.flatten() if result.ndim > 1 else result


def get_parameters():
    return {
        'max_terms': [10, 15, 20, 25, 30],
        'max_degree': [1, 2]
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    os.makedirs('Models', exist_ok=True)

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    model = EarthWrapper(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'R2': r2,
        'RMSE': rmse,
        'MSE': mse,
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, np.finfo(float).eps, y_test))) * 100,
        'Parameters': params,
        'Selected_Features': list(X_train.columns)
    }

    model_filename = os.path.join('Models', 'MARS.joblib')
    joblib.dump({
        'params': params,
        'selected_features': list(X_train.columns)
    }, model_filename)

    return model, metrics
