# Training

Model training and evaluation pipeline for banana yield prediction using 6 regression models with feature selection.

## How to run

```
python regression.py
```

Configure feature selection at the bottom of `regression.py`:
- `fs = "yes"` - Run with VIF + Boruta feature selection per fold
- `fs = "no"` - Use all features without selection

## Models

| Model | File | Library |
|-------|------|---------|
| Cubist | `models/Cubist.py` | `cubist` |
| GBM | `models/GBM.py` | `sklearn.ensemble.GradientBoostingRegressor` |
| MARS | `models/MARS.py` | R `earth` package via `rpy2` |
| Random Forest | `models/RF.py` | `sklearn.ensemble.RandomForestRegressor` |
| SVM | `models/SVM.py` | `sklearn.svm.SVR` |
| XGBoost | `models/XGB.py` | `xgboost.XGBRegressor` |

## Pipeline

1. Load `data/banana_yield_2010-2024.xlsx`
2. Run OLS regression + VIF multicollinearity check
3. Time Series 5-fold cross-validation
4. Per fold (if feature selection enabled):
   - Standardize features
   - Drop high-VIF features (threshold=5.0, `tmp` protected)
   - Boruta feature selection
   - Temperature rule: replace tmn/tmx/dtr with tmp
5. Grid search over all parameter combinations for each model
6. Save results to Word doc, generate comparison plots

## Output

- `results/` - Word documents with model performance metrics
- `plots/` - Bar charts comparing R2, RMSE, MSE, MAE across models + heatmap

## Data

- `data/banana_yield_2010-2024.xlsx` - Historical banana yield with 17 climate features per province per year
