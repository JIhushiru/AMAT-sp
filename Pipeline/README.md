# Pipeline

Production pipeline for training the best model, generating future yield predictions under SSP scenarios, and analyzing the results. This is the streamlined version of the full exploration workflow in `Training/`.

## How to run

Run these scripts in order from the project root:

```
python Pipeline/train_models.py

python Pipeline/SSP2-4.5/merge_data.py
python Pipeline/SSP2-4.5/future_predictions.py
python Pipeline/SSP2-4.5/projections.py

python Pipeline/SSP5-8.5/merge_data.py
python Pipeline/SSP5-8.5/future_predictions.py
python Pipeline/SSP5-8.5/projections.py
```

## What each script does

### 1. `train_models.py`

Trains all 6 ML models (Cubist, GBM, MARS, RF, SVM, XGBoost) using 5-fold TimeSeriesSplit cross-validation with grid search. Also evaluates a province historical mean baseline for comparison.

After ranking models by mean CV R², the best one (Cubist) is retrained on the full dataset and saved to `saved_model/best_model.joblib`.

Then it trains 100 bootstrap models (resampled with replacement) using the same best model and hyperparameters. These are saved to `saved_model/bootstrap_models.joblib` and used later for prediction intervals.

**Outputs:**
- `saved_model/best_model.joblib` (model, scaler, features, metadata)
- `saved_model/bootstrap_models.joblib` (100 bootstrap model/scaler pairs)
- `saved_model/model_comparison.csv`
- `saved_model/model_comparison.png`

### 2. `SSP*/merge_data.py`

Builds the future climate feature set for 2025 to 2034 by combining CMIP6 SSP projections with historical province averages.

Only temperature (`tmp`) and precipitation (`pre`) come directly from CMIP6. The other features are either derived using the delta change method (for `tmx`, `tmn`, `dtr`) or scaled with the Tetens equation (`vpd`, `pet`), or held at historical averages (`cld`, `wet`, `vap`, `aet`, `def`, `PDSI`, `q`, `soil`, `srad`, `ws`).

The full methodology for each feature is documented in `feature_methods.py`.

**Output:** `SSP*/merged_future_data.csv`

### 3. `SSP*/future_predictions.py`

Loads the saved best model and generates yield predictions for all 82 provinces across 2025 to 2034.

If bootstrap models exist, it runs all 100 on the future data and computes the 5th and 95th percentiles as the 90% prediction interval. The 90% level is standard in climate and crop modeling (Lobell & Burke, 2010; IPCC convention). It is wider than a confidence interval by nature since it captures uncertainty about individual predictions rather than the mean, so using 90% instead of 95% avoids overly conservative bands.

All predictions (point estimates and interval bounds) are clipped to each province's historical yield range to prevent extrapolation artifacts.

**Outputs:**
- `SSP*/banana_yield_predictions_2025-2034.xlsx`
- `SSP*/observed_vs_predicted_yield.png`

### 4. `SSP*/projections.py`

Runs the analysis and generates summary statistics and plots.

**Analysis:**
- National average yield: historical vs projected, with percent change
- Province level averages and percent change (2010 to 2024 avg vs 2025 to 2034 avg)
- Direct comparison of 2024 vs 2034 yields per province
- National trend line combining historical and projected years

**Outputs:**
- `SSP*/banana_yield_analysis.xlsx` (all analysis sheets)
- `SSP*/province_summary.csv`
- `SSP*/compare_2024_2034.csv`
- `SSP*/national_trend.csv`
- `SSP*/national_yield_trend.png` (includes 90% prediction interval band on future years)
- `SSP*/province_percent_change.png`
- `SSP*/yield_2024_vs_2034.png`

## Supporting files

### `feature_methods.py`

Documents how each of the 17 climate features is computed for future projections, with full references. Three categories:

| Category | Features | Method |
|----------|----------|--------|
| Direct from SSP | tmp, pre | Used as-is from CMIP6 |
| Delta change / Tetens | tmx, tmn, dtr, vpd, pet | Physically adjusted using SSP temperature anomaly |
| Historical average | cld, wet, vap, aet, def, PDSI, q, soil, srad, ws | Held at province 2010 to 2024 mean |

### `data/`

- `banana_yield_2010-2024.xlsx`: Training data (1,230 rows, 82 provinces, 15 years)
- `ssp245_projections.csv`: CMIP6 SSP2-4.5 temperature and precipitation projections
- `ssp585_projections.csv`: CMIP6 SSP5-8.5 temperature and precipitation projections

## Prediction intervals

The 100 bootstrap models each produce a slightly different prediction for every province-year combination. Taking the 5th and 95th percentiles across those 100 predictions gives a 90% prediction interval.

On the national trend plot, this shows up as a shaded band around the point predictions for 2025 to 2034. The band is wider when the bootstrap models disagree more, indicating higher uncertainty.

The intervals are also clipped to each province's historical range, same as the point predictions.

## Baseline comparison

The province historical mean baseline simply predicts each province's average yield from the training data. It gets a CV R² of 0.92, which beats all ML models (Cubist gets 0.67).

This happens because most yield variation is between provinces, not over time. A province like Davao del Norte consistently yields around 50 tons/ha while mountain provinces sit at 2 to 5.

The ML models are still necessary because the baseline predicts the exact same yield regardless of climate. For projecting what happens under SSP2-4.5 vs SSP5-8.5, only the climate-sensitive models are useful. The baseline is there to contextualize the R² values.
