# Pipeline

Production pipeline for training the best model, generating future yield predictions under SSP scenarios using a 5-GCM ensemble, and analyzing the results. This is the streamlined version of the full exploration workflow in `Training/`.

## Multi-model ensemble

Future climate projections use a 5-model CMIP6 ensemble from NASA/GDDP-CMIP6, following IPCC AR6 guidance that projections from a single GCM should not be used in isolation (IPCC, 2021; Tebaldi and Knutti, 2007).

Each GCM is bias-corrected independently against TerraClimate using the delta method, then fed through the trained Cubist model. The ensemble mean serves as the central yield estimate. Two uncertainty sources are quantified separately:

1. **Structural uncertainty** from the spread across 5 GCMs
2. **Statistical uncertainty** from 100 bootstrap prediction intervals

| GCM | Rationale |
|-----|-----------|
| GFDL-ESM4 | Good general tropical performance |
| MIROC6 | Strong tropical climate representation |
| MRI-ESM2-0 | Well-validated for Asian monsoon regions |
| IPSL-CM6A-LR | Good precipitation representation |
| CanESM5 | High climate sensitivity (provides ensemble spread) |

## How to run

Run these scripts in order from the project root:

```
# Step 1: Train models
python Pipeline/train_models.py

# Step 2: Extract CMIP6 data for all 5 GCMs (both scenarios)
python Pipeline/extract_cmip6.py

# Step 3: Extract historical overlap for bias correction
python Pipeline/bias_correction/extract_historical_cmip6.py

# Step 4: Compute and apply bias correction per model
python Pipeline/bias_correction/apply_correction.py

# Step 5: Merge features and predict for each scenario
python Pipeline/SSP2-4.5/merge_data.py
python Pipeline/SSP2-4.5/future_predictions.py
python Pipeline/SSP2-4.5/projections.py

python Pipeline/SSP5-8.5/merge_data.py
python Pipeline/SSP5-8.5/future_predictions.py
python Pipeline/SSP5-8.5/projections.py

# Step 6: Combined visualization
python Pipeline/combined_projections.py
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

### 2. `extract_cmip6.py`

Extracts 7 raw climate variables from NASA/GDDP-CMIP6 via Google Earth Engine for all 5 GCMs under both SSP2-4.5 and SSP5-8.5 (2025 to 2034). Derives 4 additional features (dtr, vpd, vap, pet, cld) from the raw variables.

**Outputs:** `data/{scenario}_{model}_projections.csv` (10 files: 5 models x 2 scenarios)

### 3. `bias_correction/extract_historical_cmip6.py`

Extracts the same 7 variables for the historical overlap period (2015 to 2024) for all 5 GCMs, used to compute per-model bias correction factors.

**Outputs:** `bias_correction/cmip6_historical_{model}_2015-2024.csv` (5 files)

### 4. `bias_correction/apply_correction.py`

Computes delta bias correction for each GCM independently: `corrected = future + (TerraClimate_mean - CMIP6_historical_mean)` per province per variable.

**Outputs:**
- `bias_correction/bias_factors_{model}.csv` (5 files, one per GCM)
- `bias_correction/{scenario}_{model}_corrected.csv` (10 files)

### 5. `SSP*/merge_data.py`

For each of the 5 GCMs, builds the full 17-feature future climate dataset by combining bias-corrected CMIP6 features with historical province averages for features without CMIP6 equivalents.

**Output:** `SSP*/merged_future_data_{model}.csv` (5 files per scenario)

### 6. `SSP*/future_predictions.py`

Runs the Cubist model on each GCM's merged data, then computes the ensemble mean as the central estimate. Combines model spread (5-GCM range) with bootstrap prediction intervals (100 resampled models) for combined uncertainty bounds. All predictions are clipped to province historical yield ranges.

**Outputs:**
- `SSP*/banana_yield_predictions_2025-2034.xlsx` (ensemble mean + per-GCM columns + uncertainty)
- `SSP*/observed_vs_predicted_yield.png`

### 7. `SSP*/projections.py`

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
- `SSP*/national_yield_trend.png` (includes uncertainty band on future years)
- `SSP*/province_percent_change.png`
- `SSP*/yield_2024_vs_2034.png`

### 8. `combined_projections.py`

Generates the combined national trend chart showing historical data, both SSP ensemble means with uncertainty bands, and thin traces for individual GCM projections.

**Output:** `combined_national_trend.png`

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
- `{scenario}_{model}_projections.csv`: Raw CMIP6 projections per GCM

## Prediction intervals and model spread

The output Excel files contain:
- `yield`: Ensemble mean prediction (average across 5 GCMs)
- `yield_lower`, `yield_upper`: Combined uncertainty bounds (outer envelope of bootstrap PI and GCM spread)
- `model_spread`: Standard deviation across the 5 GCMs for each province-year
- `yield_{model}`: Individual GCM predictions for transparency

On the national trend plot, the shaded band reflects combined uncertainty while thin lines show individual GCM trajectories.

## Baseline comparison

The province historical mean baseline simply predicts each province's average yield from the training data. It gets a CV R² of 0.92, which beats all ML models (Cubist gets 0.67).

This happens because most yield variation is between provinces, not over time. A province like Davao del Norte consistently yields around 50 tons/ha while mountain provinces sit at 2 to 5.

The ML models are still necessary because the baseline predicts the exact same yield regardless of climate. For projecting what happens under SSP2-4.5 vs SSP5-8.5, only the climate-sensitive models are useful. The baseline is there to contextualize the R² values.

## References

- IPCC (2021). AR6 WG1 Chapter 4: Future Global Climate. https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-4/
- Tebaldi, C. and Knutti, R. (2007). The use of the multi-model ensemble in probabilistic climate projections. Phil. Trans. R. Soc. A. doi:10.1098/rsta.2007.2076
- Maraun, D. and Widmann, M. (2018). Statistical Downscaling and Bias Correction for Climate Research. Cambridge University Press.
- Lobell, D.B. and Burke, M.B. (2010). On the use of statistical models to predict crop yield responses to climate change. Agricultural and Forest Meteorology.
- Challinor, A.J. et al. (2014). A meta-analysis of crop yield under climate change and adaptation. Nature Climate Change.
