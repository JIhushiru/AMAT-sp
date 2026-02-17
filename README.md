# Banana Yield Prediction Under Climate Change

Predicting Philippine banana yield at the province level (2025-2034) using machine learning and CMIP6 climate projections under SSP2-4.5 and SSP5-8.5 scenarios.

## Project structure

```
Training/           Model training with feature selection and cross-validation
Pipeline/            Production pipeline: train best model -> predict future yields
Mapping/             Choropleth maps of banana yield by province
SSPs Data collection/  Raw SSP climate data extraction from Google Earth Engine
area,production, yield/  Raw data processing (CRU climate + PSA crop stats)
webapp/              Web dashboard (FastAPI + React)
```

## The data

The training dataset covers 82 Philippine provinces from 2010 to 2024, with banana yield (tons/ha) and 17 climate features per province per year. That gives us 1,230 rows (82 provinces x 15 years) and 20 columns.

**Where it comes from:**

Banana yield data was downloaded from the PSA CountrySTAT portal. Climate data was extracted from CRU TS 4.08 NetCDF files (0.5-degree resolution) and clipped to province boundaries using the polygons in `philippines_provinces.geojson`. For small or island provinces where no grid cell falls inside the polygon, the script falls back to the nearest grid cell around the centroid. Additional variables (PDSI, q, soil, srad, ws, aet, def, vpd) came from TerraClimate.

All of this is handled by `area,production, yield/pipeline/main_extract_and_merge.py`. See its README for the full methodology.

**Climate features:**

| Variable | Description | Source |
|----------|-------------|--------|
| tmp | Mean temperature (C) | CRU TS |
| tmx | Max temperature (C) | CRU TS |
| tmn | Min temperature (C) | CRU TS |
| dtr | Diurnal temperature range (C) | CRU TS |
| pre | Precipitation (mm/month) | CRU TS |
| pet | Potential evapotranspiration (mm/month) | CRU TS |
| cld | Cloud cover (%) | CRU TS |
| wet | Wet day frequency (days/month) | CRU TS |
| vap | Vapor pressure (hPa) | CRU TS |
| vpd | Vapor pressure deficit (hPa) | TerraClimate |
| aet | Actual evapotranspiration (mm/month) | TerraClimate |
| def | Climatic water deficit (mm/month) | TerraClimate |
| PDSI | Palmer Drought Severity Index | TerraClimate |
| q | Runoff (mm/month) | TerraClimate |
| soil | Soil moisture (mm) | TerraClimate |
| srad | Solar radiation (W/m2) | TerraClimate |
| ws | Wind speed (m/s) | TerraClimate |

## Training

Two training workflows exist depending on what you need:

### Exploration (`Training/regression.py`)

The full comparison pipeline with feature selection. Takes several hours because of the large hyperparameter grids.

1. Loads `Training/data/banana_yield_2010-2024.xlsx`
2. Runs OLS regression and checks VIF multicollinearity
3. For each of the 5 TimeSeriesSplit folds (if feature selection is on):
   - Standardizes features
   - Drops high-VIF features iteratively (threshold = 5.0, `tmp` is protected)
   - Runs Boruta feature selection
   - Applies temperature rule: if tmn/tmx/dtr got selected, replace them with tmp
4. Grid search over all hyperparameter combinations for all 6 models
5. Exports comparison plots and a Word document with results

To run: `python Training/regression.py` (set `fs = "yes"` or `"no"` at the bottom)

### Production pipeline (`Pipeline/train_models.py`)

Faster version with trimmed hyperparameter grids, meant to feed into the prediction pipeline.

1. Loads the same training data
2. Uses the 10 features that came out of the exploration step: `cld, tmp, pre, aet, PDSI, q, soil, srad, vpd, ws`
3. Trains all 6 models with 5-fold TimeSeriesSplit cross-validation
4. Ranks by mean CV R2, retrains the best model on the full dataset
5. Saves the model, scaler, and metadata to `Pipeline/saved_model/best_model.joblib`

To run: `python Pipeline/train_models.py`

### Models

| Model | Library |
|-------|---------|
| Cubist | `cubist` |
| GBM | scikit-learn |
| MARS | R `earth` via `rpy2` |
| Random Forest | scikit-learn |
| SVM | scikit-learn |
| XGBoost | `xgboost` |

Cubist consistently comes out on top (CV R2 ~ 0.67).

## SSP predictions

Once the best model is saved, the prediction pipeline generates yield forecasts for 2025-2034 under two climate scenarios.

### How future climate features are built

CMIP6 only gives us precipitation and temperature projections. The other features need to be derived or held constant. The approach (documented in detail in `Pipeline/feature_methods.py`):

- **Direct from SSP:** `tmp`, `pre` -- used as-is from CMIP6 projections
- **Delta change method:** `tmx`, `tmn` -- the SSP temperature anomaly (SSP tmp minus historical avg tmp) is added to the historical average of each. `dtr` is then just tmx - tmn
- **Tetens equation scaling:** `vpd`, `pet` -- scaled by the ratio of saturation vapor pressure at the new temperature vs the historical temperature
- **Historical averages:** `cld, wet, vap, aet, def, PDSI, q, soil, srad, ws` -- held at each province's 2010-2024 average since SSP projections aren't available for these

This is done by `Pipeline/SSP2-4.5/merge_data.py` and `Pipeline/SSP5-8.5/merge_data.py`.

### Running predictions

```
python Pipeline/SSP2-4.5/merge_data.py          # build future climate features
python Pipeline/SSP2-4.5/future_predictions.py   # predict yields
python Pipeline/SSP2-4.5/projections.py          # analysis and plots
```

Same for SSP5-8.5. The prediction script clips outputs to each province's historical yield range to avoid extrapolation artifacts.

### SSP data collection (from scratch)

If you need to regenerate the raw SSP climate projections from Google Earth Engine, the scripts are in `SSPs Data collection/`. See its README for the 6-step pipeline. The processed projections are already in `Pipeline/data/`.

## Mapping

`Mapping/map.py` generates choropleth maps using GADM shapefiles, showing average banana yield by province. Produces a full Philippines map and regional breakdowns (Luzon, Visayas, Mindanao) at 600 DPI.

## Web app

A dashboard for exploring the results interactively.

- **Backend:** FastAPI serving historical data, model metrics, SSP predictions, and GeoJSON (`webapp/api/main.py`)
- **Frontend:** React + Tailwind CSS with Leaflet maps and Recharts charts (`webapp/frontend/`)

Pages: Dashboard, Interactive Map, Historical Data, Model Results, SSP Scenarios, Province Detail.

To start both servers: `python run.py`

## Requirements

Python packages: pandas, numpy, scikit-learn, xgboost, cubist, statsmodels, boruta, matplotlib, seaborn, openpyxl, python-docx, joblib, fastapi, uvicorn

For MARS: R with the `earth` package, plus `rpy2`

Frontend: Node.js, npm (dependencies in `webapp/frontend/package.json`)
