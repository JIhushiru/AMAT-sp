# Bias Correction for CMIP6 Projections

## Problem

The ML model was trained on historical climate data from **TerraClimate** (observational), but future projections use **GFDL-ESM4 CMIP6** (climate model output). These datasets have systematic differences even for the same location and time period, causing a discontinuity between historical and projected yield predictions.

Without bias correction, the national mean yield drops from ~12.3 t/ha (historical) to ~7.6 t/ha (projected) — an unrealistic artifact.

## Method: Delta Bias Correction (Additive)

We use the **delta method** (additive bias correction), a standard approach in climate impact studies (Maraun & Widmann, 2018).

### Steps

1. **Extract CMIP6 historical data** (`extract_historical_cmip6.py`)
   - Queries NASA/GDDP-CMIP6 for 2015-2024 (overlap with training data)
   - Uses GFDL-ESM4 model, SSP2-4.5 scenario (SSP scenarios start at 2015)
   - Same 7 variables as future extraction: pr, tas, tasmax, tasmin, rsds, sfcWind, hurs
   - Derives: dtr, vpd, vap, pet, cld

2. **Compute bias factors** (`apply_correction.py`)
   - For each province and each variable:
     ```
     bias = mean(TerraClimate_2015-2024) - mean(CMIP6_2015-2024)
     ```
   - This captures the systematic offset between the two datasets

3. **Apply correction to future projections**
   - For each province and variable:
     ```
     corrected_future = raw_CMIP6_future + bias
     ```
   - Non-negative variables (pre, srad, pet) clipped to >= 0
   - Cloud cover clipped to 0-100%

### Variables Corrected

| Variable | Description | Mean Bias |
|----------|-------------|-----------|
| tmp | Mean temperature (C) | +0.4 |
| tmx | Max temperature (C) | +2.4 |
| tmn | Min temperature (C) | -1.5 |
| pre | Precipitation (mm/month) | +20.5 |
| srad | Solar radiation (W/m2) | -27.6 |
| ws | Wind speed (m/s) | -3.0 |
| vpd | Vapor pressure deficit (kPa) | +0.2 |
| vap | Vapor pressure (hPa) | +26.0 |
| dtr | Diurnal temperature range (C) | +3.9 |
| pet | Potential evapotranspiration | -0.4 |
| cld | Cloud cover (%) | +58.3 |

Note: Large biases in `cld` and `vap` reflect different unit scales/calculation methods between TerraClimate and CMIP6.

## Results

| Period | Mean Yield (t/ha) |
|--------|-------------------|
| Historical (2010-2024) | 12.29 |
| SSP2-4.5 corrected (2025-2034) | 11.99 |
| SSP5-8.5 corrected (2025-2034) | 12.03 |

After correction, projections smoothly continue from historical values with a slight decline (~0.3 t/ha), compared to the unrealistic 1.3 t/ha drop without correction.

## Files

| File | Description |
|------|-------------|
| `extract_historical_cmip6.py` | Extract CMIP6 data for overlap period (2015-2024) |
| `apply_correction.py` | Compute bias factors and apply to future projections |
| `cmip6_historical_2015-2024.csv` | CMIP6 values for the overlap period |
| `bias_factors.csv` | Per-province, per-variable bias correction factors |
| `ssp245_corrected.csv` | Bias-corrected SSP2-4.5 projections |
| `ssp585_corrected.csv` | Bias-corrected SSP5-8.5 projections |

## How to Run

```bash
# Step 1: Extract historical CMIP6 data (~7 min, requires GEE auth)
python Pipeline/bias_correction/extract_historical_cmip6.py

# Step 2: Compute and apply bias correction
python Pipeline/bias_correction/apply_correction.py

# Step 3: Re-run merge and predictions (merge_data.py auto-detects corrected files)
python Pipeline/SSP2-4.5/merge_data.py
python Pipeline/SSP5-8.5/merge_data.py
python Pipeline/SSP2-4.5/future_predictions.py
python Pipeline/SSP5-8.5/future_predictions.py

# Step 4: Generate combined chart
python Pipeline/combined_projections.py
```

## Reference

Maraun, D. & Widmann, M. (2018). *Statistical Downscaling and Bias Correction for Climate Research*. Cambridge University Press.
