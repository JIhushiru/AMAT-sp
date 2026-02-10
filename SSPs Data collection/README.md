# SSP Climate Projection Pipeline

Predicts Philippine banana yield (2025-2034) under two climate scenarios using CMIP6 projections.

## Scenarios

| Folder | Scenario | Description |
|--------|----------|-------------|
| `SSP2-4.5/` | SSP2-4.5 | "Middle of the road" - moderate emissions |
| `SSP5-8.5/` | SSP5-8.5 | "Fossil-fueled development" - high emissions |

Each folder contains the full pipeline and is self-contained. Run each independently.

## Pipeline (run in order, inside each scenario folder)

```
[Step 1] gee_to_gdrive.py      --> .tif files on Google Drive (download to this folder)
[Step 2] tif_to_excel.py       --> province_year_all_features_2025_2034.xlsx
[Step 3] merge_excels.py       --> merged_future_data.csv
[Step 4] future_predictions.py --> banana_yield_predictions_2025-2034.xlsx
[Step 5] projections.py        --> yield analysis + plots
[Step 6] shap_analysis.py      --> SHAP feature importance plots
```

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `gee_to_gdrive.py` | Exports CMIP6 monthly climate rasters (pr, tas) from Google Earth Engine to Google Drive |
| 2 | `tif_to_excel.py` | Clips downloaded .tif rasters to province polygons, extracts yearly means |
| 3 | `merge_excels.py` | Combines SSP climate data (pre, tmp) with historical averages for all other features |
| 4 | `future_predictions.py` | Trains Cubist on historical data, predicts 2025-2034 yields |
| 5 | `projections.py` | Compares historical vs predicted yields, % change analysis |
| 6 | `shap_analysis.py` | SHAP feature importance on historical + future data |

## Key design note

Only **precipitation** (`pre`) and **temperature** (`tmp`) come from SSP projections. The other 15 climate features use their 2010-2024 historical averages, because NASA/GDDP-CMIP6 only provides `pr` and `tas`.

## Required data files (per scenario folder)

- `banana_yield_2010-2024.xlsx` - Historical yield + 17 climate features
- `philippines_provinces.geojson` - Province boundary polygons
- `gfdl_pr_2025_2034_monthly.tif` - Downloaded from Google Drive after step 1
- `gfdl_tas_2025_2034_monthly.tif` - Downloaded from Google Drive after step 1
