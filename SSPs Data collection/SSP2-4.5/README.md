# SSP2-4.5 - Moderate Emissions Scenario

"Middle of the road" scenario - CO2 emissions around current levels until mid-century, then declining.

## Run order

```
python gee_to_gdrive.py        # Step 1: Export climate rasters to Google Drive
                                #         Download .tif files from Drive into this folder
python tif_to_excel.py          # Step 2: Extract province-level climate data from .tif
python merge_excels.py          # Step 3: Merge SSP climate with historical features
python future_predictions.py    # Step 4: Train Cubist, predict 2025-2034 yields
python projections.py           # Step 5: Analyze historical vs future yield changes
python shap_analysis.py         # Step 6: SHAP feature importance analysis
```

## Scripts

| Script | Input | Output |
|--------|-------|--------|
| `gee_to_gdrive.py` | GEE (NASA/GDDP-CMIP6, GFDL-ESM4, ssp245) | `gfdl_pr_2025_2034_monthly.tif`, `gfdl_tas_2025_2034_monthly.tif` on Google Drive |
| `tif_to_excel.py` | `.tif` files + `philippines_provinces.geojson` | `province_year_all_features_2025_2034.xlsx` |
| `merge_excels.py` | `banana_yield_2010-2024.xlsx` + step 2 output | `merged_future_data.csv` |
| `future_predictions.py` | `banana_yield_2010-2024.xlsx` + `merged_future_data.csv` | `banana_yield_predictions_2025-2034.xlsx` |
| `projections.py` | Historical + predicted yield data | `banana_yield_analysis.xlsx`, CSVs, plots |
| `shap_analysis.py` | Historical + predicted yield data | `shap_results/cubist/` (plots + .npy) |

## Data files

- `banana_yield_2010-2024.xlsx` - Historical yield + 17 climate features per province per year
- `philippines_provinces.geojson` - Philippine province boundaries for spatial clipping
