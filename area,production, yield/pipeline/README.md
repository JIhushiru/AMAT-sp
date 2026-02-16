# Data Processing Pipeline

This folder contains the scripts used to build the final training dataset from raw data sources. The pipeline combines two types of data: (1) banana crop statistics from the Philippine Statistics Authority, and (2) gridded climate observations from the CRU TS dataset, to produce a province-level, yearly panel dataset suitable for regression modeling.

## Data Sources

### Banana Crop Statistics (PSA/CountrySTAT)

The raw CSV files were manually downloaded from the Philippine Statistics Authority's CountrySTAT portal. These files report area harvested, production volume, and yield per province from 2010 to 2024 in a wide format, with each year as a separate column (e.g., "2010 Annual", "2011 Annual", ...).

The geographic hierarchy in the raw data is encoded using dot prefixes in the `Geolocation` column:
- No prefix or "PHILIPPINES" = national-level aggregate
- ".." prefix = region-level aggregate (e.g., "..REGION I")
- "...." prefix = province-level data (e.g., "....Pangasinan")

Only the province-level rows (those starting with "....") are used in the analysis.

### Climate Data (CRU TS 4.08)

Monthly climate observations were obtained from the Climatic Research Unit Time-Series dataset (CRU TS 4.08), maintained by the University of East Anglia. The data is distributed as NetCDF files covering 1901-2023 at 0.5-degree spatial resolution. Each climate variable is stored in a separate file named `cru_ts4.08.1901.2023.{variable}.dat.nc`.

The climate variables extracted are:
| Variable | Description |
|----------|-------------|
| tmp | Mean temperature (deg C) |
| tmx | Maximum temperature (deg C) |
| tmn | Minimum temperature (deg C) |
| dtr | Diurnal temperature range (deg C) |
| pre | Precipitation (mm/month) |
| pet | Potential evapotranspiration (mm/month) |
| cld | Cloud cover (%) |
| wet | Wet day frequency (days/month) |
| vap | Vapor pressure (hPa) |

### Province Boundaries (GeoJSON)

A GeoJSON file (`philippines_provinces.geojson`) containing polygon geometries for each Philippine province is used to spatially match CRU grid cells to provinces.

## How the Pipeline Works

### Main Script: Extract Climate Features and Merge with Yield (`main_extract_and_merge.py`)

This script does the entire pipeline end-to-end. For each climate variable, it:

1. Opens the corresponding CRU NetCDF file and subsets it to the study period (2010-2023).
2. For each province, clips the gridded climate data to the province polygon using `rioxarray`.
3. If the polygon clip returns valid data, it computes the spatial mean across all grid cells within the province for each month (the "polygon method").
4. If no valid grid cells fall within the province boundary (common for small or island provinces), the script falls back to a "centroid method" -- it searches for the nearest valid grid cell around the province centroid within a 1-degree radius.
5. Monthly values are aggregated to yearly averages per province.

After processing all variables, the script merges them into a single dataframe keyed by (province, year).

It then loads the PSA crop data CSV, filters to province-level rows, reshapes it from wide to long format, and joins it with the climate features on (province, year) to produce the final output: `province_year_climate_features_and_yield.csv`.

### Helper: Update Yield Values (`merge_yield_data.py`)

A simpler utility script that maps yield values from a wide-format PSA CSV into an existing long-format dataset. This was used when yield data needed to be updated separately from the climate features (e.g., when newer PSA data became available).

### Helper: Visualize Area Harvested (`plot_area_harvested.py`)

Generates a time-series plot of national-level banana area harvested from 2010 to 2024, including the mean line and overall trend.

## Other Files

- `centroid_fallback_log.txt` -- Log of which provinces used the centroid fallback method for each climate variable. Useful for verifying spatial coverage.

## Output

The final output of this pipeline is a CSV with the following columns:

| Column | Description |
|--------|-------------|
| province | Province name |
| year | Year (2010-2023) |
| cld, wet, vap, ... | Yearly average climate features |
| yield | Banana yield from PSA (metric tons per hectare) |

This CSV was then used as the basis for `Training/data/banana_yield_2010-2024.xlsx`, with additional climate variables (vpd, PDSI, q, soil, srad, ws) added from other sources in later processing steps.

## Requirements

- Python 3.8+
- xarray, rioxarray, geopandas, pandas, numpy, shapely, matplotlib
- CRU TS 4.08 NetCDF files (not included in this repository due to size)
- Philippines province GeoJSON boundary file
