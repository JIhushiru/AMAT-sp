# Mapping

Choropleth maps of average banana yield (2010-2024) per Philippine province.

## Scripts

| Script | Description |
|--------|-------------|
| `map.py` | Full Philippines map |
| `luzviminda.py` | Separate maps for Luzon, Visayas, and Mindanao |

## How to run

```
python map.py          # Generates full Philippines map
python luzviminda.py   # Generates 3 regional maps
```

## Data

- `banana_yield_2010-2024.csv` - Yield data per province (from PSA/CountrySTAT)
- `philippines_provinces.geojson` - Province boundaries
- `gadm_data/` - GADM shapefiles (auto-downloaded on first run from geodata.ucdavis.edu)

## Output

- `banana_yield_map_gadm.png` - Full Philippines choropleth (600 DPI)
- `banana_yield_map_luzon.png` - Luzon region
- `banana_yield_map_visayas.png` - Visayas region
- `banana_yield_map_mindanao.png` - Mindanao region

## Color scale

| Range (tons/ha) | Color |
|-----------------|-------|
| No data | Gray (#D3D3D3) |
| 0-15 | Yellow (#fde725) |
| 15-30 | Green (#5ec962) |
| 30-46 | Teal (#21918c) |
| 46-61 | Blue (#3b528b) |
| 61-100 | Purple (#440154) |
