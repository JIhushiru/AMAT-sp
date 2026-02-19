import os
import json
import glob
import base64
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Philippine Banana Yield Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_DATA = PROJECT_ROOT / "Training" / "data" / "banana_yield_2010-2024.xlsx"
TRAINING_PLOTS = PROJECT_ROOT / "Training" / "plots"
MODELS_DIR = PROJECT_ROOT / "Training" / "models" / "top3"
MAPPING_DIR = PROJECT_ROOT / "Mapping"
MAPPING_CSV = MAPPING_DIR / "banana_yield_2010-2024.xlsx"
GEOJSON_PATH = MAPPING_DIR / "philippines_provinces.geojson"
SSP245_DIR = PROJECT_ROOT / "Pipeline" / "SSP2-4.5"
SSP585_DIR = PROJECT_ROOT / "Pipeline" / "SSP5-8.5"

# --- Model Cache ---
_model_cache = {}


def _load_models():
    """Scan Models/top3/ and load all .joblib artifacts into cache."""
    _model_cache.clear()
    if not MODELS_DIR.exists():
        return
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            artifact = joblib.load(path)
            key = path.stem  # e.g. "rank1_XGB"
            _model_cache[key] = artifact
        except Exception:
            pass


def _get_models():
    """Return cached models, reloading if cache is empty."""
    if not _model_cache:
        _load_models()
    return _model_cache


def _sanitize(obj):
    """Recursively replace NaN/inf with None for JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def safe_read_excel(path):
    """Read Excel, clean yield column."""
    df = pd.read_excel(path)
    if "yield" in df.columns:
        df["yield"] = df["yield"].replace("#DIV/0!", np.nan)
        df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
        df = df.dropna(subset=["yield"])
    return df


def safe_read_csv(path):
    df = pd.read_csv(path)
    return df


# ============================================================
# HISTORICAL DATA ENDPOINTS
# ============================================================

@app.get("/api/historical/data")
def get_historical_data():
    """Return all historical yield + climate data (2010-2024)."""
    if not TRAINING_DATA.exists():
        raise HTTPException(404, "Historical data file not found")
    df = safe_read_excel(TRAINING_DATA)
    return df.to_dict(orient="records")


@app.get("/api/historical/summary")
def get_historical_summary():
    """National-level summary statistics."""
    df = safe_read_excel(TRAINING_DATA)
    national_trend = df.groupby("year")["yield"].mean().round(2)
    province_avg = df.groupby("province")["yield"].mean().round(2).sort_values(ascending=False)
    return {
        "national_avg": round(df["yield"].mean(), 2),
        "national_trend": {int(k): v for k, v in national_trend.items()},
        "province_avg": {k: v for k, v in province_avg.items()},
        "total_provinces": df["province"].nunique(),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "total_records": len(df),
    }


@app.get("/api/historical/province/{province_name}")
def get_province_data(province_name: str):
    """Get historical data for a specific province."""
    df = safe_read_excel(TRAINING_DATA)
    province_df = df[df["province"].str.lower() == province_name.lower()]
    if province_df.empty:
        raise HTTPException(404, f"Province '{province_name}' not found")
    return {
        "province": province_name,
        "data": province_df.to_dict(orient="records"),
        "avg_yield": round(province_df["yield"].mean(), 2),
        "trend": {
            int(r["year"]): round(r["yield"], 2)
            for _, r in province_df.iterrows()
        },
    }


@app.get("/api/historical/provinces")
def list_provinces():
    """List all province names."""
    df = safe_read_excel(TRAINING_DATA)
    provinces = sorted(df["province"].unique().tolist())
    return provinces


@app.get("/api/historical/climate-features")
def get_climate_features():
    """Return the 17 climate feature names and their stats."""
    df = safe_read_excel(TRAINING_DATA)
    features = [c for c in df.columns if c not in ("province", "year", "yield")]
    stats = {}
    for f in features:
        stats[f] = {
            "mean": round(df[f].mean(), 4),
            "min": round(df[f].min(), 4),
            "max": round(df[f].max(), 4),
            "std": round(df[f].std(), 4),
        }
    return {"features": features, "stats": stats}


@app.get("/api/historical/correlation")
def get_correlation_matrix():
    """Return correlation matrix of yield + climate features."""
    df = safe_read_excel(TRAINING_DATA)
    numeric_cols = [c for c in df.columns if c not in ("province", "year")]
    corr = df[numeric_cols].corr().round(4)
    return {
        "columns": corr.columns.tolist(),
        "data": corr.values.tolist(),
    }


# ============================================================
# MAPPING ENDPOINTS
# ============================================================

@app.get("/api/map/geojson")
def get_geojson():
    """Return Philippines provinces GeoJSON."""
    if not GEOJSON_PATH.exists():
        raise HTTPException(404, "GeoJSON file not found")
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/map/yield-by-province")
def get_yield_by_province():
    """Return average yield per province for choropleth mapping."""
    if not MAPPING_CSV.exists():
        raise HTTPException(404, "Mapping data file not found")
    import re
    df = pd.read_excel(MAPPING_CSV)
    df.rename(columns={"Geolocation": "province"}, inplace=True)
    df["province"] = df["province"].apply(
        lambda x: re.sub(r"^\.*\s*", "", str(x))
    ).str.strip().str.title()
    # Fix title-case particles to match GeoJSON naming (e.g. "Del" -> "del")
    for particle in [" Del ", " De ", " Of "]:
        df["province"] = df["province"].str.replace(
            particle, particle.lower(), regex=False
        )

    province_mapping = {
        "North Cotabato": "Cotabato",
        "Compostela Valley": "Davao de Oro",
        "Maguindanao del Norte": "Maguindanao",
        "Maguindanao del Sur": "Maguindanao",
        "City of Davao": "Davao del Sur",
        "City of Zamboanga": "Zamboanga del Sur",
        "Metropolitan Manila": "Ncr",
        "Cotabato": "North Cotabato",
        "Davao Occidental": "Davao del Sur",
    }
    df["province"] = df["province"].replace(province_mapping)

    region_keywords = [
        "Region", "Autonomous Region", "Caraga", "Calabarzon",
        "Mimaropa", "Barmm", "Cordillera",
    ]
    df = df[~df["province"].str.contains("|".join(region_keywords), case=False, na=False)]
    df = df[df["province"] != "Philippines"]

    year_cols = [c for c in df.columns if "Annual" in c]
    df[year_cols] = df[year_cols].apply(pd.to_numeric, errors="coerce")

    result = {}
    for _, row in df.iterrows():
        prov = row["province"]
        yearly = {}
        for c in year_cols:
            year = c.replace(" Annual", "")
            val = row[c]
            if pd.notna(val):
                yearly[year] = round(float(val), 2)
        avg = round(float(pd.Series(list(yearly.values())).mean()), 2) if yearly else 0
        if prov in result:
            old_avg = result[prov]["average"]
            result[prov]["average"] = round((old_avg + avg) / 2, 2)
        else:
            result[prov] = {"yearly": yearly, "average": avg}
    return result


@app.get("/api/map/images")
def list_map_images():
    """List available map images."""
    images = []
    for name, label in [
        ("banana_yield_map_gadm.png", "Philippines (Full)"),
        ("banana_yield_map_luzon.png", "Luzon"),
        ("banana_yield_map_visayas.png", "Visayas"),
        ("banana_yield_map_mindanao.png", "Mindanao"),
    ]:
        path = MAPPING_DIR / name
        if path.exists():
            images.append({"name": name, "label": label})
    return images


@app.get("/api/map/image/{filename}")
def get_map_image(filename: str):
    """Serve a map image file."""
    path = MAPPING_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(path, media_type="image/png")


# ============================================================
# TRAINING / MODEL RESULTS ENDPOINTS
# ============================================================

@app.get("/api/training/plots")
def list_training_plots():
    """List available training result plots."""
    if not TRAINING_PLOTS.exists():
        return []
    plots = []
    for f in sorted(TRAINING_PLOTS.glob("*.png")):
        plots.append({"name": f.name, "label": f.stem.replace("_", " ").title()})
    return plots


@app.get("/api/training/plot/{filename}")
def get_training_plot(filename: str):
    """Serve a training plot image."""
    path = TRAINING_PLOTS / filename
    if not path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(path, media_type="image/png")


# ============================================================
# SSP SCENARIO ENDPOINTS
# ============================================================

def load_ssp_data(ssp_dir: Path):
    """Load all available SSP analysis data from a directory."""
    result = {"available": False}

    predictions_path = ssp_dir / "banana_yield_predictions_2025-2034.xlsx"
    province_summary_path = ssp_dir / "province_summary.csv"
    compare_path = ssp_dir / "compare_2024_2034.csv"
    national_trend_path = ssp_dir / "national_trend.csv"
    historical_path = ssp_dir / "banana_yield_2010-2024.xlsx"

    if predictions_path.exists():
        result["available"] = True
        pred_df = safe_read_excel(predictions_path)
        result["predictions"] = pred_df.to_dict(orient="records")
        result["future_national_trend"] = (
            pred_df.groupby("year")["yield"].mean().round(2).to_dict()
        )
        result["future_province_avg"] = (
            pred_df.groupby("province")["yield"].mean().round(2).to_dict()
        )

    if province_summary_path.exists():
        ps = pd.read_csv(province_summary_path, index_col=0)
        result["province_summary"] = ps.to_dict(orient="index")

    if compare_path.exists():
        cp = pd.read_csv(compare_path, index_col=0)
        result["compare_2024_2034"] = cp.to_dict(orient="index")

    if national_trend_path.exists():
        nt = pd.read_csv(national_trend_path, index_col=0)
        result["national_trend"] = {
            int(k): round(float(v), 2)
            for k, v in nt.iloc[:, 0].items()
        }

    if historical_path.exists():
        hist_df = safe_read_excel(historical_path)
        hist_trend = hist_df.groupby("year")["yield"].mean().round(2)
        result["historical_national_trend"] = {
            int(k): v for k, v in hist_trend.items()
        }

    # Check for SHAP images
    shap_dir = ssp_dir / "shap_results" / "cubist"
    if shap_dir.exists():
        shap_images = list(shap_dir.glob("*.png"))
        result["shap_images"] = [img.name for img in shap_images]

    # Check for projection plots
    for plot_name in [
        "national_yield_trend.png",
        "province_percent_change.png",
        "yield_2024_vs_2034.png",
        "observed_vs_predicted_yield.png",
    ]:
        if (ssp_dir / plot_name).exists():
            result.setdefault("plots", []).append(plot_name)

    return result


@app.get("/api/ssp/{scenario}")
def get_ssp_data(scenario: str):
    """Get SSP scenario data. scenario = 'ssp245' or 'ssp585'."""
    dirs = {"ssp245": SSP245_DIR, "ssp585": SSP585_DIR}
    if scenario not in dirs:
        raise HTTPException(400, "Invalid scenario. Use 'ssp245' or 'ssp585'.")
    return _sanitize(load_ssp_data(dirs[scenario]))


@app.get("/api/ssp/{scenario}/plot/{filename}")
def get_ssp_plot(scenario: str, filename: str):
    """Serve an SSP plot image."""
    dirs = {"ssp245": SSP245_DIR, "ssp585": SSP585_DIR}
    if scenario not in dirs:
        raise HTTPException(400, "Invalid scenario")
    path = dirs[scenario] / filename
    if not path.exists():
        # Check shap subdirectory
        path = dirs[scenario] / "shap_results" / "cubist" / filename
    if not path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(path, media_type="image/png")


# ============================================================
# OVERVIEW / DASHBOARD ENDPOINT
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():
    """Aggregate dashboard data for the frontend."""
    df = safe_read_excel(TRAINING_DATA)

    national_trend = df.groupby("year")["yield"].mean().round(2)
    province_avg = df.groupby("province")["yield"].mean().round(2).sort_values(ascending=False)
    top5 = province_avg.head(5)
    bottom5 = province_avg.tail(5)

    ssp245 = load_ssp_data(SSP245_DIR)
    ssp585 = load_ssp_data(SSP585_DIR)

    return {
        "historical": {
            "national_avg": round(df["yield"].mean(), 2),
            "national_trend": {int(k): v for k, v in national_trend.items()},
            "top_provinces": {k: v for k, v in top5.items()},
            "bottom_provinces": {k: v for k, v in bottom5.items()},
            "total_provinces": df["province"].nunique(),
            "year_range": [int(df["year"].min()), int(df["year"].max())],
        },
        "ssp245_available": ssp245.get("available", False),
        "ssp585_available": ssp585.get("available", False),
        "ssp245_national_trend": ssp245.get("future_national_trend"),
        "ssp585_national_trend": ssp585.get("future_national_trend"),
        "training_plots_count": len(list(TRAINING_PLOTS.glob("*.png"))) if TRAINING_PLOTS.exists() else 0,
        "map_images_count": len(list(MAPPING_DIR.glob("*.png"))),
    }


# ============================================================
# PREDICTION ENDPOINTS
# ============================================================

class PredictRequest(BaseModel):
    features: dict[str, float]
    model_key: Optional[str] = None  # e.g. "rank1_XGB"; None = best


class BatchPredictRequest(BaseModel):
    province: Optional[str] = None
    year: Optional[int] = None
    scenario: Optional[str] = None  # "ssp245" or "ssp585"
    model_key: Optional[str] = None


@app.get("/api/predict/models")
def list_available_models():
    """List all trained models available for prediction."""
    models = _get_models()
    if not models:
        return {"available": False, "models": [], "message": "No trained models found. Run training first."}
    result = []
    for key, artifact in models.items():
        result.append({
            "key": key,
            "name": key.split("_", 1)[1] if "_" in key else key,
            "rank": artifact.get("rank", 0),
            "cv_r2": round(artifact.get("cv_avg_r2", 0), 4),
            "features": artifact.get("selected_features", []),
            "params": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                       for k, v in artifact.get("best_params", {}).items()},
        })
    result.sort(key=lambda x: x["rank"])
    return {"available": True, "models": result}


@app.get("/api/predict/features")
def get_prediction_features():
    """Return the features needed for prediction, with stats for defaults."""
    models = _get_models()
    if not models:
        # Fall back to all climate features from historical data
        df = safe_read_excel(TRAINING_DATA)
        features = [c for c in df.columns if c not in ("province", "year", "yield")]
    else:
        # Use features from the best model
        best = min(models.values(), key=lambda a: a.get("rank", 99))
        features = best.get("selected_features", [])

    df = safe_read_excel(TRAINING_DATA)
    stats = {}
    for f in features:
        if f in df.columns:
            stats[f] = {
                "mean": round(float(df[f].mean()), 4),
                "min": round(float(df[f].min()), 4),
                "max": round(float(df[f].max()), 4),
                "std": round(float(df[f].std()), 4),
                "q25": round(float(df[f].quantile(0.25)), 4),
                "q75": round(float(df[f].quantile(0.75)), 4),
            }

    # Province-level averages for the "load from province" feature
    province_means = {}
    for prov, grp in df.groupby("province"):
        province_means[prov] = {
            f: round(float(grp[f].mean()), 4) for f in features if f in grp.columns
        }

    return {
        "features": features,
        "stats": stats,
        "province_defaults": province_means,
    }


@app.post("/api/predict")
def predict_yield(req: PredictRequest):
    """Predict banana yield from climate features using a trained model."""
    models = _get_models()
    if not models:
        raise HTTPException(503, "No trained models available. Training may still be in progress.")

    # Select model
    if req.model_key and req.model_key in models:
        artifact = models[req.model_key]
    else:
        artifact = min(models.values(), key=lambda a: a.get("rank", 99))

    model = artifact["model"]
    scaler = artifact["scaler"]
    expected_features = artifact["selected_features"]

    # Validate input features
    missing = [f for f in expected_features if f not in req.features]
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")

    # Build input array in the correct feature order
    X = pd.DataFrame([{f: req.features[f] for f in expected_features}])
    X_scaled = pd.DataFrame(scaler.transform(X), columns=expected_features)

    prediction = float(model.predict(X_scaled)[0])

    # Get historical context
    df = safe_read_excel(TRAINING_DATA)
    national_avg = round(float(df["yield"].mean()), 2)
    national_min = round(float(df["yield"].min()), 2)
    national_max = round(float(df["yield"].max()), 2)

    # Compute RMSE on training data for confidence interval
    try:
        train_X = df[expected_features].copy()
        train_X_scaled = pd.DataFrame(scaler.transform(train_X), columns=expected_features)
        train_preds = model.predict(train_X_scaled)
        residuals = df["yield"].values - train_preds
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
    except Exception:
        rmse = 0.0

    return _sanitize({
        "predicted_yield": round(prediction, 4),
        "model_used": artifact.get("rank", 0),
        "model_name": req.model_key or f"rank{artifact.get('rank', '?')}",
        "cv_r2": round(artifact.get("cv_avg_r2", 0), 4),
        "rmse": round(rmse, 4),
        "confidence_interval": {
            "lower": round(max(0, prediction - 1.96 * rmse), 4),
            "upper": round(prediction + 1.96 * rmse, 4),
            "level": "95%",
        },
        "context": {
            "national_avg": national_avg,
            "national_min": national_min,
            "national_max": national_max,
        },
    })


@app.post("/api/predict/batch")
def batch_predict(req: BatchPredictRequest):
    """Batch predict: all provinces for a scenario+year, or all years for a province."""
    models = _get_models()
    if not models:
        raise HTTPException(503, "No trained models available.")

    if req.model_key and req.model_key in models:
        artifact = models[req.model_key]
    else:
        artifact = min(models.values(), key=lambda a: a.get("rank", 99))

    model = artifact["model"]
    scaler = artifact["scaler"]
    expected_features = artifact["selected_features"]

    # Load SSP data if scenario specified
    if req.scenario:
        dirs = {"ssp245": SSP245_DIR, "ssp585": SSP585_DIR}
        if req.scenario not in dirs:
            raise HTTPException(400, "Invalid scenario. Use 'ssp245' or 'ssp585'.")
        ssp_dir = dirs[req.scenario]
        pred_path = ssp_dir / "banana_yield_predictions_2025-2034.xlsx"
        merged_path = ssp_dir / "merged_future_data.csv"

        # Try to load the merged future climate data
        if merged_path.exists():
            future_df = pd.read_csv(merged_path)
        elif pred_path.exists():
            future_df = safe_read_excel(pred_path)
        else:
            raise HTTPException(404, f"No SSP data found for {req.scenario}")

        # Normalize column names
        future_df.columns = [c.strip().lower() for c in future_df.columns]

        # Filter by province or year if specified
        if req.province:
            future_df = future_df[future_df["province"].str.lower() == req.province.lower()]
        if req.year:
            future_df = future_df[future_df["year"] == req.year]

        if future_df.empty:
            raise HTTPException(404, "No data matches the given filters.")

        # Check which expected features are available
        available = [f for f in expected_features if f in future_df.columns]
        if len(available) < len(expected_features):
            missing = [f for f in expected_features if f not in future_df.columns]
            raise HTTPException(400, f"SSP data missing features: {missing}")

        X = future_df[expected_features].copy()
        X_scaled = pd.DataFrame(scaler.transform(X), columns=expected_features)
        preds = model.predict(X_scaled)

        results = []
        for i, (_, row) in enumerate(future_df.iterrows()):
            results.append({
                "province": row.get("province", "Unknown"),
                "year": int(row.get("year", 0)),
                "predicted_yield": round(float(preds[i]), 4),
            })

        return _sanitize({
            "predictions": results,
            "model_name": f"rank{artifact.get('rank', '?')}",
            "scenario": req.scenario,
            "count": len(results),
        })

    # If no scenario, predict using historical averages per province
    df = safe_read_excel(TRAINING_DATA)
    if req.province:
        df = df[df["province"].str.lower() == req.province.lower()]
    if df.empty:
        raise HTTPException(404, "No data matches the given filters.")

    X = df[expected_features].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=expected_features)
    preds = model.predict(X_scaled)

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        results.append({
            "province": row["province"],
            "year": int(row["year"]),
            "actual_yield": round(float(row["yield"]), 4),
            "predicted_yield": round(float(preds[i]), 4),
        })

    return _sanitize({
        "predictions": results,
        "model_name": f"rank{artifact.get('rank', '?')}",
        "count": len(results),
    })


@app.post("/api/predict/reload-models")
def reload_models():
    """Force reload models from disk (call after training completes)."""
    _load_models()
    count = len(_model_cache)
    if count == 0:
        return {"status": "no_models", "message": "No models found in Models/top3/"}
    names = [k.split("_", 1)[1] if "_" in k else k for k in _model_cache]
    return {"status": "ok", "loaded": count, "models": names}


@app.get("/api/health")
def health():
    return {"status": "ok"}
