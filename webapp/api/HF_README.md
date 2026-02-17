---
title: Philippine Banana Yield Prediction API
emoji: 🍌
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Philippine Banana Yield Prediction API

FastAPI backend for predicting banana yields in Philippine provinces using trained ML models and climate data.

## Endpoints

- `GET /api/health` - Health check
- `GET /api/predict/models` - List available trained models
- `GET /api/predict/features` - Get required features and stats
- `POST /api/predict` - Single prediction from climate features
- `POST /api/predict/batch` - Batch prediction using SSP scenarios
- `GET /api/historical/*` - Historical data endpoints
- `GET /api/ssp/*` - SSP scenario data
