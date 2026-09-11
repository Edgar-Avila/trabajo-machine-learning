# Customer Churn Risk App

This folder contains the Software Engineer scope: FastAPI inference API, browser UI, tests, and container definition.

## Run locally

From `app`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000 and see API docs at http://localhost:8000/docs. Before the ML artifact arrives, `ALLOW_FALLBACK_MODEL=true` uses a deterministic demo model. Set it to `false` in production.

## Tests

```powershell
pytest
```

Tests inject a mock model; no `.pkl` is required.

## Docker

From repository root:

```powershell
docker build -t churn-risk-app -f app/Dockerfile app
docker run --rm -p 8000:8000 -e MODEL_PATH=/opt/model/model.pkl -v "${PWD}/model:/opt/model:ro" churn-risk-app
```

Docker disables fallback by default; add `-e ALLOW_FALLBACK_MODEL=true` for demo use.

## Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `model/model.pkl` | Serialized model path. |
| `FEATURE_LIST_PATH` | unset | Optional JSON feature list. |
| `ALLOW_FALLBACK_MODEL` | `true` locally | Enables demo model when artifact is unavailable. |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `PORT` | `8000` | Container port. |

## ML integration contract

Persona 1 supplies a joblib/pickle object at `MODEL_PATH` with `predict_proba(features)`, returning binary probabilities with churn in column 1. Prefer a serialized preprocessing-and-classifier pipeline.

The app sends one pandas row, excluding `CustomerID`, with: `Age`, `Gender`, `Tenure`, `Usage Frequency`, `Support Calls`, `Payment Delay`, `Subscription Type`, `Contract Length`, `Total Spend`, and `Last Interaction`. If ML changes names/order, provide `FEATURE_LIST_PATH` as a JSON list (or `{"features": [...]}`) and update validation aliases in `api/schemas.py`.

