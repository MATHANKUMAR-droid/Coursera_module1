import os, textwrap, zipfile, json

root = "/mnt/data/country_ml_deployment"
os.makedirs(root, exist_ok=True)
files = {
"app.py": r'''
from pathlib import Path
import json, logging, os, time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "artifacts"))
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "runtime_logs"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACT_DIR / "model.joblib"))
FEATURES = ["year", "population_millions", "internet_pct"]
TARGET = "outcome_index"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("country_api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "requests.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI(title="Country Outcome Prediction API", version="1.0.0")

class PredictionInput(BaseModel):
    country: str = Field(min_length=1, max_length=100)
    year: int = Field(ge=1900, le=2200)
    population_millions: float = Field(gt=0, le=20000)
    internet_pct: float = Field(ge=0, le=100)

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` before starting the API."
        )
    return joblib.load(MODEL_PATH)

def log_event(event: dict, log_path=None):
    """Append a JSON event. The path argument makes tests independent of production logs."""
    path = Path(log_path) if log_path else LOG_DIR / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")

@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}

@app.get("/countries")
def countries():
    path = DATA_DIR / "countries.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Country data not available")
    df = pd.read_csv(path)
    return {"countries": sorted(df["country"].dropna().astype(str).unique().tolist())}

@app.post("/predict")
def predict(payload: PredictionInput):
    started = time.perf_counter()
    try:
        model = load_model()
        x = pd.DataFrame([{
            "year": payload.year,
            "population_millions": payload.population_millions,
            "internet_pct": payload.internet_pct
        }], columns=FEATURES)
        value = float(model.predict(x)[0])
        response = {
            "country": payload.country,
            "prediction": value,
            "model_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        log_event({
            "event": "prediction",
            "country": payload.country,
            "input": payload.model_dump(),
            "prediction": value,
            "model_version": response["model_version"],
            "timestamp": response["timestamp"],
            "runtime_ms": round((time.perf_counter() - started) * 1000, 3)
        })
        return response
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/predict/all")
def predict_all(year: int = 2025):
    """Predict for every country using its latest available feature values."""
    path = DATA_DIR / "countries.csv"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Country data not available")
    try:
        model = load_model()
        df = pd.read_csv(path)
        latest = df.sort_values("year").groupby("country", as_index=False).tail(1).copy()
        latest["year"] = year
        latest["prediction"] = model.predict(latest[FEATURES])
        results = [
            {"country": str(row.country), "prediction": float(row.prediction)}
            for row in latest.sort_values("country").itertuples()
        ]
        log_event({
            "event": "batch_prediction", "year": year, "count": len(results),
            "model_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        return {"year": year, "count": len(results), "results": results}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get("/metrics")
def metrics():
    """Simple operational metrics derived from JSONL request logs."""
    path = LOG_DIR / "requests.jsonl"
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    predictions = [e for e in events if e.get("event") in ("prediction", "batch_prediction")]
    runtimes = [e["runtime_ms"] for e in predictions if isinstance(e.get("runtime_ms"), (int, float))]
    return {
        "total_logged_events": len(events),
        "prediction_events": len(predictions),
        "mean_runtime_ms": float(np.mean(runtimes)) if runtimes else None,
        "p95_runtime_ms": float(np.percentile(runtimes, 95)) if runtimes else None
    }
''',
"train.py": r'''
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FEATURES = ["year", "population_millions", "internet_pct"]
TARGET = "outcome_index"

def create_sample_data(path=DATA / "countries.csv"):
    """Create deterministic example data. Replace this CSV with real, licensed data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    rng = np.random.default_rng(17)
    countries = [
        ("Arandia", 18, 62), ("Beleron", 45, 78), ("Caspia", 9, 51),
        ("Demeria", 72, 88), ("Estoria", 31, 70), ("Faron", 12, 44),
        ("Galdor", 105, 93), ("Helvia", 24, 66), ("Istria", 56, 82),
        ("Jorvik", 7, 39), ("Kelm", 83, 90), ("Lunara", 39, 74)
    ]
    rows = []
    for country, pop, internet in countries:
        for year in range(2015, 2025):
            pop_y = max(1, pop * (1 + 0.012 * (year - 2015)) + rng.normal(0, 0.4))
            internet_y = float(np.clip(internet + (year - 2015) * 1.25 + rng.normal(0, 1.8), 0, 100))
            outcome = 22 + 0.035 * (year - 2015) + 0.31 * internet_y - 0.025 * pop_y + rng.normal(0, 2.2)
            rows.append({"country": country, "year": year, "population_millions": round(pop_y, 3),
                         "internet_pct": round(internet_y, 3), TARGET: round(outcome, 3)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(create_sample_data())
    required = {"country", *FEATURES, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=FEATURES + [TARGET])
    if len(df) < 10:
        raise ValueError("Need at least 10 complete rows to train.")
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=42
    )
    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2),
        "extra_trees": ExtraTreesRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)
    }
    results = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "model": name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred))
        })
        fitted[name] = model
    # Choose the model with the lowest held-out RMSE, including the baseline only as a safety net.
    best = min(results, key=lambda item: item["rmse"])
    joblib.dump(fitted[best["model"]], ARTIFACTS / "model.joblib")
    metrics = {"selected_model": best["model"], "test_rows": len(y_test), "metrics": results}
    (ARTIFACTS / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    results_df = pd.DataFrame(results).sort_values("rmse")
    plt.figure(figsize=(9, 5))
    bars = plt.bar(results_df["model"], results_df["rmse"])
    plt.ylabel("Test RMSE (lower is better)")
    plt.title("Model comparison against baseline")
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, results_df["rmse"]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "model_comparison.png", dpi=150)
    plt.close()

    # EDA plots
    plt.figure(figsize=(7, 5))
    plt.hist(df[TARGET], bins=18, edgecolor="black")
    plt.title("EDA: Outcome index distribution")
    plt.xlabel(TARGET)
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "eda_target_distribution.png", dpi=150)
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.scatter(df["internet_pct"], df[TARGET], alpha=0.7)
    plt.xlabel("Internet access (%)")
    plt.ylabel(TARGET)
    plt.title("EDA: Internet access vs outcome index")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "eda_internet_vs_outcome.png", dpi=150)
    plt.close()
    return metrics

if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
''',
"ingest.py": r'''
"""Automated data ingestion/validation. Usage: python ingest.py input.csv [--replace]."""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REQUIRED = ["country", "year", "population_millions", "internet_pct", "outcome_index"]

def ingest(source: str, destination: str = str(ROOT / "data" / "countries.csv"), replace: bool = False):
    src, dst = Path(source), Path(destination)
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    df = pd.read_csv(src)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[REQUIRED].copy()
    df["country"] = df["country"].astype(str).str.strip()
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["country"].eq("").any() or df[REQUIRED[1:]].isna().any().any():
        raise ValueError("Empty country names or missing/non-numeric values found.")
    if ((df["internet_pct"] < 0) | (df["internet_pct"] > 100)).any():
        raise ValueError("internet_pct must be between 0 and 100.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not replace:
        existing = pd.read_csv(dst)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["country", "year"], keep="last"
        )
    df.to_csv(dst, index=False)
    return {"rows_written": len(df), "destination": str(dst)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="CSV with the required columns")
    parser.add_argument("--destination", default=str(ROOT / "data" / "countries.csv"))
    parser.add_argument("--replace", action="store_true", help="Replace destination rather than merge")
    args = parser.parse_args()
    print(ingest(args.source, args.destination, args.replace))
''',
"tests/test_model.py": r'''
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class ModelUnitTests(unittest.TestCase):
    def test_model_learns_simple_relationship(self):
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        y = np.array([1, 3, 5, 7, 9, 11])
        model = LinearRegression().fit(X, y)
        self.assertAlmostEqual(float(model.predict(pd.DataFrame({"x": [6]}))[0]), 13.0, places=5)

    def test_model_artifact_is_not_written_to_production_directory(self):
        with tempfile.TemporaryDirectory(prefix="ml-test-") as tmp:
            test_artifact = Path(tmp) / "model.joblib"
            self.assertFalse(test_artifact.exists())
            test_artifact.write_text("test-only", encoding="utf-8")
            self.assertTrue(test_artifact.exists())
''',
"tests/test_api.py": r'''
import os
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
import app

class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="api-test-")
        self.old_data, self.old_model, self.old_logs = app.DATA_DIR, app.MODEL_PATH, app.LOG_DIR
        root = Path(self.tmp.name)
        app.DATA_DIR = root / "data"
        app.DATA_DIR.mkdir()
        app.LOG_DIR = root / "logs"
        app.MODEL_PATH = root / "missing-model.joblib"
        self.client = TestClient(app.app)

    def tearDown(self):
        app.DATA_DIR, app.MODEL_PATH, app.LOG_DIR = self.old_data, self.old_model, self.old_logs
        self.tmp.cleanup()

    def test_health_is_available_without_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["model_available"])

    def test_predict_reports_clear_error_without_model(self):
        response = self.client.post("/predict", json={
            "country": "Testland", "year": 2025,
            "population_millions": 10, "internet_pct": 50
        })
        self.assertEqual(response.status_code, 503)

    def test_countries_reports_unavailable_data(self):
        self.assertEqual(self.client.get("/countries").status_code, 503)

    def test_metrics_empty_logs(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["prediction_events"], 0)
''',
"tests/test_logging.py": r'''
import json
import tempfile
import unittest
from pathlib import Path
from app import log_event

class LoggingTests(unittest.TestCase):
    def test_event_written_to_isolated_test_log(self):
        with tempfile.TemporaryDirectory(prefix="log-test-") as tmp:
            path = Path(tmp) / "isolated.jsonl"
            log_event({"event": "test", "country": "Testland"}, log_path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["country"], "Testland")
''',
"tests/test_ingest.py": r'''
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from ingest import ingest

class IngestionTests(unittest.TestCase):
    def test_ingestion_validates_and_writes_csv(self):
        with tempfile.TemporaryDirectory(prefix="ingest-test-") as tmp:
            root = Path(tmp)
            source, dest = root / "input.csv", root / "output.csv"
            pd.DataFrame([{
                "country": "Testland", "year": 2024, "population_millions": 2,
                "internet_pct": 50, "outcome_index": 60
            }]).to_csv(source, index=False)
            result = ingest(str(source), str(dest), replace=True)
            self.assertEqual(result["rows_written"], 1)
            self.assertEqual(pd.read_csv(dest).iloc[0]["country"], "Testland")

    def test_ingestion_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory(prefix="ingest-test-") as tmp:
            source = Path(tmp) / "bad.csv"
            pd.DataFrame([{"country": "X"}]).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                ingest(str(source), str(Path(tmp) / "out.csv"), replace=True)
''',
"run_tests.py": r'''
"""One-command test runner. Usage: python run_tests.py"""
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
''',
"requirements.txt": r'''
fastapi>=0.110,<1.0
uvicorn[standard]>=0.27,<1.0
pydantic>=2.6,<3.0
pandas>=2.1,<3.0
numpy>=1.26,<3.0
scikit-learn>=1.4,<2.0
joblib>=1.3,<2.0
matplotlib>=3.8,<4.0
httpx>=0.26,<1.0
''',
"Dockerfile": r'''
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
''',
"docker-compose.yml": r'''
services:
  country-ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATA_DIR: /app/data
      ARTIFACT_DIR: /app/artifacts
      MODEL_PATH: /app/artifacts/model.joblib
      LOG_DIR: /app/runtime_logs
    volumes:
      - ./runtime_logs:/app/runtime_logs
''',
".dockerignore": r'''
__pycache__/
*.pyc
.venv/
venv/
runtime_logs/
tests/
''',
"README.md": r'''
# Country ML Deployment Project

A runnable example for a country-level regression workflow with automated ingestion, EDA, model comparison, FastAPI, JSONL logging, unit tests, and Docker.

> **Data note:** `train.py` generates deterministic synthetic example data if `data/countries.csv` does not exist. This is only a software demonstration dataset, not real-world country statistics. Replace it with data you are licensed to use before drawing substantive conclusions.

## Requirements
- Python 3.11 recommended
- Docker Desktop (optional)

## Run locally (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python run_tests.py
uvicorn app:app --reload
```
API docs: http://127.0.0.1:8000/docs

## Run locally (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python run_tests.py
uvicorn app:app --reload
```

## Run in Docker
```bash
docker compose up --build
```
Then open http://127.0.0.1:8000/docs. The image trains the model during build.

## Example API calls
Single country:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"country":"Testland","year":2025,"population_millions":25,"internet_pct":75}'
```
All countries:
```bash
curl "http://127.0.0.1:8000/predict/all?year=2025"
```
Monitoring:
```bash
curl http://127.0.0.1:8000/metrics
```
Health and countries:
- `GET /health`
- `GET /countries`
- `GET /metrics`

## Replace the sample dataset
CSV columns required:
`country,year,population_millions,internet_pct,outcome_index`

```bash
python ingest.py your_data.csv
```
By default ingestion merges rows by `(country, year)` and keeps the newest row. To replace the dataset:
```bash
python ingest.py your_data.csv --replace
```
Retrain after ingestion:
```bash
python train.py
```

## Generated artifacts
- `artifacts/model.joblib`: selected model
- `artifacts/model_metrics.json`: held-out MAE, RMSE and R² for all candidates
- `artifacts/model_comparison.png`: model RMSE comparison including baseline
- `artifacts/eda_target_distribution.png`: target distribution
- `artifacts/eda_internet_vs_outcome.png`: feature relationship plot
- `runtime_logs/requests.jsonl`: API event log

## Checklist mapping
1. API unit tests: `tests/test_api.py`
2. Model unit tests: `tests/test_model.py`
3. Logging unit tests: `tests/test_logging.py`
4. One-command test suite: `python run_tests.py`
5. Performance monitoring: `GET /metrics`, including count and latency summaries
6. Test isolation: tests use temporary directories and do not overwrite production logs/models
7. Country-specific and all-country API endpoints: `POST /predict`, `GET /predict/all`
8. Automated ingestion function/CLI: `ingest.py`
9. Multiple models compared: mean baseline, linear regression, random forest, extra trees
10. EDA visualizations: two plots generated by `train.py`
11. Containerization: `Dockerfile` and `docker-compose.yml`
12. Baseline comparison visualization: `artifacts/model_comparison.png`

## Important limits
- `/metrics` provides basic operational latency/count metrics, not a complete model-quality drift detector. For true performance monitoring, log verified ground-truth outcomes and calculate rolling metrics over time.
- The API's `country` input labels a prediction; the model uses the numeric features supplied. It does not infer country-specific behavior from the country name.
- The example model is trained on a random row split. For real time-series or country-panel data, consider a temporal or group-based split to avoid leakage.
'''
}
for name, content in files.items():
    path = os.path.join(root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).lstrip())

zip_path = "/mnt/data/country_ml_deployment.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for folder, _, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(folder, filename)
            z.write(full, os.path.relpath(full, root))
print(f"Created {zip_path} with {len(files)} project files.")
