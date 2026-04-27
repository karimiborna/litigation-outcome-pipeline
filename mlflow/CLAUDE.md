# MLflow Module

Experiment tracking and model registry for the pipeline. **The tracking server is hosted on a GCP VM.** All teammates and CI point at the same hosted server so runs and registered models are shared.

## Hosted server

```
MLFLOW_TRACKING_URI=http://35.208.251.175:5000
```

This URL is the canonical tracking endpoint for the project. It is already set in `.env.example`.

### Connect from your laptop

Either copy `.env.example` to `.env` (loaded automatically by `MLflowConfig`):

```bash
cp .env.example .env
```

…or export it for one-off shell sessions:

```bash
export MLFLOW_TRACKING_URI=http://35.208.251.175:5000
python scripts/train_classifier_real.py
python scripts/promote_models_to_production.py
```

UI: open `http://35.208.251.175:5000/` in a browser.

### If the server is unreachable

```bash
curl -m 5 http://35.208.251.175:5000/
```

If this hangs or times out, the GCE VM is stopped. Ask the team to start it back up:

```bash
gcloud compute instances start <vm-name> --zone=<zone>
```

Do **not** point your `MLFLOW_TRACKING_URI` at `localhost`. Runs and models logged to a local server are invisible to your teammates and to the API, and create a divergent registry. Get the hosted server back up before training.

## Server config (admin only)

The GCE VM runs:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts \
  --artifacts-destination gs://<artifact-bucket> \
  --cors-allowed-origins '*' \
  --host 0.0.0.0 --port 5000
```

Required flags:

- **`--serve-artifacts`** — clients upload artifacts via HTTP. Without this, experiments get an `artifact_location` only the server can write to (`file:///home/...`), and laptops fail when logging models. The helper `models/tracking.py` detects this and creates a sibling experiment with suffix `-remote-artifacts` if it sees a server-local path on a remote URI; the real fix is on the server side.
- **`--cors-allowed-origins '*'`** — MLflow 3.x UI calls `/ajax-api/...` from the browser. Without CORS, the Runs tab shows `INTERNAL_ERROR` and the server logs say `Blocked cross-origin request`.
- Static external IP attached to the VM, ingress firewall rule allowing TCP 5000.

## Conventions

- All model training must go through MLflow — no untracked experiments.
- Model registry stages: None → Staging → Production. (Stages may be replaced by aliases in a future MLflow major version; the warning during promotion is expected.)
- MLflow configuration files and server setup live here; actual model code lives in `models/`.

## End-to-end flow

```bash
cd litigation-outcome-pipeline

# (one-time) make sure your env points at the hosted server
cp .env.example .env

# train + register both API-required models from dataset.csv
python scripts/train_classifier_real.py

# promote latest registered version of each model to Production
python scripts/promote_models_to_production.py

# serve the API (loads models:/<name>/Production from the hosted registry)
uvicorn api.app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

`scripts/train_classifier_real.py` registers `litigation-win-classifier` and `litigation-monetary-regressor` from `dataset.csv` using the shared `v2 feat_*` preprocessing in `models.dataset`. It logs the dataset SHA-256 hash and the feature column list as artifacts so each run is reproducible.

> **macOS note:** use `mlflow` or `python -m mlflow` (not `python3 -m mlflow`). On macOS, `python3` resolves to `/usr/bin/python3` (system Python 3.9, outside your conda env), which can have stale or broken mlflow installs. Inside `conda activate ML`, `python` is the conda interpreter — always use that.
