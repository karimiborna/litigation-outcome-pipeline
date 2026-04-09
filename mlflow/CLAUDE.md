# MLflow Module

Experiment tracking and model registry for the pipeline.

## Server

```bash
mlflow server --config mlflow/server_config.yaml
# UI at http://localhost:5000
```

Config (`server_config.yaml`):
- Backend store: SQLite (`mlruns/mlflow.db`)
- Artifact root: `mlruns/artifacts`
- Host: `0.0.0.0:5000`

- All model training must go through MLflow — no untracked experiments
- Artifact storage should point to cloud storage (S3/GCS) in production, or use **artifact proxy** on the tracking server (see below)
- Model registry stages: None → Staging → Production (stages may be replaced by aliases in a future MLflow major version)
- Tracking URI and artifact location are environment-dependent (local dev vs cloud)
- MLflow configuration files and server setup live here; actual model code lives in `models/`

## Remote tracking server (VM / cloud)

- Clients (laptops, CI) that **log models** must not get experiment `artifact_location` that resolves to **`file:///home/...`** on the server only. Run the server with **`--serve-artifacts`** and **`--artifacts-destination <path-on-server>`** so artifacts upload over HTTP.
- **MLflow 3.x UI:** browser calls to **`/ajax-api/...`** are CORS-checked. If the Runs tab shows **INTERNAL_ERROR** and logs show **Blocked cross-origin request**, add **`--cors-allowed-origins`** (e.g. `http://YOUR_PUBLIC_IP:5000` or `*` for insecure demos) when starting `mlflow server`.
- **GCP:** reserve **static external IP**, attach to the VM, open **ingress TCP 5000** (or use SSH tunnel to `localhost:5000`). Nothing auto-opens the UI; use `http://<ip>:5000` explicitly.
