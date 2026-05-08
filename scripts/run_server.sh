#!/bin/bash
# Start the LexRatio FastAPI server with proper environment setup.
#
# This is the recommended entry point on macOS — it sets the fork-safety env
# vars (OBJC_DISABLE_INITIALIZE_FORK_SAFETY, no_proxy, KMP_DUPLICATE_LIB_OK)
# in the shell *before* Python starts, which is the only place they take effect
# in time. Without these, the lifespan model-load segfaults on macOS because
# MLflow's multiprocessing.fork() collides with libomp/CoreFoundation already
# loaded by xgboost/numpy.

set -e

# --- Project paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# --- macOS fork-safety env vars (must be set before any python invocation) ---
# OBJC_DISABLE_INITIALIZE_FORK_SAFETY  — disables Apple's fork-after-Foundation abort
# no_proxy='*'                         — stops CFNetwork proxy lookup in forked child
# KMP_DUPLICATE_LIB_OK                 — silences Intel-OMP duplicate-runtime aborts
# OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 — disables OpenMP/MKL worker threads;
#                                        XGBoost spawns these on first import, and
#                                        mlflow's artifact downloader then forks,
#                                        which crashes the child even when other
#                                        fork-safety flags are set. Single-threaded
#                                        inference is acceptable for this app size.
if [[ "$OSTYPE" == "darwin"* ]]; then
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    export no_proxy='*'
    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
fi

# --- .env is read directly by pydantic-settings inside the Python process
#     (FeaturesConfig, MLflowConfig, RetrievalConfig all use env_file=".env").
#     We deliberately do NOT `source .env` here because shell parsers choke on
#     unquoted spaces/parens in values like SCRAPER_USER_AGENT.

# --- MLflow defaults: hosted GCP server. Never fall back to localhost. ---
: "${MLFLOW_TRACKING_URI:=http://35.208.251.175:5000}"
: "${MLFLOW_REGISTRY_URI:=$MLFLOW_TRACKING_URI}"
export MLFLOW_TRACKING_URI MLFLOW_REGISTRY_URI

# --- Banner ---
echo "=========================================="
echo "Litigation Outcome Pipeline - API Server"
echo "=========================================="
echo "MLflow Server: $MLFLOW_TRACKING_URI"
echo "Frontend:      http://localhost:8000/lexratio"
echo "API root:      http://localhost:8000"
echo "Docs:          http://localhost:8000/docs"
echo "=========================================="

# --- Reload only outside production ---
RELOAD_FLAG="--reload"
if [ "$ENV" = "production" ]; then
    RELOAD_FLAG=""
fi

cd "$PROJECT_ROOT"
exec python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    $RELOAD_FLAG \
    --log-level info
