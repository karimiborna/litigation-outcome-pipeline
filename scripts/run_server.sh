#!/bin/bash
# Start the LexRatio FastAPI server with proper environment setup

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Load environment variables from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Check required environment variables
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    echo "Warning: MLFLOW_TRACKING_URI not set, using default http://localhost:5000"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

if [ -z "$LLM_API_KEY" ]; then
    echo "Warning: LLM_API_KEY not set - feature extraction may fail"
fi

# Print startup info
echo "=========================================="
echo "Litigation Outcome Pipeline - API Server"
echo "=========================================="
echo "MLflow Server: ${MLFLOW_TRACKING_URI}"
echo "API Host: 0.0.0.0"
echo "API Port: 8000"
echo "Frontend: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "=========================================="
echo ""

# Determine reload flag (development vs production)
RELOAD_FLAG="--reload"
if [ "$ENV" = "production" ]; then
    RELOAD_FLAG=""
    echo "Running in PRODUCTION mode (no auto-reload)"
else
    echo "Running in DEVELOPMENT mode (with auto-reload)"
fi

# Start the server
cd "$PROJECT_ROOT"
echo "Starting FastAPI server..."
python -m uvicorn api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    $RELOAD_FLAG \
    --log-level info
