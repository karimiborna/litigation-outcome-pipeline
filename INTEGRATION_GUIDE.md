# Frontend-Backend Integration Guide

This document describes how to run the Litigation Outcome Pipeline with the integrated LexRatio frontend and MLflow server.

## Architecture

```
Frontend (LexRatio UI)
    ↓ (HTTP POST to /api/analyze-lexratio)
FastAPI Server (http://localhost:8000)
    ↓ (Model loading from)
MLflow Server (http://35.208.251.175:5000)
    ↓ (Downloads Production models)
Scikit-learn Models
```

## Quick Start

### Prerequisites

1. **Python 3.9+** with venv or conda
2. **MLflow Server** running at `http://35.208.251.175:5000/`
3. **Models trained and registered** in MLflow Production registry
   - `litigation-win-classifier`
   - `litigation-monetary-regressor`

### Setup Environment

```bash
# 1. Navigate to project root
cd /path/to/litigation-outcome-pipeline

# 2. Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# 3. Create/update .env file with MLflow server configuration
cat > .env << 'EOF'
MLFLOW_TRACKING_URI=http://35.208.251.175:5000
MLFLOW_REGISTRY_URI=http://35.208.251.175:5000
LLM_API_KEY=sk-your-key-here
NVIDIA_API_KEY=nvapi-your-key-here
EOF

# 4. Install dependencies
pip install -e .
```

### Run the Server

```bash
# Option A: Direct Python
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Option B: Using the run script (if available)
./scripts/run_server.sh

# Option C: Docker
docker-compose -f docker/docker-compose.yml up inference
```

### Access the Frontend

Once the server is running:

- **LexRatio UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **API Root**: http://localhost:8000/api/
- **Health Check**: http://localhost:8000/health

## Configuration

### Environment Variables

Key variables in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server address |
| `MLFLOW_REGISTRY_URI` | Same as tracking URI | Model registry address |
| `LLM_API_KEY` | Required | OpenAI API key for feature extraction |
| `NVIDIA_API_KEY` | Optional | For scanned PDF extraction |

### Frontend Configuration

The frontend automatically detects the API endpoint:

- **Local dev** (`localhost`): Uses `http://localhost:8000/api/analyze-lexratio`
- **Production** (other domains): Uses `{current origin}/api/analyze-lexratio`

To override, modify the `apiUrl` calculation in `lexratio.html`:

```javascript
var apiUrl = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') 
  ? 'http://localhost:8000/api/analyze-lexratio'
  : window.location.origin + '/api/analyze-lexratio';
```

## API Endpoints

### Frontend-Facing Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves LexRatio HTML UI |
| `/lexratio` | GET | Alternative UI endpoint |
| `/health` | GET | Health check |
| `/api/` | GET | API root (JSON) |
| `/docs` | GET | Swagger UI documentation |

### Analysis Endpoints

| Endpoint | Method | Request | Purpose |
|----------|--------|---------|---------|
| `/api/analyze-lexratio` | POST | `{case_text, case_title, cause_of_action, claim_amount}` | Frontend analysis |
| `/predict` | POST | `{case_text, case_number, ...}` | Raw prediction API |
| `/predict/batch` | POST | `{cases: [...]}` | Batch predictions |
| `/similar` | POST | `{case_text, top_k}` | Similar case retrieval |
| `/counterfactual` | POST | `{case_text, perturbations}` | What-if analysis |

## Debugging

### Check MLflow Connection

```bash
python -c "
from models.config import MLflowConfig
from models.tracking import init_mlflow
config = MLflowConfig()
print(f'Tracking URI: {config.tracking_uri}')
client = init_mlflow(config)
print('MLflow connected successfully')
"
```

### Check Model Loading

```bash
curl http://localhost:8000/health | jq .
```

Expected response:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "models_loaded": true,
  "classifier_loaded": true,
  "regressor_loaded": true
}
```

### View API Logs

```bash
# Enable debug logging
LOGLEVEL=DEBUG python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Models Not Loading

**Problem**: `models_loaded: false` in health check

**Solutions**:
1. Verify MLflow server is reachable: `curl http://35.208.251.175:5000/`
2. Check that models exist in MLflow: Visit http://35.208.251.175:5000/ → Model Registry
3. Ensure model stage is set to "Production"
4. Check `MLFLOW_TRACKING_URI` in `.env` file

### Frontend Analysis Fails

**Problem**: "Analysis failed" error on frontend

**Solutions**:
1. Check browser console (F12) for network errors
2. Check server logs for exceptions
3. Verify `/health` endpoint returns `models_loaded: true`
4. Test `/api/analyze-lexratio` directly:
   ```bash
   curl -X POST http://localhost:8000/api/analyze-lexratio \
     -H "Content-Type: application/json" \
     -d '{"case_text": "I was not paid for work", "claim_amount": 500}'
   ```

### CORS Errors

**Problem**: Cross-Origin Request Blocked

**Note**: CORS is enabled for all origins in development mode. For production, update the `allow_origins` list in `api/app.py`.

## Production Deployment

### Docker Deployment

```bash
# Build inference service
docker build -f docker/Dockerfile.inference -t litigation-inference .

# Run with environment
docker run \
  -e MLFLOW_TRACKING_URI=http://35.208.251.175:5000 \
  -e LLM_API_KEY=${LLM_API_KEY} \
  -p 8000:8000 \
  litigation-inference
```

### Environment-Specific Configuration

**Development**:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
API_HOST=localhost
API_PORT=8000
```

**Production (GCP/AWS)**:
```bash
MLFLOW_TRACKING_URI=http://35.208.251.175:5000
API_HOST=0.0.0.0
API_PORT=8000
```

Update in Vercel/Cloud Run environment settings.

## Next Steps

1. **Train models** and register in MLflow:
   ```bash
   python scripts/train_binary_classifier.py
   python scripts/promote_models_to_production.py
   ```

2. **Test the full pipeline**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Deploy to cloud** (see `VERCEL_DEPLOYMENT.md` or `infra/README.md`)

## Support

For issues, check:
- `mlflow/CLAUDE.md` — MLflow server setup
- `api/CLAUDE.md` — API module documentation
- `README.md` — Overall project architecture
