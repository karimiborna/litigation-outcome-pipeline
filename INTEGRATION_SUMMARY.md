# Integration Summary

## What Was Done

Successfully integrated the LexRatio frontend with the FastAPI backend and configured it to work with the external MLflow server at `http://35.208.251.175:5000/`.

### Changes Made

#### 1. **Environment Configuration** (`.env`)
- ✅ Updated `MLFLOW_TRACKING_URI` to point to `http://35.208.251.175:5000`
- ✅ Added `MLFLOW_REGISTRY_URI` for model registry access
- ✅ Added API configuration variables for deployment flexibility

#### 2. **Frontend HTML Updates**
- **Root file** (`lexratio.html`): Updated JavaScript to call FastAPI backend instead of Anthropic's API
  - Detects localhost automatically for dev mode
  - Calls `/api/analyze-lexratio` endpoint
  - Returns structured response with predictions, confidence, strengths, weaknesses, and advice
  
- **Static file** (`api/static/lexratio.html`): Synchronized with same endpoint logic for redundancy

#### 3. **FastAPI App Enhancements** (`api/app.py`)
- ✅ Enhanced CORS middleware to accept all origins (dev-friendly)
- ✅ Mounted static files from `/api/static/`
- ✅ Added root `/` endpoint that serves `lexratio.html` as homepage
- ✅ Kept `/lexratio` endpoint as alternative UI route
- ✅ Added `/api/` endpoint for JSON-based API access

#### 4. **Dependency Injection** (`api/dependencies.py`)
- ✅ Enhanced `AppState` to initialize MLflow on startup with config
- ✅ Improved logging to show MLflow tracking URI on connection
- ✅ Added error handling for MLflow initialization failures

#### 5. **Server Startup Script** (`scripts/run_server.sh`)
- ✅ Created bash script to start the server with:
  - Automatic virtual environment activation
  - Environment variable loading from `.env`
  - Proper MLflow server information display
  - Development mode with auto-reload (configurable)

#### 6. **Documentation**
- ✅ Created `INTEGRATION_GUIDE.md` with:
  - Architecture diagram showing frontend → API → MLflow flow
  - Quick start setup instructions
  - Configuration reference
  - API endpoint documentation
  - Troubleshooting guide
  - Debugging commands
  - Production deployment instructions

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      User Browser                            │
│                   (LexRatio Frontend)                        │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP POST
                  │ /api/analyze-lexratio
                  ▼
┌─────────────────────────────────────────────────────────────┐
│             FastAPI Server (Port 8000)                       │
│  • Routes HTTP requests to ML pipeline                       │
│  • Serves HTML frontend at http://localhost:8000            │
│  • Extracts case features using LLM (OpenAI)                │
└──────┬──────────────────────────────────────────────────────┘
       │ Loads models from
       ▼
┌─────────────────────────────────────────────────────────────┐
│   MLflow Server (http://35.208.251.175:5000)               │
│  • Model Registry: litigation-win-classifier                │
│  •                 litigation-monetary-regressor            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Ensure environment is configured
```bash
cd /path/to/litigation-outcome-pipeline
source .venv/bin/activate
```

### 2. Start the server
```bash
# Option A: Using the helper script
./scripts/run_server.sh

# Option B: Direct uvicorn
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the application
- **Frontend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Testing the Integration

### Test 1: Check server is running
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

### Test 2: Test the prediction endpoint directly
```bash
curl -X POST http://localhost:8000/api/analyze-lexratio \
  -H "Content-Type: application/json" \
  -d '{
    "case_text": "I hired a contractor who never finished the work and kept my $5000 deposit.",
    "claim_amount": 5000,
    "cause_of_action": "breach_of_contract"
  }' | jq .
```

Expected response:
```json
{
  "win_probability": 72,
  "expected_award": 4500.0,
  "confidence": "high",
  "verdict_summary": "Claimant is likely to prevail with an expected recovery of $4500.",
  "strengths": [...],
  "weaknesses": [...],
  "advice": "...",
  "signals": {...}
}
```

### Test 3: Use the frontend in browser
1. Navigate to http://localhost:8000
2. Fill in case details
3. Click "Submit for analysis"
4. Receive verdict with detailed analysis

## File Changes Summary

| File | Change | Purpose |
|------|--------|---------|
| `.env` | Created/Updated | MLflow server configuration |
| `lexratio.html` | Modified | Frontend API integration |
| `api/static/lexratio.html` | Modified | Sync with root version |
| `api/app.py` | Enhanced | Better frontend serving |
| `api/dependencies.py` | Enhanced | MLflow initialization |
| `scripts/run_server.sh` | Created | Server startup helper |
| `INTEGRATION_GUIDE.md` | Created | Complete documentation |

## Next Steps

1. **Verify models are in MLflow**: Visit http://35.208.251.175:5000 to see:
   - `litigation-win-classifier` model (Production stage)
   - `litigation-monetary-regressor` model (Production stage)

2. **Train new models** (if needed):
   ```bash
   python scripts/train_binary_classifier.py
   python scripts/promote_models_to_production.py
   ```

3. **Deploy to production** (see VERCEL_DEPLOYMENT.md or infra/ directory)

## Environment Variables Checklist

Required for full functionality:
- ✅ `MLFLOW_TRACKING_URI=http://35.208.251.175:5000`
- ✅ `LLM_API_KEY=sk-...` (OpenAI)
- ⚠️ `NVIDIA_API_KEY=nvapi-...` (Optional, for scanned PDFs)
- ⚠️ `SFTC_SESSION_ID=...` (Only needed for scraping)

## Troubleshooting

### Models not loading?
1. Check if MLflow server is accessible:
   ```bash
   curl http://35.208.251.175:5000/
   ```
2. Verify models exist in MLflow registry
3. Check `/health` endpoint for detailed status

### Frontend not responding?
1. Check browser console (F12) for network errors
2. Verify API server is running: `curl http://localhost:8000/`
3. Check server logs for exceptions

### CORS errors?
- CORS is enabled for all origins in development
- For production, update `allow_origins` list in `api/app.py`

## Support Resources

- `INTEGRATION_GUIDE.md` — Detailed integration documentation
- `api/CLAUDE.md` — API module specifics
- `mlflow/CLAUDE.md` — MLflow configuration
- `README.md` — Overall project architecture
