#!/bin/bash
# Quick Reference: Running the Litigation Outcome Pipeline

# ============================================
# QUICK START (Copy & Paste)
# ============================================

# 1. Navigate to project
cd ~/Documents/GitHub/litigation-outcome-pipeline

# 2. Activate environment
source .venv/bin/activate

# 3. Start server
./scripts/run_server.sh

# 4. Open in browser
# Visit: http://localhost:8000

# ============================================
# KEY URLS
# ============================================

Frontend (LexRatio):        http://localhost:8000
API Docs (Swagger):         http://localhost:8000/docs
Health Check:               http://localhost:8000/health
Alternative UI:             http://localhost:8000/lexratio

MLflow Server:              http://35.208.251.175:5000
MLflow Model Registry:      http://35.208.251.175:5000/#/models

# ============================================
# COMMON COMMANDS
# ============================================

# Check server health
curl http://localhost:8000/health | jq .

# Test prediction endpoint
curl -X POST http://localhost:8000/api/analyze-lexratio \
  -H "Content-Type: application/json" \
  -d '{"case_text":"I paid for work that was never completed", "claim_amount":1000}'

# View API documentation
# Open: http://localhost:8000/docs

# Check if MLflow is accessible
curl http://35.208.251.175:5000/

# ============================================
# ENVIRONMENT CONFIGURATION
# ============================================

# .env file location: /Users/courtneymohun/Documents/GitHub/litigation-outcome-pipeline/.env

# Key settings:
# MLFLOW_TRACKING_URI=http://35.208.251.175:5000
# MLFLOW_REGISTRY_URI=http://35.208.251.175:5000
# LLM_API_KEY=sk-your-key-here
# API_HOST=0.0.0.0
# API_PORT=8000

# ============================================
# TROUBLESHOOTING
# ============================================

# If "Models not loaded":
# 1. Check MLflow: http://35.208.251.175:5000/
# 2. Verify models exist in registry
# 3. Check they're in "Production" stage

# If frontend errors:
# 1. Open browser console (F12)
# 2. Check server is running: curl http://localhost:8000/
# 3. Check logs in terminal

# If CORS errors:
# → CORS is enabled for all origins in development
# → For production, see api/app.py line 57-67

# ============================================
# SERVER MODES
# ============================================

# Development (with auto-reload):
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Production (no reload):
ENV=production ./scripts/run_server.sh
# OR
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Docker:
docker-compose -f docker/docker-compose.yml up inference

# ============================================
# DOCUMENTATION
# ============================================

INTEGRATION_GUIDE.md        ← Comprehensive guide
INTEGRATION_SUMMARY.md      ← What was done + tests
api/CLAUDE.md               ← API module specifics
mlflow/CLAUDE.md            ← MLflow configuration
README.md                   ← Project overview

# ============================================
# FILE STRUCTURE
# ============================================

lexratio.html               ← Frontend (root)
api/static/lexratio.html    ← Frontend (static copy)
api/app.py                  ← FastAPI server
api/dependencies.py         ← MLflow initialization
api/schemas.py              ← Request/response schemas

.env                        ← Environment configuration
scripts/run_server.sh       ← Server startup script

# ============================================
# USEFUL ENVIRONMENT VARIABLES
# ============================================

# Set custom MLflow server:
export MLFLOW_TRACKING_URI=http://35.208.251.175:5000

# Set custom API port:
# Edit .env: API_PORT=9000

# Enable debug logging:
export LOGLEVEL=DEBUG
./scripts/run_server.sh

# ============================================
