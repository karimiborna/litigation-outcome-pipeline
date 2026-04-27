# Vercel Deployment Guide

This guide covers deploying the LexRatio frontend to Vercel.

## Prerequisites

- A [Vercel account](https://vercel.com)
- The Vercel CLI installed: `npm install -g vercel`
- Git repository connected to Vercel

## Deployment Steps

### 1. Local Setup

```bash
# Install Vercel CLI (if not already installed)
npm install -g vercel

# Login to Vercel
vercel login
```

### 2. Connect Your Repository

```bash
# Link the project to Vercel
vercel link

# Follow the prompts to connect your GitHub/GitLab/Bitbucket repository
```

### 3. Environment Variables (if needed)

Set environment variables in your Vercel project dashboard:

```
REACT_APP_API_URL=https://your-backend-domain.com/api/analyze-lexratio
```

Or the frontend will auto-detect and proxy to your backend.

### 4. Deploy

```bash
# Deploy to production
vercel --prod

# Or deploy to preview environment
vercel
```

## Architecture

### Frontend (Vercel)
- Static HTML/CSS/JS deployment
- Served globally via Vercel's CDN
- No build step required (static assets only)

### Backend (Separate)
- Runs your FastAPI server with models
- Accessible via API endpoint
- Handles ML inference and MLflow model loading

### API Bridge
- Frontend POST requests → `https://your-backend-domain.com/api/analyze-lexratio`
- Returns JSON with predictions, signals, and analysis

## Configuration

### API Endpoint Detection

The frontend automatically detects the API endpoint:

1. **Local Development**: `http://localhost:8000/api/analyze-lexratio`
2. **Vercel Preview/Production**: Configured via environment variable
3. **Fallback**: Window location-based detection

### Custom API URL

Set the `REACT_APP_API_URL` environment variable in your Vercel project settings:

```
REACT_APP_API_URL=https://your-api-domain.com/api/analyze-lexratio
```

## Viewing Your Deployment

After deployment, your interface will be available at:
```
https://your-project.vercel.app
```

## Monitoring

- View logs in Vercel dashboard
- Monitor API requests in your backend server
- Check MLflow tracking for model predictions

## Troubleshooting

### CORS Issues
If you see CORS errors, update your FastAPI backend to allow Vercel domains:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### API Connection Errors
- Check that `REACT_APP_API_URL` is set correctly
- Verify backend server is running and accessible
- Test API endpoint manually: `curl https://your-backend/api/analyze-lexratio`

## Rollback

To rollback to a previous deployment:

```bash
# List recent deployments
vercel list

# Rollback to a specific deployment
vercel promote <deployment-url>
```
