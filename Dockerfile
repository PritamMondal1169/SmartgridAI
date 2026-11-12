# Production Dockerfile for SmartgridAI Full-Stack Platform
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code, models, and data
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/ ./data/
COPY quality_report.json ./
COPY exports/ ./exports/

# Copy pre-compiled production frontend bundle
COPY frontend/dist/ ./frontend/dist/

EXPOSE 8000

# Start Unified FastAPI App (Serving both API endpoints and React Dashboard SPA)
CMD ["sh", "-c", "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
