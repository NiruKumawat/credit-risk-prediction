# Multi-stage production Dockerfile for Credit Risk Prediction System
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code and artifacts
COPY src/ /app/src/
COPY api/ /app/api/
COPY dashboard/ /app/dashboard/
COPY artifacts/ /app/artifacts/
COPY data/ /app/data/
COPY pytest.ini /app/

# Expose API and Streamlit ports
EXPOSE 8000 8501

# Healthcheck for API service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
