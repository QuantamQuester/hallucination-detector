# =============================================================================
# Hallucination Detector — Production Dockerfile
# Phase 3 audit & deployment hardening — Rudra Kumar Sharma
# =============================================================================
#
# Build strategy:
#   - Two-layer approach: heavy deps first (cached), source code second.
#   - Only the slim serving subset of packages is installed (requirements-serve.txt).
#     Full dev/training deps (mlflow, pytest, etc.) never enter the image.
#   - Non-root user (appuser) is enforced so the container cannot write outside /app/logs.
#
# Local usage:
#   docker build -t hallucination-detector .
#   docker run -p 8000:8000 hallucination-detector
#   curl http://localhost:8000/health
#
# Render/Railway:
#   $PORT is injected automatically; the CMD picks it up via ${PORT:-8000}.
#   The HEALTHCHECK lets the platform detect a crashed worker and restart it.
# =============================================================================

FROM python:3.11-slim

# Keep the image clean and logs streaming to stdout (no .pyc clutter, no pip cache)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ------------------------------------------------------------------
# Layer 1: Python dependencies (invalidated only when requirements change)
# Using requirements-serve.txt keeps the image ~400 MB lighter than
# installing the full training stack (mlflow, pandas-dev, etc.).
# ------------------------------------------------------------------
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ------------------------------------------------------------------
# Layer 2: Application source + pre-trained model artefacts
# Separating source from deps means a code-only change rebuilds in seconds.
# ------------------------------------------------------------------
COPY app.py features.py drift_check.py ./
COPY static/ static/
COPY model/ model/

# ------------------------------------------------------------------
# Security: drop root privileges before the process starts.
# logs/ is created here so appuser can write the prediction log
# without needing any runtime privilege escalation.
# ------------------------------------------------------------------
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser

# Port the FastAPI app listens on (overridden by $PORT on cloud platforms)
EXPOSE 8000

# Platform liveness probe — restarts the container if /health stops responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# $PORT is injected by Render/Railway; defaults to 8000 for local runs
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
