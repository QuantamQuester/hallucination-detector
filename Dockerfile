FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# serving dependencies only (layer is cached until requirements-serve.txt changes)
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY app.py features.py drift_check.py ./
COPY static/ static/
COPY model/ model/

# run as a non-root user; logs/ receives the prediction log used for drift checks
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# $PORT is injected by hosts such as Render/Railway; defaults to 8000 locally
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
