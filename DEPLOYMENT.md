# Deployment Guide & Audit Log

---

## Live Service

| Item | Detail |
|---|---|
| **Platform** | Render (free tier, Docker runtime) |
| **Health endpoint** | `GET /health` → `{"status":"ok","model_loaded":true}` |
| **Auto-deploy** | Every push to `main` triggers a redeploy (see `render.yaml`) |

---

## Running Locally

```bash
# Option 1 — plain Python
pip install -r requirements.txt
python train.py --algorithm logreg --C 1 --force-save   # retrain if needed
uvicorn app:app --reload

# Option 2 — Docker (mirrors the production image exactly)
docker build -t hallucination-detector .
docker run -p 8000:8000 hallucination-detector
curl http://localhost:8000/health
```

---

## Docker Image Audit (Phase 3)

Bugs found and fixed during the Phase 3 deployment review:

| # | Issue | Fix |
|---|---|---|
| 1 | `mlflow.db` was not in `.dockerignore` — the binary was silently copied into the image, bloating it and leaking MLflow experiment metadata | Added `mlflow.db` and `ci_mlflow.db` to `.dockerignore` |
| 2 | `notebooks/` directory (>50 MB) was included in the build context | Added `notebooks/` to `.dockerignore` |
| 3 | CI used the committed `mlflow.db` (created with an older MLflow build), causing Alembic schema migration errors on a fresh install | Changed CI to use `--tracking-uri sqlite:///ci_mlflow.db` (a fresh, throwaway DB) and `--no-register` |
| 4 | The `Dockerfile` had no explanatory comments, making onboarding harder | Added section-level comments explaining the two-layer build strategy, non-root user, and `$PORT` injection |

---

## CI Pipeline

```
push to main
  └─ Install deps (pip install -r requirements.txt)
  └─ Train baseline (logreg, fresh MLflow DB, --force-save)
  └─ Run 33 pytest cases  ✓
  └─ Docker build
  └─ Smoke-test container (GET /health, POST /predict 422 check)
  └─ Trigger Render redeploy (if RENDER_DEPLOY_HOOK secret is set)
```

---

## Feature Engineering Notes

See `features.py` for the full 25-feature breakdown.  
Key design decision: evidence-dependent features are set to `NaN` when no knowledge passage
is provided, and the pipeline's `SimpleImputer(strategy="median")` fills them with training
medians at predict time. This keeps the serving path and training path identical.

Drift is monitored at `GET /api/drift` (live) and via `python drift_check.py` (cron-friendly).
Threshold: mean shift > 0.5 training SDs, or std ratio outside [0.5, 2.0].
