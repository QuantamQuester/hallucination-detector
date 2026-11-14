"""FastAPI service for the hallucination detector.

Routes
  GET  /                 web UI (static/index.html)
  GET  /health           liveness probe
  POST /predict          one (question, evidence, answer) -> probability, tips, signals, explanation
  POST /predict/batch    up to 50 items in one call
  POST /feedback         a user says whether a verdict was right (ground truth for monitoring)
  GET  /api/stats        live traffic statistics for the monitoring view
  GET  /api/drift        live input-drift report (same rule as drift_check.py)
  GET  /api/model        model card: metrics, experiments, global feature weights, enforced request schema
  GET  /api/examples     held-out examples for the UI
  GET  /metrics          Prometheus metrics            GET /docs  interactive OpenAPI docs
"""
import json, os, re, threading, time, uuid
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import Response

from drift_check import compute_drift
from features import BIN_COLS, NUM_COLS, STOP, cset, make_frames, stem, toks

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE, "model", "model.joblib"))
META_PATH = os.getenv("META_PATH", os.path.join(BASE, "model", "metadata.json"))
LOG_PATH = os.getenv("PREDICTION_LOG_PATH", os.path.join(BASE, "logs", "predictions.jsonl"))
LOGGED = NUM_COLS + BIN_COLS                       # the 16 numeric features written to the prediction log
MAX_BATCH = 50


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


# ---- Model is loaded ONCE at start-up (module level), never inside a request handler ----
MODEL = joblib.load(MODEL_PATH)
META = _read_json(META_PATH, {})
EXPERIMENTS = _read_json(os.path.join(BASE, "model", "experiments.json"), [])
EXAMPLES = _read_json(os.path.join(BASE, "model", "examples.json"), [])
MODEL_VERSION = f"{META.get('algorithm', 'unknown')}-{META.get('run_id', 'na')[:8]}"
STARTED = time.time()

PRED_COUNT = Counter("predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("prediction_latency_seconds", "Latency of one scored item")
REJECTED = Counter("validation_errors_total", "Requests rejected with HTTP 422")
FEEDBACK = Counter("feedback_total", "User feedback on verdicts", ["correct"])
_lock = threading.Lock()
RECENT = deque(maxlen=5000)                       # in-memory window over the prediction log (stats + drift)

app = FastAPI(title="Hallucination Detector API", version="2.0.0",
              description="Given a question, an (optional) evidence passage and an answer, estimates the probability that the answer is hallucinated.")
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")


@app.exception_handler(RequestValidationError)
async def count_rejections(request: Request, exc: RequestValidationError):
    REJECTED.inc()                                 # monitored: a spike means a client is sending bad data
    return await request_validation_exception_handler(request, exc)


# ------------------------------------------------------------------ schemas (the data contract)
class PredictRequest(BaseModel):
    """Data contract (see README). Out-of-range or wrongly typed input -> HTTP 422."""
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=5, max_length=1000, description="The question that was asked (5-1000 characters).")
    answer: str = Field(min_length=1, max_length=2000, description="The LLM answer to check (1-2000 characters).")
    knowledge: Optional[str] = Field(default=None, max_length=6000,
                                     description="Evidence passage (<= 6000 characters). Omit / null / blank = evidence unavailable.")

    @field_validator("question", "answer")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) == 0:
            raise ValueError("must not be blank")
        return v


class PredictResponse(BaseModel):
    id: str
    hallucination_probability: float
    label: str
    risk_level: str
    tips: List[str]
    signals: dict
    explanation: Optional[dict] = None
    marked: Optional[dict] = None
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[PredictRequest] = Field(min_length=1, max_length=MAX_BATCH, description=f"1-{MAX_BATCH} items.")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str = Field(min_length=6, max_length=32, pattern=r"^[0-9a-f]+$")
    correct: bool = Field(description="true = the verdict was right, false = it was wrong")


# ------------------------------------------------------------------ explainability (bonus)
LABELS = {"ans_wc": "answer length in words", "ans_chars": "answer length in characters", "avg_wlen": "average word length",
          "ans_len_cat": "answer length class", "len_ratio": "answer length relative to the evidence",
          "ctx_support": "share of answer words found in the evidence", "jaccard_ak": "answer-evidence overlap",
          "support_bin": "grounding class", "novel_content": "words found in neither source",
          "n_numbers": "numbers in the answer", "has_number": "answer contains a number", "unsup_numbers": "numbers not in the sources",
          "novel_caps": "names not in the evidence", "ans_in_q": "answer words copied from the question",
          "q_wc": "question length", "qtype": "question type", "kn_missing": "evidence missing", "ctx_wc": "evidence length",
          "negations": "negation words"}
GROUPS = [("length", "Answer length and style", ["ans_wc", "ans_chars", "avg_wlen", "ans_len_cat", "len_ratio"]),
          ("grounding", "Grounding in the evidence", ["ctx_support", "jaccard_ak", "support_bin", "novel_content"]),
          ("numbers", "Numbers", ["n_numbers", "has_number", "unsup_numbers"]),
          ("names", "New names", ["novel_caps"]),
          ("question", "Relation to the question", ["ans_in_q", "q_wc", "qtype"]),
          ("evidence", "Evidence availability", ["kn_missing", "ctx_wc"]),
          ("negation", "Negations", ["negations"]),
          ("wording", "Wording of the answer", ["answer"])]
GROUP_OF = {f: key for key, _, fs in GROUPS for f in fs}


def _source_feature(name: str) -> str:
    """'plain__ctx_support' -> 'ctx_support', 'nom__qtype_who' -> 'qtype', 'tfidf__was born' -> 'answer'."""
    block, rest = name.split("__", 1)
    if block == "tfidf":
        return "answer"
    if block == "nom":
        return "qtype" if rest.startswith("qtype") else "support_bin"
    return rest


def _linear_model(model):
    """(feature names, weights, intercept) if the classifier is linear on the transformed features, else None."""
    if "pca" in model.named_steps:
        return None
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        w, b = clf.coef_[0], float(clf.intercept_[0])
    elif hasattr(clf, "calibrated_classifiers_") and all(hasattr(c.estimator, "coef_") for c in clf.calibrated_classifiers_):
        w = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)     # average of the calibration folds
        b = float(np.mean([c.estimator.intercept_[0] for c in clf.calibrated_classifiers_]))
    else:
        return None
    return list(model.named_steps["pre"].get_feature_names_out()), np.asarray(w), b


LINEAR = _linear_model(MODEL)


def explain(frame, prob: float) -> dict:
    """Why did the model give this score? Signed contributions per feature group (+ pushes towards 'hallucinated')."""
    row = frame.iloc[0]
    per_feature, phrases = {}, []
    if LINEAR is not None:                          # exact for linear models: contribution = weight x transformed value
        names, w, _ = LINEAR
        x = MODEL.named_steps["pre"].transform(frame)
        x = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        for i in np.nonzero(x * w)[0]:
            src, c = _source_feature(names[i]), float(x[i] * w[i])
            per_feature[src] = per_feature.get(src, 0.0) + c
            if src == "answer":
                phrases.append({"phrase": names[i].split("__", 1)[1], "contribution": round(c, 4)})
        method, unit = "weight x value on the model's linear decision score", "decision score"
    else:                                           # model-agnostic fallback: replace one feature by its training mean
        stats = META.get("train_stats", {})
        probe, cols = [], [c for c in LOGGED if c in stats]
        for c in cols:
            f2 = frame.copy()
            f2[c] = stats[c]["mean"]
            probe.append(f2)
        if probe:
            import pandas as pd
            alt = MODEL.predict_proba(pd.concat(probe, ignore_index=True))[:, 1]
            per_feature = {c: float(prob - p) for c, p in zip(cols, alt)}
        method, unit = "occlusion: change in probability when a feature is replaced by its training mean", "probability"
    groups = []
    for key, label, feats in GROUPS:
        details = [{"feature": f, "label": LABELS.get(f, f), "contribution": round(per_feature[f], 4),
                    "value": None if f == "answer" else _plain(row.get(f))} for f in feats if f in per_feature]
        if details:
            groups.append({"key": key, "label": label, "contribution": round(sum(d["contribution"] for d in details), 4),
                           "details": sorted(details, key=lambda d: -abs(d["contribution"]))})
    groups.sort(key=lambda g: -abs(g["contribution"]))
    phrases.sort(key=lambda p: -abs(p["contribution"]))
    return {"method": method, "unit": unit, "groups": groups, "top_phrases": phrases[:6]}


def global_weights():
    """Model-level view for the model card: which inputs carry the most weight overall."""
    if LINEAR is not None:
        names, w, _ = LINEAR
        eng = [(n, float(v)) for n, v in zip(names, w) if not n.startswith("tfidf__")]
        txt = [(n.split("__", 1)[1], float(v)) for n, v in zip(names, w) if n.startswith("tfidf__")]
        eng.sort(key=lambda t: -abs(t[1])); txt.sort(key=lambda t: t[1])
        return {"kind": "linear weights (standardised features; + = towards hallucinated)",
                "features": [{"name": n.split("__", 1)[1], "weight": round(v, 3)} for n, v in eng[:12]],
                "phrases_hallucinated": [{"phrase": n, "weight": round(v, 3)} for n, v in txt[::-1][:8]],
                "phrases_faithful": [{"phrase": n, "weight": round(v, 3)} for n, v in txt[:8]]}
    clf = MODEL.named_steps["clf"]
    if hasattr(clf, "feature_importances_") and "pca" not in MODEL.named_steps:
        names = MODEL.named_steps["pre"].get_feature_names_out()
        top = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])[:12]
        return {"kind": "impurity-based feature importance", "features": [{"name": n.split("__", 1)[1], "weight": round(float(v), 4)} for n, v in top],
                "phrases_hallucinated": [], "phrases_faithful": []}
    return {"kind": "not available for this model type", "features": [], "phrases_hallucinated": [], "phrases_faithful": []}


GLOBAL_WEIGHTS = global_weights()


# ------------------------------------------------------------------ word-level marks for the UI's "marked proof"
_PIECES = re.compile(r"[A-Za-z0-9']+|[^A-Za-z0-9']+")


def mark_text(question: str, knowledge, answer: str) -> dict:
    """Tag every answer word with the SAME lexical rules the features use (stop-words, stemming, number / name checks).

    answer marks:   ok = found in the evidence, q = only in the question, num = number found in neither source,
                    name = capitalised term not in the evidence, new = other content word found in neither source
    evidence marks: hit = content word that the answer reuses
    """
    has_ev = isinstance(knowledge, str) and knowledge.strip() != ""
    K = cset(knowledge) if has_ev else set()
    Q, A = cset(question), cset(answer)
    kt, qt = (set(toks(knowledge)) if has_ev else set()), set(toks(question))
    out = []
    for piece in _PIECES.findall(answer):
        w = piece.lower().replace("'", "")
        mark = ""
        if re.fullmatch(r"[a-z0-9]+", w):
            if w.isdigit():
                mark = "ok" if w in kt else "q" if w in qt else ("num" if has_ev else "")
            elif w not in STOP:
                s_ = stem(w)
                if s_ in K:
                    mark = "ok"
                elif s_ in Q:
                    mark = "q"
                elif has_ev:
                    mark = "name" if piece[0].isupper() and w not in kt and w not in qt else "new"
        out.append([piece, mark])
    ev = []
    if has_ev:
        for piece in _PIECES.findall(knowledge):
            w = piece.lower().replace("'", "")
            hit = re.fullmatch(r"[a-z0-9]+", w) and w not in STOP and (stem(w) in A or (w.isdigit() and w in set(toks(answer))))
            ev.append([piece, "hit" if hit else ""])
    return {"answer": out, "evidence": ev}


# ------------------------------------------------------------------ tips, logging, scoring
def build_tips(f: dict, prob: float) -> List[str]:
    """2-3 rule-based tips for the user, derived from the engineered features."""
    tips = []
    if f["kn_missing"]:
        tips.append("No evidence passage was given, so the grounding checks were skipped. Paste the source text for a more reliable verdict.")
    else:
        if f["ctx_support"] < 0.5:
            tips.append(f"Only {f['ctx_support']:.0%} of the answer's key words appear in the evidence: the answer adds a lot of content the source does not support.")
        elif f["ctx_support"] > 0.9 and f["ans_wc"] <= 6:
            tips.append("The answer is short and almost entirely made of words from the evidence, which is typical of a faithful answer.")
        if f["unsup_numbers"] > 0:
            n = int(f["unsup_numbers"])
            tips.append(f"The answer contains {n} number{'s' if n > 1 else ''} (such as a date or a count) found in neither the evidence nor the question. Verify {'them' if n > 1 else 'it'}.")
        if f["novel_caps"] > 0:
            n = int(f["novel_caps"])
            tips.append(f"The answer uses {n} capitalised name{'s' if n > 1 else ''} (a person, place or organisation) that the evidence never mentions. Check {'each one' if n > 1 else 'it'}.")
    if f["ans_wc"] > 15:
        tips.append("This is a long, sentence-style answer. Long answers carry more room for invented details, so the detector treats them as riskier.")
    if len(tips) < 2:
        tips.append("Compare the answer with the source for the key fact asked in the question." if prob < 0.5
                    else "Treat the answer as unverified and check the key facts against a reliable source.")
    if len(tips) < 2:
        tips.append("Short answers that merely pick a wrong entity from the evidence can still be hallucinations; read the passage once more.")
    return tips[:3]


def _plain(v):
    """numpy / NaN -> JSON-safe python value."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def log_prediction(record: dict) -> None:
    RECENT.append(record)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with _lock, open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass                                    # logging must never break serving


def _load_recent() -> None:
    """Warm the in-memory window from the log file, so statistics survive a restart."""
    try:
        with open(LOG_PATH, encoding="utf-8") as fh:
            for line in deque(fh, maxlen=RECENT.maxlen):
                try:
                    RECENT.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass


_load_recent()


def score(items: List[PredictRequest], with_explanation: bool) -> List[dict]:
    """Shared by /predict and /predict/batch: features -> one vectorised model call -> response + log line per item."""
    t0 = time.perf_counter()
    frame = make_frames((it.question, it.knowledge, it.answer) for it in items)
    probs = MODEL.predict_proba(frame)[:, 1]
    per_item = (time.perf_counter() - t0) / len(items)
    out = []
    for i, prob in enumerate(map(float, probs)):
        f = {k: _plain(v) for k, v in frame.iloc[i].to_dict().items()}
        label = "hallucinated" if prob >= 0.5 else "faithful"
        risk = "high" if prob >= 0.8 else "medium" if prob >= 0.5 else "low-medium" if prob >= 0.2 else "low"
        tips = build_tips({k: (0 if f.get(k) is None else f[k]) for k in ["ctx_support", "unsup_numbers", "novel_caps", "ans_wc", "kn_missing"]}, prob)
        signals = {"grounding_in_evidence": None if f["ctx_support"] is None else round(float(f["ctx_support"]), 3),
                   "answer_words": int(f["ans_wc"]), "unsupported_numbers": None if f["unsup_numbers"] is None else int(f["unsup_numbers"]),
                   "new_named_terms": None if f["novel_caps"] is None else int(f["novel_caps"]), "evidence_provided": not bool(f["kn_missing"])}
        pid = uuid.uuid4().hex[:12]
        LATENCY.observe(per_item); PRED_COUNT.labels(label=label).inc()
        log_prediction({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "id": pid, "probability": round(prob, 4),
                        "label": label, "latency_ms": round(per_item * 1000, 2), "model_version": MODEL_VERSION,
                        "features": {k: f[k] for k in LOGGED}})
        out.append({"id": pid, "hallucination_probability": round(prob, 4), "label": label, "risk_level": risk, "tips": tips,
                    "signals": signals, "explanation": explain(frame.iloc[[i]], prob) if with_explanation else None,
                    "marked": mark_text(items[i].question, items[i].knowledge, items[i].answer) if with_explanation else None,
                    "model_version": MODEL_VERSION})
    return out


# ------------------------------------------------------------------ routes
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_version": MODEL_VERSION,
            "algorithm": META.get("algorithm"), "trained_at": META.get("trained_at"), "uptime_seconds": round(time.time() - STARTED)}


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(BASE, "static", "index.html"))


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(req: PredictRequest):
    return score([req], with_explanation=True)[0]


@app.post("/predict/batch", tags=["model"])
def predict_batch(req: BatchRequest):
    results = score(req.items, with_explanation=False)
    n_h = sum(r["label"] == "hallucinated" for r in results)
    return {"count": len(results), "hallucinated": n_h, "faithful": len(results) - n_h, "results": results}


@app.post("/feedback", tags=["monitoring"])
def feedback(fb: FeedbackRequest):
    """Human feedback = delayed ground truth. Agreement rate on labelled traffic is the most direct retraining signal."""
    rec = next((r for r in reversed(RECENT) if r.get("id") == fb.prediction_id), None)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown prediction_id")
    rec["feedback_correct"] = fb.correct
    FEEDBACK.labels(correct=str(fb.correct).lower()).inc()
    try:
        with _lock, open(os.path.join(os.path.dirname(LOG_PATH), "feedback.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "id": fb.prediction_id,
                                 "correct": fb.correct, "label": rec.get("label")}) + "\n")
    except OSError:
        pass
    return {"status": "recorded", "prediction_id": fb.prediction_id, "correct": fb.correct}


@app.get("/api/stats", tags=["monitoring"])
def stats():
    recs = list(RECENT)
    probs = [r["probability"] for r in recs]
    lat = sorted(r.get("latency_ms", 0.0) for r in recs)
    pct = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 2) if lat else None
    fb = [r["feedback_correct"] for r in recs if "feedback_correct" in r]
    return {"window": len(recs), "window_max": RECENT.maxlen,
            "labels": {"faithful": sum(p < 0.5 for p in probs), "hallucinated": sum(p >= 0.5 for p in probs)},
            "risk": {"low": sum(p < 0.2 for p in probs), "low-medium": sum(0.2 <= p < 0.5 for p in probs),
                     "medium": sum(0.5 <= p < 0.8 for p in probs), "high": sum(p >= 0.8 for p in probs)},
            "histogram": [sum(min(int(p * 10), 9) == b for p in probs) for b in range(10)],
            "latency_ms": {"mean": round(sum(lat) / len(lat), 2) if lat else None, "p50": pct(0.50), "p95": pct(0.95)},
            "evidence_missing_share": round(sum(bool(r["features"].get("kn_missing")) for r in recs) / len(recs), 3) if recs else None,
            "timeline": [{"ts": r["ts"], "p": r["probability"], "ms": r.get("latency_ms")} for r in recs[-80:]],
            "feedback": {"count": len(fb), "agree": sum(fb), "agreement_rate": round(sum(fb) / len(fb), 3) if fb else None},
            "rejected_422": int(REJECTED._value.get()), "uptime_seconds": round(time.time() - STARTED), "model_version": MODEL_VERSION}


@app.get("/api/drift", tags=["monitoring"])
def drift(min_samples: int = 30):
    rows = [r["features"] for r in RECENT if "features" in r]
    if "train_stats" not in META:
        return {"status": "unavailable", "samples": len(rows), "min_samples": min_samples, "features": []}
    if len(rows) < min_samples:
        return {"status": "insufficient_data", "samples": len(rows), "min_samples": min_samples, "features": []}
    table, found = compute_drift(rows, META["train_stats"])
    return {"status": "drift" if found else "ok", "samples": len(rows), "min_samples": min_samples, "features": table,
            "rule": "mean moved by more than 0.5 training standard deviations, or spread changed by more than 2x"}


@app.get("/api/model", tags=["model"])
def model_card():
    keep = ["algorithm", "params", "run_name", "registry_version", "metrics", "trained_at", "sklearn", "python",
            "n_train_rows", "n_test_rows", "confusion_matrix"]
    return {"model_version": MODEL_VERSION, **{k: META.get(k) for k in keep}, "experiments": EXPERIMENTS,
            "global_weights": GLOBAL_WEIGHTS, "request_schema": PredictRequest.model_json_schema(),
            "decision_threshold": 0.5, "risk_bands": {"low": [0, 0.2], "low-medium": [0.2, 0.5], "medium": [0.5, 0.8], "high": [0.8, 1]},
            "max_batch": MAX_BATCH}


@app.get("/api/examples", tags=["model"])
def examples():
    return {"count": len(EXAMPLES), "note": "held-out test questions, never used for training", "examples": EXAMPLES}
