import json, os, tempfile
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PREDICTION_LOG_PATH", os.path.join(tempfile.mkdtemp(), "predictions.jsonl"))   # keep test logs out of the repo
import app as app_module  # noqa: E402  (loads the model once at import)

client = TestClient(app_module.app)
Q = "Which magazine was started first, Arthur's Magazine or First for Women?"
EV = ("Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. "
      "First for Women is a woman's magazine published by Bauer Media Group in the USA. It was started in 1989.")


def prob(answer, knowledge=EV, question=Q):
    r = client.post("/predict", json={"question": question, "knowledge": knowledge, "answer": answer})
    assert r.status_code == 200, r.text
    return r.json()["hallucination_probability"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_ui_loads():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Hallucination Detector" in r.text
    assert client.get("/docs").status_code == 200


def test_predict_response_contract():
    r = client.post("/predict", json={"question": Q, "knowledge": EV, "answer": "Arthur's Magazine"})
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["hallucination_probability"] <= 1.0
    assert j["label"] in {"faithful", "hallucinated"}
    assert 2 <= len(j["tips"]) <= 3                       # UI requirement: 2-3 rule-based tips
    assert set(j["signals"]) >= {"grounding_in_evidence", "answer_words"}


def test_domain_logic_supported_answer_beats_invented_answer():
    """A short answer copied from the evidence must look MORE faithful than an answer with invented details."""
    supported = prob("Arthur's Magazine")
    invented = prob("First for Women was started first, in 1824, by a New York publisher in Boston.")
    assert supported < 0.5 < invented
    assert supported < invented


def test_domain_logic_more_unsupported_detail_raises_risk():
    plain = prob("It was published in Philadelphia.", question="Where was Arthur's Magazine published?")
    padded = prob("It was published in Boston, Massachusetts, by Harper and Brothers in 1901.", question="Where was Arthur's Magazine published?")
    assert padded > plain


@pytest.mark.parametrize("payload", [
    {"question": "Hi", "answer": "x"},                                           # question too short
    {"question": Q, "answer": ""},                                               # empty answer
    {"question": Q, "answer": "   "},                                            # blank answer
    {"question": Q, "answer": "a" * 2001},                                       # answer too long
    {"question": Q, "knowledge": "k" * 6001, "answer": "x"},                     # evidence too long
    {"question": Q, "answer": 12345},                                            # wrong type
    {"answer": "Arthur's Magazine"},                                             # missing required field
    {"question": Q, "answer": "x", "temperature": 0.7},                          # unknown field
])
def test_boundary_invalid_input_returns_422(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_evidence_is_handled_not_a_500():
    """A missing/NaN evidence feature is imputed inside the pipeline and flagged in the response."""
    for k in (None, "", "   "):
        r = client.post("/predict", json={"question": Q, "knowledge": k, "answer": "Arthur's Magazine"})
        assert r.status_code == 200
        assert r.json()["signals"]["evidence_provided"] is False


def test_prediction_is_logged_for_drift_checks(tmp_path, monkeypatch):
    log = tmp_path / "p.jsonl"
    monkeypatch.setattr(app_module, "LOG_PATH", str(log))
    prob("Arthur's Magazine")
    rec = json.loads(log.read_text().splitlines()[-1])
    assert "features" in rec and "probability" in rec and "ans_wc" in rec["features"]


def test_metrics_endpoint_exposes_prometheus_counters():
    prob("Arthur's Magazine")
    text = client.get("/metrics").text
    assert "predictions_total" in text and "prediction_latency_seconds" in text


# ---------------------------------------------------------------- full-stack features (v2)
def test_explanation_names_the_feature_groups_behind_the_score():
    r = client.post("/predict", json={"question": Q, "knowledge": EV, "answer": "First for Women was started first, in 1824, by a New York publisher."})
    exp = r.json()["explanation"]
    assert exp["groups"] and {"key", "label", "contribution", "details"} <= set(exp["groups"][0])
    assert sum(g["contribution"] for g in exp["groups"]) > 0          # net push is towards "hallucinated" for this answer


def test_batch_scores_every_item_and_matches_single_predictions():
    items = [{"question": Q, "knowledge": EV, "answer": a} for a in ["Arthur's Magazine", "First for Women was started first, in 1824."]]
    r = client.post("/predict/batch", json={"items": items})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["faithful"] + body["hallucinated"] == 2
    assert abs(body["results"][0]["hallucination_probability"] - prob("Arthur's Magazine")) < 1e-6


@pytest.mark.parametrize("payload", [
    {"items": []},                                                               # empty batch
    {"items": [{"question": Q, "answer": "x"}] * 51},                            # more than 50 items
    {"items": [{"question": Q, "answer": ""}]},                                  # one invalid item rejects the batch
])
def test_batch_boundaries_return_422(payload):
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_feedback_is_recorded_and_unknown_ids_are_rejected():
    pid = client.post("/predict", json={"question": Q, "knowledge": EV, "answer": "Arthur's Magazine"}).json()["id"]
    assert client.post("/feedback", json={"prediction_id": pid, "correct": True}).status_code == 200
    assert client.get("/api/stats").json()["feedback"]["count"] >= 1
    assert client.post("/feedback", json={"prediction_id": "ffffffffffff", "correct": True}).status_code == 404
    assert client.post("/feedback", json={"prediction_id": pid, "correct": "maybe"}).status_code == 422


def test_stats_and_drift_reports_have_the_expected_shape():
    prob("Arthur's Magazine")
    s = client.get("/api/stats").json()
    assert s["window"] >= 1 and len(s["histogram"]) == 10 and sum(s["histogram"]) == s["window"]
    d = client.get("/api/drift", params={"min_samples": 1}).json()
    assert d["status"] in {"ok", "drift"} and d["features"] and d["features"][0]["status"] in {"ok", "DRIFT"}
    assert client.get("/api/drift", params={"min_samples": 10**6}).json()["status"] == "insufficient_data"


def test_model_card_exposes_metrics_schema_and_examples():
    m = client.get("/api/model").json()
    assert m["metrics"]["test_f1"] > 0.9 and m["request_schema"]["properties"]["question"]["minLength"] == 5
    ex = client.get("/api/examples").json()["examples"]
    assert ex and {"question", "answer", "truth"} <= set(ex[0])
