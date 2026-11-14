"""Question-type study: do WHY and POLAR questions hallucinate differently, and which features catch them?

Writes model/qtype_study.json (shown on the app's "Question types" page) and logs the pipeline comparison to MLflow.

  1. audit        question-type distribution of HaluEval-QA with the old (v1) and the new (v2) rule
  2. dataset      composition of the WHY/POLAR extension
  3. contrasts    per question group, how each feature differs between faithful and hallucinated answers
                  (Cohen's d and single-feature AUC, on TRAIN rows only)
  4. families     per question group, grouped-CV accuracy of a model that sees ONE feature family only
                  (lexical / semantic / contextual) and of all three together
  5. pipelines    the same test set scored by
                    A  first deployment : v1 features, trained on HaluEval-QA only
                    B  + extension data : v1 features, trained on HaluEval-QA + WHY/POLAR
                    C  + new features   : v2 features, trained on HaluEval-QA + WHY/POLAR
                    BERT                : read from model/bert_results.json when train_bert.py has been run
Run:  python qtype_study.py
"""
import argparse, json, os, warnings
from datetime import datetime, timezone

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score

from features import FEATURE_GROUPS, qtype, qtype_v1
from train import FEATURE_SETS, SEED, build_model, load_split, qgroup, slice_metrics

warnings.filterwarnings("ignore")
GROUPS = ["WHY", "POLAR", "OTHER"]


def audit(base_path):
    q = pd.read_csv(base_path).drop_duplicates("group_id")["question"]
    v1, v2 = q.map(qtype_v1).value_counts(), q.map(qtype).value_counts()
    order = ["what", "which", "who", "polar", "how", "when", "where", "why", "other"]
    return {"n_questions": int(len(q)), "rows": [{"type": t, "v1": int(v1.get(t, 0)), "v2": int(v2.get(t, 0))} for t in order]}


def contrasts(train):
    out = {}
    g = qgroup(train["qtype"].values)
    for name in GROUPS:
        d = train[g == name]
        y = d["hallucination"].values
        rows = []
        for fam, cols in FEATURE_GROUPS.items():
            for c in cols:
                x = d[c].astype(float)
                ok = x.notna().values
                if ok.sum() < 20 or x[ok].nunique() < 2:
                    continue
                a, b = x[ok & (y == 0)], x[ok & (y == 1)]
                sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
                auc = roc_auc_score(y[ok], x[ok])
                rows.append({"feature": c, "family": fam, "faithful": round(float(a.mean()), 3), "hallucinated": round(float(b.mean()), 3),
                             "cohen_d": round(float((b.mean() - a.mean()) / sd), 3), "auc": round(float(max(auc, 1 - auc)), 3)})
        rows.sort(key=lambda r: -r["auc"])
        out[name] = {"n_rows": int(len(d)), "features": rows}
    return out


def families(train, folds=3):
    out = {}
    g = qgroup(train["qtype"].values)
    sets = {**FEATURE_GROUPS, "all three": sorted({c for v in FEATURE_GROUPS.values() for c in v})}
    for name in GROUPS:
        d = train[g == name]
        out[name] = {}
        for fam, cols in sets.items():
            clf = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.1, random_state=SEED)
            acc = cross_val_score(clf, d[cols].astype(float), d["hallucination"], groups=d["group_id"], cv=GroupKFold(folds), scoring="accuracy")
            out[name][fam] = round(float(acc.mean()), 4)
    return out


def pipeline(train, test, feature_set, algorithm="svm", C=0.5):
    a = argparse.Namespace(algorithm=algorithm, C=C, feature_set=feature_set)
    log_cols = [c for c in FEATURE_SETS[feature_set][0] if train[c].min() >= 0 and skew(train[c].dropna()) > 1]
    model = build_model(a, log_cols).fit(train, train["hallucination"].values)
    prob = model.predict_proba(test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    y = test["hallucination"].values
    by_q, by_s = slice_metrics(test, pred, prob)
    return {"overall": {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
                        "roc_auc": float(roc_auc_score(y, prob))}, "by_question_type": by_q, "recall_by_strategy": by_s}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/halueval_qa.csv")
    ap.add_argument("--extra", default="data/why_polar_dataset.csv")
    ap.add_argument("--bert", default="model/bert_results.json")
    ap.add_argument("--out", default="model/qtype_study.json")
    ap.add_argument("--experiment", default="question-type-study")
    ap.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    df, train, test = load_split(a.data, a.extra)
    ext = pd.read_csv(a.extra)
    comp = ext.assign(strategy=ext["strategy"].str.split(":").str[0]).groupby(["qtype", "source", "strategy"]).size().reset_index(name="rows")
    pipes = [
        {"key": "A", "name": "First deployment", "features": "v1 (16 features)", "train_data": "HaluEval-QA only",
         **pipeline(train[train["in_base"] == 1], test, "v1")},
        {"key": "B", "name": "+ WHY/POLAR data", "features": "v1 (16 features)", "train_data": "HaluEval-QA + WHY/POLAR",
         **pipeline(train, test, "v1")},
        {"key": "C", "name": "+ new features", "features": "v2 (25 features + question-type interactions)", "train_data": "HaluEval-QA + WHY/POLAR",
         **pipeline(train, test, "v2")},
    ]
    bert = json.load(open(a.bert)) if os.path.exists(a.bert) else None
    study = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "audit": audit(a.data),
             "dataset": {"file": a.extra, "rows": int(len(ext)), "questions": int(ext["group_id"].nunique()),
                         "composition": comp.to_dict("records"),
                         "split": {"train_rows": int(len(train)), "test_rows": int(len(test)),
                                   "test_groups": pd.Series(qgroup(test["qtype"].values)).value_counts().to_dict()}},
             "feature_families": FEATURE_GROUPS, "contrasts": contrasts(train), "family_accuracy": families(train),
             "pipelines": pipes, "pipeline_algorithm": "calibrated linear SVM, C = 0.5 (same algorithm for A, B and C)", "bert": bert}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(study, open(a.out, "w"), indent=1)
    if not a.no_mlflow:
        mlflow.set_tracking_uri(a.tracking_uri); mlflow.set_experiment(a.experiment)
        for p in pipes:
            with mlflow.start_run(run_name=f"pipeline-{p['key']}"):
                mlflow.log_params({"pipeline": p["key"], "features": p["features"], "train_data": p["train_data"], "algorithm": "svm", "C": 0.5})
                mlflow.log_metrics({"test_accuracy": p["overall"]["accuracy"], "test_f1": p["overall"]["f1"], "test_roc_auc": p["overall"]["roc_auc"],
                                    **{f"test_{k}_{q.lower()}": v[k] for q, v in p["by_question_type"].items() for k in ("accuracy", "f1", "roc_auc")}})
        with mlflow.start_run(run_name="feature-families"):
            mlflow.log_metrics({f"cv_acc_{q.lower()}_{f.replace(' ', '_')}": v for q, d in study["family_accuracy"].items() for f, v in d.items()})
            mlflow.log_dict(study["contrasts"], "feature_contrasts.json")
    print("question types (v1 -> v2):", {r["type"]: (r["v1"], r["v2"]) for r in study["audit"]["rows"]})
    for p in pipes:
        print(f"{p['key']} {p['name']:18s} overall={p['overall']['accuracy']:.3f}  " + "  ".join(f"{q}={v['accuracy']:.3f}" for q, v in p["by_question_type"].items()))
        print("     recall by strategy:", {k: round(v["recall"], 3) for k, v in p["recall_by_strategy"].items()})
    if bert:
        print(f"BERT ({bert['n_test_rows']} test rows) overall={bert['metrics']['test_accuracy']:.3f}  " + "  ".join(f"{q}={v['accuracy']:.3f}" for q, v in bert["by_question_type"].items()))
    print("family accuracy:", json.dumps(study["family_accuracy"]))
    for q in GROUPS:
        print(q, [(r["feature"], r["auc"], r["cohen_d"]) for r in study["contrasts"][q]["features"][:6]])


if __name__ == "__main__":
    main()
