"""Training script with MLflow experiment tracking (SQLite backend).

Examples
  python train.py                                   # default: gradient boosting
  python train.py --algorithm rf --n_estimators 300 --max_depth 16
  python train.py --algorithm logreg --C 10
  python train.py --sweep                           # 10 tracked runs, 6 algorithms
Hyper-parameters are CLI arguments. Every run logs params + metrics to MLflow, registers the model in the
MLflow Model Registry and, if it has the best validation (grouped CV) F1 so far, is saved to model/model.joblib.
Training data = the HaluEval-QA sample + the WHY/POLAR extension (build_why_polar.py); every run also reports
accuracy / F1 / AUC separately for WHY, POLAR and all other questions.
The run table is also exported to model/experiments.json so the serving image can show it without MLflow.
"""
import argparse, json, logging, os, platform, time, warnings
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import skew
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

from features import (BIN_COLS, INT_COLS, LEN_CATEGORIES, NOM_COLS, NUM_COLS, ORD_COL, V1_BIN_COLS, V1_NOM_COLS, V1_NUM_COLS,
                      add_features)

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
SEED = 42
MODEL_PATH = "model/model.joblib"
META_PATH = "model/metadata.json"
REGISTERED_NAME = "hallucination-detector"

SWEEP = [  # (algorithm, hyper-parameters)  -> 10 configurations, 6 algorithms
    ("logreg", {"C": 0.1}), ("logreg", {"C": 1.0}), ("logreg", {"C": 10.0}),
    ("svm", {"C": 0.5}), ("nb", {"var_smoothing": 1e-3}), ("knn", {"n_neighbors": 15}),
    ("rf", {"n_estimators": 200, "max_depth": 8}), ("rf", {"n_estimators": 300, "max_depth": 16}),
    ("gb", {"learning_rate": 0.05, "max_leaf_nodes": 15}), ("gb", {"learning_rate": 0.1, "max_leaf_nodes": 31}),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", default="gb", choices=["logreg", "svm", "nb", "knn", "rf", "gb"])
    p.add_argument("--C", type=float, default=1.0, help="logreg / svm regularisation strength")
    p.add_argument("--n_estimators", type=int, default=200, help="rf")
    p.add_argument("--max_depth", type=int, default=16, help="rf")
    p.add_argument("--min_samples_leaf", type=int, default=1, help="rf")
    p.add_argument("--learning_rate", type=float, default=0.1, help="gb")
    p.add_argument("--max_leaf_nodes", type=int, default=31, help="gb")
    p.add_argument("--n_neighbors", type=int, default=15, help="knn")
    p.add_argument("--var_smoothing", type=float, default=1e-3, help="nb")
    p.add_argument("--data", default="data/halueval_qa.csv")
    p.add_argument("--extra", default="data/why_polar_dataset.csv",
                   help="WHY/POLAR extension built by build_why_polar.py ('' or 'none' = HaluEval-QA only)")
    p.add_argument("--feature-set", default="v2", choices=["v1", "v2"],
                   help="v1 = first deployment (16 features), v2 = + lexical/semantic/contextual features and question-type interactions")
    p.add_argument("--cv-folds", type=int, default=3, help="grouped CV folds used for model selection (>=2)")
    p.add_argument("--experiment", default="hallucination-detection-v2")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--force-save", action="store_true", help="overwrite model/model.joblib even if not better")
    p.add_argument("--no-save", action="store_true", help="track in MLflow only")
    p.add_argument("--no-register", action="store_true", help="skip MLflow Model Registry")
    p.add_argument("--sweep", action="store_true", help="run the 10 preset configurations")
    return p.parse_args()


FEATURE_SETS = {"v1": (V1_NUM_COLS, V1_BIN_COLS, V1_NOM_COLS),
                "v2": (NUM_COLS + INT_COLS, BIN_COLS, NOM_COLS)}
REQUIRED = ["group_id", "question", "knowledge", "answer", "hallucination"]


def load_data(path, extra=None):
    """HaluEval-QA sample (+ the WHY/POLAR extension). Extension rows of questions already in the base file are dropped."""
    df = pd.read_csv(path)
    assert set(REQUIRED) <= set(df.columns), "unexpected columns"
    df = df[REQUIRED].assign(source="HaluEval-QA", strategy=np.where(df["hallucination"] == 1, "chatgpt", "gold_answer"), in_base=1)
    if extra and extra.lower() != "none" and os.path.exists(extra):
        ex = pd.read_csv(extra)
        ex = ex[~ex["group_id"].isin(set(df["group_id"]))]
        df = pd.concat([df, ex[REQUIRED + ["source", "strategy"]].assign(in_base=0)], ignore_index=True)
    return df


def load_split(path, extra=None):
    df = load_data(path, extra)
    # Split BEFORE anything is learned. Grouped by question: both answers of a question stay on the same side.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(df, df["hallucination"], groups=df["group_id"]))
    train = add_features(df.iloc[tr].reset_index(drop=True))
    test = add_features(df.iloc[te].reset_index(drop=True))
    return df, train, test


def qgroup(qtypes):
    """The three groups of the question-type study."""
    return np.where(qtypes == "why", "WHY", np.where(qtypes == "polar", "POLAR", "OTHER"))


def slice_metrics(test, pred, prob):
    """Accuracy / F1 / AUC per question-type group, and recall per way the hallucination was produced."""
    y, g = test["hallucination"].values, qgroup(test["qtype"].values)
    by_q = {}
    for name in ["WHY", "POLAR", "OTHER"]:
        m = g == name
        if m.sum() < 10 or len(set(y[m])) < 2:
            continue
        by_q[name] = {"n": int(m.sum()), "accuracy": float(accuracy_score(y[m], pred[m])), "f1": float(f1_score(y[m], pred[m])),
                      "precision": float(precision_score(y[m], pred[m], zero_division=0)), "recall": float(recall_score(y[m], pred[m])),
                      "roc_auc": float(roc_auc_score(y[m], prob[m]))}
    strat = test["strategy"].str.split(":").str[0].values
    by_s = {}
    for name in sorted(set(strat[y == 1])):
        for qg in ["WHY", "POLAR", "OTHER"]:
            m = (strat == name) & (y == 1) & (g == qg)
            if m.sum() >= 10:
                by_s[f"{qg}/{name}"] = {"n": int(m.sum()), "recall": float(pred[m].mean())}
    return by_q, by_s


def make_pre(log_cols, text, feature_set="v2"):
    num_cols, bin_cols, nom_cols = FEATURE_SETS[feature_set]
    plain_cols = [c for c in num_cols if c not in log_cols]
    log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())])
    plain_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    ord_pipe = Pipeline([("ord", OrdinalEncoder(categories=[LEN_CATEGORIES])), ("sc", StandardScaler())])
    parts = [("log", log_pipe, log_cols), ("plain", plain_pipe, plain_cols), ("bin", "passthrough", bin_cols),
             ("ord", ord_pipe, ORD_COL), ("nom", OneHotEncoder(handle_unknown="ignore"), nom_cols)]
    if text:
        parts.append(("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=3000, sublinear_tf=True), "answer"))
    return ColumnTransformer(parts, sparse_threshold=0.3 if text else 0.0)


def build_model(a, log_cols):
    alg, fs = a.algorithm, getattr(a, "feature_set", "v2")
    if alg == "logreg":
        return Pipeline([("pre", make_pre(log_cols, True, fs)), ("clf", LogisticRegression(C=a.C, max_iter=2000, solver="liblinear"))])
    if alg == "svm":   # calibrated so the API can return probabilities
        return Pipeline([("pre", make_pre(log_cols, True, fs)), ("clf", CalibratedClassifierCV(LinearSVC(C=a.C, max_iter=5000), cv=3))])
    if alg == "nb":
        return Pipeline([("pre", make_pre(log_cols, False, fs)), ("pca", PCA(n_components=0.95, random_state=SEED)), ("clf", GaussianNB(var_smoothing=a.var_smoothing))])
    if alg == "knn":
        return Pipeline([("pre", make_pre(log_cols, False, fs)), ("pca", PCA(n_components=0.95, random_state=SEED)),
                         ("clf", KNeighborsClassifier(n_neighbors=a.n_neighbors, weights="distance"))])
    if alg == "rf":
        return Pipeline([("pre", make_pre(log_cols, False, fs)),
                         ("clf", RandomForestClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth, min_samples_leaf=a.min_samples_leaf, random_state=SEED, n_jobs=-1))])
    return Pipeline([("pre", make_pre(log_cols, False, fs)),
                     ("clf", HistGradientBoostingClassifier(learning_rate=a.learning_rate, max_leaf_nodes=a.max_leaf_nodes, random_state=SEED))])


HPARAMS = {"logreg": ["C"], "svm": ["C"], "nb": ["var_smoothing"], "knn": ["n_neighbors"],
           "rf": ["n_estimators", "max_depth", "min_samples_leaf"], "gb": ["learning_rate", "max_leaf_nodes"]}


def export_experiments(experiment):
    """Write the MLflow runs to model/experiments.json so the (MLflow-free) serving image can show the comparison."""
    runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time ASC"])
    out = []
    for _, r in runs.iterrows():
        alg = r.get("params.algorithm")
        if not isinstance(alg, str) or pd.isna(r.get("metrics.cv_f1")):
            continue
        out.append({"run": r["tags.mlflow.runName"], "algorithm": alg,
                    "params": {k: r[f"params.{k}"] for k in HPARAMS[alg] if isinstance(r.get(f"params.{k}"), str)},
                    **{m: round(float(r[f"metrics.{m}"]), 4) for m in ["cv_f1", "test_accuracy", "test_f1", "test_roc_auc", "train_seconds"]},
                    **{m: round(float(r[f"metrics.{m}"]), 4) for m in ["test_accuracy_why", "test_accuracy_polar", "test_accuracy_other"]
                       if f"metrics.{m}" in r and not pd.isna(r[f"metrics.{m}"])}})
    os.makedirs("model", exist_ok=True)
    json.dump(out, open("model/experiments.json", "w"), indent=1)


def export_examples(test, per_group=(("WHY", 5), ("POLAR", 5), ("OTHER", 6))):
    """A few HELD-OUT question pairs (never used for training) for the UI's 'try a real example' button."""
    g = pd.Series(qgroup(test["qtype"].values), index=test.index)
    gids = []
    for name, n in per_group:
        pool = test.loc[g == name, "group_id"].drop_duplicates()
        gids += pool.sample(min(n, len(pool)), random_state=SEED).tolist()
    rows = test[test["group_id"].isin(gids)]
    ex = [{"question": r.question, "knowledge": None if pd.isna(r.knowledge) else r.knowledge, "answer": r.answer,
           "truth": "hallucinated" if r.hallucination == 1 else "faithful", "question_type": r.qtype,
           "source": r.source, "strategy": r.strategy} for r in rows.itertuples()]
    json.dump(ex, open("model/examples.json", "w"), indent=1)


def train_stats(train):
    cols = NUM_COLS + BIN_COLS
    return {c: {"mean": float(train[c].mean()), "std": float(train[c].std() or 1.0)} for c in cols}


def run(a, df, train, test):
    # skewed non-negative columns are decided from TRAIN only, then frozen
    log_cols = [c for c in FEATURE_SETS[a.feature_set][0] if train[c].min() >= 0 and skew(train[c].dropna()) > 1]
    model = build_model(a, log_cols)
    ytr, yte = train["hallucination"].values, test["hallucination"].values
    hp = {k: getattr(a, k) for k in HPARAMS[a.algorithm]}
    qg_all = np.concatenate([qgroup(train["qtype"].values), qgroup(test["qtype"].values)])
    run_name = a.algorithm + "-" + "-".join(f"{k}{v}" for k, v in hp.items())
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.log_params({"algorithm": a.algorithm, **hp, "n_rows": len(df), "test_size": 0.2, "split": "GroupShuffleSplit",
                           "seed": SEED, "cv_folds": a.cv_folds, "log1p_cols": len(log_cols), "feature_set": a.feature_set,
                           "n_why_rows": int((qg_all == "WHY").sum()), "n_polar_rows": int((qg_all == "POLAR").sum())})
        mlflow.set_tags({"task": "hallucination-detection", "sklearn": sklearn.__version__,
                         "dataset": "HaluEval-QA + WHY/POLAR extension" if (df["source"] != "HaluEval-QA").any() else "HaluEval-QA"})
        t0 = time.time()
        cv = cross_validate(model, train, ytr, groups=train["group_id"].values, cv=GroupKFold(n_splits=a.cv_folds),
                            scoring={"f1": "f1", "acc": "accuracy"}, n_jobs=1)
        cv_f1, cv_acc = float(cv["test_f1"].mean()), float(cv["test_acc"].mean())
        model.fit(train, ytr)
        fit_s = time.time() - t0
        pred = model.predict(test)
        prob = model.predict_proba(test)[:, 1]
        m = {"cv_f1": cv_f1, "cv_acc": cv_acc, "cv_f1_std": float(cv["test_f1"].std()),
             "test_accuracy": accuracy_score(yte, pred), "test_precision": precision_score(yte, pred),
             "test_recall": recall_score(yte, pred), "test_f1": f1_score(yte, pred), "test_roc_auc": roc_auc_score(yte, prob),
             "train_seconds": fit_s}
        by_q, by_s = slice_metrics(test, pred, prob)
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        mlflow.log_metrics({f"test_{k}_{q.lower()}": v[k] for q, v in by_q.items() for k in ("accuracy", "f1", "roc_auc")})
        mlflow.log_dict({"by_question_type": by_q, "recall_by_strategy": by_s}, "question_type_metrics.json")
        mlflow.log_text(classification_report(yte, pred, target_names=["faithful", "hallucinated"], digits=4), "classification_report.txt")
        mlflow.log_dict({"confusion_matrix": confusion_matrix(yte, pred).tolist()}, "confusion_matrix.json")
        version = None
        if not a.no_register:
            try:
                info = mlflow.sklearn.log_model(model, name="model", registered_model_name=REGISTERED_NAME,
                                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
                version = getattr(info, "registered_model_version", None)
            except Exception as e:                       # registry is a convenience, never block training
                print("MLflow registry skipped:", repr(e)[:120])
        saved = False
        if not a.no_save:
            best = json.load(open(META_PATH))["metrics"]["cv_f1"] if os.path.exists(META_PATH) else -1.0
            if a.force_save or cv_f1 > best:
                os.makedirs("model", exist_ok=True)
                tfidf = model.named_steps["pre"].named_transformers_.get("tfidf")
                if tfidf is not None:
                    tfidf.stop_words_ = None                 # pruned vocabulary: megabytes on disk, never used to predict
                joblib.dump(model, MODEL_PATH, compress=3)
                meta = {"algorithm": a.algorithm, "params": hp, "run_id": r.info.run_id, "run_name": run_name,
                        "registry_version": version, "metrics": {k: float(v) for k, v in m.items()},
                        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "sklearn": sklearn.__version__, "python": platform.python_version(), "n_train_rows": len(train),
                        "n_test_rows": len(test), "confusion_matrix": confusion_matrix(yte, pred).tolist(),
                        "feature_set": a.feature_set, "by_question_type": by_q, "recall_by_strategy": by_s,
                        "data": {"sources": df["source"].value_counts().to_dict(),
                                 "question_groups": pd.Series(qg_all).value_counts().to_dict()},
                        "train_stats": train_stats(train)}
                json.dump(meta, open(META_PATH, "w"), indent=2)
                export_examples(test)
                saved = True
                if version is not None:
                    try:
                        mlflow.MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion", str(version))
                    except Exception:
                        pass
        mlflow.set_tag("saved_as_best", str(saved))
    if not a.no_save:
        export_experiments(a.experiment)
    print(f"{run_name:38s} cv_f1={cv_f1:.4f}  test_acc={m['test_accuracy']:.4f}  test_f1={m['test_f1']:.4f}  "
          f"auc={m['test_roc_auc']:.4f}  " + " ".join(f"{q}={v['accuracy']:.3f}" for q, v in by_q.items()) + f"  {'[saved -> ' + MODEL_PATH + ']' if saved else ''}")
    return m


def main():
    a = parse_args()
    df, train, test = load_split(a.data, a.extra)
    print("sources:", df["source"].value_counts().to_dict(), "| test question groups:", pd.Series(qgroup(test["qtype"].values)).value_counts().to_dict())
    print(f"data: {len(df)} rows | train {len(train)} / test {len(test)} (grouped split) | evidence missing: {df['knowledge'].isna().mean():.1%}")
    if a.sweep:
        for alg, hp in SWEEP:
            b = argparse.Namespace(**vars(a)); b.algorithm = alg
            for k, v in hp.items():
                setattr(b, k, v)
            run(b, df, train, test)
    else:
        run(a, df, train, test)


if __name__ == "__main__":
    main()
