"""Training script with MLflow experiment tracking (SQLite backend).

Examples
  python train.py                                   # default: gradient boosting
  python train.py --algorithm rf --n_estimators 300 --max_depth 16
  python train.py --algorithm logreg --C 10
  python train.py --sweep                           # 10 tracked runs, 6 algorithms
Hyper-parameters are CLI arguments. Every run logs params + metrics to MLflow, registers the model in the
MLflow Model Registry and, if it has the best validation (grouped CV) F1 so far, is saved to model/model.joblib.
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

from features import BIN_COLS, LEN_CATEGORIES, NOM_COLS, NUM_COLS, ORD_COL, add_features

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
    p.add_argument("--cv-folds", type=int, default=3, help="grouped CV folds used for model selection (>=2)")
    p.add_argument("--experiment", default="hallucination-detection")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--force-save", action="store_true", help="overwrite model/model.joblib even if not better")
    p.add_argument("--no-save", action="store_true", help="track in MLflow only")
    p.add_argument("--no-register", action="store_true", help="skip MLflow Model Registry")
    p.add_argument("--sweep", action="store_true", help="run the 10 preset configurations")
    return p.parse_args()


def load_split(path):
    df = pd.read_csv(path)
    assert {"group_id", "question", "knowledge", "answer", "hallucination"} <= set(df.columns), "unexpected columns"
    # Split BEFORE anything is learned. Grouped by question: both answers of a question stay on the same side.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED).split(df, df["hallucination"], groups=df["group_id"]))
    train = add_features(df.iloc[tr].reset_index(drop=True))
    test = add_features(df.iloc[te].reset_index(drop=True))
    return df, train, test


def make_pre(log_cols, text):
    plain_cols = [c for c in NUM_COLS if c not in log_cols]
    log_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("sc", StandardScaler())])
    plain_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    ord_pipe = Pipeline([("ord", OrdinalEncoder(categories=[LEN_CATEGORIES])), ("sc", StandardScaler())])
    parts = [("log", log_pipe, log_cols), ("plain", plain_pipe, plain_cols), ("bin", "passthrough", BIN_COLS),
             ("ord", ord_pipe, ORD_COL), ("nom", OneHotEncoder(handle_unknown="ignore"), NOM_COLS)]
    if text:
        parts.append(("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=3000, sublinear_tf=True), "answer"))
    return ColumnTransformer(parts, sparse_threshold=0.3 if text else 0.0)


def build_model(a, log_cols):
    alg = a.algorithm
    if alg == "logreg":
        return Pipeline([("pre", make_pre(log_cols, True)), ("clf", LogisticRegression(C=a.C, max_iter=2000, solver="liblinear"))])
    if alg == "svm":   # calibrated so the API can return probabilities
        return Pipeline([("pre", make_pre(log_cols, True)), ("clf", CalibratedClassifierCV(LinearSVC(C=a.C, max_iter=5000), cv=3))])
    if alg == "nb":
        return Pipeline([("pre", make_pre(log_cols, False)), ("pca", PCA(n_components=0.95, random_state=SEED)), ("clf", GaussianNB(var_smoothing=a.var_smoothing))])
    if alg == "knn":
        return Pipeline([("pre", make_pre(log_cols, False)), ("pca", PCA(n_components=0.95, random_state=SEED)),
                         ("clf", KNeighborsClassifier(n_neighbors=a.n_neighbors, weights="distance"))])
    if alg == "rf":
        return Pipeline([("pre", make_pre(log_cols, False)),
                         ("clf", RandomForestClassifier(n_estimators=a.n_estimators, max_depth=a.max_depth, min_samples_leaf=a.min_samples_leaf, random_state=SEED, n_jobs=-1))])
    return Pipeline([("pre", make_pre(log_cols, False)),
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
                    **{m: round(float(r[f"metrics.{m}"]), 4) for m in ["cv_f1", "test_accuracy", "test_f1", "test_roc_auc", "train_seconds"]}})
    os.makedirs("model", exist_ok=True)
    json.dump(out, open("model/experiments.json", "w"), indent=1)


def export_examples(test, n_questions=12):
    """A few HELD-OUT question pairs (never used for training) for the UI's 'try a real example' button."""
    gids = test["group_id"].drop_duplicates().sample(n_questions, random_state=SEED)
    rows = test[test["group_id"].isin(gids)]
    ex = [{"question": r.question, "knowledge": None if pd.isna(r.knowledge) else r.knowledge, "answer": r.answer,
           "truth": "hallucinated" if r.hallucination == 1 else "faithful"} for r in rows.itertuples()]
    json.dump(ex, open("model/examples.json", "w"), indent=1)


def train_stats(train):
    cols = NUM_COLS + BIN_COLS
    return {c: {"mean": float(train[c].mean()), "std": float(train[c].std() or 1.0)} for c in cols}


def run(a, df, train, test):
    # skewed non-negative columns are decided from TRAIN only, then frozen
    log_cols = [c for c in NUM_COLS if train[c].min() >= 0 and skew(train[c].dropna()) > 1]
    model = build_model(a, log_cols)
    ytr, yte = train["hallucination"].values, test["hallucination"].values
    hp = {k: getattr(a, k) for k in HPARAMS[a.algorithm]}
    run_name = a.algorithm + "-" + "-".join(f"{k}{v}" for k, v in hp.items())
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=run_name) as r:
        mlflow.log_params({"algorithm": a.algorithm, **hp, "n_rows": len(df), "test_size": 0.2, "split": "GroupShuffleSplit",
                           "seed": SEED, "cv_folds": a.cv_folds, "log1p_cols": len(log_cols)})
        mlflow.set_tags({"task": "hallucination-detection", "dataset": "HaluEval-QA", "sklearn": sklearn.__version__})
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
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
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
                joblib.dump(model, MODEL_PATH, compress=3)
                meta = {"algorithm": a.algorithm, "params": hp, "run_id": r.info.run_id, "run_name": run_name,
                        "registry_version": version, "metrics": {k: float(v) for k, v in m.items()},
                        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "sklearn": sklearn.__version__, "python": platform.python_version(), "n_train_rows": len(train),
                        "n_test_rows": len(test), "confusion_matrix": confusion_matrix(yte, pred).tolist(),
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
          f"auc={m['test_roc_auc']:.4f}  {'[saved -> ' + MODEL_PATH + ']' if saved else ''}")
    return m


def main():
    a = parse_args()
    df, train, test = load_split(a.data)
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
