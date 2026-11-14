"""Stateless feature engineering shared by train.py and app.py.

Nothing in this module learns from data (fixed rules and fixed bin edges), so it is safe to apply
before the train/test split and identically at prediction time.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP = set(ENGLISH_STOP_WORDS)          # deterministic: no NLTK corpus download needed
_ps = PorterStemmer()

NUM_COLS = ["ans_wc", "ans_chars", "avg_wlen", "q_wc", "ctx_wc", "len_ratio", "ctx_support", "jaccard_ak",
            "novel_content", "ans_in_q", "n_numbers", "unsup_numbers", "novel_caps", "negations"]
BIN_COLS = ["has_number", "kn_missing"]
ORD_COL = ["ans_len_cat"]
NOM_COLS = ["qtype", "support_bin"]
EVIDENCE_COLS = ["ctx_wc", "len_ratio", "ctx_support", "jaccard_ak", "novel_content", "unsup_numbers", "novel_caps"]
SUPPORT_EDGES = [-0.001, 0.10, 0.90, 1.001]
SUPPORT_LABELS = ["ungrounded", "partial", "full"]
LEN_CATEGORIES = ["short", "medium", "long"]

TOK = re.compile(r"[a-z0-9]+")
NEG = {"not", "no", "never", "neither", "nor", "none", "cannot", "isnt", "wasnt", "didnt", "doesnt", "dont", "wont"}
POLAR = {"is", "are", "was", "were", "did", "does", "do", "has", "have", "had", "can", "could", "will", "would", "should"}
WH = {"who", "whom", "whose", "what", "which", "when", "where", "how", "why"}


@lru_cache(maxsize=None)
def stem(w):
    return _ps.stem(w)


def toks(s):
    return TOK.findall(str(s).lower().replace("'", ""))


def cset(s):
    return {stem(t) for t in toks(s) if t not in STOP}


def qtype(q):
    t = toks(q)
    for w in t[:4]:
        if w in WH:
            return "who" if w in {"who", "whom", "whose"} else w
    if t and t[0] in POLAR:
        return "polar"
    return "other"


def row_features(q, k, a):
    kn_ok = isinstance(k, str) and k.strip() != ""
    A, Q = cset(a), cset(q)
    at = toks(a)
    f = {"ans_wc": len(at), "ans_chars": len(str(a)), "avg_wlen": float(np.mean([len(t) for t in at])) if at else 0.0,
         "q_wc": len(toks(q)), "ans_in_q": len(A & Q) / len(A) if A else 0.0, "negations": sum(t in NEG for t in at)}
    nums = re.findall(r"\d+", str(a))
    f["n_numbers"] = len(nums)
    f["has_number"] = int(len(nums) > 0)
    caps = {w.lower() for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", str(a))}
    if kn_ok:
        K = cset(k); kt = set(toks(k)); qt = set(toks(q))
        f["ctx_wc"] = len(toks(k))
        f["len_ratio"] = len(at) / (f["ctx_wc"] + 1)
        f["ctx_support"] = len(A & K) / len(A) if A else 0.0
        f["jaccard_ak"] = len(A & K) / len(A | K) if (A | K) else 0.0
        f["novel_content"] = len(A - K - Q)
        f["unsup_numbers"] = sum(n not in kt and n not in qt for n in nums)
        f["novel_caps"] = sum((c.replace("'", "") not in kt) and (c not in qt) for c in caps)
    else:                                    # evidence missing -> NaN, imputed by the pipeline with TRAIN medians
        for c in EVIDENCE_COLS:
            f[c] = np.nan
    f["kn_missing"] = int(not kn_ok)
    return f


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame([row_features(q, k, a) for q, k, a in zip(d["question"], d["knowledge"], d["answer"])], index=d.index)
    out = pd.concat([d, feats], axis=1)
    out["qtype"] = out["question"].map(qtype)
    out["ans_len_cat"] = pd.cut(out["ans_wc"], bins=[-1, 3, 10, np.inf], labels=LEN_CATEGORIES).astype(str)
    b = pd.cut(out["ctx_support"], bins=SUPPORT_EDGES, labels=SUPPORT_LABELS).astype(object)
    out["support_bin"] = b.where(out["ctx_support"].notna(), "unknown")
    return out


def make_frames(rows) -> pd.DataFrame:
    """Build the model input for MANY requests. rows = iterable of (question, knowledge, answer)."""
    rows = list(rows)
    kn = [k if (isinstance(k, str) and k.strip()) else np.nan for _, k, _ in rows]
    return add_features(pd.DataFrame({"question": [q for q, _, _ in rows], "knowledge": pd.Series(kn, dtype=object),
                                      "answer": [a for _, _, a in rows]}))


def make_frame(question, knowledge, answer) -> pd.DataFrame:
    """Build the model input for ONE request."""
    return make_frames([(question, knowledge, answer)])
