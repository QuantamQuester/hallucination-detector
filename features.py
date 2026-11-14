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

# ---- feature set v1 (mid-term / first deployment) -------------------------------------------------------------
V1_NUM_COLS = ["ans_wc", "ans_chars", "avg_wlen", "q_wc", "ctx_wc", "len_ratio", "ctx_support", "jaccard_ak",
               "novel_content", "ans_in_q", "n_numbers", "unsup_numbers", "novel_caps", "negations"]
V1_BIN_COLS = ["has_number", "kn_missing"]
V1_NOM_COLS = ["qtype_v1", "support_bin"]
# ---- feature set v2 (question-type study): adds lexical, semantic and contextual signals --------------------------
LEXICAL_NEW = ["causal_markers", "hedges", "ttr"]                                  # surface form of the answer
SEMANTIC_NEW = ["soft_overlap", "q_ctx_overlap"]                                   # fuzzy meaning match
CONTEXTUAL_NEW = ["best_sent_support", "span_support"]                             # consistency with the evidence
NUM_COLS = V1_NUM_COLS + LEXICAL_NEW + SEMANTIC_NEW + CONTEXTUAL_NEW
BIN_COLS = V1_BIN_COLS + ["evid_causal", "neg_mismatch"]
ORD_COL = ["ans_len_cat"]
NOM_COLS = ["qtype", "support_bin", "polarity"]
# question-type interactions: let a linear model weigh a signal differently for WHY and for POLAR questions
INT_BASE = ["ans_wc", "ctx_support", "best_sent_support", "span_support", "novel_content"]
INT_COLS = [f"{t}_x_{c}" for t in ("why", "polar") for c in INT_BASE]
EVIDENCE_COLS = ["ctx_wc", "len_ratio", "ctx_support", "jaccard_ak", "novel_content", "unsup_numbers", "novel_caps",
                 "soft_overlap", "q_ctx_overlap", "best_sent_support", "span_support"]
FEATURE_GROUPS = {   # the three families used in the question-type study
    "lexical": ["ans_wc", "ans_chars", "avg_wlen", "n_numbers", "has_number", "negations", "causal_markers", "hedges", "ttr", "q_wc"],
    "semantic": ["ans_in_q", "jaccard_ak", "soft_overlap", "q_ctx_overlap"],
    "contextual": ["ctx_support", "best_sent_support", "span_support", "novel_content", "unsup_numbers", "novel_caps",
                   "len_ratio", "ctx_wc", "kn_missing", "evid_causal", "neg_mismatch"],
}
SUPPORT_EDGES = [-0.001, 0.10, 0.90, 1.001]
SUPPORT_LABELS = ["ungrounded", "partial", "full"]
LEN_CATEGORIES = ["short", "medium", "long"]

TOK = re.compile(r"[a-z0-9]+")
NEG = {"not", "no", "never", "neither", "nor", "none", "cannot", "isnt", "wasnt", "didnt", "doesnt", "dont", "wont"}
POLAR = {"is", "are", "was", "were", "did", "does", "do", "has", "have", "had", "can", "could", "will", "would", "should"}
WH = {"who", "whom", "whose", "what", "which", "when", "where", "how", "why"}


CAUSAL = {"because", "due", "since", "caused", "cause", "causes", "result", "resulted", "led", "therefore", "thus",
          "owing", "reason", "hence", "consequently"}
HEDGE = {"may", "might", "possibly", "probably", "likely", "perhaps", "could", "appears", "seems", "suggests",
         "unclear", "approximately", "reportedly", "allegedly", "maybe"}
_CAUSAL_PHRASE = re.compile(r"\b(because|due to|since|as a result|so that|led to|caused|causing|in order to|therefore|"
                            r"thus|owing to|thanks to|result(?:ed|ing)? (?:in|from)|reason)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")
_WHO = {"who", "whom", "whose"}


@lru_cache(maxsize=None)
def stem(w):
    return _ps.stem(w)


def toks(s):
    return TOK.findall(str(s).lower().replace("'", ""))


def cset(s):
    return {stem(t) for t in toks(s) if t not in STOP}


def qtype_v1(q):
    """First version (kept for the before/after comparison): wh-word in the first 4 tokens, else auxiliary-initial."""
    t = toks(q)
    for w in t[:4]:
        if w in WH:
            return "who" if w in _WHO else w
    if t and t[0] in POLAR:
        return "polar"
    return "other"


def qtype(q):
    """Question type, v2. HotpotQA-style questions often keep the wh-word at the END ("... was born in which year?"),
    which v1 filed under 'other'. Order of the rules:
      1. 'why' among the first 4 tokens           -> why    (explanation-seeking)
      2. the question starts with an auxiliary     -> polar  (yes/no, including "Is X older or younger than Y?")
      3. another wh-word among the first 4 tokens  -> that word
      4. otherwise the LAST wh-word in the question (wh-in-situ)
    """
    t = toks(q)
    if not t:
        return "other"
    if "why" in t[:4]:
        return "why"
    if t[0] in POLAR:
        return "polar"
    for w in t[:4]:
        if w in WH:
            return "who" if w in _WHO else w
    for w in reversed(t):
        if w in WH:
            return "who" if w in _WHO else w
    return "other"


def polarity(a):
    """Does the answer open with an explicit yes / no?"""
    t = toks(a)
    return "yes" if t[:1] == ["yes"] else "no" if t[:1] == ["no"] else "none"


def _trigrams(s):
    s = " " + " ".join(toks(s)) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _longest_run(at, kt_list):
    """Length of the longest run of consecutive answer tokens that appears verbatim in the evidence."""
    if not at or not kt_list:
        return 0
    text = " " + " ".join(kt_list) + " "
    best, i, n = 0, 0, len(at)
    while i < n and n - i > best:
        j = i + best + 1
        if (" " + " ".join(at[i:j]) + " ") in text:
            while j < n and (" " + " ".join(at[i:j + 1]) + " ") in text:
                j += 1
            best = j - i
        i += 1
    return best


def row_features(q, k, a):
    kn_ok = isinstance(k, str) and k.strip() != ""
    A, Q = cset(a), cset(q)
    at = toks(a)
    f = {"ans_wc": len(at), "ans_chars": len(str(a)), "avg_wlen": float(np.mean([len(t) for t in at])) if at else 0.0,
         "q_wc": len(toks(q)), "ans_in_q": len(A & Q) / len(A) if A else 0.0, "negations": sum(t in NEG for t in at)}
    nums = re.findall(r"\d+", str(a))
    f["n_numbers"] = len(nums)
    f["has_number"] = int(len(nums) > 0)
    f["causal_markers"] = sum(t in CAUSAL for t in at)
    f["hedges"] = sum(t in HEDGE for t in at)
    f["ttr"] = len(set(at)) / len(at) if at else 0.0
    f["evid_causal"] = 0
    f["neg_mismatch"] = 0
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
        f["q_ctx_overlap"] = len(Q & K) / len(Q) if Q else 0.0
        f["span_support"] = _longest_run(at, toks(k)) / len(at) if at else 0.0
        # sentence-level view: the single evidence sentence that supports the answer best
        best, best_sent = 0.0, ""
        for s_ in _SENT.split(k.strip()):
            sc = len(A & cset(s_)) / len(A) if A else 0.0
            if sc > best or not best_sent:
                best, best_sent = sc, s_
        f["best_sent_support"] = best
        ta, tb = _trigrams(a), _trigrams(best_sent)
        f["soft_overlap"] = len(ta & tb) / len(ta) if ta else 0.0        # character 3-grams: tolerant to inflection
        f["evid_causal"] = int(bool(_CAUSAL_PHRASE.search(best_sent)))  # is the supporting sentence a causal statement?
        f["neg_mismatch"] = int(any(t in NEG for t in at) != any(t in NEG for t in toks(best_sent)))   # one negates, the other does not
    else:                                    # evidence missing -> NaN, imputed by the pipeline with TRAIN medians
        for c in EVIDENCE_COLS:
            f[c] = np.nan
    f["kn_missing"] = int(not kn_ok)
    return f


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame([row_features(q, k, a) for q, k, a in zip(d["question"], d["knowledge"], d["answer"])], index=d.index)
    out = pd.concat([d, feats], axis=1)
    out["qtype"] = out["question"].map(qtype)
    out["qtype_v1"] = out["question"].map(qtype_v1)
    out["polarity"] = out["answer"].map(polarity)
    for t in ("why", "polar"):
        is_t = (out["qtype"] == t).astype(float)
        for c in INT_BASE:
            out[f"{t}_x_{c}"] = is_t * out[c]          # NaN (missing evidence) propagates and is imputed like its base
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
