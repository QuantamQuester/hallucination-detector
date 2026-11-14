"""Build the WHY / POLAR research dataset: data/why_polar_dataset.csv (+ data/why_polar_features.csv).

Why this exists
  HaluEval-QA comes from HotpotQA, which is almost entirely entity questions: of its 10,000 questions only 4 are
  WHY questions, so question-type effects cannot be studied on it alone. This script builds an extension:

  POLAR (yes/no)  every polar question of the full HaluEval-QA file. Both answers are the originals:
                  the right answer and the ChatGPT-generated hallucinated answer.            strategy = chatgpt
  WHY             (a) the few WHY questions of HaluEval-QA, same as above                    strategy = chatgpt
                  (b) WikiWhy (Ho et al., ICLR 2023): "why" questions with a Wikipedia passage and the annotated
                      cause. WikiWhy has NO hallucinated answers, so one is CONSTRUCTED per question with three
                      rule-based strategies that mirror how explanations go wrong:
                        foreign_cause        the cause of the most similar other question (invented cause,
                                             plausible topic, not in the evidence)
                        evidence_distractor  another clause of the SAME passage that is not the cause
                                             (grounded wording, wrong reason)
                        altered_fact         the true cause with one fact changed (number, name, or polarity)

  The constructed answers are NOT model generations. They are labelled per strategy so every result can be
  reported separately for real (chatgpt) and constructed hallucinations.

Sources
  https://github.com/RUCAIBox/HaluEval  (MIT)      https://github.com/matt-seb-ho/WikiWhy  (MIT)
Run:  python build_why_polar.py [--n-why 2000]
"""
import argparse, io, json, os, re, urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features import FEATURE_GROUPS, add_features, cset, qtype, toks

HALUEVAL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
WIKIWHY = "https://raw.githubusercontent.com/matt-seb-ho/WikiWhy/main/dataset/v1.2/"
WW_FILES = {"context.json": ["ctx", "title", "topic"], "question.json": ["question"], "cause.json": ["cause"], "effect.json": ["effect"]}
WW_OFFSET = 100_000                    # group_id of a WikiWhy row = 100000 + its index (HaluEval ids are < 10000)
SEED = 42
ANTONYMS = [("increase", "decrease"), ("increased", "decreased"), ("more", "less"), ("high", "low"), ("higher", "lower"),
            ("large", "small"), ("larger", "smaller"), ("before", "after"), ("success", "failure"), ("successful", "unsuccessful"),
            ("popular", "unpopular"), ("positive", "negative"), ("strong", "weak"), ("rise", "fall"), ("win", "lose"),
            ("won", "lost"), ("good", "poor"), ("early", "late"), ("supported", "opposed"), ("support", "oppose"),
            ("able", "unable"), ("many", "few"), ("first", "last"), ("best", "worst"), ("long", "short"), ("hot", "cold"),
            ("heavy", "light"), ("fast", "slow"), ("rich", "poor"), ("north", "south"), ("east", "west"), ("old", "new")]
ANT = {a: b for a, b in ANTONYMS} | {b: a for a, b in ANTONYMS}
NEGATE = [(r"\bwas not\b", "was"), (r"\bwere not\b", "were"), (r"\bdid not\b", "did"), (r"\bis not\b", "is"),
          (r"\bcould not\b", "could"), (r"\bwas\b", "was not"), (r"\bwere\b", "were not"), (r"\bis\b", "is not"),
          (r"\bare\b", "are not"), (r"\bhad\b", "had not"), (r"\bcould\b", "could not"), (r"\bwould\b", "would not")]
CLAUSE = re.compile(r"(?<=[.!?;])\s+|,\s+(?:and|but|which|while|although|as|so)\s+|\s+\(|\)\s*")
LEAD = {"is", "are", "was", "were", "has", "have", "had", "did", "does", "of", "in", "to", "for", "with", "by", "than"}
CAUSAL = re.compile(r"\b(because|due to|since|as a result|so that|led to|caused|in order to|therefore|thus|owing to)\b", re.I)


def fetch(url):
    with urllib.request.urlopen(url, timeout=180) as r:
        return r.read().decode("utf-8")


def load_halueval(path=None):
    text = open(path, encoding="utf-8").read() if path else fetch(HALUEVAL)
    df = pd.DataFrame([json.loads(l) for l in text.splitlines() if l.strip()])
    return df.reset_index(drop=True).rename_axis("group_id").reset_index()


def load_wikiwhy(directory=None):
    df = pd.DataFrame()
    for name, cols in WW_FILES.items():
        raw = open(os.path.join(directory, name), encoding="utf-8").read() if directory else fetch(WIKIWHY + name)
        df[cols] = pd.read_json(io.StringIO(raw))[cols]
    for c in ["ctx", "question", "cause", "effect"]:
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["question"].map(qtype) == "why"]
    df = df[(df["cause"].str.split().str.len() >= 2) & (df["ctx"].str.split().str.len().between(25, 400))]
    return df.drop_duplicates("question").reset_index().rename(columns={"index": "ww_id"})


def match_style(text, like):
    """Give a constructed answer the same surface style as the true cause (case of first letter, final full stop)."""
    text = text.strip().strip(",;:")
    if not text:
        return text
    if like[:1].islower() and not re.match(r"[A-Z][a-z]*\s+[A-Z]", text) and not text[:2].isupper():
        text = text[0].lower() + text[1:]
    elif like[:1].isupper():
        text = text[0].upper() + text[1:]
    text = text.rstrip(".")
    return text + "." if like.rstrip().endswith(".") else text


def overlap(a, b):
    A, B = cset(a), cset(b)
    return len(A & B) / len(A) if A else 0.0


def foreign_cause(i, ww, sims):
    """Cause of the most similar question from ANOTHER article that the passage does not support."""
    order = np.argsort(-sims)
    for j in order[1:40]:
        o = ww.iloc[j]
        if o["title"] != ww.iloc[i]["title"] and overlap(o["cause"], ww.iloc[i]["ctx"]) < 0.5 and overlap(o["cause"], ww.iloc[i]["cause"]) < 0.5:
            return match_style(o["cause"], ww.iloc[i]["cause"])
    return None


def evidence_distractor(r, rng):
    """A clause of the same passage that is neither the cause nor the effect, as close in length to the cause as possible."""
    n = len(r["cause"].split())
    cands = []
    for c in CLAUSE.split(r["ctx"]):
        c = re.sub(r"^(?:(?:and|but|which|while|although|as|so|that|or|who|where|when)\s+)+", "", c.strip(), flags=re.I) if c else ""
        w = c.split()
        if not (4 <= len(w) <= max(8, 2 * n + 4)) or CAUSAL.search(c) or not re.match(r"[A-Za-z]", c) or not re.search(r"[a-z]{4,}", c):
            continue
        if overlap(c, r["cause"]) < 0.34 and overlap(r["cause"], c) < 0.34 and overlap(c, r["effect"]) < 0.6:
            cands.append((abs(len(w) - n) + rng.random(), c))
    cands = [(d, c) for d, c in cands if c.split()[0].lower() not in LEAD and not c.endswith((" and", " or", " the", " of", ","))] or cands
    return match_style(min(cands)[1], r["cause"]) if cands else None


def altered_fact(r, rng):
    """The true cause with exactly one fact changed: a number, a named entity, an antonym, or the polarity."""
    cause = r["cause"]
    nums = re.findall(r"\d+", cause)
    if nums:
        n = nums[int(rng.integers(len(nums)))]
        new = str(int(n) + int(rng.choice([-7, -3, -2, 2, 3, 5, 9]))) if len(n) != 4 else str(int(n) + int(rng.choice([-12, -6, -3, 4, 8, 15])))
        return re.sub(rf"\b{n}\b", new.lstrip("-") or "1", cause, count=1), "number"
    ents = [m.group(0) for m in re.finditer(r"\b[A-Z][a-z]{2,}(?: [A-Z][a-z]+)*", cause)]
    ents = [e for e in ents if " " in e or not re.match(re.escape(e), cause)]     # a lone capital at the start is not a name
    pool = sorted({m for m in re.findall(r"(?<=[a-z,] )[A-Z][a-z]{2,}(?: [A-Z][a-z]+)*", r["ctx"])} - set(ents))
    pool = [p for p in pool if p not in cause and all(p not in e and e not in p for e in ents)]
    if ents and pool:
        e = ents[int(rng.integers(len(ents)))]
        same = [p for p in pool if len(p.split()) == len(e.split())] or pool       # swap like for like
        return cause.replace(e, same[int(rng.integers(len(same)))], 1), "entity"
    words = cause.split()
    idx = [k for k, w in enumerate(words) if w.lower().strip(".,") in ANT]
    if idx:
        k = idx[int(rng.integers(len(idx)))]
        core = words[k].lower().strip(".,")
        rep = ANT[core]
        words[k] = words[k].lower().replace(core, rep) if words[k][0].islower() else words[k].lower().replace(core, rep).capitalize()
        return " ".join(words), "antonym"
    for pat, rep in NEGATE:
        if re.search(pat, cause):
            return re.sub(pat, rep, cause, count=1), "polarity"
    return None, None


def build_why(ww, n_why):
    rng = np.random.default_rng(SEED)
    # sample evenly over topics so that no single domain dominates
    pick = (ww.groupby("topic", group_keys=False)[ww.columns.tolist()]
              .apply(lambda g: g.sample(n=min(len(g), int(np.ceil(n_why * len(g) / len(ww))) + 1), random_state=SEED)))
    pick = pick.sample(n=min(n_why, len(pick)), random_state=SEED)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(ww["question"])
    M = vec.transform(ww["question"])
    pos = {wid: p for p, wid in enumerate(ww["ww_id"])}
    rows, order = [], ["foreign_cause", "evidence_distractor", "altered_fact"]
    for k, (_, r) in enumerate(pick.iterrows()):
        i = pos[r["ww_id"]]
        made = {}
        for strat in order[k % 3:] + order[:k % 3]:              # round-robin, with fall-back to the next strategy
            if strat == "foreign_cause":
                ans, detail = foreign_cause(i, ww, (M @ M[i].T).toarray().ravel()), ""
            elif strat == "evidence_distractor":
                ans, detail = evidence_distractor(r, rng), ""
            else:
                ans, detail = altered_fact(r, rng)
            if ans and ans.strip().lower().rstrip(".") != r["cause"].strip().lower().rstrip("."):
                made = {"answer": ans, "strategy": strat + (f":{detail}" if detail else "")}
                break
        if not made:
            continue
        base = {"group_id": WW_OFFSET + int(r["ww_id"]), "question": r["question"], "knowledge": r["ctx"],
                "source": "WikiWhy", "topic": r["topic"]}
        rows.append({**base, "answer": r["cause"], "hallucination": 0, "strategy": "gold_cause"})
        rows.append({**base, "answer": made["answer"], "hallucination": 1, "strategy": made["strategy"]})
    return pd.DataFrame(rows)


def build_halueval_part(h):
    h = h.assign(qtype=h["question"].map(qtype))
    part = h[h["qtype"].isin(["why", "polar"])]
    base = part[["group_id", "question", "knowledge"]].assign(source="HaluEval-QA", topic="")
    return pd.concat([base.assign(answer=part["right_answer"], hallucination=0, strategy="gold_answer"),
                      base.assign(answer=part["hallucinated_answer"], hallucination=1, strategy="chatgpt")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-why", type=int, default=2000, help="number of WikiWhy questions to use")
    ap.add_argument("--halueval", default=None, help="local qa_data.json (default: download)")
    ap.add_argument("--wikiwhy", default=None, help="local WikiWhy dataset/v1.2 directory (default: download)")
    ap.add_argument("--out", default="data/why_polar_dataset.csv")
    a = ap.parse_args()
    df = pd.concat([build_halueval_part(load_halueval(a.halueval)), build_why(load_wikiwhy(a.wikiwhy), a.n_why)], ignore_index=True)
    df["qtype"] = df["question"].map(qtype).str.upper()
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df = df[["group_id", "question", "qtype", "knowledge", "answer", "hallucination", "source", "strategy", "topic"]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    # the same rows with every engineered feature, grouped as lexical / semantic / contextual
    feats = add_features(df.drop(columns=["qtype"]))
    cols = [c for g in ("lexical", "semantic", "contextual") for c in FEATURE_GROUPS[g]]
    out = feats[["group_id", "question", "knowledge", "answer", "hallucination", "source", "strategy"]].copy()
    out.insert(2, "question_type", df["qtype"])
    for g in ("lexical", "semantic", "contextual"):
        for c in FEATURE_GROUPS[g]:
            out[f"{g[:3]}_{c}"] = feats[c].round(4)
    out.to_csv(a.out.replace("_dataset.csv", "_features.csv"), index=False)
    print(f"saved {a.out}: {df.shape}")
    print(df.groupby(["qtype", "source", "strategy"]).size().to_string())
    print("feature file columns:", len(out.columns), "| engineered features:", len(cols))


if __name__ == "__main__":
    main()
