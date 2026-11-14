"""Download HaluEval-QA and build data/halueval_qa.csv (one row per answer).

Source: Li et al., "HaluEval" (EMNLP 2023), https://github.com/RUCAIBox/HaluEval  (data/qa_data.json)
Steps (identical to the mid-term notebook):
  1. sample N_ITEMS questions (seed 42), 2. one faithful (label 0) + one hallucinated (label 1) row per question,
  3. shuffle, 4. SIMULATE an evidence-retrieval failure: blank `knowledge` for 5 % of the questions (seed 42).
Run:  python download_data.py [--n-items 6000]
"""
import argparse, json, os, urllib.request
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=6000)
    ap.add_argument("--out", default="data/halueval_qa.csv")
    ap.add_argument("--missing-frac", type=float, default=0.05)
    a = ap.parse_args()
    with urllib.request.urlopen(URL, timeout=120) as r:
        rows = [json.loads(l) for l in r.read().decode("utf-8").splitlines() if l.strip()]
    raw = pd.DataFrame(rows).reset_index(drop=True).rename_axis("group_id").reset_index()
    raw = raw.sample(n=min(a.n_items, len(raw)), random_state=SEED)
    faithful = raw.assign(answer=raw["right_answer"], hallucination=0)
    halluc = raw.assign(answer=raw["hallucinated_answer"], hallucination=1)
    df = pd.concat([faithful, halluc])[["group_id", "question", "knowledge", "answer", "hallucination"]]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    groups = df["group_id"].unique()
    miss = rng.choice(groups, size=int(a.missing_frac * len(groups)), replace=False)
    df.loc[df["group_id"].isin(miss), "knowledge"] = np.nan
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"saved {a.out}: {df.shape}, missing evidence rows: {df['knowledge'].isna().sum()}")


if __name__ == "__main__":
    main()
