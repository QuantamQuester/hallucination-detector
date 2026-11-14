"""Simple data-drift check: compare mean/std of logged request features against the training data.

Usage: python drift_check.py [--log logs/predictions.jsonl] [--min-samples 30]
Flags a feature when its mean moved by more than 0.5 training standard deviations, or its std changed by more than 2x.
Exit code 1 if drift is found (usable in a cron job / CI step). Drift is a signal to inspect and possibly retrain.
The same compute_drift() function powers the live GET /api/drift endpoint of the service.
"""
import argparse, json, math, sys


def compute_drift(rows, stats, mean_shift=0.5):
    """rows: list of logged feature dicts; stats: {feature: {mean, std}} from training. Returns (table, drift_found)."""
    table, drift = [], False
    for c, s in stats.items():
        x = [float(r[c]) for r in rows if r.get(c) is not None and not (isinstance(r[c], float) and math.isnan(r[c]))]
        if len(x) < 2:
            continue
        mean = sum(x) / len(x)
        std = math.sqrt(sum((v - mean) ** 2 for v in x) / (len(x) - 1))
        sd = s["std"] or 1.0
        shift, ratio = abs(mean - s["mean"]) / sd, (std or 1e-9) / sd
        flag = shift > mean_shift or ratio > 2.0 or ratio < 0.5
        drift |= flag
        table.append({"feature": c, "train_mean": round(s["mean"], 3), "live_mean": round(mean, 3), "train_std": round(s["std"], 3),
                      "live_std": round(std, 3), "mean_shift_in_sd": round(shift, 3), "std_ratio": round(ratio, 3),
                      "status": "DRIFT" if flag else "ok"})
    return table, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/predictions.jsonl")
    ap.add_argument("--meta", default="model/metadata.json")
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--mean-shift", type=float, default=0.5)
    a = ap.parse_args()
    stats = json.load(open(a.meta))["train_stats"]
    rows = [json.loads(l)["features"] for l in open(a.log, encoding="utf-8") if l.strip()]
    if len(rows) < a.min_samples:
        print(f"only {len(rows)} logged predictions (< {a.min_samples}); not enough for a drift check"); return 0
    table, drift = compute_drift(rows, stats, a.mean_shift)
    cols = ["feature", "train_mean", "live_mean", "train_std", "live_std", "mean_shift_in_sd", "std_ratio", "status"]
    print(f"{len(rows)} logged predictions vs training data\n")
    print("".join(f"{c:>17s}" for c in cols))
    for r in table:
        print("".join(f"{r[c]:>17}" for c in cols))
    print("\nRESULT:", "DRIFT DETECTED - inspect the flagged features and consider retraining" if drift else "no drift detected")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
