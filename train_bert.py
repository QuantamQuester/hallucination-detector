"""Fine-tune a BERT transformer as a hallucination classifier and compare it with the feature pipeline.

Input of the model (a standard BERT sentence pair):
    [CLS] question [SEP] answer [SEP] evidence [SEP]            (evidence is truncated to fit --max-len)
The same grouped 80/20 split as train.py is used, so the numbers are directly comparable with the sklearn runs.
Results (overall, per question type, per hallucination strategy) go to MLflow and to model/bert_results.json,
which the serving app shows on its "Question types" page. The BERT weights are NOT part of the serving image
(420 MB, more memory than the free hosting tier has); the deployed model stays the calibrated linear SVM.

Examples
  python train_bert.py                                    # full fine-tuning (use a GPU, e.g. Colab: ~10 min)
  python train_bert.py --freeze-layers 8 --max-train-rows 3600 --max-test-rows 1800   # CPU-sized run
  python train_bert.py --model /path/to/bert-base-uncased  # local copy of the weights (no Hugging Face access)
Extra requirements: pip install -r requirements-bert.txt
"""
import argparse, json, os, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train import SEED, load_split, qgroup, slice_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="bert-base-uncased", help="Hugging Face model name or a local directory")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="keep the embeddings and the first N encoder layers fixed (0 = full fine-tuning). "
                        "Their outputs are computed once and cached, which makes CPU training practical.")
    p.add_argument("--max-train-rows", type=int, default=0, help="0 = all; otherwise a sample balanced over WHY / POLAR / OTHER")
    p.add_argument("--max-test-rows", type=int, default=0, help="0 = all; otherwise all WHY and POLAR test rows + a sample of the rest")
    p.add_argument("--time-budget-min", type=float, default=0, help="stop training after this many minutes (0 = no limit)")
    p.add_argument("--data", default="data/halueval_qa.csv")
    p.add_argument("--extra", default="data/why_polar_dataset.csv")
    p.add_argument("--out", default="model/bert_results.json")
    p.add_argument("--save-dir", default="", help="optionally save the fine-tuned weights here")
    p.add_argument("--experiment", default="hallucination-detection-bert")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()


def subsample(df, n, seed, keep_minor=False):
    """Balanced sample over the three question groups. keep_minor=True keeps every WHY and POLAR row."""
    if not n or n >= len(df):
        return df
    g = pd.Series(qgroup(df["qtype"].values), index=df.index)
    if keep_minor:
        minor = df[g != "OTHER"]
        rest = df[g == "OTHER"]
        return pd.concat([minor, rest.sample(min(len(rest), max(n - len(minor), 200)), random_state=seed)])
    share = {"WHY": 0.45, "POLAR": 0.15, "OTHER": 0.40}
    parts = [df[g == k].sample(min(int((g == k).sum()), int(n * s)), random_state=seed) for k, s in share.items()]
    return pd.concat(parts)


def encode(tok, df, max_len):
    a = (df["question"].astype(str) + " " + tok.sep_token + " " + df["answer"].astype(str)).tolist()
    b = df["knowledge"].fillna("").astype(str).tolist()
    enc = tok(a, b, truncation="only_second", max_length=max_len)
    return enc["input_ids"], enc["token_type_ids"]


def batches(order, size):
    for i in range(0, len(order), size):
        yield order[i:i + size]


def pad(seqs, value=0):
    n = max(len(s) for s in seqs)
    out = torch.full((len(seqs), n), value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s)
    return out


class Split:
    """BERT cut in two: a frozen lower part (run without gradients, cacheable) and a trainable upper part."""

    def __init__(self, model, n_frozen):
        self.m, self.n = model, n_frozen
        bert = model.bert
        for p in bert.embeddings.parameters():
            p.requires_grad = n_frozen == 0
        for layer in bert.encoder.layer[:n_frozen]:
            for p in layer.parameters():
                p.requires_grad = False

    def mask(self, attn):
        return (1.0 - attn[:, None, None, :].float()) * torch.finfo(torch.float32).min

    def lower(self, ids, types, attn):
        bert = self.m.bert
        with torch.set_grad_enabled(self.n == 0 and self.m.training):
            h = bert.embeddings(input_ids=ids, token_type_ids=types)
        if self.n:
            with torch.no_grad():
                was = self.m.training
                self.m.eval()                                  # frozen layers: no dropout, deterministic -> cacheable
                for layer in bert.encoder.layer[:self.n]:
                    h = layer(h, attention_mask=self.mask(attn))[0]
                self.m.train(was)
        return h

    def upper(self, h, attn):
        bert = self.m.bert
        for layer in bert.encoder.layer[self.n:]:
            h = layer(h, attention_mask=self.mask(attn))[0]
        return self.m.classifier(self.m.dropout(bert.pooler(h)))


def main():
    a = parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df, train, test = load_split(a.data, a.extra)
    train_s = subsample(train, a.max_train_rows, SEED).reset_index(drop=True)
    test_s = subsample(test, a.max_test_rows, SEED, keep_minor=True).reset_index(drop=True)
    print(f"device={dev} | train {len(train_s)}/{len(train)} rows | test {len(test_s)}/{len(test)} rows | "
          f"train groups {pd.Series(qgroup(train_s['qtype'].values)).value_counts().to_dict()}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
    net = Split(model, a.freeze_layers)
    ids_tr, ty_tr = encode(tok, train_s, a.max_len)
    ids_te, ty_te = encode(tok, test_s, a.max_len)
    ytr = torch.tensor(train_s["hallucination"].values)
    t0 = time.time()

    def lower_cached(ids, types, order, tag):
        """Outputs of the frozen part for every row (half precision, variable length), computed once."""
        cache = [None] * len(ids)
        by_len = sorted(order, key=lambda i: len(ids[i]))
        for k, b in enumerate(batches(by_len, 32)):
            x, t = pad([ids[i] for i in b]).to(dev), pad([types[i] for i in b]).to(dev)
            attn = (x != 0).long()
            h = net.lower(x, t, attn)
            for j, i in enumerate(b):
                cache[i] = h[j, :len(ids[i])].half().cpu()
            if k % 20 == 0:
                print(f"  [{tag}] frozen layers: {min((k + 1) * 32, len(ids))}/{len(ids)} rows, {time.time() - t0:.0f}s", flush=True)
        return cache

    def forward(b, ids, types, cache):
        lens = [len(ids[i]) for i in b]
        attn = torch.zeros(len(b), max(lens), dtype=torch.long, device=dev)
        for j, n in enumerate(lens):
            attn[j, :n] = 1
        if cache is not None:
            h = torch.zeros(len(b), max(lens), cache[b[0]].shape[-1], device=dev)
            for j, i in enumerate(b):
                h[j, :lens[j]] = cache[i].float().to(dev)
        else:
            h = net.lower(pad([ids[i] for i in b]).to(dev), pad([types[i] for i in b]).to(dev), attn)
        return net.upper(h, attn)

    use_cache = a.freeze_layers > 0
    model.eval()
    cache_tr = lower_cached(ids_tr, ty_tr, list(range(len(ids_tr))), "train") if use_cache else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(ids_tr) / a.batch_size))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(1, 0.06 * steps), max(0.0, (steps - s) / (0.94 * steps))))
    rng = np.random.default_rng(SEED)
    done_epochs, step, stopped = 0.0, 0, False
    for ep in range(a.epochs):
        model.train()
        order = rng.permutation(len(ids_tr)).tolist()
        # bucket by length inside chunks of 50 batches: little padding, still shuffled
        chunks = [sorted(order[i:i + a.batch_size * 50], key=lambda i: len(ids_tr[i])) for i in range(0, len(order), a.batch_size * 50)]
        bl = [b for c in chunks for b in batches(c, a.batch_size)]
        rng.shuffle(bl)
        run = 0.0
        for k, b in enumerate(bl):
            loss = torch.nn.functional.cross_entropy(forward(b, ids_tr, ty_tr, cache_tr), ytr[b].to(dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            run = 0.95 * run + 0.05 * loss.item() if k else loss.item()
            if k % 20 == 0:
                print(f"  epoch {ep + 1}/{a.epochs} batch {k}/{len(bl)} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
            if a.time_budget_min and time.time() - t0 > a.time_budget_min * 60:
                stopped = True
                break
        done_epochs = ep + (k + 1) / len(bl)
        if stopped:
            print(f"time budget reached after {done_epochs:.2f} epochs", flush=True)
            break
    train_s_ = time.time() - t0
    del cache_tr

    model.eval()
    order = sorted(range(len(ids_te)), key=lambda i: len(ids_te[i]))
    prob = np.zeros(len(ids_te))
    with torch.no_grad():
        for k, b in enumerate(batches(order, 32)):
            logits = forward(b, ids_te, ty_te, None)
            prob[b] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
            if k % 10 == 0:
                print(f"  eval {min((k + 1) * 32, len(order))}/{len(order)} {time.time() - t0:.0f}s", flush=True)
    y = test_s["hallucination"].values
    pred = (prob >= 0.5).astype(int)
    by_q, by_s = slice_metrics(test_s, pred, prob)
    res = {"model": os.path.basename(os.path.normpath(a.model)), "task_input": "[CLS] question [SEP] answer [SEP] evidence [SEP]",
           "max_len": a.max_len, "epochs_requested": a.epochs, "epochs_done": round(float(done_epochs), 2),
           "batch_size": a.batch_size, "lr": a.lr, "frozen_layers": a.freeze_layers,
           "trainable_parameters": int(sum(p.numel() for p in params)), "total_parameters": int(sum(p.numel() for p in model.parameters())),
           "device": dev, "n_train_rows": len(train_s), "n_train_rows_available": len(train), "n_test_rows": len(test_s),
           "n_test_rows_available": len(test), "train_minutes": round(train_s_ / 60, 1),
           "metrics": {"test_accuracy": float(accuracy_score(y, pred)), "test_f1": float(f1_score(y, pred)),
                       "test_precision": float(precision_score(y, pred)), "test_recall": float(recall_score(y, pred)),
                       "test_roc_auc": float(roc_auc_score(y, prob))},
           "confusion_matrix": confusion_matrix(y, pred).tolist(), "by_question_type": by_q, "recall_by_strategy": by_s,
           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "torch": torch.__version__}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    pd.DataFrame({"group_id": test_s["group_id"], "answer": test_s["answer"], "hallucination": y, "bert_prob": prob.round(5)}
                 ).to_csv(a.out.replace(".json", "_predictions.csv"), index=False)
    if a.save_dir:
        model.save_pretrained(a.save_dir); tok.save_pretrained(a.save_dir)
    if not a.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(a.tracking_uri); mlflow.set_experiment(a.experiment)
        with mlflow.start_run(run_name=f"{res['model']}-freeze{a.freeze_layers}-len{a.max_len}"):
            mlflow.log_params({"algorithm": "bert", "model": res["model"], "max_len": a.max_len, "epochs": res["epochs_done"],
                               "batch_size": a.batch_size, "lr": a.lr, "freeze_layers": a.freeze_layers,
                               "n_train_rows": len(train_s), "n_test_rows": len(test_s), "seed": SEED, "split": "GroupShuffleSplit"})
            mlflow.log_metrics({**res["metrics"], "train_seconds": train_s_,
                                **{f"test_{k}_{q.lower()}": v[k] for q, v in by_q.items() for k in ("accuracy", "f1", "roc_auc")}})
            mlflow.log_dict(res, "bert_results.json")
    print(json.dumps({"metrics": res["metrics"], "by_question_type": by_q, "recall_by_strategy": by_s}, indent=1))


if __name__ == "__main__":
    main()
