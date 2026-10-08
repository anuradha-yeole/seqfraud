"""End-to-end training + out-of-time evaluation.

    python -m seqfraud.train --data data/transactions.parquet --out artifacts/

Steps: Spark feature backfill -> time-based split -> LightGBM baseline ->
PyTorch sequence model -> thresholds picked on validation at a fixed
false-positive (block) rate -> test report with per-typology recall and a
card-level bootstrap CI for the recall lift.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from .features import AGG_FEATURES, CAT_FEATURES, spark_features
from .metrics import cluster_bootstrap_diff, summarize, threshold_at_fpr
from .model import SeqFraudModel
from .sequences import build_sequences

DAY = 86_400
TRAIN_END, VAL_END = 56, 63          # days; test = [63, 90)


def split(df):
    d = df["ts"] // DAY
    return (d < TRAIN_END).to_numpy(), ((d >= TRAIN_END) & (d < VAL_END)).to_numpy(), (d >= VAL_END).to_numpy()


def train_lgbm(df, tr, va, seed):
    cols = AGG_FEATURES + CAT_FEATURES
    dtr = lgb.Dataset(df.loc[tr, cols], df.loc[tr, "label"], categorical_feature=CAT_FEATURES)
    dva = lgb.Dataset(df.loc[va, cols], df.loc[va, "label"], reference=dtr)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=seed,
                  verbose=-1, metric="average_precision", num_threads=os.cpu_count(),
                  deterministic=True, force_row_wise=True)
    booster = lgb.train(params, dtr, 1000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    return booster, cols


def _tensors(num, cat, mask, agg, idx):
    return (torch.from_numpy(num[idx]), torch.from_numpy(cat[idx]),
            torch.from_numpy(mask[idx]), torch.from_numpy(agg[idx]))


@torch.no_grad()
def predict_torch(model, num, cat, mask, agg, idx, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        out.append(torch.sigmoid(model(*_tensors(num, cat, mask, agg, idx[i:i + bs]))).numpy())
    return np.concatenate(out) if out else np.array([])


def train_seq(df, num, cat, mask, agg, tr, va, epochs, seed, neg_rate=0.25, bs=1024):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    y = df["label"].to_numpy().astype(np.float32)
    tr_idx = np.flatnonzero(tr)
    va_idx = np.flatnonzero(va)
    model = SeqFraudModel(n_agg=agg.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, log = -1.0, None, []
    for ep in range(epochs):
        # negative down-sampling keeps every fraud example, 25% of legit ones
        keep = tr_idx[(y[tr_idx] == 1) | (rng.random(len(tr_idx)) < neg_rate)]
        rng.shuffle(keep)
        model.train()
        t0, tot = time.time(), 0.0
        for i in range(0, len(keep), bs):
            b = keep[i:i + bs]
            logits = model(*_tensors(num, cat, mask, agg, b))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        ap = average_precision_score(y[va_idx], predict_torch(model, num, cat, mask, agg, va_idx))
        log.append({"epoch": ep + 1, "loss": tot / len(keep), "val_pr_auc": ap,
                    "secs": time.time() - t0})
        print(f"  epoch {ep + 1}: loss {tot / len(keep):.4f}  val PR-AUC {ap:.4f}  ({time.time() - t0:.0f}s)")
        if ap > best:
            best, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.parquet")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--fpr", type=float, default=0.001, help="target block rate on legit traffic")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())

    raw = pd.read_parquet(a.data)
    feat_path = os.path.join(a.out, "features.parquet")
    t0 = time.time()
    if os.path.exists(feat_path):
        df = pd.read_parquet(feat_path)
    else:
        df = spark_features(raw)
        df.to_parquet(feat_path, index=False)
    print(f"features: {len(df):,} rows in {time.time() - t0:.0f}s")
    tr, va, te = split(df)
    y = df["label"].to_numpy()
    amount = df["amount"].to_numpy()
    print(f"train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}  "
          f"| token_theft in train: {(df.loc[tr, 'fraud_type'] == 'token_theft').sum()}")

    print("LightGBM baseline ...")
    booster, cols = train_lgbm(df, tr, va, a.seed)
    s_lgb = booster.predict(df[cols], num_threads=os.cpu_count())
    booster.save_model(os.path.join(a.out, "lgbm.txt"))

    print("Sequence model ...")
    num, cat, mask = build_sequences(df)
    agg = df[AGG_FEATURES].to_numpy(dtype=np.float32)
    model, log = train_seq(df, num, cat, mask, agg, tr, va, a.epochs, a.seed)
    s_seq = predict_torch(model, num, cat, mask, agg, np.arange(len(df)))

    report = {"config": vars(a), "rows": int(len(df)), "train_log": log, "models": {}}
    thr = {}
    # simple blend: average of the two models' percentile ranks
    from scipy.stats import rankdata
    s_blend = (rankdata(s_lgb) + rankdata(s_seq)) / (2 * len(s_lgb))
    for name, s in [("lightgbm", s_lgb), ("seq_transformer", s_seq), ("blend", s_blend)]:
        thr[name] = threshold_at_fpr(y[va], s[va], a.fpr)
        m = summarize(y[te], s[te], amount[te], thr[name])
        m["recall_by_type"] = {
            ft: float((s[te & (df.fraud_type == ft).to_numpy()] >= thr[name]).mean())
            for ft in ["card_testing", "stolen_card", "token_theft"]}
        report["models"][name] = m
    point, lo, hi = cluster_bootstrap_diff(y[te], s_lgb[te], s_seq[te], df.loc[te, "card_id"].to_numpy(),
                                           thr["lightgbm"], thr["seq_transformer"])
    report["recall_lift_seq_vs_lgbm"] = {"point": point, "ci95": [lo, hi]}
    tt = te & (df.fraud_type == "token_theft").to_numpy()
    tt_or_legit = te & ((df.fraud_type == "token_theft") | (df.label == 0)).to_numpy()
    p, l, h = cluster_bootstrap_diff(y[tt_or_legit], s_lgb[tt_or_legit], s_seq[tt_or_legit],
                                     df.loc[tt_or_legit, "card_id"].to_numpy(),
                                     thr["lightgbm"], thr["seq_transformer"])
    report["token_theft_recall_lift"] = {"point": p, "ci95": [l, h], "n_fraud_txns": int(tt.sum())}
    p, l, h = cluster_bootstrap_diff(y[te], s_lgb[te], s_blend[te], df.loc[te, "card_id"].to_numpy(),
                                     thr["lightgbm"], thr["blend"])
    report["recall_lift_blend_vs_lgbm"] = {"point": p, "ci95": [l, h]}

    pd.DataFrame({"txn_id": df["txn_id"], "lightgbm": s_lgb, "seq_transformer": s_seq}
                 ).to_parquet(os.path.join(a.out, "scores.parquet"), index=False)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2)
    json.dump({"threshold": thr["seq_transformer"], "agg_features": AGG_FEATURES},
              open(os.path.join(a.out, "serving_config.json"), "w"), indent=2)
    model.eval()
    torch.save(model.state_dict(), os.path.join(a.out, "seq_model_state.pt"))
    # Trace with batch size 1: the serving path scores one authorisation at a time.
    # check_trace is off because the Transformer fast path picks different kernels per
    # batch size; benchmark.py verifies online scores match the offline batch scores.
    ex = _tensors(num, cat, mask, agg, np.arange(1))
    torch.jit.trace(model, ex, check_trace=False).save(os.path.join(a.out, "seq_model.pt"))
    print(json.dumps({k: v for k, v in report.items() if k != "train_log"}, indent=2))


if __name__ == "__main__":
    main()
