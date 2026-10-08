"""Emerging-fraud adaptation experiment.

Token theft first appears on day 70 and is invisible to models trained before
then. In production the first confirmed cases arrive as chargebacks/analyst
labels. This script asks: *given labels for only N compromised cards, how much
token theft can each model catch afterwards?*

    python -m seqfraud.adapt --artifacts artifacts

* Adaptation window: days [0, 77). Token-theft transactions are kept only for
  N randomly chosen compromised cards (labelled); the rest are dropped as
  "not yet reported" rather than mislabelled as legit.
* Evaluation window: days [77, 90). Metric: recall at a fixed 0.1% false
  positive rate on legit traffic in that window, overall and for token theft.
* LightGBM is retrained from scratch; the sequence model is fine-tuned from
  the checkpoint trained by ``train.py`` for 2 epochs.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .features import AGG_FEATURES
from .metrics import threshold_at_fpr
from .model import SeqFraudModel
from .sequences import build_sequences
from .train import DAY, predict_torch, train_lgbm, _tensors

ADAPT_END = 77


def finetune(state, num, cat, mask, agg, y, idx, epochs, seed, neg_rate=0.25, bs=1024, lr=3e-4):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SeqFraudModel(n_agg=agg.shape[1])
    model.load_state_dict(state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    yf = y.astype(np.float32)
    for _ in range(epochs):
        keep = idx[(y[idx] == 1) | (rng.random(len(idx)) < neg_rate)]
        rng.shuffle(keep)
        model.train()
        for i in range(0, len(keep), bs):
            b = keep[i:i + bs]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(*_tensors(num, cat, mask, agg, b)), torch.from_numpy(yf[b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def recall_at(y, s, fpr, sel):
    thr = threshold_at_fpr(y, s, fpr)
    return float((s[sel] >= thr).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n-cards", default="0,10,30,100,all")
    ap.add_argument("--fpr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())

    df = pd.read_parquet(os.path.join(a.artifacts, "features.parquet"))
    num, cat, mask = build_sequences(df)
    agg = df[AGG_FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    day = (df["ts"] // DAY).to_numpy()
    tt = (df["fraud_type"] == "token_theft").to_numpy()
    ev = day >= ADAPT_END
    ev_y, ev_tt = y[ev], tt[ev]
    ev_other_fraud = (ev_y == 1) & ~ev_tt
    base_state = torch.load(os.path.join(a.artifacts, "seq_model_state.pt"))

    rng = np.random.default_rng(a.seed)
    tt_cards = df.loc[tt & (day < ADAPT_END), "card_id"].unique()
    rng.shuffle(tt_cards)
    results = []
    for n in a.n_cards.split(","):
        k = len(tt_cards) if n == "all" else min(int(n), len(tt_cards))
        labelled = set(tt_cards[:k].tolist())
        drop = tt & (day < ADAPT_END) & ~df["card_id"].isin(labelled).to_numpy()
        fit = (day < ADAPT_END) & ~drop
        # LightGBM: retrain from scratch; early-stop on the last adaptation week
        tr = fit & (day < ADAPT_END - 7)
        va = fit & (day >= ADAPT_END - 7)
        booster, cols = train_lgbm(df, tr, va, a.seed)
        s_lgb = booster.predict(df.loc[ev, cols])
        # Sequence model: fine-tune the day-56 checkpoint
        if k == 0:
            model = SeqFraudModel(n_agg=agg.shape[1])
            model.load_state_dict(base_state)
        else:
            model = finetune(base_state, num, cat, mask, agg, y, np.flatnonzero(fit), a.epochs, a.seed)
        s_seq = predict_torch(model, num, cat, mask, agg, np.flatnonzero(ev))
        row = {"labelled_cards": k, "labelled_txns": int((tt & fit).sum())}
        for name, s in [("lightgbm", s_lgb), ("seq_transformer", s_seq)]:
            row[f"{name}_token_theft_recall"] = recall_at(ev_y, s, a.fpr, ev_tt)
            row[f"{name}_other_fraud_recall"] = recall_at(ev_y, s, a.fpr, ev_other_fraud)
        print(row, flush=True)
        results.append(row)
    out = {"fpr": a.fpr, "eval_token_theft_txns": int(ev_tt.sum()),
           "eval_token_theft_cards": int(df.loc[ev & tt, "card_id"].nunique()),
           "compromised_cards_in_adapt_window": int(len(tt_cards)), "results": results}
    json.dump(out, open(os.path.join(a.artifacts, "adaptation.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
