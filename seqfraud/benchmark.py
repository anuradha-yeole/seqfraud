"""Replay the test period through the online scorer and measure latency.

    python -m seqfraud.benchmark --artifacts artifacts --n 20000

Warms per-card state with the history before the replay window, then scores
transactions one at a time (batch size 1, single thread), as an
authorisation path would. Also checks online scores match the offline batch
scores, i.e. no training/serving skew.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .serve import Scorer
from .train import VAL_END, DAY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transactions.parquet")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n", type=int, default=20_000)
    a = ap.parse_args()
    torch.set_num_threads(1)

    df = pd.read_parquet(a.data).sort_values("txn_id")
    sc = Scorer(a.artifacts)
    start = VAL_END * DAY
    # warm per-card state with all history before the replay window
    for rec in df[df.ts < start].to_dict("records"):
        sc.feats.update(rec)
        sc.seqs.update(rec)
    replay = df[df.ts >= start].head(a.n).to_dict("records")
    lat, scores = [], []
    t_all = time.perf_counter()
    for rec in replay:
        t0 = time.perf_counter()
        r = sc.score(rec)
        lat.append((time.perf_counter() - t0) * 1e3)
        scores.append(r["risk_score"])
    wall = time.perf_counter() - t_all
    lat = np.array(lat[200:])                    # drop first requests (lazy init)
    out = {"n": len(replay), "p50_ms": float(np.percentile(lat, 50)),
           "p95_ms": float(np.percentile(lat, 95)), "p99_ms": float(np.percentile(lat, 99)),
           "throughput_per_core_rps": len(replay) / wall}
    offline = pd.read_parquet(os.path.join(a.artifacts, "scores.parquet")).set_index("txn_id")
    ids = [r["txn_id"] for r in replay]
    out["max_abs_online_offline_diff"] = float(np.max(np.abs(
        np.array(scores) - offline.loc[ids, "seq_transformer"].to_numpy())))
    print(json.dumps(out, indent=2))
    json.dump(out, open(os.path.join(a.artifacts, "latency.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
