"""Real-time scoring service.

    uvicorn seqfraud.serve:app --port 8000

``POST /v1/score`` takes one authorisation, computes velocity + sequence
features from in-memory per-card state (the same code paths the parity tests
check against the Spark backfill), runs the TorchScript model and returns a
risk score and an allow/block decision. State is updated after scoring, so
each request only ever sees strictly earlier events.
"""
from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from .features import OnlineFeatureState
from .sequences import OnlineSequenceState


class Txn(BaseModel):
    txn_id: int
    card_id: int
    ts: int
    amount: float
    mcc: int
    merchant: int
    device: int
    ip_country: int
    card_country: int
    channel: int


class Scorer:
    def __init__(self, artifacts: str = "artifacts"):
        cfg = json.load(open(os.path.join(artifacts, "serving_config.json")))
        self.threshold = cfg["threshold"]
        self.agg_features = cfg["agg_features"]
        self.model = torch.jit.load(os.path.join(artifacts, "seq_model.pt"))
        self.model.eval()
        self.feats = OnlineFeatureState()
        self.seqs = OnlineSequenceState()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def score(self, txn: dict) -> dict:
        t0 = time.perf_counter()
        with self.lock:                       # per-process state; shard by card_id to scale out
            f = self.feats.features(txn)
            num, cat, mask = self.seqs.sequence(txn)
            self.feats.update(txn)
            self.seqs.update(txn)
        agg = np.array([[f[c] for c in self.agg_features]], dtype=np.float32)
        logit = self.model(torch.from_numpy(num[None]), torch.from_numpy(cat[None]),
                           torch.from_numpy(mask[None]), torch.from_numpy(agg))
        p = float(torch.sigmoid(logit)[0])
        return {"txn_id": txn["txn_id"], "risk_score": p,
                "decision": "block" if p >= self.threshold else "allow",
                "latency_ms": (time.perf_counter() - t0) * 1e3}


torch.set_num_threads(1)                      # one request per core; avoids thread oversubscription
app = FastAPI(title="seqfraud scoring")
_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer(os.environ.get("SEQFRAUD_ARTIFACTS", "artifacts"))
    return _scorer


@app.post("/v1/score")
def score(txn: Txn):
    return get_scorer().score(txn.model_dump())


@app.get("/healthz")
def health():
    return {"ok": True}
