"""Smoke test for the scoring API using a randomly initialised model."""
import json

import numpy as np
import torch
from fastapi.testclient import TestClient

from seqfraud.features import AGG_FEATURES
from seqfraud.model import SeqFraudModel
from seqfraud.sequences import K, N_NUM


def _artifacts(tmp_path):
    m = SeqFraudModel(n_agg=len(AGG_FEATURES)).eval()
    ex = (torch.zeros(1, K + 1, N_NUM), torch.zeros(1, K + 1, 2, dtype=torch.long),
          torch.ones(1, K + 1, dtype=torch.bool), torch.zeros(1, len(AGG_FEATURES)))
    torch.jit.trace(m, ex, check_trace=False).save(tmp_path / "seq_model.pt")
    json.dump({"threshold": 0.5, "agg_features": AGG_FEATURES}, open(tmp_path / "serving_config.json", "w"))


def test_score_endpoint(tmp_path, monkeypatch):
    _artifacts(tmp_path)
    monkeypatch.setenv("SEQFRAUD_ARTIFACTS", str(tmp_path))
    import seqfraud.serve as serve
    serve._scorer = None
    client = TestClient(serve.app)
    base = dict(card_id=7, amount=12.5, mcc=3, merchant=11, device=70, ip_country=1,
                card_country=1, channel=1)
    for i in range(20):                                  # history builds up across calls
        r = client.post("/v1/score", json={**base, "txn_id": i, "ts": 1_000 + 60 * i})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["decision"] in {"allow", "block"}
    st = serve._scorer.feats.cards[7]
    assert st.cnt_all == 20 and len(serve._scorer.seqs.hist[7]) == K
    assert np.isfinite(body["latency_ms"])
