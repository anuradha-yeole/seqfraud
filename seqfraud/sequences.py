"""Per-card transaction sequences for the sequence model.

For every transaction we take the card's previous ``K`` transactions (arrival
order) plus the current one. Each step is described *relative to the current
transaction* (same device? same IP country? how long ago?), which is what lets
the model notice a token being replayed from a new device even when amount and
merchant look normal.

``encode_steps`` is the single source of truth used by both the batch builder
and the online scorer, so the two cannot drift apart.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

K = 16                       # history length; sequence length is K + 1
N_NUM = 6                    # numeric features per step
RAW = ["ts", "amount", "mcc", "channel", "device", "ip_country", "card_country"]


def encode_steps(hist: np.ndarray, cur: np.ndarray):
    """hist: (..., L, 7) raw history rows, cur: (..., 7) current row.

    Returns numeric (..., L, N_NUM) float32 and categorical (..., L, 2) int64.
    """
    ts, amt, mcc, chan, dev, ipc, cc = [hist[..., i] for i in range(7)]
    c = cur[..., None, :]
    dt = np.maximum(c[..., 0] - ts, 0)
    num = np.stack([
        np.log1p(amt) / 5.0,
        np.log1p(dt / 60.0) / 7.0,                 # minutes ago, log-scaled
        (dev == c[..., 4]).astype(np.float32),     # same device as current txn
        (ipc == c[..., 5]).astype(np.float32),     # same IP country as current txn
        (ipc != cc).astype(np.float32),            # foreign IP for this card
        ((ts % 86_400) // 3_600) / 23.0,
    ], axis=-1).astype(np.float32)
    cat = np.stack([mcc, chan], axis=-1).astype(np.int64)
    return num, cat


def build_sequences(df: pd.DataFrame, k: int = K):
    """Vectorised batch builder. Returns arrays aligned with ``df`` row order."""
    order = np.lexsort((df["txn_id"].to_numpy(), df["card_id"].to_numpy()))
    raw = df[RAW].to_numpy(dtype=np.float64)[order]
    card = df["card_id"].to_numpy()[order]
    n = len(df)
    L = k + 1
    hist = np.zeros((n, L, 7), dtype=np.float64)
    mask = np.zeros((n, L), dtype=bool)
    pos = np.arange(n)
    for j in range(k, -1, -1):                       # slot L-1-j holds lag j
        src = pos - j
        ok = src >= 0
        ok[ok] &= card[src[ok]] == card[ok]
        slot = L - 1 - j
        hist[ok, slot] = raw[src[ok]]
        mask[:, slot] = ok
    num, cat = encode_steps(hist, raw)
    num[~mask] = 0.0
    cat[~mask] = 0
    inv = np.empty(n, dtype=np.int64)
    inv[order] = pos
    return num[inv], cat[inv], mask[inv]


class OnlineSequenceState:
    """Keeps the last ``k`` raw events per card for real-time scoring."""

    def __init__(self, k: int = K):
        self.k = k
        self.hist: dict[int, deque] = {}

    def sequence(self, txn: dict):
        cur = np.array([txn[c] for c in RAW], dtype=np.float64)
        past = list(self.hist.get(txn["card_id"], ()))
        L = self.k + 1
        hist = np.zeros((L, 7))
        mask = np.zeros(L, dtype=bool)
        rows = past + [cur]
        hist[L - len(rows):] = np.array(rows)
        mask[L - len(rows):] = True
        num, cat = encode_steps(hist, cur)
        num[~mask] = 0.0
        cat[~mask] = 0
        return num, cat, mask

    def update(self, txn: dict) -> None:
        d = self.hist.setdefault(txn["card_id"], deque(maxlen=self.k))
        d.append(np.array([txn[c] for c in RAW], dtype=np.float64))
