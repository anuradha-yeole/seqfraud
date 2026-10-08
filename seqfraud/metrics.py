"""Fraud-specific evaluation: operating-point metrics and cluster bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Score threshold that blocks ``fpr`` of legitimate traffic."""
    return float(np.quantile(s[y == 0], 1.0 - fpr))


def operating_point(y, s, amount, thr):
    block = s >= thr
    fraud = y == 1
    return {
        "recall": float(block[fraud].mean()) if fraud.any() else float("nan"),
        "fpr": float(block[~fraud].mean()),
        "dollar_recall": float(amount[block & fraud].sum() / max(amount[fraud].sum(), 1e-9)),
        "precision": float(fraud[block].mean()) if block.any() else float("nan"),
    }


def summarize(y, s, amount, thr) -> dict:
    out = {"pr_auc": float(average_precision_score(y, s)),
           "roc_auc": float(roc_auc_score(y, s))}
    out.update(operating_point(y, s, amount, thr))
    return out


def cluster_bootstrap_diff(y, s_a, s_b, groups, thr_a, thr_b, n_boot=500, seed=0):
    """95% CI for recall(B) - recall(A), resampling *cards* (not rows) because
    fraud transactions on the same card are strongly correlated."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    fraud = y == 1
    hit_a = ((s_a >= thr_a) & fraud).astype(float)
    hit_b = ((s_b >= thr_b) & fraud).astype(float)
    per_n = np.bincount(inv, weights=fraud.astype(float), minlength=len(uniq))
    per_a = np.bincount(inv, weights=hit_a, minlength=len(uniq))
    per_b = np.bincount(inv, weights=hit_b, minlength=len(uniq))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(len(uniq), size=len(uniq)), minlength=len(uniq))
        n = (w * per_n).sum()
        diffs[i] = ((w * per_b).sum() - (w * per_a).sum()) / max(n, 1)
    point = (per_b.sum() - per_a.sum()) / max(per_n.sum(), 1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)
