"""Point-in-time velocity features.

Two implementations share one contract:

* ``spark_features``  - batch backfill over the full history with PySpark
                        window functions (training data).
* ``OnlineFeatureState`` - incremental per-card state used by the scoring
                        service (one event at a time, O(window) memory).

Every aggregate only looks at events with ``ts`` strictly earlier than the
current event (window ``[t - W, t - 1]``), so there is no label/future leakage
and both paths produce identical values. ``tests/test_parity.py`` enforces
this, which is how training/serving skew is kept at zero.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

HOUR, DAY = 3_600, 86_400
NO_HISTORY_SECS = 30 * DAY      # value for "secs_since_last" on a card's first txn

AGG_FEATURES = [
    "log_amount", "hour", "ip_foreign", "cnt_1h", "cnt_24h", "log_sum_amt_24h",
    "n_merch_1h", "log_secs_since_last", "new_device", "new_ip_country",
    "amt_ratio_30d", "log_cnt_all",
]
CAT_FEATURES = ["mcc", "channel"]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Shared transforms applied after the raw aggregates are computed."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["hour"] = (out["ts"] % DAY) // HOUR
    out["ip_foreign"] = (out["ip_country"] != out["card_country"]).astype("int64")
    out["log_sum_amt_24h"] = np.log1p(out["sum_amt_24h"])
    out["log_secs_since_last"] = np.log1p(out["secs_since_last"])
    out["amt_ratio_30d"] = out["amount"] / (out["avg_amt_30d"] + 1.0)
    out["log_cnt_all"] = np.log1p(out["cnt_all"])
    return out


# --------------------------------------------------------------------------- batch
def spark_features(df: pd.DataFrame, spark=None) -> pd.DataFrame:
    """Backfill features with PySpark window functions."""
    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F

    own = spark is None
    spark = spark or (SparkSession.builder.master("local[*]").appName("seqfraud-features")
                      .config("spark.sql.shuffle.partitions", "8")
                      .config("spark.driver.memory", "4g").getOrCreate())
    raw_cols = ["txn_id", "card_id", "ts", "amount", "merchant", "device",
                "ip_country", "card_country", "mcc", "channel"]
    sdf = spark.createDataFrame(df[raw_cols])

    by_card = Window.partitionBy("card_id").orderBy("ts")
    w1h = by_card.rangeBetween(-HOUR, -1)
    w24h = by_card.rangeBetween(-DAY, -1)
    w30d = by_card.rangeBetween(-30 * DAY, -1)
    wall = by_card.rangeBetween(Window.unboundedPreceding, -1)
    first_dev = Window.partitionBy("card_id", "device")
    first_ipc = Window.partitionBy("card_id", "ip_country")

    feats = (sdf
             .withColumn("cnt_1h", F.count("*").over(w1h))
             .withColumn("cnt_24h", F.count("*").over(w24h))
             .withColumn("sum_amt_24h", F.coalesce(F.sum("amount").over(w24h), F.lit(0.0)))
             .withColumn("n_merch_1h", F.size(F.collect_set("merchant").over(w1h)))
             .withColumn("avg_amt_30d", F.coalesce(F.avg("amount").over(w30d), F.lit(0.0)))
             .withColumn("cnt_all", F.count("*").over(wall))
             .withColumn("secs_since_last",
                         F.coalesce(F.col("ts") - F.max("ts").over(wall), F.lit(NO_HISTORY_SECS)))
             .withColumn("new_device",
                         (F.min("ts").over(first_dev) >= F.col("ts")).cast("long"))
             .withColumn("new_ip_country",
                         (F.min("ts").over(first_ipc) >= F.col("ts")).cast("long")))
    agg = feats.select("txn_id", "cnt_1h", "cnt_24h", "sum_amt_24h", "n_merch_1h",
                       "avg_amt_30d", "cnt_all", "secs_since_last", "new_device",
                       "new_ip_country").toPandas()
    if own:
        spark.stop()                          # release the JVM heap before model training
    out = df.merge(agg, on="txn_id", how="left").sort_values("txn_id").reset_index(drop=True)
    return _finalize(out)


# --------------------------------------------------------------------------- online
class _CardState:
    __slots__ = ("events", "first_dev", "first_ipc", "cnt_all", "last_ts", "prev_ts",
                 "cnt_at_last")

    def __init__(self):
        self.events: deque = deque()          # (ts, amount, merchant) within 30 days
        self.first_dev: dict[int, int] = {}
        self.first_ipc: dict[int, int] = {}
        self.cnt_all = 0
        self.last_ts: int | None = None       # most recent event second
        self.prev_ts: int | None = None       # most recent second strictly before last_ts
        self.cnt_at_last = 0                  # events seen at last_ts


class OnlineFeatureState:
    """Per-card rolling state; ``features`` reads, ``update`` writes."""

    def __init__(self):
        self.cards: dict[int, _CardState] = {}

    def _state(self, card_id: int) -> _CardState:
        st = self.cards.get(card_id)
        if st is None:
            st = self.cards[card_id] = _CardState()
        return st

    def features(self, txn: dict) -> dict:
        st = self._state(txn["card_id"])
        t = txn["ts"]
        cnt_1h = cnt_24h = n30 = 0
        sum24 = sum30 = 0.0
        merch_1h = set()
        # iterate newest -> oldest so we can stop once outside the 30-day window
        for ts, amt, merch in reversed(st.events):
            if ts > t - 1:            # strictly earlier only (same-second ties excluded)
                continue
            if ts < t - 30 * DAY:
                break
            n30 += 1
            sum30 += amt
            if ts >= t - DAY:
                cnt_24h += 1
                sum24 += amt
            if ts >= t - HOUR:
                cnt_1h += 1
                merch_1h.add(merch)
        # events arrive in ts order, so only events at exactly ts == t are "same second"
        if st.last_ts == t:
            prior_all = st.cnt_all - st.cnt_at_last
            last = st.prev_ts
        else:
            prior_all = st.cnt_all
            last = st.last_ts
        dev_first = st.first_dev.get(txn["device"])
        ipc_first = st.first_ipc.get(txn["ip_country"])
        raw = dict(txn)
        raw.update(
            cnt_1h=cnt_1h, cnt_24h=cnt_24h, sum_amt_24h=sum24, n_merch_1h=len(merch_1h),
            avg_amt_30d=(sum30 / n30) if n30 else 0.0, cnt_all=prior_all,
            secs_since_last=(t - last) if last is not None else NO_HISTORY_SECS,
            new_device=int(dev_first is None or dev_first >= t),
            new_ip_country=int(ipc_first is None or ipc_first >= t),
        )
        a = raw["amount"]
        raw["log_amount"] = math.log1p(a)
        raw["hour"] = (t % DAY) // HOUR
        raw["ip_foreign"] = int(txn["ip_country"] != txn["card_country"])
        raw["log_sum_amt_24h"] = math.log1p(sum24)
        raw["log_secs_since_last"] = math.log1p(raw["secs_since_last"])
        raw["amt_ratio_30d"] = a / (raw["avg_amt_30d"] + 1.0)
        raw["log_cnt_all"] = math.log1p(prior_all)
        return raw

    def update(self, txn: dict) -> None:
        st = self._state(txn["card_id"])
        t = txn["ts"]
        st.events.append((t, txn["amount"], txn["merchant"]))
        while st.events and st.events[0][0] < t - 30 * DAY - 1:
            st.events.popleft()
        st.first_dev.setdefault(txn["device"], t)
        st.first_ipc.setdefault(txn["ip_country"], t)
        st.cnt_all += 1
        if st.last_ts == t:
            st.cnt_at_last += 1
        else:
            st.prev_ts, st.last_ts, st.cnt_at_last = st.last_ts, t, 1
