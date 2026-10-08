"""Synthetic card-payments simulator with labelled fraud typologies.

Real payment logs are not public, so this generator produces a transaction
stream whose fraud patterns follow typologies described in public fraud
literature:

* ``card_testing``   - bursts of tiny authorisations across many merchants.
* ``stolen_card``    - card-not-present spend from a new device / foreign IP
                       on high-risk merchant categories.
* ``token_theft``    - a stolen wallet token replayed from a new device. Amounts
                       and merchant categories mimic the victim's normal
                       behaviour, so per-transaction features look benign;
                       only the *sequence* (device switch, geo jump, timing)
                       gives it away. This pattern only appears in the final
                       weeks of the simulation, so it is unseen at training
                       time ("emerging fraud").

Every transaction carries ``label`` (1 = fraud) and ``fraud_type``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

N_MCC = 40              # merchant category codes
HIGH_RISK_MCC = np.array([3, 7, 11, 19, 23, 31])   # e.g. gift cards, electronics
N_COUNTRIES = 30
CHANNELS = {"card_present": 0, "cnp": 1, "wallet_token": 2}
DAY = 86_400


@dataclass
class SimConfig:
    n_cards: int = 40_000
    n_days: int = 90
    n_merchants: int = 8_000
    p_card_testing: float = 0.012
    p_stolen_card: float = 0.012
    p_token_theft: float = 0.010
    token_theft_start_day: int = 70   # emerging pattern: only in the last 20 days
    seed: int = 7


def _legit_card(rng: np.random.Generator, cfg: SimConfig, card_id: int):
    home = rng.integers(N_COUNTRIES)
    rate = rng.lognormal(mean=np.log(0.35), sigma=0.6)          # txns/day
    n = rng.poisson(rate * cfg.n_days)
    if n == 0:
        return None
    ts = np.sort(rng.uniform(0, cfg.n_days * DAY, size=n))
    # diurnal pattern: shift times toward the card's active hours
    active_hour = rng.integers(8, 22)
    day_start = (ts // DAY) * DAY
    hour = np.clip(rng.normal(active_hour, 3.0, size=n), 0, 23.99)
    ts = np.sort(day_start + hour * 3600)
    mu = rng.normal(3.3, 0.7)
    amount = np.round(rng.lognormal(mu, 0.8, size=n), 2)
    fav_mcc = rng.choice(N_MCC, size=4, replace=False)
    mcc = np.where(rng.random(n) < 0.85, rng.choice(fav_mcc, size=n), rng.integers(N_MCC, size=n))
    fav_merch = rng.integers(cfg.n_merchants, size=6)
    merchant = np.where(rng.random(n) < 0.7, rng.choice(fav_merch, size=n),
                        rng.integers(cfg.n_merchants, size=n))
    n_dev = 1 + (rng.random() < 0.3)
    devices = card_id * 10 + np.arange(n_dev)
    device = rng.choice(devices, size=n)
    # legit device upgrade: a new phone/laptop takes over part-way through history
    if rng.random() < 0.25:
        switch = rng.uniform(ts[0], ts[-1] + 1)
        new_dev = card_id * 10 + 5
        device = np.where(ts >= switch, new_dev, device)
        devices = np.append(devices, new_dev)
    ip_country = np.where(rng.random(n) < 0.97, home, rng.integers(N_COUNTRIES, size=n))
    # legit travel: a few days abroad, some purchases from a hotel/borrowed device
    if rng.random() < 0.08:
        t_trip = rng.uniform(ts[0], ts[-1] + 1)
        on_trip = (ts >= t_trip) & (ts < t_trip + rng.uniform(2, 10) * DAY)
        ip_country = np.where(on_trip, (home + rng.integers(1, N_COUNTRIES)) % N_COUNTRIES, ip_country)
        borrowed = on_trip & (rng.random(n) < 0.3)
        device = np.where(borrowed, card_id * 10 + 7, device)
        amount = np.where(on_trip, np.round(amount * rng.uniform(1.0, 2.5), 2), amount)
    chan_p = rng.dirichlet([3, 2, 1.5])
    channel = rng.choice(3, size=n, p=chan_p)
    return dict(ts=ts, amount=amount, mcc=mcc, merchant=merchant, device=device,
                ip_country=ip_country, channel=channel, home=home, mu=mu,
                fav_mcc=fav_mcc, fav_merch=fav_merch, devices=devices)


def _card_testing(rng, cfg, t0):
    k = rng.integers(5, 20)
    ts = t0 + np.cumsum(rng.exponential(25, size=k))             # seconds apart
    amount = np.round(rng.uniform(0.5, 3.0, size=k), 2)
    if rng.random() < 0.5:                                        # cash-out
        ts = np.append(ts, ts[-1] + rng.exponential(600))
        amount = np.append(amount, np.round(rng.lognormal(5.5, 0.5), 2))
    k = len(ts)
    return dict(ts=ts, amount=amount, mcc=rng.integers(N_MCC, size=k),
                merchant=rng.integers(cfg.n_merchants, size=k),
                device=np.full(k, -1_000_000 - rng.integers(10_000)),
                ip_country=np.full(k, rng.integers(N_COUNTRIES)),
                channel=np.full(k, CHANNELS["cnp"]))


def _stolen_card(rng, cfg, t0, home):
    k = rng.integers(2, 6)
    ts = t0 + np.cumsum(rng.exponential(900, size=k))
    amount = np.round(rng.lognormal(5.0, 0.6, size=k), 2)
    # half of fraudsters route through a proxy in the cardholder's own country
    ipc = home if rng.random() < 0.5 else rng.integers(N_COUNTRIES)
    return dict(ts=ts, amount=amount, mcc=rng.choice(HIGH_RISK_MCC, size=k),
                merchant=rng.integers(cfg.n_merchants, size=k),
                device=np.full(k, -2_000_000 - rng.integers(10_000)),
                ip_country=np.full(k, ipc),
                channel=np.full(k, CHANNELS["cnp"]))


def _token_theft(rng, cfg, t0, card):
    """Replay a stolen wallet token: amounts/MCCs look like the victim's own,
    and most attackers use a residential proxy in the victim's country."""
    k = rng.integers(3, 7)
    ts = t0 + np.cumsum(rng.exponential(1200, size=k))
    amount = np.round(rng.lognormal(card["mu"] + 0.6, 0.5, size=k), 2)
    foreign = (card["home"] + rng.integers(1, N_COUNTRIES)) % N_COUNTRIES
    ipc = card["home"] if rng.random() < 0.7 else foreign
    return dict(ts=ts, amount=amount, mcc=rng.choice(card["fav_mcc"], size=k),
                merchant=rng.choice(card["fav_merch"], size=k),
                device=np.full(k, -3_000_000 - rng.integers(10_000)),
                ip_country=np.full(k, ipc),
                channel=np.full(k, CHANNELS["wallet_token"]))


def simulate(cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    cols = {c: [] for c in ["card_id", "ts", "amount", "mcc", "merchant", "device",
                            "ip_country", "card_country", "channel", "label", "fraud_type"]}

    def add(card_id, home, ev, label, ftype):
        k = len(ev["ts"])
        cols["card_id"].append(np.full(k, card_id))
        for c in ["ts", "amount", "mcc", "merchant", "device", "ip_country", "channel"]:
            cols[c].append(np.asarray(ev[c]))
        cols["card_country"].append(np.full(k, home))
        cols["label"].append(np.full(k, label))
        cols["fraud_type"].append(np.full(k, ftype, dtype=object))

    for cid in range(cfg.n_cards):
        card = _legit_card(rng, cfg, cid)
        if card is None:
            continue
        add(cid, card["home"], card, 0, "legit")
        if rng.random() < cfg.p_card_testing:
            add(cid, card["home"], _card_testing(rng, cfg, rng.uniform(0, cfg.n_days * DAY)), 1, "card_testing")
        if rng.random() < cfg.p_stolen_card:
            add(cid, card["home"], _stolen_card(rng, cfg, rng.uniform(0, cfg.n_days * DAY), card["home"]), 1, "stolen_card")
        if rng.random() < cfg.p_token_theft:
            t0 = rng.uniform(cfg.token_theft_start_day * DAY, cfg.n_days * DAY)
            add(cid, card["home"], _token_theft(rng, cfg, t0, card), 1, "token_theft")

    df = pd.DataFrame({c: np.concatenate(v) for c, v in cols.items()})
    df = df[df.ts < cfg.n_days * DAY].sort_values(["ts"], kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", np.arange(len(df)))
    df["ts"] = df["ts"].astype("int64")
    for c in ["mcc", "merchant", "device", "ip_country", "card_country", "channel", "label", "card_id"]:
        df[c] = df[c].astype("int64")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/transactions.parquet")
    ap.add_argument("--n-cards", type=int, default=SimConfig.n_cards)
    ap.add_argument("--seed", type=int, default=SimConfig.seed)
    a = ap.parse_args()
    df = simulate(SimConfig(n_cards=a.n_cards, seed=a.seed))
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"{len(df):,} transactions | fraud rate {df.label.mean():.3%}")
    print(df.fraud_type.value_counts().to_string())


if __name__ == "__main__":
    main()
