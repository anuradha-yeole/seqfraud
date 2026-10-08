# seqfraud

**Transaction-sequence models for real-time payment fraud.**

[![tests](https://github.com/anuradha-yeole/seqfraud/actions/workflows/tests.yml/badge.svg)](https://github.com/anuradha-yeole/seqfraud/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> How fast can a fraud model catch a pattern it has never seen?

This repo builds an end-to-end card-fraud pipeline — Spark feature backfill, a
LightGBM baseline, a PyTorch Transformer over each card's recent transactions,
and a real-time FastAPI scorer — then uses it to study one question. When a new
attack such as **wallet-token theft** appears, how many labelled cases does
each model need before it can block the attack?

**Short answer:** neither model catches token theft before seeing it. After
**30 labelled compromised cards**, the sequence model catches **82%** of token
theft at a 0.1% false-positive rate, against **9%** for the gradient-boosted
baseline — while keeping 99.9% recall on the fraud types it already knew.

| | LightGBM | Seq Transformer |
|---|---|---|
| Token theft, never seen | 2.9% | 0.0% |
| Token theft, 30 labelled cards | 8.7% | **81.9%** |
| Known fraud recall (retained) | 99.6% | 99.9% |
| p99 scoring latency | — | 2.57 ms |

---

## Contents

- [Why sequences](#why-sequences)
- [Data](#data)
- [Pipeline](#pipeline)
- [Results](#results)
- [Quickstart](#quickstart)
- [Serving API](#serving-api)
- [Repo layout](#repo-layout)
- [Limitations](#limitations)

---

## Why sequences

Stolen-token fraud is built to look normal. The attacker replays the victim's
token at merchants the victim already uses, for amounts in the victim's range,
usually through a residential proxy in the victim's country. Taken one
transaction at a time, little looks unusual.

What gives it away is **context**. A device that first appeared on this card
twenty minutes ago is suddenly paying through a wallet token, several times in
a row. The sequence model describes each of the last 16 transactions *relative
to the current one* — same device? same IP country? how long ago? — so it can
represent that context directly. Per-transaction velocity aggregates cannot.

## Data

Real payment logs aren't public, so [`seqfraud/simulate.py`](seqfraud/simulate.py)
generates a stream of **1.52M transactions** (40k cards, 90 days, 0.63% fraud).
Legitimate behaviour includes realistic noise: diurnal timing, favourite
merchants, device upgrades and trips abroad. Three fraud typologies are injected:

| Typology | What it looks like | When it appears |
|---|---|---|
| `card_testing` | bursts of $0.50–$3 authorisations across many merchants, sometimes a cash-out | whole period |
| `stolen_card` | card-not-present spend on high-risk categories from a new device, half via in-country proxies | whole period |
| `token_theft` | stolen wallet token replayed from a new device; victim-like merchants and amounts; 70% in-country proxy | **day 70 onward only** |

Holding `token_theft` back until day 70 is what makes the emerging-fraud
experiment possible: models trained on days 0–56 have never seen it.

## Pipeline

```
simulate.py ─▶ features.py (PySpark windows) ─▶ train.py ─▶ adapt.py
                    │                               │
                    └── OnlineFeatureState ◀──── serve.py (FastAPI, TorchScript)
                        OnlineSequenceState          benchmark.py
```

**Features (Spark).** Point-in-time velocity features use window functions over
strictly earlier events (`rangeBetween(-W, -1)`): counts and spend over 1 h and
24 h, distinct merchants in the past hour, time since the last transaction,
first time seen for a device or IP country, and amount relative to the 30-day
mean. A 1.5M-row backfill takes 35 s on 2 cores.

**No training/serving skew.** The scorer recomputes the same features
incrementally from per-card state. [`tests/test_parity.py`](tests/test_parity.py)
checks that the online values equal the Spark backfill exactly, and the replay
benchmark confirms online and offline model scores agree to within `1.5e-7`.
`encode_steps` in [`sequences.py`](seqfraud/sequences.py) is the single source
of truth shared by the batch builder and the online scorer, so the two cannot
drift apart.

**Models.** LightGBM on the 12 aggregates (early-stopped on validation PR-AUC),
and `SeqFraudModel`: a 2-layer pre-norm Transformer encoder (d=64, 4 heads) over
the last 16 transactions plus the current one. The current-transaction token's
output is concatenated with the aggregates and fed to an MLP head. Training uses
negative down-sampling (all fraud, 25% of legit).

**Evaluation.** Out-of-time split (train days 0–56, validation 56–63, test
63–90). The decision threshold is set on validation at a fixed **0.1% false
positive rate** — the share of good customers who would be blocked. Reported
metrics are recall, dollar-weighted recall and PR-AUC. Confidence intervals come
from a **card-level cluster bootstrap**, because fraud transactions on the same
card are not independent.

## Results

### 1. Known vs. emerging fraud

Models trained on days 0–56, evaluated on days 63–90 at 0.1% FPR:

| Metric | LightGBM | Seq Transformer |
|---|---|---|
| PR-AUC | 0.657 | 0.644 |
| ROC-AUC | 0.846 | 0.951 |
| Overall recall | 61.0% | 58.0% |
| Dollar-weighted recall | 52.2% | 47.6% |
| Card testing recall | 99.9% | 100% |
| Stolen card recall | 99.6% | 100% |
| **Token theft recall** | **7.5%** | **0.1%** |

Both models handle the typologies they were trained on and miss the new one
almost entirely. This is the expected failure mode for supervised fraud models,
and it motivates the next experiment. (A score blend does not help here either —
see `artifacts/report.json`.)

### 2. Adapting to token theft from a few labels

[`adapt.py`](seqfraud/adapt.py) releases labels for **N** compromised cards from
days 70–77, simulating the first chargebacks to come back. Unreported cases are
dropped rather than mislabelled. LightGBM is retrained from scratch; the
Transformer is fine-tuned from its day-56 checkpoint for 2 epochs. Evaluation
covers days 77–90: 1,132 token-theft transactions on 248 cards.

| Labelled cards (txns) | LightGBM token theft | Seq token theft | Seq other-fraud recall |
|---|---|---|---|
| 0 (0) | 2.9% | 0.0% | 100% |
| 10 (44) | 8.5% | 44.4% | 99.9% |
| **30 (128)** | **8.7%** | **81.9%** | **99.9%** |
| 100 (431) | 9.5% | 85.1% | 99.9% |
| 132, all (578) | 9.5% | 88.3% | 99.9% |

More labels barely help LightGBM, because the signal is not in its features.
`new_device` fires only on the first transaction of an attack, and the
in-country proxy hides the geography. The Transformer already encodes "device
first seen minutes ago, now paying with a wallet token", so a handful of labels
is enough to make that signal useful.

**Practical takeaway:** a GBDT would need hand-built device-tenure features to
compete — and those features only exist after a human has studied the attack.
The sequence model buys back that analyst lead time.

### 3. Real-time serving

[`benchmark.py`](seqfraud/benchmark.py) replays 20k test-period authorisations
one at a time through the online scorer, covering the feature update, sequence
encoding and TorchScript inference on a **single CPU core**:

| p50 | p95 | p99 | Throughput |
|---|---|---|---|
| 0.99 ms | 1.68 ms | 2.57 ms | ~870 req/s per core |

## Quickstart

Requires Python 3.12 and **Java 17+** (for PySpark).

```bash
pip install -r requirements.txt
```

```bash
make data      # 1.52M synthetic transactions (~20 s)
make train     # Spark backfill + LightGBM + Transformer (~12 min on 2 CPU cores)
make bench     # online latency + online/offline parity
make test      # parity + API tests
make serve     # POST /v1/score on :8000
```

The label-efficiency experiment is a separate step (~20 min):

```bash
python -m seqfraud.adapt --n-cards 0,10,30,100,all
```

Every stage is also a module with its own flags, so results are reproducible
end to end:

| Module | Purpose | Key flags |
|---|---|---|
| `seqfraud.simulate` | generate the transaction stream | `--out`, `--n-cards`, `--seed` |
| `seqfraud.train` | Spark backfill + train both models | `--data`, `--out`, `--epochs`, `--fpr`, `--seed` |
| `seqfraud.adapt` | label-efficiency experiment | `--artifacts`, `--n-cards`, `--fpr`, `--epochs`, `--seed` |
| `seqfraud.benchmark` | latency + online/offline parity | `--data`, `--artifacts`, `--n` |
| `seqfraud.serve` | FastAPI scoring service | `SEQFRAUD_ARTIFACTS` env var |

### Pre-trained artifacts

Committed under [`artifacts/`](artifacts/), so you can serve and inspect results
without running training:

| File | Contents |
|---|---|
| `report.json` | per-model metrics, per-typology recall, bootstrap CIs, training log |
| `adaptation.json` | label-efficiency curve (section 2) |
| `latency.json` | latency percentiles + max online/offline score difference |
| `seq_model.pt` | TorchScript Transformer used by the scorer |
| `seq_model_state.pt` | `state_dict` checkpoint, for fine-tuning |
| `lgbm.txt` | LightGBM baseline model |
| `serving_config.json` | decision threshold + aggregate feature order |

## Serving API

```bash
make serve    # or: uvicorn seqfraud.serve:app --port 8000
```

`POST /v1/score` scores one authorisation. `GET /healthz` is a liveness probe.

```bash
curl -X POST localhost:8000/v1/score -H 'content-type: application/json' -d \
 '{"txn_id":1,"card_id":42,"ts":6100000,"amount":84.2,"mcc":3,"merchant":17,
   "device":-3000001,"ip_country":5,"card_country":5,"channel":2}'
```

```json
{"txn_id": 1, "risk_score": 6.1e-08, "decision": "allow", "latency_ms": 1.04}
```

Request fields:

| Field | Type | Meaning |
|---|---|---|
| `txn_id` | int | authorisation id, echoed back |
| `card_id` | int | key for the per-card feature and sequence state |
| `ts` | int | unix seconds |
| `amount` | float | authorisation amount |
| `mcc` | int | merchant category |
| `merchant` | int | merchant id |
| `device` | int | device fingerprint |
| `ip_country` / `card_country` | int | geography, compared to detect proxies |
| `channel` | int | 0 card-present, 1 card-not-present, 2 wallet token |

`decision` compares `risk_score` against the threshold calibrated at 0.1% FPR
(`serving_config.json`). State is updated **after** scoring, so each request
only ever sees strictly earlier events — and each call updates that card's
state, so a later burst from the same new device is scored with the earlier
calls as context. The first call pays a one-time model warm-up cost.

### Docker

The image installs CPU-only Torch and skips Spark and LightGBM, since serving
needs neither:

```bash
docker build -t seqfraud .
```

```bash
docker run -p 8000:8000 seqfraud
```

## Repo layout

```
seqfraud/
├── simulate.py     synthetic transaction stream + 3 fraud typologies
├── features.py     Spark point-in-time windows + OnlineFeatureState (parity twin)
├── sequences.py    per-card history encoding; encode_steps shared batch/online
├── model.py        SeqFraudModel: Transformer encoder + aggregate MLP head
├── metrics.py      operating-point metrics + card-level cluster bootstrap
├── train.py        backfill, train LightGBM + Transformer, calibrate threshold
├── adapt.py        label-efficiency experiment on emerging token theft
├── benchmark.py    online latency + online/offline score parity
└── serve.py        FastAPI scorer over TorchScript
tests/
├── test_parity.py  online features == Spark backfill, exactly
└── test_serve.py   scoring API contract
```

## Limitations

- **The data is simulated.** The typologies follow publicly documented patterns,
  but absolute numbers will not transfer to real traffic. The *relative*
  findings are what this repo is about: supervised models are blind to novel
  attacks, and sequence context improves label efficiency.
- **Labels are assumed to arrive cleanly.** Real chargebacks arrive 30–90 days
  late and are noisy.
- **Serving state is in-process.** The scorer keeps per-card state in memory in
  a single process, behind a lock. In production that state would live in a
  low-latency key-value store, sharded by card.

## License

MIT — see [LICENSE](LICENSE).
