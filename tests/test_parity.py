"""Training/serving parity: the online scorer must see exactly the features the
model was trained on."""
import numpy as np
import pandas as pd
import pytest

from seqfraud.features import AGG_FEATURES, OnlineFeatureState
from seqfraud.sequences import OnlineSequenceState, build_sequences
from seqfraud.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def small():
    return simulate(SimConfig(n_cards=400, n_days=40, token_theft_start_day=25, p_card_testing=0.05, p_stolen_card=0.05, p_token_theft=0.05, seed=3))


def test_simulator_has_all_typologies(small):
    big = simulate(SimConfig(n_cards=3000, seed=1))
    assert set(big.fraud_type.unique()) == {"legit", "card_testing", "stolen_card", "token_theft"}
    # token theft is an emerging pattern: never before its start day
    tt = big[big.fraud_type == "token_theft"]
    assert (tt.ts >= SimConfig.token_theft_start_day * 86_400).all()


def test_online_features_match_spark(small):
    pytest.importorskip("pyspark")
    from seqfraud.features import spark_features
    batch = spark_features(small).set_index("txn_id")
    state = OnlineFeatureState()
    online = []
    for rec in small.sort_values("txn_id").to_dict("records"):
        online.append({"txn_id": rec["txn_id"], **{k: state.features(rec)[k] for k in AGG_FEATURES}})
        state.update(rec)
    online = pd.DataFrame(online).set_index("txn_id").loc[batch.index]
    np.testing.assert_allclose(online[AGG_FEATURES].to_numpy(float),
                               batch[AGG_FEATURES].to_numpy(float), rtol=1e-6, atol=1e-6)


def test_online_sequences_match_batch(small):
    df = small.sort_values("txn_id").reset_index(drop=True)
    num, cat, mask = build_sequences(df)
    st = OnlineSequenceState()
    for i, rec in enumerate(df.to_dict("records")):
        n, c, m = st.sequence(rec)
        st.update(rec)
        np.testing.assert_array_equal(m, mask[i])
        np.testing.assert_allclose(n, num[i], atol=1e-6)
        np.testing.assert_array_equal(c, cat[i])
