"""Cached batch evaluation must be identical to independent model/half inference."""

import numpy as np
import pytest

from mtah import A, H
from mtah.config import load_config
from mtah.external import predict_external, predict_many_external
from mtah.load import Tracking
from mtah.paths import CONFIG_DIR
from mtah.predict import prepare_features, run_pipeline


def tracking():
    n = 160
    players = np.tile([[20.0, 30.0], [40.0, 30.0]], (n, 1, 1))
    ball = np.tile([20.2, 30.0], (n, 1))
    ball[80:] = [40.2, 30.0]
    ball[30:40] = np.nan
    return Tracking(players, np.full((n, 2), 2.0), ball, np.zeros(n), np.array([H, A]), ("H", "A"))


def test_batch_rules_and_hmm_match_independent_models_in_both_orders_and_halves():
    tr = tracking()
    runs = [(5, 75), (85, 155)]
    configs = [
        load_config(CONFIG_DIR / name)
        for name in ["default.toml", "exp_bm_out.toml", "exp_pos_control.toml", "exp_hmm_v2.toml"]
    ]
    expected = np.stack([predict_external(c, tr, runs) for c in configs])
    np.testing.assert_array_equal(predict_many_external(configs, tr, runs), expected)
    np.testing.assert_array_equal(predict_many_external(configs[::-1], tr, runs), expected[::-1])


def test_explicit_feature_cache_rejects_a_different_tracking_input():
    first, second = tracking(), tracking()
    with pytest.raises(ValueError, match="different tracking"):
        run_pipeline(load_config(), second, [], prepare_features(first))
