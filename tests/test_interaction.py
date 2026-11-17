"""Interaction features must preserve symmetry, missingness, and halftime isolation."""

import numpy as np
import pytest
from train_supervised import possession_balanced_samples, training_weights

from mtah import A, D, H, X
from mtah.config import load_config
from mtah.interaction import interaction_matrix, interaction_names
from mtah.load import Tracking
from mtah.paths import CONFIG_DIR
from mtah.supervised import feature_matrix, feature_names, swap_features


def fixture():
    n = 180
    players = np.tile([[10.0, 20.0], [30.0, 20.0]], (n, 1, 1))
    ball = np.c_[np.linspace(10, 30, n), np.full(n, 20.0)]
    tracking = Tracking(players, np.zeros((n, 2)), ball, np.zeros(n), np.array([H, A]), ("H", "A"))
    cfg = load_config(CONFIG_DIR / "exp_supervised_v3.toml")["supervised"]["features"]
    return tracking, cfg


def test_interactions_preserve_team_symmetry_and_ignore_unused_channels_and_identities():
    tr, cfg = fixture()
    first = feature_matrix(tr, cfg)
    tr.slot_team = 1 - tr.slot_team
    tr.ball_extra[:] = np.arange(tr.n_frames) % 3
    tr.team_names = ("SECRET", "KEY")
    np.testing.assert_allclose(
        feature_matrix(tr, cfg),
        swap_features(first, feature_names(cfg)),
        atol=1e-14,
        rtol=1e-14,
    )


def test_interactions_never_observe_contacts_across_halftime_or_unknown_gap():
    tr, cfg = fixture()
    runs = [(5, 80), (100, 170)]
    full = feature_matrix(tr, cfg, runs)
    assert np.isnan(full[80:100]).all()
    for a, b in runs:
        part = Tracking(
            tr.players[a:b],
            tr.player_speed[a:b],
            tr.ball[a:b],
            tr.ball_extra[a:b],
            tr.slot_team,
            tr.team_names,
        )
        np.testing.assert_array_equal(full[a:b], feature_matrix(part, cfg))


def test_contact_age_tracks_evidence_instead_of_persisting_opposite_owner():
    tr, cfg = fixture()
    tr.ball[:] = [20, 20]
    tr.ball[20] = [10.1, 20]
    tr.ball[100] = [29.9, 20]
    original = tr.ball.copy()
    values = interaction_matrix(tr, cfg)
    names = interaction_names(cfg)
    assert values[30, names.index("H_contact_past_age_0.75")] == 10 / 250
    assert values[30, names.index("A_contact_past_age_0.75")] == 1
    assert values[30, names.index("A_contact_future_age_0.75")] == 70 / 250
    np.testing.assert_array_equal(tr.ball, original)


def test_missing_ball_or_team_cannot_create_a_contact():
    tr, cfg = fixture()
    tr.ball[:] = np.nan
    tr.players[:, 1] = np.nan
    values = interaction_matrix(tr, cfg)
    assert not np.isinf(values).any()
    for i, name in enumerate(interaction_names(cfg)):
        if "_age_" in name:
            assert (values[:, i] == 1).all()


@pytest.mark.parametrize(
    "key,value",
    [("contact_radii_m", [-1]), ("interaction_offsets", [0]), ("interaction_offsets", [1, 1])],
)
def test_invalid_interaction_configuration_fails(key, value):
    tr, cfg = fixture()
    cfg[key] = value
    with pytest.raises(ValueError):
        interaction_matrix(tr, cfg)


def test_short_spell_training_weights_use_only_known_training_runs():
    reference = np.array([H] * 100 + [A] * 10 + [D] * 10 + [X] * 10 + [H] * 10)
    weights = training_weights(reference, [(0, 120), (130, 140)], 0.5, 1.5, 3)
    assert weights[104] > weights[50]
    assert weights[100] > weights[104]
    assert weights[110] == 1.5
    assert (weights[120:130] == 1).all()
    assert weights[130] == weights[135]  # A new half does not manufacture a transition.


def test_possession_balanced_sampling_keeps_single_frame_spells_and_equal_object_mass():
    reference = np.array([H] * 200 + [A] + [H] * 5 + [D] * 50 + [X] * 10)
    cfg = {"stride": 5, "possession_mass": 125.0, "short_spell_frames": 50}
    selected, weights = possession_balanced_samples(reference, [(0, 256)], cfg)
    assert 200 in selected and set(range(201, 206)) <= set(selected)
    for a, b in [(0, 200), (200, 201), (201, 206)]:
        assert weights[(selected >= a) & (selected < b)].sum() == pytest.approx(125.0)
    assert selected.max() < 256
    assert (weights[reference[selected] == D] == 1).all()
