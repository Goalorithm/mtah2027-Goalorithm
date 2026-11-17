"""Independent decoder checks and regression coverage for Tech 1's sequence model."""

import itertools

import numpy as np
import pytest

from mtah import A, D, H, X, features, hmm
from mtah.config import load_config
from mtah.io import sha256, write_submission
from mtah.load import Tracking
from mtah.paths import CONFIG_DIR
from mtah.predict import run_pipeline


def config(version=1):
    return load_config(CONFIG_DIR / f"exp_hmm_v{version}.toml")


def tracking(n=120):
    players = np.zeros((n, 2, 2), dtype=float)
    players[:, 0] = [20, 30]
    players[:, 1] = [40, 30]
    ball = np.tile([20.2, 30.0], (n, 1))
    return Tracking(
        players,
        np.full((n, 2), 2.5),
        ball,
        np.zeros(n),
        np.array([H, A]),
        ("Barcelona", "Athletic Club"),
    )


def test_viterbi_agrees_with_exhaustive_path_search():
    rng = np.random.default_rng(17)
    transition = rng.dirichlet(np.ones(3), size=3)
    initial = rng.dirichlet(np.ones(3))
    evidence = rng.normal(size=(6, 3))

    def path_score(path):
        total = np.log(initial[path[0]]) + evidence[0, path[0]]
        for t in range(1, len(path)):
            total += np.log(transition[path[t - 1], path[t]]) + evidence[t, path[t]]
        return total

    candidates = itertools.product(range(3), repeat=len(evidence))
    expected = max(candidates, key=path_score)
    np.testing.assert_array_equal(hmm.viterbi(evidence, transition, initial), expected)


def test_viterbi_rejects_bad_probabilities_and_unreachable_paths():
    with pytest.raises(ValueError, match="sum to 1"):
        hmm.viterbi(np.zeros((4, 3)), np.ones((3, 3)), [1, 0, 0])
    with pytest.raises(ValueError, match="no reachable"):
        hmm.viterbi([[0, -np.inf], [-np.inf, 0]], np.eye(2), [1, 0])
    with pytest.raises(ValueError, match="negative infinity"):
        hmm.viterbi([[np.nan, 0]], np.eye(2), [1, 0])


def test_viterbi_ties_and_long_sequences_are_deterministic():
    tied = hmm.viterbi(np.zeros((12, 3)), np.full((3, 3), 1 / 3), np.full(3, 1 / 3))
    np.testing.assert_array_equal(tied, np.zeros(12))
    emissions = np.tile([-1000.0, -1001.0], (12000, 1))
    result = hmm.viterbi(emissions, [[0.999, 0.001], [0.001, 0.999]], [0.5, 0.5])
    assert (result == 0).all()
    assert hmm.viterbi(np.empty((0, 2)), np.eye(2), [1, 0]).size == 0


def test_ball_history_and_interpolation_preserve_observations():
    ball = np.array(
        [
            [np.nan, np.nan],
            [0, 0],
            [np.nan, np.nan],
            [2, 0],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan],
            [6, 0],
            [np.nan, np.nan],
        ]
    )
    original = ball.copy()
    extra = hmm.ball_features(ball, max_gap=1, max_speed_mps=100)
    np.testing.assert_array_equal(extra["ball_gap"], [-1, 0, 1, 0, 1, 2, 3, 0, 1])
    assert np.isnan(extra["ball_last_xy"][0]).all()
    np.testing.assert_array_equal(extra["ball_last_xy"][4:7], [[2, 0]] * 3)
    np.testing.assert_array_equal(extra["ball_xy_filled"][2], [1, 0])
    assert np.isnan(extra["ball_xy_filled"][4:7]).all()
    assert np.isnan(extra["ball_xy_filled"][-1]).all()
    np.testing.assert_array_equal(ball, original)


def test_interpolation_resets_at_boundaries_and_rejects_teleports():
    ball = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [3, 0]])
    extra = hmm.ball_features(ball, 10, 100, runs=[(0, 2), (2, 4)])
    assert np.isnan(extra["ball_xy_filled"][1:3]).all()
    assert extra["ball_gap"][2] == -1
    assert np.isnan(extra["ball_last_xy"][2]).all()
    teleport = np.array([[0, 0], [np.nan, np.nan], [100, 0]])
    assert np.isnan(hmm.ball_features(teleport, 1, 45)["ball_xy_filled"][1]).all()
    partial = np.array([[1, np.nan], [2, 2]])
    assert hmm.ball_features(partial, 1, 45)["ball_gap"][0] == -1


@pytest.mark.parametrize("version", [1, 2])
def test_sequence_model_suppresses_a_single_opponent_touch(version):
    tr = tracking()
    tr.ball[60] = [39.8, 30]
    feats = features.build(tr)
    raw_distances = feats["team_dist"].copy()
    codes = hmm.infer(tr, feats, [], config(version)["hmm"])
    assert (codes == H).all()
    np.testing.assert_array_equal(feats["team_dist"], raw_distances)


@pytest.mark.parametrize("version", [1, 2])
def test_sequence_model_detects_sustained_control_and_outside_ball(version):
    tr = tracking(180)
    tr.ball[60:120] = [39.8, 30]
    tr.ball[120:] = [-3, 30]
    result = hmm.infer(tr, features.build(tr), [], config(version)["hmm"])
    assert (result[10:50] == H).all()
    assert (result[70:110] == A).all()
    assert (result[130:] == D).all()


def test_half_decoding_does_not_bridge_ungraded_frames():
    tr = tracking(100)
    tr.ball[60:] = [39.8, 30]
    runs = [(10, 40), (60, 90)]
    codes = hmm.infer(tr, features.build(tr), [], config()["hmm"], runs=runs)
    assert (codes[10:40] == H).all() and (codes[60:90] == A).all()
    assert (codes[:10] == X).all() and (codes[40:60] == X).all()
    assert (codes[90:] == X).all()
    with pytest.raises(ValueError, match="ordered, disjoint"):
        hmm.sequence_runs(100, [(10, 40), (30, 50)])


def test_defensive_tag_inverts_team_and_unknown_team_is_ignored():
    tr = tracking(20)
    tagging = [
        {
            "code": "defensive transition",
            "start_frame": 2,
            "end_frame": 8,
            "tags": {"Team": "Barcelona"},
        },
        {"code": "SET PIECES", "start_frame": 0, "end_frame": 20, "tags": {"Team": "N/A"}},
    ]
    signal = hmm._tagging_evidence(tr, tagging, config()["hmm"])
    np.testing.assert_array_equal(signal[2:8], np.full(6, -1))
    assert (signal[:2] == 0).all() and (signal[8:] == 0).all()


def test_two_adapters_share_one_decode_and_reproduce_bytes(monkeypatch, tmp_path):
    real_infer = hmm.infer
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real_infer(*args, **kwargs)

    monkeypatch.setattr(hmm, "infer", counted)
    tr = tracking()
    first = run_pipeline(config(), tr, [])
    assert len(calls) == 1
    second = run_pipeline(config(), tr, [])
    assert len(calls) == 2
    a = write_submission(tmp_path / "a.csv", first)
    b = write_submission(tmp_path / "b.csv", second)
    assert sha256(a) == sha256(b)


def test_configuration_change_invalidates_shared_decode():
    tr = tracking()
    feats, tagging = features.build(tr), []
    cfg = config()["hmm"]
    hmm.shared_inference(tr, feats, tagging, cfg)
    first = feats["_hmm_result"]
    cfg["team_weight"] += 0.1
    hmm.shared_inference(tr, feats, tagging, cfg)
    assert feats["_hmm_result"] is not first


def test_invalid_model_config_is_reported_clearly():
    tr = tracking()
    cfg = config()["hmm"]
    cfg["distance_scale_m"] = 0
    with pytest.raises(ValueError, match="distance_scale_m"):
        hmm.infer(tr, features.build(tr), [], cfg)


def test_outside_memory_expires_without_altering_raw_history():
    ball = np.array([[-3.0, 30], *[[np.nan, np.nan]] * 7, [20.2, 30]])
    history = hmm.ball_features(ball, max_gap=0, max_speed_mps=45)
    original = {key: value.copy() for key, value in history.items()}
    legacy = hmm.outside_signal(history, margin_m=0.5)
    bounded = hmm.outside_signal(history, margin_m=0.5, memory_frames=4)
    assert legacy[0] == bounded[0] == 1
    assert (np.diff(bounded[:5]) < 0).all()
    assert (bounded[4:] == 0).all() and (legacy[:8] == 1).all()
    observed_only = hmm.outside_signal(history, 0.5, memory_frames=0)
    assert np.count_nonzero(observed_only) == 1
    for key, value in original.items():
        np.testing.assert_array_equal(history[key], value)


def test_expired_outside_observation_changes_only_its_evidence_channel():
    tr = tracking(400)
    tr.ball[50] = [-3, 30]
    tr.ball[51:320] = np.nan
    feats = features.build(tr)
    old = hmm.log_evidence(tr, feats, [], config(1)["hmm"])
    new = hmm.log_evidence(tr, feats, [], config(2)["hmm"])
    np.testing.assert_array_equal(old[:, :2], new[:, :2])
    np.testing.assert_array_equal(old[:51], new[:51])
    np.testing.assert_array_equal(old[320:], new[320:])
    assert (new[101:320, D] < old[101:320, D]).all()
    # Removing stale geometry must restore the same evidence as an old inside observation.
    inside = tracking(400)
    inside.ball[51:320] = np.nan
    neutral = hmm.log_evidence(inside, features.build(inside), [], config(2)["hmm"])
    np.testing.assert_array_equal(new[101:300], neutral[101:300])


def test_outside_memory_does_not_cross_halves_or_invent_unknown_observations():
    ball = np.full((120, 2), np.nan)
    ball[10:40] = [-3, 30]
    history = hmm.ball_features(ball, 0, 45, runs=[(10, 40), (60, 110)])
    signal = hmm.outside_signal(history, 0.5, memory_frames=50)
    assert (signal[10:40] == 1).all()
    assert (signal[60:] == 0).all() and (history["ball_gap"][60:] == -1).all()


@pytest.mark.parametrize("version", [1, 2])
def test_long_ball_dropout_does_not_force_dead_when_other_evidence_supports_live_play(version):
    tr = tracking(5000)
    tr.ball[250:4750] = np.nan
    codes = hmm.infer(tr, features.build(tr), [], config(version)["hmm"])
    assert (codes == H).all()


def test_expired_geometry_does_not_force_a_restart_during_a_slow_stoppage():
    tr = tracking(1000)
    tr.player_speed[:] = 0
    tr.ball[:20] = [-3, 30]
    tr.ball[20:] = np.nan
    codes = hmm.infer(tr, features.build(tr), [], config(2)["hmm"])
    assert (codes == D).all()


def test_v2_tracks_sustained_control_stoppage_and_restart_despite_brief_touches():
    tr = tracking(1200)
    tr.ball[400:800] = [39.8, 30]
    tr.ball[800:1000] = [-3, 30]
    tr.ball[[60, 120, 180, 240, 300]] = [39.8, 30]
    codes = hmm.infer(tr, features.build(tr), [], config(2)["hmm"])
    for start, stop, expected in [(10, 350, H), (450, 750, A), (850, 950, D), (1050, 1150, H)]:
        assert (codes[start:stop] == expected).all()


@pytest.mark.parametrize("bad", [-1, True, 2.5, float("inf"), "50", None])
def test_v2_rejects_invalid_outside_memory_configuration(bad):
    tr = tracking()
    cfg = config(2)["hmm"]
    cfg["outside_memory_frames"] = bad
    with pytest.raises(ValueError, match="outside_memory_frames"):
        hmm.infer(tr, features.build(tr), [], cfg)
