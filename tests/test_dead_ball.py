"""Tech 2: boundary noise, finite memory, halftime isolation and observed restarts."""

import numpy as np
import pytest

from mtah import A, H, features, hmm
from mtah.dead_ball import out_of_bounds, set_piece_restart
from mtah.load import Tracking


def tracking(n=200):
    players = np.tile([[20.0, 30.0], [40.0, 30.0]], (n, 1, 1))
    return Tracking(
        players,
        np.full((n, 2), 2.5),
        np.tile([20.2, 30.0], (n, 1)),
        np.zeros(n),
        np.array([H, A]),
        ("Home", "Away"),
    )


def test_outside_confirmation_backdates_and_expires_without_borrowing_raw_positions():
    tr = tracking()
    tr.ball[30:35] = [-2, 30]
    tr.ball[35:100] = np.nan
    original = tr.ball.copy()
    cfg = {"include_missing": False, "outside_memory_frames": 20}
    dead = out_of_bounds(tr, features.build(tr), [], cfg)
    assert not dead[:30].any()
    assert dead[30:55].all()
    assert not dead[55:].any()
    np.testing.assert_array_equal(tr.ball, original)


def test_single_outside_spike_and_tolerance_margin_do_not_create_stoppage():
    tr = tracking()
    tr.ball[30] = [-3, 30]
    tr.ball[80:90] = [-0.25, 30]
    assert not out_of_bounds(tr, features.build(tr), [], {"include_missing": False}).any()


def test_outside_state_releases_at_first_confirmed_inside_frame():
    tr = tracking()
    tr.ball[20:40] = [-2, 30]
    dead = out_of_bounds(tr, features.build(tr), [], {"include_missing": False})
    assert dead[20:40].all()
    assert not dead[40:].any()


def test_missing_ball_fallback_is_half_local(monkeypatch):
    tr = tracking(120)
    monkeypatch.setattr(hmm, "N_FRAMES", 120)
    monkeypatch.setattr(hmm, "match_runs", lambda: [(10, 50), (70, 110)])
    tr.ball[45:50] = [-2, 30]
    tr.ball[70:110] = np.nan
    dead = out_of_bounds(tr, features.build(tr), [], {"window": 100})
    assert dead[70:110].all()  # Ball in another half cannot rescue this half's missing frames.
    assert not dead[:10].any() and not dead[50:70].any() and not dead[110:].any()
    isolated = out_of_bounds(tr, features.build(tr), [], {"include_missing": False})
    assert not isolated[70:110].any()


def restart_scene():
    tr = tracking()
    tr.ball[20:60] = np.nan
    tr.player_speed[20:61] = 0
    tr.ball[60:90, 0] = 20.2 + np.arange(30) * 0.2  # 5 m/s after a stopped/missing spell.
    tag = {
        "code": " set pieces ",
        "start_frame": 30,
        "end_frame": 85,
        "tags": {"Type": "throw-in", "Team": "N/A"},
    }
    return tr, tag


def test_restart_uses_motion_not_tag_start_and_accepts_unknown_team():
    tr, tag = restart_scene()
    feats = features.build(tr)
    dead = set_piece_restart(tr, feats, [tag], {"include_missing": False})
    assert dead[20:61].all()
    assert not dead[61:85].any()
    assert feats["_restart_hints"][0]["frame"] == 61
    assert feats["_restart_hints"][0]["tags"]["Team"] == "N/A"


def test_false_tag_does_not_paint_live_ball_dead():
    tr = tracking()
    tr.ball[:, 0] = 10 + np.arange(200) * 0.2
    tag = {
        "code": "SET PIECES",
        "start_frame": 30,
        "end_frame": 80,
        "tags": {"Type": "Free Kick", "Team": "Home"},
    }
    assert not set_piece_restart(tr, features.build(tr), [tag], {"include_missing": False}).any()


def test_unknown_tag_and_reversed_tag_order_do_not_change_detector():
    tr, tag = restart_scene()
    unknown = {**tag, "tags": {"Type": "Unknown", "Team": "Home"}}
    cfg = {"include_missing": False}
    assert not set_piece_restart(tr, features.build(tr), [unknown], cfg).any()
    a = set_piece_restart(tr, features.build(tr), [unknown, tag], cfg)
    b = set_piece_restart(tr, features.build(tr), [tag, unknown], cfg)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "cfg",
    [
        {"margin_m": -1},
        {"outside_confirm_frames": 0},
        {"outside_memory_frames": True},
        {"include_missing": "yes"},
        {"max_ball_speed_mps": float("nan")},
    ],
)
def test_invalid_detector_parameters_fail_clearly(cfg):
    tr = tracking()
    with pytest.raises(ValueError):
        out_of_bounds(tr, features.build(tr), [], cfg)


@pytest.mark.parametrize(
    "cfg",
    [
        {"restart_types": "Throw-in"},
        {"restart_types": [None]},
        {"restart_ball_mps": 100},
        {"still_ball_mps": 5},
    ],
)
def test_invalid_restart_configuration_is_not_silently_ignored(cfg):
    tr = tracking()
    with pytest.raises(ValueError):
        set_piece_restart(tr, features.build(tr), [], cfg)


def test_restart_tag_can_overlap_start_of_a_graded_half(monkeypatch):
    tr, tag = restart_scene()
    monkeypatch.setattr(hmm, "N_FRAMES", 200)
    monkeypatch.setattr(hmm, "match_runs", lambda: [(25, 100), (120, 180)])
    tag["start_frame"] = 20
    feats = features.build(tr)
    result = set_piece_restart(tr, feats, [tag], {"include_missing": False})
    assert result[25:61].all() and not result[:25].any()
    assert feats["_restart_hints"][0]["frame"] == 61
