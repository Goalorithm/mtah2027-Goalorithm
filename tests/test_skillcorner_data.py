"""Sparse provider labels must not fabricate dead balls or bridge period boundaries."""

import copy

import numpy as np
from skillcorner_data import convert


def fixture():
    metadata = {
        "home_team": {"id": 1, "name": "Home"},
        "away_team": {"id": 2, "name": "Away"},
        "pitch_length": 105.3,
        "pitch_width": 68,
        "players": [{"id": 11, "team_id": 1}, {"id": 22, "team_id": 2}],
    }
    rows = [
        {
            "period": 1,
            "frame": 10 + t,
            "player_data": [{"player_id": 11, "x": t, "y": 0}, {"player_id": 22, "x": 10, "y": 0}],
            "ball_data": {"x": t, "y": 0, "is_detected": True},
            "possession": {"group": "home team"},
        }
        for t in range(4)
    ]
    phases = [{"period": 1, "frame_start": 9, "frame_end": 15, "team_in_possession_id": 1}]
    return metadata, rows, phases


def test_unknown_owner_and_inferred_ball_remain_unknown_and_last_anchor_is_kept():
    meta, rows, phases = fixture()
    rows[1]["possession"]["group"] = None
    rows[2]["ball_data"]["is_detected"] = False
    tracking, reference, runs, report = convert(meta, rows, phases)
    assert len(reference) == 9 and runs == [(0, 9)]
    np.testing.assert_array_equal(np.flatnonzero(reference != 3), [0, 8])
    assert np.isfinite(tracking.ball[8]).all()
    assert not (reference == 2).any() and report["D_reference_available"] is False


def test_missing_phase_never_creates_dead_ball_truth():
    meta, rows, _ = fixture()
    _, reference, _, _ = convert(meta, rows, [])
    assert (reference == 3).all()


def test_phase_owner_conflict_excludes_silver_reference():
    meta, rows, phases = fixture()
    phases.append({**phases[0], "team_in_possession_id": 2})
    _, reference, _, _ = convert(meta, rows, phases)
    assert (reference == 3).all()


def test_player_interpolation_and_speed_do_not_cross_periods():
    meta, rows, phases = fixture()
    other = copy.deepcopy(rows)
    for row in other:
        row["period"], row["frame"] = 2, row["frame"] + 100
        row["player_data"][0]["x"] += 20
    tracking, reference, runs, _ = convert(meta, rows + other, phases)
    assert len(runs) == 2
    assert reference[runs[0][1]] == 3
    assert np.isnan(tracking.players[runs[0][1]]).all()
    assert np.isnan(tracking.player_speed[runs[1][0]]).all()
    assert tracking.players[runs[1][0], 0, 0] == 20 + 105.3 / 2
