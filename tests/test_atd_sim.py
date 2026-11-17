"""ATD-like simulation, Metrica-derived references and team features: invariants only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import atd_sim  # noqa: E402
from atd_features import team_feature_matrix, team_feature_names  # noqa: E402
from metrica_labels import bridge_ball, reference_labels  # noqa: E402

from mtah import A, D, H, X  # noqa: E402
from mtah.config import load_config  # noqa: E402
from mtah.load import Tracking  # noqa: E402
from mtah.paths import PITCH  # noqa: E402
from mtah.supervised import swap_features  # noqa: E402

CFG = load_config("config/atd.toml")


def game(n=900, seed=3):
    rng = np.random.default_rng(seed)
    players = np.cumsum(rng.normal(0, 0.1, (n, 6, 2)), axis=0) + rng.uniform(20, 60, (1, 6, 2))
    speeds = np.abs(rng.normal(2, 1, (n, 6)))
    ball = np.cumsum(rng.normal(0, 0.3, (n, 2)), axis=0) + [50.0, 30.0]
    reference = np.full(n, H, dtype=np.int8)
    reference[300:500] = D
    reference[500:] = A
    tracking = Tracking(
        players, speeds, ball, np.full(n, np.nan), np.array([H, H, H, A, A, A]), ("h", "a")
    )
    return tracking, reference, [(0, n)]


def test_renewal_mask_is_seeded_and_near_its_fraction():
    first = atd_sim.renewal_mask(200000, 0.1, 100, 0.7, np.random.default_rng(5))
    second = atd_sim.renewal_mask(200000, 0.1, 100, 0.7, np.random.default_rng(5))
    assert np.array_equal(first, second)
    assert 0.07 < first.mean() < 0.13
    assert not atd_sim.renewal_mask(100, 0.0, 100, 0.7, np.random.default_rng(5)).any()


def test_stoppage_path_starts_at_the_stop_and_ends_at_the_restart():
    rng = np.random.default_rng(1)
    previous, restart = np.array([10.0, 67.0]), np.array([14.0, 68.0])
    path, held = atd_sim.dead_ball_path(
        previous, np.array([0.2, 0.3]), restart, 300, rng, CFG["simulation"]
    )
    assert held == 300  # nobody carries the ball when no carrier track is given
    assert np.isfinite(path).all()
    assert np.linalg.norm(path[0] - previous) < 1.0
    np.testing.assert_allclose(path[-1], restart)
    blank, _ = atd_sim.dead_ball_path(None, None, None, 5, rng, CFG["simulation"])
    assert np.isnan(blank).all()


def test_low_pass_keeps_blanks_and_short_stretches():
    ball = np.cumsum(np.random.default_rng(2).normal(0, 0.5, (400, 2)), axis=0)
    ball[100:120] = np.nan
    ball[130:140] = np.nan
    result = atd_sim.low_pass(ball, 0.66, 4, 30)
    assert np.array_equal(np.isnan(result), np.isnan(ball))
    np.testing.assert_array_equal(result[120:130], ball[120:130])
    jerk = np.abs(np.diff(result[140:], n=2, axis=0)).mean()
    assert jerk < 0.2 * np.abs(np.diff(ball[140:], n=2, axis=0)).mean()


def test_simulation_is_deterministic_and_leaves_the_source_untouched():
    tracking, reference, runs = game()
    before = tracking.ball.copy()
    params = CFG["simulation"]["benchmark"]
    first = atd_sim.simulate(tracking, reference, runs, params, CFG["simulation"], 11)
    second = atd_sim.simulate(tracking, reference, runs, params, CFG["simulation"], 11)
    np.testing.assert_array_equal(first.ball, second.ball)
    np.testing.assert_array_equal(first.players, second.players)
    np.testing.assert_array_equal(tracking.ball, before)
    assert first.ball.shape == before.shape
    assert np.isnan(first.player_speed[np.isnan(first.players[:, :, 0])]).all()


def test_missing_stoppage_hides_the_ball_and_frozen_positions_are_replaced():
    tracking, reference, runs = game()
    tracking.ball[300:500] = tracking.ball[299]  # provider-style frozen stoppage ball
    params = {**CFG["simulation"]["benchmark"], "stoppage_missing": 1.0, "stoppage_visible": 0.0}
    params.update(live_gap_fraction=0.0, player_loss_fraction=0.0, outage_per_minute=0.0)
    # No early re-acquisition, so the whole stoppage after the first frames stays lost.
    quiet = {**CFG["simulation"], "reacquire_just_before": 0.0, "reacquire_early": 0.0}
    result = atd_sim.simulate(tracking, reference, runs, params, quiet, 4)
    assert np.isnan(result.ball[345:480]).all()
    params.update(stoppage_missing=0.0, stoppage_visible=1.0, outside_visible_margin_m=1000.0)
    result = atd_sim.simulate(tracking, reference, runs, params, CFG["simulation"], 4)
    assert np.isfinite(result.ball[300:500]).all()
    assert np.ptp(result.ball[300:500], axis=0).max() > 0


def test_sampled_parameters_stay_inside_their_ranges():
    params = atd_sim.sample_parameters(CFG["simulation"]["train"], np.random.default_rng(9))
    for key, (low, high) in CFG["simulation"]["train"].items():
        assert low <= params[key] <= high


def test_metrica_reference_uses_blank_ball_for_dead_and_events_for_teams():
    n = 200
    ball = np.tile([[0.5, 0.5]], (n, 1)).astype(float)
    ball[50:53] = np.nan  # too short to be a stoppage
    ball[100:150] = np.nan
    bridged, dead = bridge_ball(ball, [(0, n)], 12)
    assert np.isfinite(bridged[50:53]).all() and dead[100:150].all() and not dead[:100].any()
    tracking = Tracking(
        np.zeros((n, 2, 2)), np.zeros((n, 2)), ball, np.full(n, np.nan), np.array([H, A]), ("", "")
    )
    events = pd.DataFrame(
        {
            "Type": ["PASS", "CHALLENGE", "RECOVERY", "SET PIECE"],
            "Team": ["Home", "Away", "Away", "Home"],
            "Start Frame": [11, 31, 61, 156],
        }
    )
    _, reference, _ = reference_labels(tracking, [(0, n)], events, 12)
    assert (reference[:60] == H).all()  # frames before the first event take its team
    assert (reference[60:100] == A).all()
    assert (reference[100:150] == D).all()
    assert (reference[150:] == H).all()  # restart owner, not the team from before the stop
    assert not (reference == X).any()


def test_team_features_swap_with_the_teams_and_ignore_mirroring():
    tracking, _, runs = game()
    names = team_feature_names()
    values = team_feature_matrix(tracking, CFG["team_features"], runs)
    swapped = Tracking(
        tracking.players,
        tracking.player_speed,
        tracking.ball,
        tracking.ball_extra,
        np.where(tracking.slot_team == H, A, H),
        tracking.team_names,
    )
    np.testing.assert_allclose(
        team_feature_matrix(swapped, CFG["team_features"], runs),
        swap_features(values, names),
        equal_nan=True,
    )
    mirrored_players = tracking.players.copy()
    mirrored_players[:, :, 0] = PITCH[0] - mirrored_players[:, :, 0]
    mirrored_ball = tracking.ball.copy()
    mirrored_ball[:, 0] = PITCH[0] - mirrored_ball[:, 0]
    mirrored = Tracking(
        mirrored_players,
        tracking.player_speed,
        mirrored_ball,
        tracking.ball_extra,
        tracking.slot_team,
        tracking.team_names,
    )
    np.testing.assert_allclose(
        team_feature_matrix(mirrored, CFG["team_features"], runs), values, atol=1e-9, equal_nan=True
    )


def test_blank_speed_of_a_tracked_player_is_derived_from_positions():
    from atd_features import harmonise_speeds

    tracking, _, runs = game(n=50)
    tracking.players[:, 0, :] = [10.0, 10.0]  # standing still
    tracking.player_speed[:, 0] = np.nan
    tracking.players[20:25, 1] = np.nan
    tracking.player_speed[20:25, 1] = np.nan
    filled = harmonise_speeds(tracking, runs)
    assert (filled.player_speed[:, 0] == 0).all()
    assert np.isnan(filled.player_speed[20:25, 1]).all()  # an untracked player stays blank
    np.testing.assert_array_equal(filled.player_speed[:, 2], tracking.player_speed[:, 2])


def test_ball_state_features_mark_a_ball_off_the_pitch_and_a_resting_ball():
    from atd_features import ball_state_features

    ball = np.tile([[50.0, 30.0]], (200, 1)).astype(float)
    ball[:60, 0] += np.arange(60, 0, -1) * 0.5  # moving until frame 60, then at rest
    ball[100:110] = [-3.0, 30.0]  # behind the goal line
    ball[150:160] = np.nan
    cfg = CFG["team_features"]
    past, future, still_past, still_future = ball_state_features(ball, cfg)
    assert past[99] == 1.0 and past[105] == 0.0
    assert np.isclose(past[120], 11 / cfg["outside_age_cap_frames"])
    assert np.isclose(future[90], 10 / cfg["outside_age_cap_frames"])
    assert still_past[30] == 0 and still_past[80] > still_past[65] > 0
    assert still_future[65] > still_future[80] > 0
    assert np.isnan(still_past[155]) and np.isnan(still_future[155])


def test_broadcast_ball_is_synchronised_and_fills_only_tactical_gaps():
    from broadcast_ball import fill_ball

    rng = np.random.default_rng(12)
    n, lag = 6000, 137
    truth = np.cumsum(rng.normal(0, 0.4, (n + 500, 2)), axis=0) + [50.0, 30.0]
    tactical_ball = truth[lag : lag + n].copy()
    tactical_ball[1000:1400] = np.nan
    broadcast_ball = truth[: n + 300] + rng.normal(0, 0.2, (n + 300, 2))
    broadcast_ball[3000:3300] = np.nan

    def tracking(ball):
        size = len(ball)
        return Tracking(
            np.zeros((size, 2, 2)),
            np.zeros((size, 2)),
            ball,
            np.full(size, np.nan),
            np.array([H, A]),
            ("", ""),
        )

    merged, report = fill_ball(
        tracking(tactical_ball),
        tracking(broadcast_ball),
        [(0, n)],
        CFG["broadcast"] | {"min_shared_frames": 1000},
    )
    # tactical frame t shows broadcast frame t + lag, so tactical = broadcast - lag
    assert report["halves"][0]["lag_frames"] == -lag
    assert report["ball_frames_added"] == 400
    kept = np.isfinite(tactical_ball).all(axis=1)
    np.testing.assert_array_equal(merged.ball[kept], tactical_ball[kept])
    assert np.abs(merged.ball[1000:1400] - truth[lag + 1000 : lag + 1400]).max() < 1.5


def test_profile_plan_keeps_roles_apart_and_repeats_exactly():
    from prepare_atd import profile_plan

    first, second = profile_plan(CFG), profile_plan(CFG)
    assert first == second
    games = {}
    for split, name, _, _, _ in first:
        games.setdefault(split, set()).add(name)
    assert not games["train"] & (games["validation"] | games["test"])
    assert not games["validation"] & games["test"]
    assert games["refit"] == games["validation"]  # final refit only, never used for selection
    seeds = [seed for _, _, _, seed, _ in first]
    assert len(set(seeds)) == len(seeds)
    benchmark = CFG["simulation"]["benchmark"]
    for split, _, _, _, params in first:
        if split in ("validation", "test"):
            assert params == benchmark
        else:
            assert params != benchmark


def test_cutaways_follow_stoppages_and_their_features_are_well_formed():
    from broadcast_cut import (
        CUT_NAMES,
        cutaway_features,
        cutaway_statistics,
        measured_cutaways,
        simulate_cutaways,
    )

    reference = np.full(6000, H, dtype=np.int8)
    for start in range(500, 5500, 1000):
        reference[start : start + 400] = D
    runs = [(0, 6000)]
    params = {**CFG["simulation"]["benchmark"], "cut_stoppage_share": 1.0}
    params["cut_live_fraction"] = 0.0
    cut = simulate_cutaways(reference, runs, params, CFG["simulation"], 3)
    np.testing.assert_array_equal(
        cut, simulate_cutaways(reference, runs, params, CFG["simulation"], 3)
    )
    statistics = cutaway_statistics(cut, reference == D, runs)
    assert statistics["stoppages_with_cutaway"] == 1.0
    assert statistics["cut_share_in_stoppage"] > 5 * statistics["cut_share_in_live"]
    features = cutaway_features(cut, runs, CFG["team_features"])
    assert features.shape == (6000, len(CUT_NAMES))
    assert set(np.unique(features[:, 0])) == {0.0, 1.0}
    assert (features[cut > 0.5, 1] == 0).all() and (features[cut > 0.5, 3] > 0).all()
    assert (features[cut < 0.5, 3] == 0).all()
    unknown = cut.copy()
    unknown[:100] = np.nan
    assert np.isnan(cutaway_features(unknown, runs, CFG["team_features"])[:100]).all()
    valid = np.ones(50, dtype=bool)
    valid[10:20] = False
    measured = measured_cutaways(60, valid, [(0, 60)], [5])
    assert np.isnan(measured[:5]).all() and np.isnan(measured[55:]).all()
    assert (measured[15:25] == 1).all() and (measured[25:55] == 0).all()


def test_touches_credit_the_team_that_changes_the_ball_velocity():
    from touch_features import touch_feature_matrix, touch_feature_names, touches

    settings = CFG["team_features"]["touch"]
    count = 200
    ball = np.zeros((count, 2))
    ball[:, 1] = 30.0
    ball[:60, 0] = 20 + 0.4 * np.arange(60)  # rolling at 10 m/s
    ball[60:140, 0] = ball[59, 0] - 0.4 * np.arange(1, 81)  # an away player turns it round
    ball[140:, 0] = ball[139, 0]
    ball[140:, 1] = 30 + 0.5 * np.arange(1, 61)  # a home player sends it sideways
    players = np.full((count, 2, 2), 100.0)
    players[:, 0] = ball[140] + [0.5, 0.0]  # home player waits where the second kink happens
    players[:, 1] = ball[60] + [0.5, 0.0]  # away player stands where the first kink happens
    strength, team = touches(ball, players, np.array([H, A]), settings)
    assert team[59] == A and team[139] == H  # the last frame of the old direction
    assert set(np.flatnonzero(team >= 0)) <= set(range(57, 64)) | set(range(137, 144))
    assert strength[59] > settings["kick_mps"]
    names = touch_feature_names()
    values = touch_feature_matrix(ball, players, np.array([H, A]), settings, [(0, count)])
    cap = settings["age_cap_frames"]
    assert values.shape == (count, len(names))
    assert values[100, names.index("A_touch_past_age")] == 41 / cap
    assert values[100, names.index("H_touch_future_age")] == 39 / cap
    assert values[100, names.index("H_touch_past_age")] == 1.0  # home has not touched yet
    far = players.copy()
    far[:] = 100.0  # nobody in reach: the kinks are bounces
    assert (touches(ball, far, np.array([H, A]), settings)[1] == -1).all()


def test_handled_stoppage_ball_moves_but_still_starts_and_ends_in_place():
    cfg = CFG["simulation"]
    out = atd_sim.stroll(200, 2.0, cfg, np.random.default_rng(4))
    assert out.shape == (200, 2) and np.allclose(out[-1], 0)  # it comes back to where it waits
    pace = np.linalg.norm(np.diff(out, axis=0), axis=1) * 25
    assert 1.0 < np.sqrt((pace**2).mean()) < 3.0
    assert np.allclose(atd_sim.stroll(2, 2.0, cfg, np.random.default_rng(4)), 0)

    previous, restart = np.array([20.0, 30.0]), np.array([60.0, 5.0])
    velocity = np.array([0.2, 0.0])
    lying, _ = atd_sim.dead_ball_path(
        previous, velocity, restart, 600, np.random.default_rng(9), cfg, None, 0.0
    )
    handled, _ = atd_sim.dead_ball_path(
        previous, velocity, restart, 600, np.random.default_rng(9), cfg, None, 1.0
    )
    assert np.allclose(handled[-1], restart) and np.allclose(lying[-1], restart)
    still = lambda path: (np.linalg.norm(np.diff(path, axis=0), axis=1) * 25 < 1.0).mean()  # noqa: E731
    # Only the wait before the ball is brought back is handled; at the restart spot it lies.
    assert still(lying) > 0.6 and still(handled) < still(lying) - 0.05
    assert still(handled[-100:]) == 1.0


def test_the_workflow_has_no_video_ball_and_no_cutaway_input():
    # Nothing in this workflow is trained, fitted or calibrated on the provided video, and the
    # broadcast cutaway cue (timed on the provided match) is not an input either.
    assert CFG["feature_groups"] == ["atd"]
    tracking, reference, runs = game()
    params = atd_sim.sample_parameters(CFG["simulation"]["train"], np.random.default_rng(1))
    _, second = atd_sim.simulate_game(tracking, reference, runs, params, CFG["simulation"], 2)
    assert np.isnan(second).all()
    flat = {**CFG["simulation"], **CFG["simulation"]["train"], **CFG["simulation"]["benchmark"]}
    assert not [key for key in flat if key.startswith(("video_", "rest_")) or "wander" in key]
    assert not hasattr(atd_sim, "camera_view") and not hasattr(atd_sim, "simulate_video_ball")
