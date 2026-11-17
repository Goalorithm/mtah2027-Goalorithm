"""Tech 3: controlled handovers, loose balls, dead-ball reset and bounded abstention."""

import numpy as np
import pytest

from mtah import A, H, X, features, hmm
from mtah.load import Tracking
from mtah.possession import hold_to_switch, hold_to_switch_x


def tracking(n=300):
    players = np.tile([[20.0, 30.0], [40.0, 30.0]], (n, 1, 1))
    return Tracking(
        players,
        np.full((n, 2), 2.5),
        np.tile([20.2, 30.0], (n, 1)),
        np.zeros(n),
        np.array([H, A]),
        ("Home", "Away"),
    )


def assign(tr, cfg=None, dead=None, abstain=False):
    fn = hold_to_switch_x if abstain else hold_to_switch
    return fn(
        tr,
        features.build(tr),
        np.zeros(tr.n_frames, dtype=bool) if dead is None else dead,
        [],
        cfg or {},
    )


def test_single_touch_does_not_transfer_possession_but_sustained_control_does():
    tr = tracking()
    tr.ball[60] = [40.2, 30]
    tr.ball[120:200] = [40.2, 30]
    result = assign(tr)
    assert (result[:120] == H).all()
    assert (result[121:200] == A).all()
    assert (result[201:] == H).all()


def test_control_confirmation_backdates_and_never_changes_dead_frames():
    tr = tracking()
    tr.ball[100:180] = [40.2, 30]
    cfg = {"control_ball_mps": 1000.0, "max_ball_speed_mps": 1000.0}
    result = assign(tr, cfg)
    assert result[99] == H and result[100] == A
    cfg["backdate_switch"] = False
    delayed = assign(tr, cfg)
    assert (delayed[100:114] == H).all() and delayed[114] == A


def test_loose_ball_and_brief_occlusion_retain_last_controlled_team():
    tr = tracking()
    tr.ball[50:100] = [30, 30]  # Equidistant and too far away to show control.
    tr.ball[100:130] = np.nan
    assert (assign(tr) == H).all()
    tr.ball[:] = [30, 30]
    assert (assign(tr) == X).all()  # No initial owner is invented from ties.


def test_dead_ball_resets_owner_and_fresh_restart_can_belong_to_other_team():
    tr = tracking()
    dead = np.zeros(tr.n_frames, dtype=bool)
    dead[80:120] = True
    tr.ball[80:120] = np.nan
    tr.ball[120:] = [40.2, 30]
    result = assign(tr, dead=dead)
    assert (result[:80] == H).all() and (result[80:120] == X).all()
    assert (result[120:] == A).all()


def test_half_boundary_resets_owner_and_pending_confirmation(monkeypatch):
    tr = tracking(120)
    monkeypatch.setattr(hmm, "N_FRAMES", 120)
    monkeypatch.setattr(hmm, "match_runs", lambda: [(10, 50), (70, 110)])
    tr.ball[70:110] = np.nan
    result = assign(tr)
    assert (result[10:50] == H).all() and (result[70:110] == X).all()
    assert (result[:10] == X).all() and (result[50:70] == X).all()


def test_x_is_bounded_to_same_owner_gaps_and_preserves_filled_sequence():
    tr = tracking(1000)
    tr.ball[100:110] = np.nan
    tr.ball[200:240] = np.nan
    tr.ball[-10:] = np.nan
    baseline = assign(tr)
    result = assign(tr, abstain=True)
    assert (result[100:110] == X).all()
    assert (result[200:240] == H).all() and (result[-10:] == H).all()
    from mtah.paths import pm

    np.testing.assert_array_equal(pm.fill_abstentions(result, [(0, 1000)]), baseline)


def test_x_never_hides_handover_gap_or_exceeds_share_budget():
    tr = tracking(1000)
    tr.ball[100:110] = np.nan
    tr.ball[110:500] = [40.2, 30]
    for start in range(600, 900, 20):
        tr.ball[start : start + 5] = np.nan
    result = assign(tr, abstain=True)
    assert not (result[90:125] == X).any()
    assert np.sum(result == X) <= 30


@pytest.mark.parametrize(
    "cfg",
    [
        {"min_frames": 0},
        {"initial_min_frames": True},
        {"control_m": -1},
        {"max_control_gap_frames": 1.5},
        {"backdate_switch": 1},
    ],
)
def test_invalid_control_parameters_are_rejected(cfg):
    with pytest.raises(ValueError):
        assign(tracking(), cfg)


def test_shared_history_helper_is_opt_in_and_does_not_modify_existing_feature_keys():
    tr = tracking()
    tr.ball[20:22] = np.nan
    original = features.build(tr)
    context = features.ball_context(tr, max_gap=2)
    assert np.isfinite(context["ball_xy_filled"][20:22]).all()
    assert (context["ball_gap"][20:22] == [1, 2]).all()
    for name, values in original.items():
        np.testing.assert_array_equal(features.build(tr)[name], values)


def test_control_is_equivariant_to_swapping_team_identities_and_player_slots():
    tr = tracking()
    tr.ball[100:200] = [40.2, 30]
    original = assign(tr)
    tr.slot_team[:] = 1 - tr.slot_team
    swapped = assign(tr)
    np.testing.assert_array_equal(swapped, np.where(original == X, X, 1 - original))
    tr.players = tr.players[:, ::-1].copy()
    tr.player_speed = tr.player_speed[:, ::-1].copy()
    tr.slot_team = tr.slot_team[::-1].copy()
    np.testing.assert_array_equal(assign(tr), swapped)
