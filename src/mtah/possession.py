"""Which team is in possession while the ball is alive?  (owner: Tech 3)

Output contract
---------------
assign(tracking, features, dead, tagging, config) returns an int8 array of length n_frames with
H, A or X (from mtah). Frames where `dead` is True are overwritten with D afterwards, so their
value here does not matter.

X means "repeat my previous answer" to the ranking metric (spellF1): free in the middle of a
possession, costly across a hand-over. Use it when the ball is lost inside a possession, not at
a transition. See EVALUATION.md, "How an abstention is scored".

How to add an idea
------------------
Same as dead_ball.py: write a function, register it in METHODS, select it in a config.

Ideas worth trying:
  * a possession changes only when the other team keeps the ball for a while ("control"),
    not on a single touch, deflection or aerial duel
  * after a stoppage, the team that restarts play has the ball (Smart Tagging set pieces)
  * a hysteresis / state machine instead of a symmetric majority vote
"""

import numpy as np

from mtah import A, H, X
from mtah.dead_ball import window_count


def nearest_majority(tracking, features, dead, tagging, cfg):
    """Baseline: majority vote of the nearest team over +/- window frames. Ties go to H."""
    nearest = features["nearest_team"]
    window = int(cfg.get("window", 50))
    if window <= 0:
        home, away = (nearest == H).astype(np.int64), (nearest == A).astype(np.int64)
    else:
        home, away = window_count(nearest == H, window), window_count(nearest == A, window)

    out = np.full(len(nearest), X, dtype=np.int8)
    voted = (home + away) > 0
    out[voted] = np.where(home[voted] >= away[voted], H, A)
    return out


def hold_to_switch(tracking, features, dead, tagging, cfg):
    """Tech 3: control needs proximity, separation and sustained observations, not one touch.

    Confirmation backdates to the first supported frame (offline decoding). Missing frames
    retain the last owner; dead ball and half boundaries reset all ownership/streak history.
    """
    from mtah.hmm import sequence_runs
    from mtah.rule_support import boolean, geometry, number

    need = number(cfg, "min_frames", 15, integer=True, minimum=1)
    first_need = number(cfg, "initial_min_frames", 3, integer=True, minimum=1)
    allowed_gap = number(cfg, "max_control_gap_frames", 2, integer=True)
    distance = number(cfg, "control_m", 2.5, strict=True)
    advantage = number(cfg, "advantage_m", 0.5)
    maximum_speed = number(cfg, "control_ball_mps", 15.0, strict=True)
    backdate = boolean(cfg, "backdate_switch", True)
    context = geometry(tracking, cfg)
    # Raw observations provide control evidence; interpolation never manufactures a touch.
    ball = context["ball_xy_clean"]
    squared = np.sum((tracking.players - ball[:, None, :]) ** 2, axis=2)
    distances = np.full((tracking.n_frames, 2), np.inf)
    for team in (H, A):
        slots = tracking.slot_team == team
        if slots.any():
            distances[:, team] = (
                np.min(np.where(np.isfinite(squared[:, slots]), squared[:, slots], np.inf), axis=1)
                ** 0.5
            )
    nearest = np.argmin(distances, axis=1)
    own = distances[np.arange(len(distances)), nearest]
    other = distances[np.arange(len(distances)), 1 - nearest]
    credible = (
        context["inside"]
        & (own <= distance)
        & (other >= own + advantage)
        & (~np.isfinite(context["speed"]) | (context["speed"] <= maximum_speed))
    )
    result = np.full(tracking.n_frames, X, dtype=np.int8)
    events = []
    for start, stop in sequence_runs(tracking.n_frames):
        current, pending, count, onset, gap = X, X, 0, start, 0
        for i in range(start, stop):
            if dead[i]:
                current, pending, count, gap = X, X, 0, 0
                continue
            if credible[i]:
                observed = int(nearest[i])
                gap = 0
                if observed == current:
                    pending, count = X, 0
                else:
                    if observed != pending:
                        pending, count, onset = observed, 0, i
                    count += 1
                    threshold = first_need if current == X else need
                    if count >= threshold:
                        previous, current = current, observed
                        first = onset if backdate else i
                        result[first : i + 1] = current
                        events.append(
                            {
                                "frame": int(first),
                                "confirmed_frame": i,
                                "previous": int(previous),
                                "team": current,
                            }
                        )
                        pending, count = X, 0
            else:
                gap += 1
                if gap > allowed_gap:
                    pending, count = X, 0
            result[i] = current
    features["_control_events"] = events
    return result


def hold_to_switch_x(tracking, features, dead, tagging, cfg):
    """Tech 3: abstain only in short missing-ball gaps with matching owners at both ends."""
    from mtah.hmm import sequence_runs
    from mtah.rule_support import number, segments

    result = hold_to_switch(tracking, features, dead, tagging, cfg)
    low = number(cfg, "abstain_min_gap_frames", 3, integer=True, minimum=1)
    high = number(cfg, "abstain_max_gap_frames", 25, integer=True, minimum=low)
    guard = number(cfg, "transition_guard_frames", 15, integer=True)
    share = number(cfg, "max_abstain_fraction", 0.03)
    if share > 1:
        raise ValueError("max_abstain_fraction must be at most 1")
    seen = np.isfinite(tracking.ball).all(axis=1)
    for start, stop in sequence_runs(tracking.n_frames):
        labels = result[start:stop].copy()
        changes = np.zeros(stop - start, dtype=bool)
        changes[1:] = (labels[1:] != labels[:-1]) | (
            dead[start + 1 : stop] != dead[start : stop - 1]
        )
        changes[0] = True
        protected = window_count(changes, guard) > 0
        budget = int(np.sum(~dead[start:stop]) * share)
        used = 0
        for a, b in segments(~seen[start:stop]):
            if not low <= b - a <= high or a == 0 or b == len(labels):
                continue
            team = labels[a - 1]
            if (
                team not in (H, A)
                or labels[b] != team
                or not (labels[a:b] == team).all()
                or dead[start + a : start + b].any()
                or protected[a:b].any()
                or used + b - a > budget
            ):
                continue
            result[start + a : start + b] = X
            used += b - a
    return result


def hmm(tracking, features, dead, tagging, cfg):
    """Thin adapter sharing the joint H/A/D decode with the dead-ball method."""
    from mtah import D
    from mtah.hmm import shared_inference

    states = shared_inference(tracking, features, tagging, cfg)
    return np.where(states == D, X, states).astype(np.int8)


def supervised(tracking, features, dead, tagging, cfg):
    """External-only supervised joint decoder, shared with dead-ball inference."""
    from mtah import D
    from mtah.supervised import shared_inference

    states = shared_inference(tracking, features, tagging, cfg)
    return np.where(states == D, X, states).astype(np.int8)


METHODS = {
    "nearest_majority": nearest_majority,
    "hold_to_switch": hold_to_switch,
    "hold_to_switch_x": hold_to_switch_x,
    "hmm": hmm,
    "supervised": supervised,
}


def assign(tracking, features, dead, tagging, config):
    cfg = config.get("possession", {})
    method = cfg.get("method", "nearest_majority")
    if method not in METHODS:
        raise KeyError(f"unknown possession.method {method!r}; have {sorted(METHODS)}")
    if method == "hmm":
        cfg = config.get("hmm", {})
    elif method == "supervised":
        cfg = config
    team = np.asarray(METHODS[method](tracking, features, dead, tagging, cfg), dtype=np.int8)
    if team.shape != (tracking.n_frames,):
        raise ValueError(f"possession.{method} returned shape {team.shape}")
    if not np.isin(team, (H, A, X)).all():
        raise ValueError(f"possession.{method} returned codes outside H/A/X")
    return team
