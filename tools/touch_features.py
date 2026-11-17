"""Explicit ball touches read from a ball track: who changed the ball's velocity, and when.

A provider's owning team changes when a player of the other team plays the ball. In a sharp ball
track that moment is a kink: the velocity just after differs from the velocity just before. A
kink with a tracked player close to the ball is credited to that player's team; a kink with
nobody near is a bounce and is ignored. The features are the time since and until each team's
nearest touch, at two strengths, plus the kink strength itself.

A band-limited ball (the automatic one) has no kinks, so these columns stay quiet for it; they
carry information where the ball was re-detected in the video.
"""

import numpy as np
from scipy.ndimage import maximum_filter1d

from mtah import A, H
from mtah.hmm import sequence_runs
from mtah.paths import FPS

TOUCH_KEYS = ("touch_past_age", "touch_future_age", "kick_past_age", "kick_future_age")


def touch_feature_names():
    names = ["kink_strength"]
    for key in TOUCH_KEYS:
        names += [f"{side}_{key}" for side in ("H", "A")]
    return names


def _ages(mask, cap):
    index = np.arange(len(mask))
    past = np.maximum.accumulate(np.where(mask, index, -cap))
    future = np.minimum.accumulate(np.where(mask, index, len(mask) + cap)[::-1])[::-1]
    return np.minimum(index - past, cap) / cap, np.minimum(future - index, cap) / cap


def kink_strength(ball, lag):
    """Speed change in m/s between the `lag` frames before and after each frame; NaN if unknown."""
    count = len(ball)
    strength = np.full(count, np.nan)
    if count > 2 * lag:
        before = (ball[lag:-lag] - ball[: -2 * lag]) / lag
        after = (ball[2 * lag :] - ball[lag:-lag]) / lag
        strength[lag:-lag] = np.linalg.norm(after - before, axis=1) * FPS
    return strength


def touches(ball, players, slot_team, cfg):
    """(strength, toucher team per frame or -1): local kink maxima with a player in reach."""
    strength = kink_strength(ball, int(cfg["lag_frames"]))
    filled = np.where(np.isfinite(strength), strength, -np.inf)
    peak = filled >= maximum_filter1d(filled, 2 * int(cfg["peak_frames"]) + 1, mode="nearest")
    peak &= np.isfinite(strength) & (strength >= cfg["touch_mps"])
    team = np.full(len(ball), -1, dtype=np.int64)
    rows = np.flatnonzero(peak)
    if len(rows):
        gap = np.linalg.norm(players[rows] - ball[rows][:, None, :], axis=2)
        gap = np.where(np.isfinite(gap), gap, np.inf)
        nearest = gap.argmin(axis=1)
        close = gap[np.arange(len(rows)), nearest] <= cfg["reach_m"]
        team[rows[close]] = slot_team[nearest[close]]
    return strength, team


def touch_feature_matrix(ball, players, slot_team, cfg, runs=None):
    names = touch_feature_names()
    out = np.full((len(ball), len(names)), np.nan)
    cap = int(cfg["age_cap_frames"])
    for start, stop in sequence_runs(len(ball), runs):
        strength, team = touches(ball[start:stop], players[start:stop], slot_team, cfg)
        columns = {"kink_strength": np.minimum(strength, cfg["strength_cap_mps"])}
        columns["kink_strength"] = columns["kink_strength"] / cfg["strength_cap_mps"]
        for side, label in ((H, "H"), (A, "A")):
            soft = team == side
            hard = soft & (np.nan_to_num(strength) >= cfg["kick_mps"])
            past, future = _ages(soft, cap)
            columns[f"{label}_touch_past_age"], columns[f"{label}_touch_future_age"] = past, future
            past, future = _ages(hard, cap)
            columns[f"{label}_kick_past_age"], columns[f"{label}_kick_future_age"] = past, future
        out[start:stop] = np.column_stack([columns[name] for name in names])
    return out
