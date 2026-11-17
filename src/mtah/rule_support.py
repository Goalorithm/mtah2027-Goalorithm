"""Geometry and configuration validation shared by Tech 2/3's rule candidates."""

import numpy as np

from mtah.features import ball_context
from mtah.hmm import sequence_runs
from mtah.paths import FPS, PITCH


def number(cfg, key, default, *, integer=False, minimum=0, strict=False):
    value = cfg.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or (integer and type(value) is not int)
        or (value <= minimum if strict else value < minimum)
    ):
        raise ValueError(
            f"{key} must be {'an integer' if integer else 'finite'} "
            f"{'greater than' if strict else 'at least'} {minimum}"
        )
    return value


def boolean(cfg, key, default):
    value = cfg.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be boolean")
    return value


def segments(mask):
    """Half-open true segments, including edges and an empty input."""
    bounds = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    return list(zip(bounds[::2], bounds[1::2], strict=True))


def geometry(tracking, cfg):
    """Reject isolated two-sided teleports; preserve all raw inputs and half isolation."""
    gap = number(cfg, "max_gap", 25, integer=True)
    limit = number(cfg, "max_ball_speed_mps", 45.0, strict=True)
    history = ball_context(tracking, gap, limit)
    ball = np.array(tracking.ball, dtype=float, copy=True)
    seen = np.isfinite(ball).all(axis=1)
    for start, stop in sequence_runs(tracking.n_frames):
        if stop - start < 3:
            continue
        part = ball[start:stop]
        speed = np.linalg.norm(np.diff(part, axis=0), axis=1) * FPS
        # A point is a spike only if both adjacent jumps are implausible and its neighbours agree.
        neighbours = np.linalg.norm(part[2:] - part[:-2], axis=1) * FPS / 2
        spikes = (speed[:-1] > limit) & (speed[1:] > limit) & (neighbours <= limit)
        seen[np.flatnonzero(spikes) + start + 1] = False
    ball[~seen] = np.nan
    clean_tracking = type(tracking)(
        tracking.players,
        tracking.player_speed,
        ball,
        tracking.ball_extra,
        tracking.slot_team,
        tracking.team_names,
    )
    clean = ball_context(clean_tracking, gap, limit)
    speed = np.full(tracking.n_frames, np.nan)
    for start, stop in sequence_runs(tracking.n_frames):
        speed[start + 1 : stop] = np.linalg.norm(np.diff(ball[start:stop], axis=0), axis=1) * FPS
    length, width = PITCH
    margin = number(cfg, "margin_m", 0.5)
    outside = seen & (
        (ball[:, 0] < -margin)
        | (ball[:, 0] > length + margin)
        | (ball[:, 1] < -margin)
        | (ball[:, 1] > width + margin)
    )
    inside = seen & (
        (ball[:, 0] >= 0) & (ball[:, 0] <= length) & (ball[:, 1] >= 0) & (ball[:, 1] <= width)
    )
    return {
        **clean,
        "raw_ball_last_xy": history["ball_last_xy"],
        "ball_xy_clean": ball,
        "seen": seen,
        "speed": speed,
        "outside": outside,
        "inside": inside,
    }
