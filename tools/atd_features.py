"""Ball-independent team-shape features appended to the shared interaction features.

The automatic ball is absent on 41% of graded frames, in stretches of ten seconds and more.
These columns describe what both teams do without it: how fast they move, how spread they
are, how far each block is pushed up and which way it is travelling. Attack direction is read
from the data per period (the team whose mean position is lower in x attacks towards +x), so
every column is invariant to mirroring the pitch and swaps cleanly between H and A.
"""

import numpy as np
from broadcast_cut import CUT_NAMES, cutaway_features
from touch_features import touch_feature_matrix, touch_feature_names

from mtah import A, H
from mtah.hmm import sequence_runs
from mtah.load import Tracking
from mtah.paths import FPS, PITCH
from mtah.supervised import feature_matrix, feature_names

GLOBAL_NAMES = [
    "speed_q25",
    "speed_q50",
    "speed_q75",
    "speed_max",
    "fast_fraction",
    "outside_past_age",
    "outside_future_age",
    "still_past",
    "still_future",
]
TEAM_KEYS = [
    "team_speed",
    "team_speed_max",
    "spread_x",
    "spread_y",
    "push",
    "advance",
    "ball_territory",
    "count",
]


def team_feature_names():
    return GLOBAL_NAMES + [f"{side}_{key}" for key in TEAM_KEYS for side in ("H", "A")]


VIDEO_SUFFIX = "_video"
GROUPS = ("atd", "video", "cut", "touch")


def atd_feature_names(feature_cfg):
    """All model inputs in column order: automatic ball, team shape, video ball, cutaway,
    explicit touches of the video ball."""
    base = feature_names(feature_cfg)
    return (
        base
        + team_feature_names()
        + [name + VIDEO_SUFFIX for name in base]
        + ["balls_apart"]
        + CUT_NAMES
        + touch_feature_names()
    )


def group_columns(names, groups):
    """Column indexes of the requested input groups, in the original order."""
    if not groups or not set(groups) <= set(GROUPS) or len(set(groups)) != len(groups):
        raise ValueError(f"feature groups must be distinct members of {GROUPS}")
    keep = []
    touch = set(touch_feature_names())
    for index, name in enumerate(names):
        if name in CUT_NAMES:
            group = "cut"
        elif name in touch:
            group = "touch"
        elif name.endswith(VIDEO_SUFFIX) or name == "balls_apart":
            group = "video"
        else:
            group = "atd"
        if group in groups:
            keep.append(index)
    return keep


def _row_quantiles(values, quantiles):
    """Per-row quantiles ignoring NaN, by sorting once (NaN sorts last)."""
    ordered = np.sort(values, axis=1)
    count = np.isfinite(values).sum(axis=1)
    rows = np.arange(len(values))
    result = []
    for q in quantiles:
        index = np.clip(np.floor(q * (count - 1)).astype(np.int64), 0, values.shape[1] - 1)
        column = ordered[rows, index]
        column[count == 0] = np.nan
        result.append(column)
    return result


def _nan_mean(values):
    count = np.isfinite(values).sum(axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0), count


def _ages(mask, cap):
    """Frames since and until the nearest True, each capped and scaled to 0..1."""
    index = np.arange(len(mask))
    past = np.maximum.accumulate(np.where(mask, index, -cap))
    future = np.minimum.accumulate(np.where(mask, index, len(mask) + cap)[::-1])[::-1]
    return np.minimum(index - past, cap) / cap, np.minimum(future - index, cap) / cap


def _still_runs(still, cap):
    """How long the ball has rested so far and how long it still will; 0 while it moves."""
    index = np.arange(len(still))
    began = np.maximum.accumulate(np.where(~still, index, -1))
    ends = np.minimum.accumulate(np.where(~still, index, len(still))[::-1])[::-1]
    past = np.where(still, np.minimum(index - began, cap), 0) / cap
    future = np.where(still, np.minimum(ends - index, cap), 0) / cap
    return past, future


def ball_state_features(ball, cfg):
    """Ages of the last/next ball seen off the pitch, and the length of a resting ball.

    A ball that crossed a line stays dead until the restart even when it is tracked on its
    way back, and a ball lying still for seconds is waiting for a set piece.
    """
    length, width = PITCH
    margin = cfg["outside_margin_m"]
    seen = np.isfinite(ball).all(axis=1)
    with np.errstate(invalid="ignore"):
        outside = seen & (
            (ball[:, 0] < -margin)
            | (ball[:, 0] > length + margin)
            | (ball[:, 1] < -margin)
            | (ball[:, 1] > width + margin)
        )
        step = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS]
        still = seen & (step < cfg["still_ball_mps"])
    past, future = _ages(outside, cfg["outside_age_cap_frames"])
    rest_past, rest_future = _still_runs(still, cfg["still_cap_frames"])
    rest_past[~seen] = np.nan
    rest_future[~seen] = np.nan
    return past, future, rest_past, rest_future


def team_feature_matrix(tracking, cfg, runs=None):
    lag = int(cfg["advance_lag_frames"])
    speed_scale, spread_scale = cfg["speed_scale_mps"], cfg["spread_scale_m"]
    if lag < 1 or speed_scale <= 0 or spread_scale <= 0:
        raise ValueError("invalid team feature configuration")
    length = PITCH[0]
    names = team_feature_names()
    out = np.full((tracking.n_frames, len(names)), np.nan)
    speeds = np.where(np.isfinite(tracking.players[:, :, 0]), tracking.player_speed, np.nan)
    for start, stop in sequence_runs(tracking.n_frames, runs):
        size = stop - start
        local = np.arange(size)
        left, right = np.maximum(0, local - lag), np.minimum(size - 1, local + lag)
        elapsed = (right - left) / FPS
        speed = speeds[start:stop]
        q25, q50, q75, top = _row_quantiles(speed, [0.25, 0.5, 0.75, 1.0])
        tracked = np.isfinite(speed).sum(axis=1)
        fast = np.divide(
            np.sum(speed > cfg["fast_player_mps"], axis=1),
            tracked,
            out=np.full(size, np.nan),
            where=tracked > 0,
        )
        columns = {
            "speed_q25": q25 / speed_scale,
            "speed_q50": q50 / speed_scale,
            "speed_q75": q75 / speed_scale,
            "speed_max": np.minimum(top, 12.0) / 12.0,
            "fast_fraction": fast,
        }
        for name, column in zip(
            GLOBAL_NAMES[5:], ball_state_features(tracking.ball[start:stop], cfg), strict=True
        ):
            columns[name] = column
        centre = {}
        for side in (H, A):
            centre[side] = _nan_mean(
                tracking.players[start:stop][:, tracking.slot_team == side, 0].reshape(size, -1)
            )[0]
        gap = centre[A] - centre[H]
        direction = 1.0 if not np.isfinite(gap).any() or np.nanmean(gap) >= 0 else -1.0
        ball_x = tracking.ball[start:stop, 0]
        for side, label in ((H, "H"), (A, "A")):
            sign = direction if side == H else -direction
            slots = tracking.slot_team == side
            xy = tracking.players[start:stop][:, slots]
            side_speed = speed[:, slots]
            mean_speed, count = _nan_mean(side_speed)
            top_speed = _row_quantiles(side_speed, [1.0])[0] if slots.any() else mean_speed
            mean_x = centre[side]
            with np.errstate(invalid="ignore"):
                spread_x = np.sqrt(_nan_mean((xy[:, :, 0] - mean_x[:, None]) ** 2)[0])
                mean_y = _nan_mean(xy[:, :, 1])[0]
                spread_y = np.sqrt(_nan_mean((xy[:, :, 1] - mean_y[:, None]) ** 2)[0])
                step = (xy[right, :, 0] - xy[left, :, 0]) / np.maximum(elapsed, 1 / FPS)[:, None]
            advance = _nan_mean(step)[0]
            columns.update(
                {
                    f"{label}_team_speed": mean_speed / speed_scale,
                    f"{label}_team_speed_max": np.minimum(top_speed, 12.0) / 12.0,
                    f"{label}_spread_x": spread_x / spread_scale,
                    f"{label}_spread_y": spread_y / spread_scale,
                    f"{label}_push": sign * (mean_x - length / 2) / (length / 2),
                    f"{label}_advance": np.clip(sign * advance, -speed_scale, speed_scale)
                    / speed_scale,
                    f"{label}_ball_territory": sign * (ball_x - length / 2) / (length / 2),
                    f"{label}_count": count / 11,
                }
            )
        out[start:stop] = np.column_stack([columns[name] for name in names])
    if np.isinf(out).any():
        raise ValueError("team features must be finite or missing")
    return out


def harmonise_speeds(tracking, runs=None):
    """Fill a missing exported speed of a tracked player from its own positions.

    The automatic export leaves the speed blank on 8% of tracked positions, almost all of them
    players standing still (median derived speed 0.11 m/s). Averages over the remaining speeds
    would overstate how much a team moves exactly when play is stopped.
    """
    speeds = np.array(tracking.player_speed, dtype=np.float64, copy=True)
    for start, stop in sequence_runs(tracking.n_frames, runs):
        xy = tracking.players[start:stop]
        local = np.arange(stop - start)
        left, right = np.maximum(0, local - 1), np.minimum(stop - start - 1, local + 1)
        derived = np.full(xy.shape[:2], np.nan)
        for lo, hi in ((left, right), (local, right), (left, local)):
            elapsed = ((hi - lo) / FPS)[:, None]
            step = np.linalg.norm(xy[hi] - xy[lo], axis=2)
            estimate = np.divide(step, elapsed, out=np.full_like(step, np.nan), where=elapsed > 0)
            derived = np.where(np.isfinite(derived), derived, estimate)
        blank = np.isfinite(xy[:, :, 0]) & ~np.isfinite(speeds[start:stop])
        speeds[start:stop][blank] = derived[blank]
    return Tracking(
        tracking.players,
        speeds,
        tracking.ball,
        tracking.ball_extra,
        tracking.slot_team,
        tracking.team_names,
    )


def atd_feature_matrix(tracking, feature_cfg, team_cfg, runs=None, video_ball=None, cut=None):
    """Every model input. video_ball and cut default to unknown (all columns missing).

    The interaction features are computed twice: once with the automatic ball, once with the
    ball re-detected in the video. The two disagree when the ball is off the grass, so their
    distance is an input as well.
    """
    tracking = harmonise_speeds(tracking, runs)
    count = tracking.n_frames
    if video_ball is None:
        video_ball = np.full((count, 2), np.nan)
    if cut is None:
        cut = np.full(count, np.nan)
    if len(video_ball) != count or len(cut) != count:
        raise ValueError("video ball and cutaway arrays must cover every tracking frame")
    seen_in_video = Tracking(
        tracking.players,
        tracking.player_speed,
        np.asarray(video_ball, dtype=np.float64),
        tracking.ball_extra,
        tracking.slot_team,
        tracking.team_names,
    )
    cap = float(feature_cfg["distance_cap_m"])
    apart = np.linalg.norm(seen_in_video.ball - tracking.ball, axis=1)
    return np.concatenate(
        [
            feature_matrix(tracking, feature_cfg, runs),
            team_feature_matrix(tracking, team_cfg, runs),
            feature_matrix(seen_in_video, feature_cfg, runs),
            (np.minimum(apart, cap) / cap)[:, None],
            cutaway_features(cut, runs, team_cfg),
            touch_feature_matrix(
                seen_in_video.ball, tracking.players, tracking.slot_team, team_cfg["touch"], runs
            ),
        ],
        axis=1,
    )
