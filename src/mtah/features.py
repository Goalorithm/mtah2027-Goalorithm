"""Per-frame features shared by the dead-ball and possession modules.

Everything here is a plain numpy array of length n_frames, indexed by video frame. Add new
features as new keys; never change the meaning of an existing key, other modules read them.

Keys
----
ball_seen        bool     the tracking located the ball
ball_xy          (n, 2)   ball position in metres, NaN when not seen
ball_speed       float    ball speed in m/s from consecutive positions, NaN when unknown
nearest_team     int8     team (H/A) of the tracked object closest to the ball, -1 if none
nearest_dist     float    distance in metres from the ball to that object, NaN if none
team_dist        (n, 2)   closest distance from the ball to each team [H, A], NaN if none
mean_speed       float    mean speed of all tracked objects in m/s, NaN if none
n_tracked        int16    number of tracked objects on the frame
"""

import numpy as np

from mtah import A, H
from mtah.paths import FPS

NOBODY = -1


def nearest_to_ball(tracking):
    """(nearest_team, nearest_dist, team_dist) for every frame.

    Squared distances are computed in exactly the order starter/baseline.py uses, so the nearest
    team agrees with the organisers' baseline frame for frame (ties go to the lowest slot).
    """
    dx = tracking.players[:, :, 0] - tracking.ball[:, None, 0]
    dy = tracking.players[:, :, 1] - tracking.ball[:, None, 1]
    squared = dx * dx + dy * dy  # NaN where the slot or the ball is missing

    valid = ~np.isnan(squared)
    has_any = valid.any(axis=1)
    filled = np.where(valid, squared, np.inf)
    best_slot = np.argmin(filled, axis=1)
    rows = np.arange(len(squared))

    nearest_team = np.full(len(squared), NOBODY, dtype=np.int8)
    nearest_team[has_any] = tracking.slot_team[best_slot[has_any]]
    nearest_dist = np.full(len(squared), np.nan)
    nearest_dist[has_any] = np.sqrt(filled[rows, best_slot][has_any])

    team_dist = np.full((len(squared), 2), np.nan)
    for team in (H, A):
        in_team = valid & (tracking.slot_team == team)[None, :]
        team_has = in_team.any(axis=1)
        team_min = np.where(in_team, squared, np.inf).min(axis=1)
        team_dist[team_has, team] = np.sqrt(team_min[team_has])
    return nearest_team, nearest_dist, team_dist


def ball_speed(ball_xy, fps=FPS):
    """Ball speed in m/s from consecutive frames; NaN where either frame lacks the ball."""
    step = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps
    return np.concatenate(([np.nan], step))


def ball_context(tracking, max_gap=25, max_speed_mps=45.0, runs=None):
    """Tech 1's opt-in history features; existing build() keys/meaning stay unchanged.

    Shares the tested short-gap geometry implementation with the independent HMM lane.
    Histories reset at half boundaries. This does not correct or overwrite raw tracking.
    """
    from mtah.hmm import ball_features

    return ball_features(tracking.ball, max_gap, max_speed_mps, runs=runs)


def build(tracking):
    """All features as a dict of arrays. Cheap enough to rebuild on every run (seconds)."""
    nearest_team, nearest_dist, team_dist = nearest_to_ball(tracking)
    seen = tracking.player_seen
    with np.errstate(invalid="ignore"):
        speeds = np.where(seen, tracking.player_speed, np.nan)
        counts = seen.sum(axis=1)
        mean_speed = np.where(counts > 0, np.nansum(speeds, axis=1) / np.maximum(counts, 1), np.nan)
    return {
        "ball_seen": tracking.ball_seen,
        "ball_xy": tracking.ball,
        "ball_speed": ball_speed(tracking.ball),
        "nearest_team": nearest_team,
        "nearest_dist": nearest_dist,
        "team_dist": team_dist,
        "mean_speed": mean_speed,
        "n_tracked": counts.astype(np.int16),
    }
