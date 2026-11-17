"""Fill tactical ball gaps with the broadcast camera's ball, after synchronising the two feeds.

The broadcast recording has its own timeline. Its ball track, mapped to the pitch by its own
homography, is the same physical trajectory, so the frame offset of each half is the lag that
maximises the cross-correlation of the two ball tracks, refined to the lag with the smallest
median distance between the two balls. Everything is computed from the provided tracking;
no label is read and nothing is fitted.
"""

import _src_path  # noqa: F401
import numpy as np

from mtah.load import Tracking


def _standardise(values):
    result = np.zeros(len(values))
    finite = np.isfinite(values)
    if finite.sum() > 1 and values[finite].std() > 0:
        result[finite] = (values[finite] - values[finite].mean()) / values[finite].std()
    return result


def _median_distance(tactical, broadcast, start, stop, lag):
    frames = np.arange(start, stop)
    source = frames - lag
    inside = (source >= 0) & (source < len(broadcast))
    gap = np.linalg.norm(tactical[frames[inside]] - broadcast[source[inside]], axis=1)
    gap = gap[np.isfinite(gap)]
    return (float(np.median(gap)), len(gap)) if len(gap) else (np.inf, 0)


def half_lags(tactical_ball, broadcast_ball, runs, cfg):
    """[lag per half] with tactical frame = broadcast frame + lag, plus a report."""
    size = 1 << int(np.ceil(np.log2(len(tactical_ball) + len(broadcast_ball))))
    lags, report = [], []
    for start, stop in runs:
        score = np.zeros(size)
        for axis in range(2):
            half = np.full(len(tactical_ball), np.nan)
            half[start:stop] = tactical_ball[start:stop, axis]
            score += np.fft.irfft(
                np.fft.rfft(_standardise(half), size)
                * np.conj(np.fft.rfft(_standardise(broadcast_ball[:, axis]), size)),
                size,
            )
        coarse = int(np.argmax(score))
        coarse = coarse if coarse < size // 2 else coarse - size
        radius = cfg["refine_frames"]
        candidates = [
            (_median_distance(tactical_ball, broadcast_ball, start, stop, lag)[0], lag)
            for lag in range(coarse - radius, coarse + radius + 1)
        ]
        distance, lag = min(candidates)
        shared = _median_distance(tactical_ball, broadcast_ball, start, stop, lag)[1]
        if distance > cfg["max_median_distance_m"] or shared < cfg["min_shared_frames"]:
            raise ValueError("broadcast and tactical ball tracks could not be synchronised")
        lags.append(lag)
        report.append(
            {
                "half": [int(start), int(stop)],
                "lag_frames": int(lag),
                "median_ball_distance_m": distance,
                "frames_with_both_balls": int(shared),
            }
        )
    return lags, report


def fill_ball(tactical, broadcast, runs, cfg):
    """Tactical tracking whose missing ball takes the synchronised broadcast ball."""
    lags, report = half_lags(tactical.ball, broadcast.ball, runs, cfg)
    ball = np.array(tactical.ball, dtype=np.float64, copy=True)
    added = 0
    for (start, stop), lag in zip(runs, lags, strict=True):
        frames = np.arange(start, stop)
        source = frames - lag
        inside = (source >= 0) & (source < broadcast.n_frames)
        frames, source = frames[inside], source[inside]
        use = ~np.isfinite(ball[frames]).all(axis=1) & np.isfinite(broadcast.ball[source]).all(
            axis=1
        )
        ball[frames[use]] = broadcast.ball[source[use]]
        added += int(use.sum())
    merged = Tracking(
        tactical.players,
        tactical.player_speed,
        ball,
        tactical.ball_extra,
        tactical.slot_team,
        tactical.team_names,
    )
    graded = sum(stop - start for start, stop in runs)
    return merged, {"halves": report, "ball_frames_added": added, "share_added": added / graded}
