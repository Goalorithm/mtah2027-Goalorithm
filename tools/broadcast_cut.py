"""Broadcast cutaways (replays, close-ups) as a stoppage cue: measured and simulated.

A TV director leaves the wide shot when play stops. The broadcast homography is invalid on
exactly those frames, so after synchronising the two recordings every tactical frame knows
whether the broadcast showed the pitch. On BAR-ATH, measured against the model's own stoppage
predictions (no ground truth), 91% of stoppages contain a cutaway, it starts a median 68 frames
after play stops and ends a median 18 frames before the restart, and only 3% of live frames lie
in a cutaway that touches no stoppage.

External games have no broadcast feed, so the cue is simulated from their reference stoppages
with those timings, randomised. Nothing here labels a frame; the cue is one more model input.
"""

import _src_path  # noqa: F401
import numpy as np
from atd_sim import _runs_of, renewal_mask

from mtah import D
from mtah.hmm import sequence_runs

CUT_NAMES = [
    "broadcast_cut",
    "cut_past_age",
    "cut_future_age",
    "cut_run_past",
    "cut_run_future",
]


def measured_cutaways(n_frames, broadcast_valid, runs, lags):
    """1 where the synchronised broadcast shows no pitch, 0 where it does, NaN where unknown."""
    cut = np.full(n_frames, np.nan)
    valid = np.asarray(broadcast_valid).astype(bool)
    for (start, stop), lag in zip(runs, lags, strict=True):
        frames = np.arange(start, stop)
        source = frames - lag
        inside = (source >= 0) & (source < len(valid))
        cut[frames[inside]] = ~valid[source[inside]]
    return cut


def simulate_cutaways(reference, runs, params, cfg, seed):
    """Cutaway indicator for a labelled external game; its own random stream, seeded."""
    reference = np.asarray(reference)
    rng = np.random.default_rng([int(seed), 7])
    cut = np.zeros(len(reference), dtype=bool)
    least = cfg["cut_minimum_frames"]
    for first, last in sequence_runs(len(reference), runs):
        cut[first:last] = renewal_mask(
            last - first,
            params["cut_live_fraction"],
            cfg["cut_live_median_frames"],
            cfg["cut_live_spread"],
            rng,
        )
        for a, b in _runs_of(reference[first:last] == D):
            start, stop = first + a, first + b
            if rng.random() >= params["cut_stoppage_share"]:
                continue
            begin = start + int(
                round(rng.normal(params["cut_delay_frames"], cfg["cut_delay_spread"]))
            )
            if rng.random() < params["cut_early_return"]:
                low, high = cfg["cut_early_frames"]
                high = int(max(low + 1, min(high, cfg["cut_early_share"] * (stop - start))))
                end = stop - int(rng.integers(low, high))
            else:
                end = stop + int(
                    round(rng.normal(cfg["cut_overrun_mean"], cfg["cut_overrun_spread"]))
                )
            begin, end = max(begin, first), min(end, last)
            if end - begin < least:
                end = min(last, begin + int(rng.integers(least, 3 * least)))
            cut[begin:end] = True
            if stop - start >= cfg["long_stoppage_frames"] and rng.random() < params["cut_break"]:
                size = int(rng.integers(cfg["cut_break_frames"][0], cfg["cut_break_frames"][1]))
                if end - begin > size + 2 * least:
                    at = int(rng.integers(begin + least, end - size - least))
                    cut[at : at + size] = False
    return cut.astype(np.float64)


def cutaway_features(cut, runs, cfg):
    """Indicator, ages of the nearest cutaway and the elapsed/remaining length of this one."""
    from atd_features import _ages, _still_runs

    cut = np.asarray(cut, dtype=np.float64)
    cap = int(cfg["cut_age_cap_frames"])
    out = np.full((len(cut), len(CUT_NAMES)), np.nan)
    for start, stop in sequence_runs(len(cut), runs):
        part = cut[start:stop]
        known = np.isfinite(part)
        active = known & (part > 0.5)
        past, future = _ages(active, cap)
        run_past, run_future = _still_runs(active, cap)
        block = np.column_stack([active.astype(np.float64), past, future, run_past, run_future])
        block[~known] = np.nan
        out[start:stop] = block
    return out


def cutaway_statistics(cut, stoppage, runs):
    """How a cutaway indicator sits against a stoppage mask (reference or predicted)."""
    cut = np.asarray(cut, dtype=np.float64)
    frames = np.concatenate([np.arange(a, b) for a, b in runs])
    frames = frames[np.isfinite(cut[frames])]
    active, dead = cut[frames] > 0.5, np.asarray(stoppage)[frames]
    delays, ends, hit, total = [], [], 0, 0
    for first, last in runs:
        on = np.nan_to_num(cut[first:last]) > 0.5
        starts, stops = (
            (np.asarray(v) for v in zip(*_runs_of(on), strict=True)) if on.any() else ([], [])
        )
        for a, b in _runs_of(np.asarray(stoppage)[first:last]):
            total += 1
            touching = [(x, y) for x, y in zip(starts, stops, strict=True) if y > a and x < b]
            if touching:
                hit += 1
                delays.append(touching[0][0] - a)
                ends.append(touching[-1][1] - b)
    return {
        "cut_share_of_frames": float(active.mean()),
        "cut_share_in_stoppage": float(active[dead].mean()),
        "cut_share_in_live": float(active[~dead].mean()),
        "stoppages_with_cutaway": hit / max(total, 1),
        "median_delay_frames": float(np.median(delays)) if delays else None,
        "median_end_minus_restart_frames": float(np.median(ends)) if ends else None,
    }
