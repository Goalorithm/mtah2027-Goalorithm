"""Which frames is the ball out of play?  (owner: Tech 2)

Output contract
---------------
detect(tracking, features, tagging, config) returns a bool array of length n_frames:
True = the ball is dead (the frame will be labelled D), False = the ball is alive.

How to add an idea
------------------
1. Write a function with the signature below, e.g. `def out_of_bounds(tracking, features,
   tagging, cfg)`, where `cfg` is the [dead_ball] section of the config.
2. Register it in METHODS under a short name.
3. Select it in a config file: `[dead_ball] method = "out_of_bounds"`, plus its parameters.
Keep the old methods: the baseline stays runnable, so every PR can be compared against it.

Signals worth trying (see docs/ for the team's possession rules):
  * the ball's last known position is beyond a touchline or goal line
  * every player slows down at once (features["mean_speed"])
  * Smart Tagging set pieces mark the restart that ends a stoppage
"""

import numpy as np

from mtah.features import NOBODY


def window_count(mask, window):
    """For every frame, how many True frames lie within +/- window (cumulative-sum trick)."""
    cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    index = np.arange(len(mask))
    low = np.maximum(0, index - window)
    high = np.minimum(len(mask), index + window + 1)
    return cumulative[high] - cumulative[low]


def no_ball_window(tracking, features, tagging, cfg):
    """Baseline: dead when no frame within +/- window has a ball with a nearest player."""
    located = features["nearest_team"] != NOBODY
    window = int(cfg.get("window", 50))
    if window <= 0:
        return ~located
    return window_count(located, window) == 0


def out_of_bounds(tracking, features, tagging, cfg):
    """Tech 2: confirmed outside observations, bounded memory and optional missing-ball fallback."""
    from mtah.hmm import sequence_runs
    from mtah.rule_support import boolean, geometry, number

    context = geometry(tracking, cfg)
    need = number(cfg, "outside_confirm_frames", 2, integer=True, minimum=1)
    release = number(cfg, "inside_confirm_frames", 3, integer=True, minimum=1)
    memory = number(cfg, "outside_memory_frames", 125, integer=True)
    missing = boolean(cfg, "include_missing", True)
    window = number(cfg, "window", 50, integer=True)
    out = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in sequence_runs(tracking.n_frames):
        active, outside_streak, inside_streak, last_outside = False, 0, 0, -1
        for i in range(start, stop):
            outside_streak = outside_streak + 1 if context["outside"][i] else 0
            inside_streak = inside_streak + 1 if context["inside"][i] else 0
            if context["outside"][i]:
                last_outside = i
            if outside_streak == need:
                active = True
                out[i - outside_streak + 1 : i + 1] = True
            if active and (i - last_outside > memory or inside_streak >= release):
                active = False
                if inside_streak >= release:
                    out[i - inside_streak + 1 : i + 1] = False
            out[i] = active
        if missing:
            # Never let the baseline's symmetric window borrow observations from another half.
            located = context["seen"][start:stop] & (features["nearest_team"][start:stop] != NOBODY)
            out[start:stop] |= window_count(located, window) == 0
    return out


def set_piece_restart(tracking, features, tagging, cfg):
    """Tech 2: tag windows locate searches; observed motion resolves restart timing.

    Tags alone never paint an entire interval D. A nearby sustained stoppage and a plausible
    inside-pitch moving ball are required. Team N/A is valid and never converted to an owner.
    """
    from mtah.hmm import sequence_runs
    from mtah.rule_support import geometry, number, segments

    dead = out_of_bounds(tracking, features, tagging, cfg)
    context = geometry(tracking, cfg)
    pre = number(cfg, "restart_pre_frames", 75, integer=True)
    post = number(cfg, "restart_post_frames", 25, integer=True)
    minimum = number(cfg, "stopped_min_frames", 10, integer=True, minimum=1)
    moving_need = number(cfg, "restart_confirm_frames", 3, integer=True, minimum=1)
    moving = number(cfg, "restart_ball_mps", 3.0, strict=True)
    slow = number(cfg, "slow_player_mps", 1.0, strict=True)
    still = number(cfg, "still_ball_mps", 0.75)
    maximum = number(cfg, "max_ball_speed_mps", 45.0, strict=True)
    types = cfg.get(
        "restart_types",
        [
            "Throw-in",
            "Corner Kick",
            "Goal Kick",
            "Free Kick",
            "Kick Off (Start)",
            "Kick Off (After Goal)",
        ],
    )
    if not isinstance(types, list) or not all(isinstance(value, str) for value in types):
        raise ValueError("restart_types must be a list of strings")
    if moving > maximum or still >= moving:
        raise ValueError("restart speed thresholds must satisfy still < restart <= maximum")
    supported = {value.strip().casefold() for value in types}
    # Compute from the unmodified base mask so overlapping tag order cannot change the result.
    stopped = dead | (
        (features["mean_speed"] <= slow) & (~context["seen"] | (context["speed"] <= still))
    )
    motion = context["inside"] & (context["speed"] >= moving) & (context["speed"] <= maximum)
    hints = []
    for start, stop in sequence_runs(tracking.n_frames):
        pieces = sorted(
            (
                item
                for item in tagging
                if str(item.get("code", "")).strip().casefold() == "set pieces"
                and str(item.get("tags", {}).get("Type", "")).strip().casefold() in supported
                and item["start_frame"] < stop
                and item["end_frame"] > start
                and item["start_frame"] < item["end_frame"]
            ),
            key=lambda item: (item["start_frame"], item["end_frame"]),
        )
        for item in pieces:
            low, high = max(start, item["start_frame"] - pre), min(stop, item["end_frame"] + post)
            stops = [(low + a, low + b) for a, b in segments(stopped[low:high]) if b - a >= minimum]
            for a, b in stops:
                search = max(item["start_frame"], a + minimum)
                candidates = [
                    (search + x, search + y)
                    for x, y in segments(motion[search:high])
                    if y - x >= moving_need and search + x >= b
                ]
                if not candidates:
                    continue
                restart = candidates[0][0]
                # Keep the extension local to this tagged restart, never bridge a later stoppage.
                if restart - b > pre:
                    continue
                dead[a:restart] = True
                dead[restart : restart + moving_need] = False
                hints.append(
                    {
                        "frame": int(restart),
                        "tags": dict(item.get("tags", {})),
                        "origin": "automatic_tag_and_tracking_motion",
                    }
                )
                break
    features["_restart_hints"] = sorted(hints, key=lambda item: item["frame"])
    return dead


def hmm(tracking, features, tagging, cfg):
    """Thin adapter for Tech 1's joint sequence decoder; parameters live in [hmm]."""
    from mtah import D
    from mtah.hmm import shared_inference

    return shared_inference(tracking, features, tagging, cfg) == D


def supervised(tracking, features, tagging, cfg):
    """External-label boosted trees with a learned sequence decoder."""
    from mtah import D
    from mtah.supervised import shared_inference

    return shared_inference(tracking, features, tagging, cfg) == D


METHODS = {
    "no_ball_window": no_ball_window,
    "out_of_bounds": out_of_bounds,
    "set_piece_restart": set_piece_restart,
    "hmm": hmm,
    "supervised": supervised,
}


def detect(tracking, features, tagging, config):
    cfg = config.get("dead_ball", {})
    method = cfg.get("method", "no_ball_window")
    if method not in METHODS:
        raise KeyError(f"unknown dead_ball.method {method!r}; have {sorted(METHODS)}")
    if method == "hmm":
        cfg = config.get("hmm", {})
    elif method == "supervised":
        cfg = config
    dead = np.asarray(METHODS[method](tracking, features, tagging, cfg), dtype=bool)
    if dead.shape != (tracking.n_frames,):
        raise ValueError(f"dead_ball.{method} returned shape {dead.shape}")
    return dead
