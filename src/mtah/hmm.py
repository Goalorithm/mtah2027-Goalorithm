"""Tech 1's independent H/A/D sequence model, using NumPy and log-space Viterbi.

Emissions are hand-specified evidence potentials, not a model trained on ground truth.
All weights and thresholds come from [hmm] in an experiment config. Tracking, features,
and tagging are the only inference inputs. Each half is decoded independently.
"""

import json

import numpy as np

from mtah import A, D, H, X
from mtah.paths import FPS, N_FRAMES, PITCH, match_runs


def sequence_runs(n_frames, runs=None):
    """Validated half-open spans; full-length submissions use the published half bounds."""
    if runs is None:
        runs = match_runs() if n_frames == N_FRAMES else ([(0, n_frames)] if n_frames else [])
    result = []
    previous = 0
    for start, stop in runs:
        if not isinstance(start, int | np.integer) or not isinstance(stop, int | np.integer):
            raise ValueError("sequence boundaries must be integers")
        if not 0 <= previous <= start < stop <= n_frames:
            raise ValueError("sequence spans must be ordered, disjoint, and inside the video")
        result.append((int(start), int(stop)))
        previous = stop
    return result


def ball_features(ball_xy, max_gap, max_speed_mps, runs=None, fps=FPS):
    """Private ball history and bounded interpolation, without changing raw observations.

    ball_gap is -1 before the first complete observation in a span, 0 when observed.
    Interpolation needs two observed endpoints, a short gap, and plausible endpoint speed.
    It never extrapolates or bridges a half boundary. Coordinates remain in metres.
    """
    ball = np.asarray(ball_xy, dtype=np.float64)
    if ball.ndim != 2 or ball.shape[1] != 2:
        raise ValueError("ball_xy must have shape (n_frames, 2)")
    if isinstance(max_gap, bool) or not isinstance(max_gap, int | np.integer) or max_gap < 0:
        raise ValueError("max_gap must be a nonnegative integer")
    if not np.isfinite(max_speed_mps) or max_speed_mps <= 0 or fps <= 0:
        raise ValueError("interpolation speed limit and fps must be positive")
    seen = np.isfinite(ball).all(axis=1)
    filled = np.where(seen[:, None], ball, np.nan)
    last_xy = np.full_like(ball, np.nan)
    gap = np.full(len(ball), -1, dtype=np.int64)
    for start, stop in sequence_runs(len(ball), runs):
        index = np.arange(start, stop)
        last = np.maximum.accumulate(np.where(seen[start:stop], index, -1))
        known = last >= 0
        last_xy[index[known]] = ball[last[known]]
        gap[index[known]] = index[known] - last[known]
        anchors = index[seen[start:stop]]
        for left, right in zip(anchors[:-1], anchors[1:], strict=True):
            missing = right - left - 1
            if missing <= 0 or missing > max_gap:
                continue
            speed = np.linalg.norm(ball[right] - ball[left]) * fps / (right - left)
            if speed > max_speed_mps:
                continue
            alpha = np.arange(1, missing + 1)[:, None] / (right - left)
            filled[left + 1 : right] = ball[left] + alpha * (ball[right] - ball[left])
    return {"ball_last_xy": last_xy, "ball_gap": gap, "ball_xy_filled": filled}


def outside_signal(history, margin_m, memory_frames=None):
    """Outside-pitch evidence with optional linear age decay and a finite horizon.

    An observed position has age zero. At the configured horizon its evidence is zero;
    this removes one source of dead-ball evidence, without asserting that play restarted.
    None preserves v1's unbounded memory; zero trusts only current observations.
    """
    if memory_frames is not None and (type(memory_frames) is not int or memory_frames < 0):
        raise ValueError("outside_memory_frames must be a nonnegative integer")
    length, width = PITCH
    last, age = history["ball_last_xy"], history["ball_gap"]
    outside = (
        (last[:, 0] < -margin_m)
        | (last[:, 0] > length + margin_m)
        | (last[:, 1] < -margin_m)
        | (last[:, 1] > width + margin_m)
    )
    signal = (outside & (age >= 0)).astype(np.float64)
    if memory_frames == 0:
        signal *= age == 0
    elif memory_frames is not None:
        signal *= np.clip(1.0 - age / memory_frames, 0.0, 1.0)
    return signal


def _probabilities(values, shape, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite, nonnegative, and have shape {shape}")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"{name} must sum to 1 on every row")
    return values


def viterbi(log_emissions, transition, initial):
    """Most likely state path. Transition rows are previous state, columns are next state.

    Zero-probability transitions are forbidden. Exact ties use the lowest state index.
    Row-wise rescaling avoids score drift on long sequences and leaves the optimum intact.
    """
    emissions = np.asarray(log_emissions, dtype=np.float64)
    if emissions.ndim != 2 or emissions.shape[1] == 0:
        raise ValueError("log_emissions must have shape (frames, states)")
    n, k = emissions.shape
    if k > np.iinfo(np.int16).max:
        raise ValueError("too many states")
    if np.isnan(emissions).any() or np.isposinf(emissions).any():
        raise ValueError("log_emissions may contain finite numbers or negative infinity only")
    transition = _probabilities(transition, (k, k), "transition")
    initial = _probabilities(initial, (k,), "initial")
    if not n:
        return np.empty(0, dtype=np.int16)
    with np.errstate(divide="ignore"):
        log_transition, log_initial = np.log(transition), np.log(initial)
    back = np.empty((n, k), dtype=np.int16)
    score = log_initial + emissions[0]
    if not np.isfinite(score.max()):
        raise ValueError("no reachable state on frame 0")
    score -= score.max()
    columns = np.arange(k)
    for t in range(1, n):
        candidates = score[:, None] + log_transition
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], columns] + emissions[t]
        if not np.isfinite(score.max()):
            raise ValueError(f"no reachable state on frame {t}")
        score -= score.max()
    path = np.empty(n, dtype=np.int16)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _validate_config(cfg):
    positive = (
        "max_interp_speed_mps",
        "distance_scale_m",
        "control_distance_m",
        "distance_cap_m",
        "evidence_clip",
        "slow_player_mps",
        "moving_player_mps",
        "still_ball_mps",
        "moving_ball_mps",
        "max_ball_speed_mps",
    )
    nonnegative = (
        "margin_m",
        "team_weight",
        "interpolated_weight",
        "visible_weight",
        "missing_weight",
        "outside_weight",
        "slow_weight",
        "moving_weight",
        "still_weight",
        "ball_motion_weight",
        "tagging_weight",
        "emission_scale",
    )
    for name in positive + nonnegative:
        if name not in cfg or not np.isfinite(cfg[name]):
            raise ValueError(f"hmm.{name} must be a finite number; use config/exp_hmm_v1.toml")
        if cfg[name] < 0 or (name in positive and cfg[name] == 0):
            raise ValueError(f"hmm.{name} is outside its permitted range")
    for name in ("max_gap", "missing_window"):
        if name not in cfg or type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"hmm.{name} must be a nonnegative integer")
    if "outside_memory_frames" in cfg and (
        type(cfg["outside_memory_frames"]) is not int or cfg["outside_memory_frames"] < 0
    ):
        raise ValueError("hmm.outside_memory_frames must be a nonnegative integer")
    if cfg["slow_player_mps"] >= cfg["moving_player_mps"]:
        raise ValueError("slow_player_mps must be below moving_player_mps")
    if not cfg["still_ball_mps"] < cfg["moving_ball_mps"] <= cfg["max_ball_speed_mps"]:
        raise ValueError("ball motion thresholds must be ordered")
    _probabilities(cfg["transition"], (3, 3), "hmm.transition")
    _probabilities(cfg["initial"], (3,), "hmm.initial")


def _team_distances(tracking, feats, filled):
    """Reuse raw features; compute extra distances only on short interpolated gaps."""
    distances = np.array(feats["team_dist"], dtype=np.float64, copy=True)
    indices = np.flatnonzero(
        np.isfinite(filled).all(axis=1) & ~np.isfinite(tracking.ball).all(axis=1)
    )
    # Chunking bounds temporary memory; it does not alter the model or its arithmetic.
    for offset in range(0, len(indices), 4096):
        rows = indices[offset : offset + 4096]
        squared = np.sum((tracking.players[rows] - filled[rows, None, :]) ** 2, axis=2)
        for team in (H, A):
            values = squared[:, tracking.slot_team == team]
            if values.shape[1]:
                best = np.min(np.where(np.isfinite(values), values, np.inf), axis=1)
                distances[rows, team] = np.where(np.isfinite(best), np.sqrt(best), np.nan)
    return distances


def _tagging_evidence(tracking, tagging, cfg):
    """Soft team evidence; defensive phases imply the opposite possessor. Restarts are excluded."""
    counts = np.zeros((tracking.n_frames + 1, 2), dtype=np.float64)
    offensive = {s.casefold() for s in cfg["offensive_codes"]}
    defensive = {s.casefold() for s in cfg["defensive_codes"]}
    aliases = {name.casefold(): i for i, name in enumerate(tracking.team_names)}
    for label, names in cfg.get("team_aliases", {}).items():
        if label not in ("H", "A"):
            raise ValueError("hmm.team_aliases keys must be H or A")
        for name in names:
            aliases[name.casefold()] = H if label == "H" else A
    for item in tagging:
        code = (item.get("code") or "").strip().casefold()
        team = aliases.get((item.get("tags", {}).get("Team") or "").strip().casefold())
        if team is None or code not in offensive | defensive:
            continue
        if code in defensive:
            team = A if team == H else H
        start = max(0, int(item["start_frame"]))
        stop = min(tracking.n_frames, int(item["end_frame"]))
        if start < stop:
            counts[start, team] += 1
            counts[stop, team] -= 1
    counts = np.cumsum(counts, axis=0)[:-1]
    total = counts.sum(axis=1)
    return (counts[:, H] - counts[:, A]) / np.maximum(total, 1)


def log_evidence(tracking, feats, tagging, cfg, runs=None):
    """Three per-frame log evidence potentials, in H/A/D order. Never reads labels/."""
    _validate_config(cfg)
    runs = sequence_runs(tracking.n_frames, runs)
    history = ball_features(tracking.ball, cfg["max_gap"], cfg["max_interp_speed_mps"], runs=runs)
    filled = history["ball_xy_filled"]
    seen = np.isfinite(tracking.ball).all(axis=1)
    located = np.isfinite(filled).all(axis=1)
    distances = _team_distances(tracking, feats, filled)
    bounded = np.where(np.isfinite(distances), distances, cfg["distance_cap_m"])
    bounded = np.minimum(bounded, cfg["distance_cap_m"])
    margin = np.clip(
        (bounded[:, A] - bounded[:, H]) / cfg["distance_scale_m"],
        -cfg["evidence_clip"],
        cfg["evidence_clip"],
    )
    reliability = np.exp(-bounded.min(axis=1) / cfg["control_distance_m"])
    reliability *= np.where(seen, 1.0, cfg["interpolated_weight"])
    team_signal = cfg["team_weight"] * margin * reliability * located
    team_signal += cfg["tagging_weight"] * _tagging_evidence(tracking, tagging, cfg)
    out = np.zeros((tracking.n_frames, 3), dtype=np.float64)
    out[:, H], out[:, A] = team_signal, -team_signal

    outside = outside_signal(history, cfg["margin_m"], cfg.get("outside_memory_frames"))
    out[:, D] -= cfg["visible_weight"] * seen
    out[:, D] += cfg["outside_weight"] * outside
    speed = np.asarray(feats["mean_speed"])
    out[:, D] += cfg["slow_weight"] * (speed < cfg["slow_player_mps"])
    out[:, D] -= cfg["moving_weight"] * (speed > cfg["moving_player_mps"])
    ball_motion = np.array(feats["ball_speed"], copy=True)
    for start, stop in runs:
        ball_motion[start] = np.nan
        observations = seen[start:stop]
        cumulative = np.concatenate(([0], np.cumsum(observations, dtype=np.int64)))
        index = np.arange(stop - start)
        low = np.maximum(0, index - cfg["missing_window"])
        high = np.minimum(stop - start, index + cfg["missing_window"] + 1)
        missing = cumulative[high] == cumulative[low]
        out[start:stop, D] += cfg["missing_weight"] * missing
    out[:, D] += cfg["still_weight"] * (ball_motion < cfg["still_ball_mps"])
    moving = (ball_motion > cfg["moving_ball_mps"]) & (ball_motion <= cfg["max_ball_speed_mps"])
    out[:, D] -= cfg["ball_motion_weight"] * moving
    return out * cfg["emission_scale"]


def infer(tracking, feats, tagging, cfg, runs=None):
    """Joint H/A/D decoding, X outside graded halves. No state crosses a half boundary."""
    runs = sequence_runs(tracking.n_frames, runs)
    evidence = log_evidence(tracking, feats, tagging, cfg, runs)
    out = np.full(tracking.n_frames, X, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = viterbi(evidence[start:stop], cfg["transition"], cfg["initial"])
    return out


def shared_inference(tracking, feats, tagging, cfg):
    """One decode per pipeline feature dict, shared by the two method adapters."""
    key = (id(tracking), id(tagging), json.dumps(cfg, sort_keys=True, allow_nan=False))
    cached = feats.get("_hmm_result")
    if cached is None or cached[0] != key:
        cached = (key, infer(tracking, feats, tagging, cfg))
        feats["_hmm_result"] = cached
    return cached[1]
