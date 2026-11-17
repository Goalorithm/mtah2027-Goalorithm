"""External-only supervised tracking features, portable boosted trees and sequence decoding.

No reference labels, event fields, identity strings, ball_extra or video enter inference.
Provider labels are passed explicitly to the separate training tool on other matches only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from mtah import A, H, X, features
from mtah.dead_ball import window_count
from mtah.hmm import _probabilities, ball_features, sequence_runs, viterbi
from mtah.io import sha256
from mtah.load import Tracking
from mtah.paths import FPS, PITCH, ROOT
from mtah.rule_support import number

BASE_FEATURES = [
    "ball_seen",
    "interpolated",
    "gap_past",
    "gap_future",
    "H_distance",
    "A_distance",
    "H_last_vote",
    "A_last_vote",
    "H_next_vote",
    "A_next_vote",
    "edge_last",
    "edge_next",
    "ball_speed",
    "player_speed",
    "slow_players",
    "tracked_fraction",
]


def feature_names(cfg):
    names = BASE_FEATURES + [
        f"{name}_{window}"
        for window in cfg["windows"]
        for name in ("seen", "H_vote", "A_vote", "player_motion", "ball_motion")
    ]
    if cfg.get("motion_lag_frames", 1) > 1:
        names += ["net_ball_speed", *[f"net_ball_motion_{window}" for window in cfg["windows"]]]
    if cfg.get("interactions", False):
        from mtah.interaction import interaction_names

        names += interaction_names(cfg)
    return names


def swap_features(values, names):
    indexes = []
    for name in names:
        other = (
            "A_" + name[2:]
            if name.startswith("H_")
            else ("H_" + name[2:] if name.startswith("A_") else name)
        )
        indexes.append(names.index(other))
    return np.asarray(values)[:, indexes]


def feature_matrix(tracking, cfg, runs=None):
    max_gap = number(cfg, "max_gap", 25, integer=True)
    max_speed = number(cfg, "max_ball_speed_mps", 45.0, strict=True)
    cap = number(cfg, "distance_cap_m", 30.0, strict=True)
    gap_cap = number(cfg, "gap_cap_frames", 250, integer=True, minimum=1)
    slow = number(cfg, "slow_player_mps", 1.0, strict=True)
    moving = number(cfg, "moving_ball_mps", 3.0, strict=True)
    player_scale = number(cfg, "player_speed_scale_mps", 12.0, strict=True)
    motion_lag = number(cfg, "motion_lag_frames", 1, integer=True, minimum=1)
    windows = cfg["windows"]
    if (
        not isinstance(windows, list)
        or not windows
        or any(type(w) is not int or w < 0 for w in windows)
    ):
        raise ValueError("feature windows must be a nonempty list of nonnegative integers")
    if len(set(windows)) != len(windows):
        raise ValueError("feature windows must be distinct")
    runs = sequence_runs(tracking.n_frames, runs)
    context = ball_features(tracking.ball, max_gap, max_speed, runs)
    ball = context["ball_xy_filled"]
    seen = np.isfinite(tracking.ball).all(axis=1)
    located = np.isfinite(ball).all(axis=1)
    squared = np.sum((tracking.players - ball[:, None, :]) ** 2, axis=2)
    distances = np.full((tracking.n_frames, 2), cap)
    for side in (H, A):
        slots = tracking.slot_team == side
        if slots.any():
            closest = np.min(
                np.where(np.isfinite(squared[:, slots]), squared[:, slots], np.inf), axis=1
            )
            distances[:, side] = np.minimum(np.sqrt(closest), cap)
    tied = distances[:, H] == distances[:, A]
    home = (distances[:, H] < distances[:, A]).astype(float) + 0.5 * tied
    valid_vote = seen & (distances.min(axis=1) < cap)
    home[~valid_vote] = np.nan
    feats = features.build(tracking)
    valid_players = np.isfinite(tracking.players).all(axis=2) & np.isfinite(tracking.player_speed)
    counts = valid_players.sum(axis=1)
    slow_players = np.sum(valid_players & (tracking.player_speed <= slow), axis=1) / np.maximum(
        counts, 1
    )
    out = np.full((tracking.n_frames, len(feature_names(cfg))), np.nan)
    length, width = PITCH

    def edge(xy):
        return (
            np.clip(
                np.min(np.c_[xy[:, 0], length - xy[:, 0], xy[:, 1], width - xy[:, 1]], axis=1),
                -cap,
                cap,
            )
            / cap
        )

    for start, stop in runs:
        index = np.arange(start, stop)
        last = np.maximum.accumulate(np.where(valid_vote[start:stop], index, -1))
        next_seen = np.minimum.accumulate(np.where(valid_vote[start:stop], index, stop)[::-1])[::-1]
        past_gap = np.where(last >= 0, index - last, gap_cap)
        future_gap = np.where(next_seen < stop, next_seen - index, gap_cap)
        last_vote = np.where(past_gap < gap_cap, home[np.clip(last, start, stop - 1)], 0.5)
        next_vote = np.where(future_gap < gap_cap, home[np.clip(next_seen, start, stop - 1)], 0.5)
        last_xy = np.array(context["ball_last_xy"][start:stop], copy=True)
        last_xy[context["ball_gap"][start:stop] >= gap_cap] = np.nan
        next_xy = np.array(tracking.ball[np.clip(next_seen, start, stop - 1)], copy=True)
        next_xy[future_gap >= gap_cap] = np.nan
        speed = np.r_[np.nan, np.linalg.norm(np.diff(ball[start:stop], axis=0), axis=1) * FPS]
        speed[speed > max_speed] = np.nan
        out[start:stop, : len(BASE_FEATURES)] = np.c_[
            seen[start:stop],
            located[start:stop] & ~seen[start:stop],
            np.minimum(past_gap, gap_cap) / gap_cap,
            np.minimum(future_gap, gap_cap) / gap_cap,
            distances[start:stop] / cap,
            last_vote,
            1 - last_vote,
            next_vote,
            1 - next_vote,
            edge(last_xy),
            edge(next_xy),
            speed / max_speed,
            np.minimum(feats["mean_speed"][start:stop], player_scale) / player_scale,
            slow_players[start:stop],
            np.minimum(feats["n_tracked"][start:stop] / 22, 1),
        ]
        column = len(BASE_FEATURES)
        for window in windows:
            size = window_count(np.ones(stop - start, dtype=bool), window)
            vote_count = window_count(valid_vote[start:stop], window)
            # Weighted sums are needed for ties (0.5), while window_count accepts numeric input.
            h = np.nan_to_num(home[start:stop])
            cumulative = np.r_[0.0, np.cumsum(h)]
            local = np.arange(stop - start)
            low, high = np.maximum(0, local - window), np.minimum(stop - start, local + window + 1)
            h_mean = np.divide(
                cumulative[high] - cumulative[low],
                vote_count,
                out=np.full(len(local), np.nan),
                where=vote_count > 0,
            )
            players_moving = feats["mean_speed"][start:stop] > slow
            out[start:stop, column : column + 5] = np.c_[
                window_count(seen[start:stop], window) / size,
                h_mean,
                1 - h_mean,
                window_count(players_moving, window) / size,
                window_count(np.isfinite(speed) & (speed > moving), window) / size,
            ]
            column += 5
        if motion_lag > 1:
            local = np.arange(stop - start)
            left = np.maximum(0, local - motion_lag // 2)
            right = np.minimum(stop - start - 1, local + (motion_lag + 1) // 2)
            elapsed = (right - left) / FPS
            net = np.divide(
                np.linalg.norm(ball[start:stop][right] - ball[start:stop][left], axis=1),
                elapsed,
                out=np.full(len(local), np.nan),
                where=elapsed > 0,
            )
            net[net > max_speed] = np.nan
            out[start:stop, column] = net / max_speed
            for offset, window in enumerate(windows, 1):
                size = window_count(np.ones(stop - start, dtype=bool), window)
                out[start:stop, column + offset] = (
                    window_count(np.isfinite(net) & (net > moving), window) / size
                )
    if cfg.get("interactions", False):
        from mtah.interaction import interaction_matrix, interaction_names

        count = len(interaction_names(cfg))
        out[:, -count:] = interaction_matrix(tracking, cfg, runs)
    if np.isinf(out).any():
        raise ValueError("supervised features must be finite or explicitly missing")
    return out


def hide_ball(tracking, runs, fraction, mean_frames, seed):
    """Label-independent Markov occlusion augmentation, resetting at each half."""
    if not 0 <= fraction < 1 or mean_frames <= 0:
        raise ValueError("invalid dropout fraction or mean duration")
    if fraction / max(1 - fraction, np.finfo(float).tiny) / mean_frames > 1:
        raise ValueError("dropout parameters imply an invalid transition probability")
    rng = np.random.default_rng(seed)
    ball = tracking.ball.copy()
    for start, stop in sequence_runs(tracking.n_frames, runs):
        missing = rng.random() < fraction
        draws = rng.random(stop - start)
        for i, draw in enumerate(draws, start):
            if missing:
                ball[i] = np.nan
                missing = draw >= 1 / mean_frames
            else:
                missing = draw < fraction / max(1 - fraction, np.finfo(float).tiny) / mean_frames
    return Tracking(
        tracking.players,
        tracking.player_speed,
        ball,
        np.full(tracking.n_frames, np.nan),
        tracking.slot_team,
        tracking.team_names,
    )


def perturb_ball(tracking, std_m, seed):
    """Label-independent coordinate jitter; missing observations remain missing."""
    if isinstance(std_m, bool) or not np.isfinite(std_m) or std_m < 0:
        raise ValueError("ball coordinate noise must be finite and nonnegative")
    ball = tracking.ball + np.random.default_rng(seed).normal(0, std_m, tracking.ball.shape)
    return Tracking(
        tracking.players,
        tracking.player_speed,
        ball,
        np.full(tracking.n_frames, np.nan),
        tracking.slot_team,
        tracking.team_names,
    )


def export_classifier(classifier, names):
    if not np.array_equal(classifier.classes_, [H, A, 2]):
        raise ValueError("classifier must have H/A/D classes in order")
    trees = []
    for stage in classifier._predictors:
        for side, predictor in enumerate(stage):
            nodes = predictor.nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical trees are not supported by the portable format")
            threshold = nodes["num_threshold"]
            # sklearn uses +/-inf for missing-only splits. Preserve that routing explicitly,
            # while keeping the portable JSON strictly finite and standards-compliant.
            finite_route = np.where(
                np.isposinf(threshold), 1, np.where(np.isneginf(threshold), -1, 0)
            )
            trees.append(
                {
                    "class": side,
                    **{
                        new: nodes[old].tolist()
                        for new, old in (
                            ("feature", "feature_idx"),
                            ("missing_left", "missing_go_to_left"),
                            ("left", "left"),
                            ("right", "right"),
                            ("value", "value"),
                            ("leaf", "is_leaf"),
                        )
                    },
                    "threshold": np.where(np.isinf(threshold), 0.0, threshold).tolist(),
                    **({"finite_route": finite_route.tolist()} if finite_route.any() else {}),
                }
            )
    return {
        "version": 1,
        "method": "external_supervised_boosted_trees",
        "feature_names": list(names),
        "baseline": classifier._baseline_prediction.ravel().tolist(),
        "trees": trees,
    }


def validate_classifier(model):
    if model.get("version") != 1 or model.get("method") != "external_supervised_boosted_trees":
        raise ValueError("unsupported supervised checkpoint schema")
    names = model["feature_names"]
    if not names or len(set(names)) != len(names) or not model["trees"]:
        raise ValueError("missing/duplicate supervised features or empty tree ensemble")
    base = np.asarray(model["baseline"], dtype=float)
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError("invalid boosted-tree baseline")
    for tree in model["trees"]:
        n = len(tree["leaf"])
        if tree["class"] not in (H, A, 2) or not n:
            raise ValueError("invalid boosted tree class or size")
        if any(
            len(tree[key]) != n
            for key in ("feature", "threshold", "missing_left", "left", "right", "value")
        ):
            raise ValueError("boosted tree arrays have different sizes")
        if not np.isfinite(tree["threshold"]).all() or not np.isfinite(tree["value"]).all():
            raise ValueError("nonfinite boosted tree value")
        if "finite_route" in tree and (
            len(tree["finite_route"]) != n or not np.isin(tree["finite_route"], [-1, 0, 1]).all()
        ):
            raise ValueError("invalid missing-only split routing")
        for i, leaf in enumerate(tree["leaf"]):
            if leaf not in (0, 1) or tree["missing_left"][i] not in (0, 1):
                raise ValueError("invalid boosted tree flags")
            if not leaf and (
                type(tree["feature"][i]) is not int
                or not 0 <= tree["feature"][i] < len(names)
                or any(
                    type(tree[key][i]) is not int or not i < tree[key][i] < n
                    for key in ("left", "right")
                )
            ):
                raise ValueError("invalid or cyclic boosted tree")


def raw_probabilities(values, model):
    validate_classifier(model)
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(model["feature_names"]) or np.isinf(values).any():
        raise ValueError("invalid supervised feature matrix")
    scores = np.tile(model["baseline"], (len(values), 1))
    for tree in model["trees"]:
        feature, threshold, missing_left, left, right, leaf, weight = (
            np.asarray(tree[key])
            for key in ("feature", "threshold", "missing_left", "left", "right", "leaf", "value")
        )
        node = np.zeros(len(values), dtype=np.int64)
        finite_route = np.asarray(tree.get("finite_route", np.zeros(len(leaf), dtype=np.int8)))
        for _ in range(len(leaf)):
            active = np.flatnonzero(leaf[node] == 0)
            if not len(active):
                break
            at = node[active]
            x = values[active, feature[at]]
            finite_left = np.where(finite_route[at] == 0, x <= threshold[at], finite_route[at] > 0)
            go_left = np.where(np.isnan(x), missing_left[at].astype(bool), finite_left)
            node[active] = np.where(go_left, left[at], right[at])
        scores[:, tree["class"]] += weight[node]
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def symmetric_probabilities(values, model):
    first = raw_probabilities(values, model)
    swapped = raw_probabilities(swap_features(values, model["feature_names"]), model)[:, [A, H, 2]]
    return (first + swapped) / 2


def decode_probabilities(probabilities, model, runs):
    probabilities = np.asarray(probabilities, dtype=float)
    _probabilities(probabilities, (len(probabilities), 3), "supervised probabilities")
    prior = _probabilities(model["class_prior"], (3,), "supervised class prior")
    scale = model["decoder_scale"]
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0 or (prior <= 0).any():
        raise ValueError("invalid supervised decoder scale/prior")
    emissions = (np.log(np.maximum(probabilities, np.finfo(float).tiny)) - np.log(prior)) * scale
    result = np.full(len(probabilities), X, dtype=np.int8)
    for start, stop in sequence_runs(len(probabilities), runs):
        result[start:stop] = viterbi(emissions[start:stop], model["transition"], model["initial"])
    return result


def shared_inference(tracking, feats, tagging, config):
    path = Path(config["supervised"]["model_path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError("supervised checkpoint missing; run tools/train_supervised.py")
    cfg = config["supervised"]["features"]
    key = (id(tracking), sha256(path), json.dumps(cfg, sort_keys=True))
    cached = feats.get("_supervised_result")
    if cached is None or cached[0] != key:
        model = json.loads(path.read_text(encoding="utf-8"))
        if model["feature_config"] != cfg or model["feature_names"] != feature_names(cfg):
            raise ValueError("supervised feature configuration changed; retrain")
        runs = sequence_runs(tracking.n_frames)
        feature_key = (id(tracking), json.dumps(cfg, sort_keys=True))
        feature_cache = feats.setdefault("_supervised_features", {})
        if feature_key not in feature_cache:
            feature_cache[feature_key] = feature_matrix(tracking, cfg, runs)
        values = feature_cache[feature_key]
        from mtah.compiled_trees import compiled_symmetric_probabilities

        probability_cache = feats.setdefault("_supervised_probabilities", {})

        def probabilities_for(classifier):
            numeric = {key: classifier[key] for key in ("feature_names", "baseline", "trees")}
            digest = hashlib.sha256(json.dumps(numeric, sort_keys=True).encode()).hexdigest()
            probability_key = (feature_key, digest)
            if probability_key not in probability_cache:
                probability_cache[probability_key] = compiled_symmetric_probabilities(
                    values, classifier
                )
            return probability_cache[probability_key]

        probabilities = probabilities_for(model)
        if "boundary_classifier" in model:
            from mtah.boundary import decode_boundaries

            event_probabilities = probabilities_for(model["boundary_classifier"])
            prediction = decode_boundaries(probabilities, event_probabilities, model, runs)
        else:
            prediction = decode_probabilities(probabilities, model, runs)
        cached = key, prediction
        feats["_supervised_result"] = cached
    return cached[1]
