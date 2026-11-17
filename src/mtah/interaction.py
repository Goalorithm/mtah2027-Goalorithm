"""Tracking-only player/ball interactions; every temporal operation resets within a run."""

import numpy as np
from scipy.ndimage import minimum_filter1d

from mtah import A, H
from mtah.hmm import ball_features, sequence_runs
from mtah.paths import FPS


def interaction_names(cfg):
    radii = cfg.get("contact_radii_m", [0.75, 1.5, 2.5])
    offsets = cfg.get("interaction_offsets", [-25, -10, -3, 3, 10, 25])
    if (
        not isinstance(radii, list)
        or not radii
        or any(isinstance(r, bool) or not np.isfinite(r) or r <= 0 for r in radii)
        or len(set(radii)) != len(radii)
        or not isinstance(offsets, list)
        or any(type(o) is not int or o == 0 for o in offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ValueError("invalid interaction radii or temporal offsets")
    names = ["ball_turn", "ball_acceleration"]
    for side in ("H", "A"):
        names += [
            f"{side}_{key}"
            for key in (
                "second_distance",
                "approach",
                "relative_speed",
                "velocity_alignment",
                "close_fraction",
                "distance_min_5",
                "distance_min_15",
            )
        ]
        for radius in radii:
            names += [
                f"{side}_{key}_{radius}"
                for key in (
                    "contact_past_age",
                    "contact_future_age",
                    "minimum_past_age",
                    "minimum_future_age",
                )
            ]
        for offset in offsets:
            names += [f"{side}_distance_offset_{offset}"]
    return names


def _ages(mask, cap):
    index = np.arange(len(mask))
    past = np.maximum.accumulate(np.where(mask, index, -cap))
    future = np.minimum.accumulate(np.where(mask, index, len(mask) + cap)[::-1])[::-1]
    return np.minimum(index - past, cap) / cap, np.minimum(future - index, cap) / cap


def interaction_matrix(tracking, cfg, runs=None):
    """Distances, relative motion, and ages of candidate contacts, without reference labels."""
    names = interaction_names(cfg)
    runs = sequence_runs(tracking.n_frames, runs)
    cap = float(cfg["distance_cap_m"])
    age_cap = int(cfg["gap_cap_frames"])
    context = ball_features(tracking.ball, cfg["max_gap"], cfg["max_ball_speed_mps"], runs)
    ball = context["ball_xy_filled"]
    out = np.full((tracking.n_frames, len(names)), np.nan)
    radii = cfg.get("contact_radii_m", [0.75, 1.5, 2.5])
    offsets = cfg.get("interaction_offsets", [-25, -10, -3, 3, 10, 25])
    for start, stop in runs:
        b, players = ball[start:stop], tracking.players[start:stop]
        local = np.arange(stop - start)
        left, right = np.maximum(0, local - 2), np.minimum(len(local) - 1, local + 2)
        elapsed = (right - left)[:, None] / FPS
        ball_v = np.divide(
            b[right] - b[left], elapsed, out=np.full_like(b, np.nan), where=elapsed > 0
        )
        player_v = np.divide(
            players[right] - players[left],
            elapsed[:, None, :],
            out=np.full_like(players, np.nan),
            where=elapsed[:, None, :] > 0,
        )
        past, future = ball_v[left], ball_v[right]
        past_norm, future_norm = np.linalg.norm(past, axis=1), np.linalg.norm(future, axis=1)
        turn = np.sum(past * future, axis=1) / np.maximum(past_norm * future_norm, 1e-6)
        acceleration = np.linalg.norm(future - past, axis=1) / np.maximum(elapsed[:, 0], 1 / FPS)
        columns = [turn, np.minimum(acceleration, 300) / 300]
        all_dist = np.linalg.norm(players - b[:, None, :], axis=2)
        all_dist[~np.isfinite(all_dist)] = np.inf
        side_dist = []
        side_slots = []
        for side in (H, A):
            slots = np.flatnonzero(tracking.slot_team == side)
            side_slots.append(slots)
            side_dist.append(
                all_dist[:, slots].min(axis=1) if len(slots) else np.full(len(local), np.inf)
            )
        observed = np.isfinite(tracking.ball[start:stop]).all(axis=1)
        for side in (H, A):
            slots = side_slots[side]
            distance = side_dist[side]
            if len(slots):
                ordered = np.sort(all_dist[:, slots], axis=1)
                nearest = slots[np.argmin(all_dist[:, slots], axis=1)]
                second = ordered[:, min(1, len(slots) - 1)]
                near_v = player_v[local, nearest]
                near_xy = players[local, nearest]
                displacement = b - near_xy
                relative = ball_v - near_v
                approach = np.sum(relative * displacement, axis=1) / np.maximum(distance, 0.01)
                relative_speed = np.linalg.norm(relative, axis=1)
                alignment = np.sum(ball_v * near_v, axis=1) / np.maximum(
                    np.linalg.norm(ball_v, axis=1) * np.linalg.norm(near_v, axis=1),
                    1e-6,
                )
                close = np.mean(all_dist[:, slots] <= 3.0, axis=1)
            else:
                second = np.full(len(local), cap)
                approach = relative_speed = alignment = np.full(len(local), np.nan)
                close = np.zeros(len(local))
            min5 = minimum_filter1d(distance, size=11, mode="constant", cval=np.inf)
            min15 = minimum_filter1d(distance, size=31, mode="constant", cval=np.inf)
            columns += [
                np.minimum(second, cap) / cap,
                np.clip(approach, -45, 45) / 45,
                np.minimum(relative_speed, 45) / 45,
                alignment,
                close,
                np.minimum(min5, cap) / cap,
                np.minimum(min15, cap) / cap,
            ]
            closest = observed & (distance < side_dist[1 - side])
            minimum = distance <= minimum_filter1d(distance, size=7, mode="constant", cval=np.inf)
            for radius in radii:
                contact = closest & (distance <= radius)
                columns += [*_ages(contact, age_cap), *_ages(contact & minimum, age_cap)]
            for offset in offsets:
                shifted = local + offset
                values = np.minimum(distance[np.clip(shifted, 0, len(local) - 1)], cap) / cap
                values[(shifted < 0) | (shifted >= len(local))] = np.nan
                columns.append(values)
        out[start:stop] = np.column_stack(columns)
    if np.isinf(out).any():
        raise ValueError("interaction features must be finite or missing")
    return out
