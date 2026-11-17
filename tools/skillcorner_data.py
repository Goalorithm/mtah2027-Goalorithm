"""Partial, algorithmic H/A references from pinned broadcast data; never infer D from gaps."""

import csv
import hashlib
import json

import _src_path  # noqa: F401
import numpy as np
from train_boundaries import profiles

from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.io import sha256
from mtah.load import Tracking
from mtah.paths import OUTPUTS_DIR, PITCH, ROOT
from mtah.supervised import feature_matrix


def convert(metadata, frames, phases):
    home, away = metadata["home_team"]["id"], metadata["away_team"]["id"]
    players = [p for p in metadata["players"] if p["team_id"] in (home, away)]
    identifiers = {p["id"]: i for i, p in enumerate(players)}
    if len(identifiers) != len(players):
        raise ValueError("duplicate player identifiers")
    teams = np.array([0 if p["team_id"] == home else 1 for p in players])
    scale = np.asarray(PITCH) / [metadata["pitch_length"], metadata["pitch_width"]]
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid pitch dimensions")
    parts, balls, references, runs, reports = [], [], [], [], []
    cursor = 0
    for period in (1, 2):
        rows = [row for row in frames if row["period"] == period]
        if not rows:
            continue
        first, last = rows[0]["frame"], rows[-1]["frame"]
        ids = np.array([r["frame"] for r in rows])
        if (np.diff(ids) <= 0).any():
            raise ValueError("source frames must be ordered and unique within a period")
        n = last - first + 1
        original = np.full((n, len(players), 2), np.nan)
        ball = np.full((n, 2), np.nan)
        labels = np.full(n, 3, dtype=np.int8)
        votes = np.zeros((n, 2), dtype=bool)
        for phase in phases:
            if int(phase["period"]) != period:
                continue
            side = {home: 0, away: 1}.get(int(phase["team_in_possession_id"]))
            if side is None:
                continue
            # Keep only positive in-play interiors. Phase absence never creates a D label.
            a = max(0, int(phase["frame_start"]) - first + 1)
            b = min(n, int(phase["frame_end"]) - first)
            if a < b:
                votes[a:b, side] = True
        phase_owner = np.where(votes.sum(axis=1) == 1, votes.argmax(axis=1), 3)
        for row in rows:
            t = row["frame"] - first
            for player in row["player_data"]:
                if player["player_id"] in identifiers:
                    xy = [player["x"], player["y"]]
                    if all(x is not None and np.isfinite(x) for x in xy):
                        original[t, identifiers[player["player_id"]]] = xy
            source_ball = row["ball_data"]
            xy = [source_ball["x"], source_ball["y"]]
            if source_ball["is_detected"] is True and all(
                x is not None and np.isfinite(x) for x in xy
            ):
                ball[t] = xy
                group = row["possession"]["group"]
                owner = {"home team": 0, "away team": 1}.get(group, 3)
                if owner != 3 and phase_owner[t] == owner:
                    labels[t] = owner
        # Source is 10Hz. Interpolate player positions only between adjacent finite source
        # observations. Ball observations and reference labels stay sparse at source times.
        target = np.arange(int(np.rint((n - 1) * 2.5)) + 1) / 2.5
        lo = np.floor(target).astype(int)
        hi = np.minimum(lo + 1, n - 1)
        fraction = (target - lo)[:, None, None]
        located = original[lo] * (1 - fraction) + original[hi] * fraction
        exact = target == lo
        located[exact] = original[lo[exact]]
        observed = np.rint(np.arange(n) * 2.5).astype(int)
        valid = observed < len(target)
        target_ball = np.full((len(target), 2), np.nan)
        target_reference = np.full(len(target), 3, dtype=np.int8)
        target_ball[observed[valid]] = ball[valid]
        target_reference[observed[valid]] = labels[valid]
        located = located * scale + np.asarray(PITCH) / 2
        target_ball = target_ball * scale + np.asarray(PITCH) / 2
        if parts:
            parts.append(np.full((1, len(players), 2), np.nan))
            balls.append(np.full((1, 2), np.nan))
            references.append(np.array([3], dtype=np.int8))
            cursor += 1
        runs.append((cursor, cursor + len(target)))
        cursor += len(target)
        parts.append(located)
        balls.append(target_ball)
        references.append(target_reference)
        reports.append(
            {
                "period": period,
                "source_frames": n,
                "target_frames": len(target),
                "known_H_A": np.bincount(labels[labels != 3], minlength=2).tolist(),
                "ball_source_observed": int(np.isfinite(ball).all(axis=1).sum()),
            }
        )
    if not parts:
        raise ValueError("no playable period in source")
    coordinates = np.concatenate(parts)
    speeds = np.full(coordinates.shape[:2], np.nan)
    for a, b in runs:
        speeds[a + 1 : b] = np.linalg.norm(np.diff(coordinates[a:b], axis=0), axis=2) * 25
    tracking = Tracking(
        coordinates,
        speeds,
        np.concatenate(balls),
        np.full(cursor, np.nan),
        teams,
        (metadata["home_team"]["name"], metadata["away_team"]["name"]),
    )
    return (
        tracking,
        np.concatenate(references),
        runs,
        {
            "periods": reports,
            "source_fps": 10,
            "model_fps": 25,
            "label_time_quantization_max_s": 0.02,
            "reference": (
                "algorithmic known owner, detected ball and matching positive phase interior; "
                "all other frames X"
            ),
            "D_reference_available": False,
            "complete_spell_reference_available": False,
        },
    )


def read_game(game):
    directory = ROOT / "data/external/skillcorner_new"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    record = next(r for r in manifest["files"] if r["game"] == game)
    metadata_path = directory / f"{game}_match.json"
    tracking_path = directory / f"{game}_tracking_extrapolated.jsonl"
    phase_path = directory / f"{game}_phases_of_play.csv"
    if (
        sha256(metadata_path) != record["metadata_sha256"]
        or sha256(tracking_path) != record["tracking_sha256"]
    ):
        raise ValueError("source files no longer match pinned SkillCorner manifest")
    if sha256(phase_path) != record["phases_sha256"]:
        raise ValueError("phase source changed")
    with tracking_path.open(encoding="utf-8") as handle:
        frames = [json.loads(line) for line in handle]
    with phase_path.open(encoding="utf-8", newline="") as handle:
        phases = list(csv.DictReader(handle))
    return convert(json.loads(metadata_path.read_text(encoding="utf-8")), frames, phases)


def load_profiles(game, split, parent):
    cfg = load_config("config/skillcorner.toml")
    if game not in cfg[f"{split}_games"]:
        raise ValueError("SkillCorner game has a different declared role")
    directory = OUTPUTS_DIR / "skillcorner_cache"
    directory.mkdir(exist_ok=True)
    manifest_path = ROOT / "data/external/skillcorner_new/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(
        cfg[key] != manifest["roles"][key]
        for key in ("train_games", "validation_games", "test_games")
    ):
        raise ValueError("source roles differ from pinned download manifest")
    row = next(r for r in manifest["files"] if r["game"] == game)
    source_directory = ROOT / "data/external/skillcorner_new"
    for suffix, key in (
        ("match.json", "metadata_sha256"),
        ("tracking_extrapolated.jsonl", "tracking_sha256"),
        ("phases_of_play.csv", "phases_sha256"),
    ):
        if sha256(source_directory / f"{game}_{suffix}") != row[key]:
            raise ValueError("SkillCorner source changed before cache lookup")
    signature = {
        "parser": sha256(__file__),
        "model_features": parent["feature_config"],
        "manifest": sha256(ROOT / "data/external/skillcorner_new/manifest.json"),
        "parent": hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest(),
        "source": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "src/mtah").glob("*.py")},
    }
    outputs = []
    names = ("clean", "occluded", "occluded_jittered")
    paths = [directory / f"{game}_{name}.npz" for name in names]
    try:
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if json.loads(str(data["signature"])) != signature:
                    raise ValueError("source cache changed")
                outputs.append({key: data[key] for key in data.files if key != "signature"})
        return outputs
    except (OSError, ValueError, KeyError):
        outputs = []
    print(f"Preparing partial SkillCorner {split}/{game}", flush=True)
    tracking, reference, runs, report = read_game(game)
    augmentation = {
        "dropout_fraction": 0.4,
        "dropout_mean_frames": 25,
        "coordinate_noise_std_m": 0.15,
    }
    for (profile, current), path in zip(
        profiles(tracking, runs, augmentation, 20261200 + game), paths, strict=True
    ):
        values = feature_matrix(current, parent["feature_config"], runs)
        item = {
            "values": values.astype(np.float32),
            "state": compiled_symmetric_probabilities(values, parent).astype(np.float32),
            "event": compiled_symmetric_probabilities(values, parent["boundary_classifier"]).astype(
                np.float32
            ),
            "reference": reference,
            "runs": np.asarray(runs),
            "target": np.full(len(reference), 3, dtype=np.int8),
            "weights": (reference != 3).astype(np.float32),
            "changes": np.flatnonzero(np.r_[False, np.diff(reference) != 0]),
            "game": np.asarray(str(game)),
            "profile": np.asarray(profile),
        }
        np.savez_compressed(path, **item, signature=json.dumps(signature, sort_keys=True))
        outputs.append(item)
    (directory / f"{game}_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return outputs
