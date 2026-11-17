"""IDSSE/Sportec external-game adapter. Provider labels are kept outside inference Tracking.

DFL/Sportec data: Bassek et al. (2025), doi:10.1038/s41597-025-04505-y, CC-BY-4.0.
Coordinates are centered metres; speed S is km/h. BallStatus/Possession enter references only.
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from mtah import A, D, H, X
from mtah.io import sha256
from mtah.load import Tracking
from mtah.paths import CACHE_DIR, FPS, PITCH, ROOT


def game_assets(directory, game):
    directory = Path(directory)
    result = {}
    for key, pattern in (
        ("metadata", f"*matchinformation*{game}.xml"),
        ("positions", f"*positions_raw_observed*{game}.xml"),
    ):
        files = list(directory.glob(pattern))
        if len(files) != 1:
            raise ValueError(f"expected one {key} file for {game}, found {len(files)}")
        result[key] = files[0]
    return result


def read_metadata(path):
    root = ET.parse(path).getroot()
    general, environment = root.find(".//General"), root.find(".//Environment")
    if general is None or environment is None:
        raise ValueError("missing IDSSE general or pitch metadata")
    teams, people = {}, {}
    for team in root.findall(".//Teams/Team"):
        role = team.get("Role")
        if role not in ("home", "guest"):
            raise ValueError(f"unknown team role {role}")
        side = H if role == "home" else A
        teams[team.get("TeamId")] = (side, team.get("TeamName"))
        for player in team.findall("Players/Player"):
            person = player.get("PersonId")
            if not person or person in people:
                raise ValueError("missing or duplicate player identity in external metadata")
            people[person] = (side, team.get("TeamId"))
    if sorted(value[0] for value in teams.values()) != [H, A]:
        raise ValueError("external match needs exactly one home and away team")
    pitch = np.array([float(environment.get("PitchX")), float(environment.get("PitchY"))])
    if not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("invalid external pitch dimensions")
    names = tuple(next(name for side, name in teams.values() if side == k) for k in (H, A))
    return general.get("MatchId"), pitch, names, people


def read_blocks(path):
    """Stream each frame immediately; do not materialize the multi-gigabyte XML tree."""
    blocks, attributes, rows, times = [], None, [], []
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            attributes, rows, times = dict(element.attrib), [], []
        elif event == "end" and element.tag == "Frame" and attributes is not None:
            row = element.attrib
            is_ball = attributes.get("TeamId") == "BALL"
            rows.append(
                (
                    int(row["N"]),
                    float(row["X"]),
                    float(row["Y"]),
                    float(row.get("S", "nan")),
                    int(row.get("BallStatus", -1)) if is_ball else -1,
                    int(row.get("BallPossession", -1)) if is_ball else -1,
                    round(datetime.fromisoformat(row["T"]).timestamp() * 1000),
                )
            )
            if is_ball:
                times.append(row["T"])
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            if not rows:
                raise ValueError("empty external FrameSet")
            blocks.append((attributes, np.asarray(rows, dtype=float), times))
            attributes, rows, times = None, [], []
            element.clear()
    return blocks


def parse_game(metadata, positions):
    match_id, pitch, names, people = read_metadata(metadata)
    blocks = read_blocks(positions)
    if any(attrs.get("MatchId") != match_id for attrs, _, _ in blocks):
        raise ValueError("external metadata and tracking match IDs differ")
    available = {attrs.get("GameSection") for attrs, _, _ in blocks}
    if not available <= {"firstHalf", "secondHalf"} or not available:
        raise ValueError("unsupported external game section")
    ids = sorted(people)
    slots = {name: i for i, name in enumerate(ids)}
    slot_team = np.array([people[name][0] for name in ids], dtype=np.int8)
    parts, references, runs, reports = [], [], [], []
    cursor = 0
    for section in ("firstHalf", "secondHalf"):
        section_blocks = [
            (attrs, rows, times) for attrs, rows, times in blocks if attrs["GameSection"] == section
        ]
        if not section_blocks:
            continue
        balls = [
            (rows, times) for attrs, rows, times in section_blocks if attrs["TeamId"] == "BALL"
        ]
        if len(balls) != 1:
            raise ValueError("each external half needs one ball/reference timeline")
        ball_rows, times = balls[0]
        frame_ids = ball_rows[:, 0].astype(np.int64)
        if not np.array_equal(frame_ids, np.arange(frame_ids[0], frame_ids[0] + len(frame_ids))):
            raise ValueError("external ball frame IDs are not ordered and contiguous")
        clocks = ball_rows[:, 6].astype(np.int64)
        if not (np.diff(clocks) == round(1000 / FPS)).all():
            raise ValueError("external ball timestamps do not match 25 fps")
        n = len(frame_ids)
        players = np.full((n, len(ids), 2), np.nan)
        speeds = np.full((n, len(ids)), np.nan)
        occupied = set()
        for attrs, rows, _ in section_blocks:
            if attrs["TeamId"] == "BALL" or attrs["TeamId"].casefold() == "referee":
                continue
            person = attrs.get("PersonId")
            if person not in people or people[person][1] != attrs["TeamId"]:
                raise ValueError("external player does not belong to the declared metadata team")
            if person in occupied:
                raise ValueError("duplicate player FrameSet in one half")
            occupied.add(person)
            indexes = rows[:, 0].astype(np.int64) - frame_ids[0]
            if ((indexes < 0) | (indexes >= n)).any() or (np.diff(indexes) <= 0).any():
                raise ValueError("external player frames are outside the ball clock or repeated")
            if not np.array_equal(rows[:, 6].astype(np.int64), clocks[indexes]):
                raise ValueError("external player timestamps differ from ball frame timestamps")
            players[indexes, slots[person]] = (rows[:, 1:3] + pitch / 2) * (PITCH / pitch)
            speeds[indexes, slots[person]] = rows[:, 3] / 3.6
        ball = (ball_rows[:, 1:3] + pitch / 2) * (PITCH / pitch)
        status, owner = ball_rows[:, 4], ball_rows[:, 5]
        if not np.isin(status, [0, 1]).all() or not np.isin(owner, [0, 1, 2]).all():
            raise ValueError("unknown external ball status or possession code")
        reference = np.full(n, X, dtype=np.int8)
        reference[status == 0] = D
        reference[(status == 1) & (owner == 1)] = H
        reference[(status == 1) & (owner == 2)] = A
        if parts:
            # Explicit unknown separator keeps the organiser's scorer from joining the halves.
            parts.append(
                (
                    np.full((1, len(ids), 2), np.nan),
                    np.full((1, len(ids)), np.nan),
                    np.full((1, 2), np.nan),
                )
            )
            references.append(np.array([X], dtype=np.int8))
            cursor += 1
        parts.append((players, speeds, ball))
        references.append(reference)
        runs.append((cursor, cursor + n))
        cursor += n
        reports.append(
            {
                "section": section,
                "source_first_frame": int(frame_ids[0]),
                "frames": n,
                "unlabelled_live_frames": int(np.sum(reference == X)),
                "excluded_referee_framesets": sum(
                    attrs["TeamId"].casefold() == "referee" for attrs, _, _ in section_blocks
                ),
                "label_counts_H_A_D_X": np.bincount(reference, minlength=4).tolist(),
            }
        )
    tracking = Tracking(
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
        np.full(cursor, np.nan),
        slot_team,
        names,
    )
    report = {
        "match_id": match_id,
        "pitch_source_m": pitch.tolist(),
        "model_pitch_m": list(PITCH),
        "frames_with_separator": cursor,
        "runs": runs,
        "halves": reports,
        "reference": "provider BallStatus and BallPossession; no event-derived or hand edits",
        "inference_excludes": ["BallStatus", "BallPossession", "ball Z", "events", "identities"],
    }
    return tracking, np.concatenate(references), runs, report


def read_game(directory, game, use_cache=True):
    assets = game_assets(directory, game)
    signature = {key: sha256(path) for key, path in assets.items()}
    source = json.loads((ROOT / "config/idsse_sources.json").read_text(encoding="utf-8"))
    expected = {row["name"]: row["sha256"] for row in source["files"]}
    if any(signature[key] != expected.get(path.name) for key, path in assets.items()):
        raise ValueError("external game does not match pinned IDSSE provider files")
    signature["parser_sha256"] = sha256(__file__)
    cache = CACHE_DIR / "idsse" / f"{game}.npz"
    if use_cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if json.loads(str(stored["signature"])) == signature:
                tracking = Tracking(
                    stored["players"],
                    stored["player_speed"],
                    stored["ball"],
                    stored["ball_extra"],
                    stored["slot_team"],
                    tuple(stored["team_names"]),
                )
                report = json.loads(str(stored["metadata"]))
                return tracking, stored["reference"], [tuple(row) for row in report["runs"]], report
    result = parse_game(assets["metadata"], assets["positions"])
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tracking, reference, _, metadata = result
        np.savez_compressed(
            cache,
            players=tracking.players,
            player_speed=tracking.player_speed,
            ball=tracking.ball,
            ball_extra=tracking.ball_extra,
            slot_team=tracking.slot_team,
            team_names=np.asarray(tracking.team_names),
            reference=reference,
            metadata=json.dumps(metadata),
            signature=json.dumps(signature, sort_keys=True),
        )
    return result
