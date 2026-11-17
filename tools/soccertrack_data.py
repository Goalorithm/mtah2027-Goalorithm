"""SoccerTrack v2 halves as model inputs and an independent possession reference.

SoccerTrack v2 (Scott et al., arXiv:2508.01802, CC BY 4.0) films ten amateur matches with a
fixed panoramic camera and ships the provider's game state: every player's pitch position per
frame with a team side, and a ball record whose `status` is the provider's possession state.
Measured on matches 117092 and 117093: `OUT` events fall on status 0, one team's passes and
drives on status 1, the other team's on status 2 (93-96% of them), and blocks, headers and
shots largely on status 3.

    status 0 -> D (ball out of play)      status 1 or 2 -> the team it belongs to
    status 3 or missing -> X (contested, not graded)

Status 1 and 2 follow the two clubs, while the tracking names teams by the side they defend
and the sides swap at half-time. So which side a status code means is read per half from the
pass and drive events: the side that makes most events under that code. H is the left side
of the half, A the right side; the names carry no home/away meaning here.

The ball positions in that record are a reconstruction between annotated events, not a
measurement, so they are never a model input: the ball comes from the video. The reference is
read only by the evaluator. No BAR-ATH data is involved and nothing is fitted on these games.
"""

import json

import _src_path  # noqa: F401
import ijson
import numpy as np

from mtah import A, D, H, X
from mtah.config import load_config
from mtah.io import sha256
from mtah.load import Tracking
from mtah.paths import CACHE_DIR, FPS, PITCH, ROOT

SOURCE_PITCH = (105.0, 68.0)
TEAM_TO_SIDE = {"left": H, "right": A}
OWNING_EVENTS = ("PASS", "DRIVE")
PLAYING_ROLES = ("player", "goalkeeper")


def half_paths(cfg, match, half):
    root = ROOT / cfg["directory"]
    return {
        "gsr": root / "gsr" / match / f"{match}_{half}.json",
        "ball": root / "ball" / f"{match}_{half}_ball.npz",
        "events": root / "bas" / match / f"{match}_12_class_events.json",
        "video": root / "videos" / match / f"{match}_panorama_{half}_half.mp4",
    }


def to_model_pitch(xy):
    """Centre-origin metres on a 105 x 68 pitch -> corner-origin metres on the model pitch."""
    xy = np.asarray(xy, dtype=np.float64)
    scale = np.asarray(PITCH) / np.asarray(SOURCE_PITCH)
    return (xy + np.asarray(SOURCE_PITCH) / 2) * scale


def status_sides(events_path, half, status):
    """{status code: H or A} for codes 1 and 2 in one half, by majority of pass/drive events."""
    events = json.loads(events_path.read_text(encoding="utf-8"))["actions"]
    period = "1" if half == "1st" else "2"
    votes = np.zeros((3, 2), dtype=np.int64)
    for event in events:
        frame = int(event["frame"])
        if (
            event["gameTime"].startswith(period)
            and event["label"] in OWNING_EVENTS
            and event.get("team") in TEAM_TO_SIDE
            and 1 <= frame <= len(status)
            and status[frame - 1] in (1, 2)
        ):
            votes[status[frame - 1], TEAM_TO_SIDE[event["team"]]] += 1
    first, second = int(votes[1].argmax()), int(votes[2].argmax())
    if first == second or votes[1].sum() == 0 or votes[2].sum() == 0:
        raise ValueError("possession status codes do not separate the two teams in this half")
    agreement = (votes[1, first] + votes[2, second]) / votes[1:].sum()
    return {1: first, 2: second}, float(agreement)


def reference_from_status(status, sides):
    reference = np.full(len(status), X, dtype=np.int8)
    reference[status == 0] = D
    for value, label in sides.items():
        reference[status == value] = label
    return reference


def keep_last_owner(reference):
    """Contested frames take the previous graded label; leading ones stay ungraded."""
    filled = np.array(reference, copy=True)
    last = X
    for index, label in enumerate(reference):
        if label == X:
            filled[index] = last
        else:
            last = label
    return filled


def with_provider_ball(item):
    """The half's tracking with the provider's reconstructed ball, for the simulator only."""
    tracking = item["tracking"]
    ball = np.where((item["status"] >= 0)[:, None], item["provider_ball"], np.nan)
    return Tracking(
        tracking.players,
        tracking.player_speed,
        ball,
        tracking.ball_extra,
        tracking.slot_team,
        tracking.team_names,
    )


def parse_game_state(path, n_frames):
    """(pitch positions, feet pixels, team per slot, frame size) streamed from a 2.7 GB JSON."""
    slots, rows = {}, []
    size = None
    with open(path, "rb") as handle:
        for record in ijson.items(handle, "annotations.item", use_float=True):
            if record.get("supercategory") != "object":
                continue
            attributes = record.get("attributes") or {}
            if attributes.get("role") not in PLAYING_ROLES:
                continue
            side = TEAM_TO_SIDE.get(attributes.get("team"))
            pitch, image = record.get("bbox_pitch"), record.get("bbox_image")
            if side is None or not pitch or pitch.get("x_bottom_middle") is None:
                continue
            frame = int(record["image_id"][-6:]) - 1  # image_id ends in the 1-based video frame
            if not 0 <= frame < n_frames:
                continue
            key = (int(record["track_id"]), side)
            slot = slots.setdefault(key, len(slots))
            feet = (np.nan, np.nan)
            if image:
                feet = (image["x"] + image["w"] / 2, image["y"] + image["h"])
            rows.append(
                (frame, slot, pitch["x_bottom_middle"], pitch["y_bottom_middle"], feet[0], feet[1])
            )
    with open(path, "rb") as handle:
        for image in ijson.items(handle, "images.item"):
            size = (int(image["width"]), int(image["height"]))
            break
    table = np.asarray(rows, dtype=np.float64)
    players = np.full((n_frames, len(slots), 2), np.nan)
    pixels = np.full((n_frames, len(slots), 2), np.nan)
    frame, slot = table[:, 0].astype(np.int64), table[:, 1].astype(np.int64)
    players[frame, slot] = to_model_pitch(table[:, 2:4])
    pixels[frame, slot] = table[:, 4:6]
    team = np.empty(len(slots), dtype=np.int8)
    for (_, side), index in slots.items():
        team[index] = side
    return players, pixels, team, size


def read_half(match, half, cfg=None):
    """One half: Tracking without a ball, feet pixels, the reference and bookkeeping."""
    cfg = cfg or load_config(ROOT / "config/soccertrack.toml")
    paths = half_paths(cfg, match, half)
    signature = json.dumps(
        {
            "gsr": paths["gsr"].stat().st_size,
            "ball": sha256(paths["ball"]),
            "reader": sha256(__file__),
        }
    )
    cache = CACHE_DIR / "soccertrack" / f"{match}_{half}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["signature"]) == signature:
                return _assemble({key: stored[key] for key in stored.files}, paths, half)
    with np.load(paths["ball"], allow_pickle=False) as ball:
        status = ball["status"].astype(np.int64)
        provider_ball = to_model_pitch(np.stack([ball["x"], ball["y"]], axis=1))
    players, pixels, team, size = parse_game_state(paths["gsr"], len(status))
    data = {
        "players": players.astype(np.float32),
        "pixels": pixels.astype(np.float32),
        "team": team,
        "status": status.astype(np.int8),
        "provider_ball": provider_ball.astype(np.float32),
        "frame_size": np.asarray(size),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data, signature=signature)
    return _assemble(data, paths, half)


def _assemble(data, paths, half):
    players = data["players"].astype(np.float64)
    status = data["status"].astype(np.int64)
    sides, agreement = status_sides(paths["events"], half, status)
    count = len(players)
    speed = np.full(players.shape[:2], np.nan)
    speed[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS
    tracking = Tracking(
        players,
        speed,
        np.full((count, 2), np.nan),
        np.full(count, np.nan),
        data["team"].astype(np.int8),
        ("left", "right"),
    )
    return {
        "tracking": tracking,
        "pixels": data["pixels"].astype(np.float64),
        "reference": reference_from_status(status, sides),
        "status": status,
        "status_sides": sides,
        "event_status_agreement": agreement,
        "provider_ball": data["provider_ball"].astype(np.float64),
        "frame_size": tuple(int(v) for v in data["frame_size"]),
        "runs": [(0, count)],
    }


if __name__ == "__main__":
    config = load_config(ROOT / "config/soccertrack.toml")
    for name in config["matches"]:
        for part in ("1st", "2nd"):
            item = read_half(name, part, config)
            reference = item["reference"]
            seen = np.isfinite(item["tracking"].players[:, :, 0]).sum(axis=1)
            share = np.bincount(reference, minlength=4) / len(reference)
            print(
                f"{name} {part}: {len(reference)} frames, {item['tracking'].n_slots} tracks, "
                f"{seen.mean():.1f} players per frame, H/A/D/X {share.round(3).tolist()}, "
                f"status sides {item['status_sides']}, "
                f"events agreeing {item['event_status_agreement']:.3f}",
                flush=True,
            )
