"""Read the match data once, cache it, and hand everyone the same arrays.

Parsing the raw ATD (186 MB of text) takes a minute or two. The first call writes the parsed
arrays to cache/ as a compressed .npz; every later call reads that in seconds. The cache is
rebuilt automatically when the source file changes or when PARSER_VERSION is bumped.

    python -m mtah.load            # build the cache and print a summary

Conventions (shared by every module, do not change them locally):
  * index 0 of every array is video frame 0, i.e. row 0 of a submission
    (the ATD counts from 1, so frame_id 1 lands on index 0)
  * positions are in metres, x along the pitch (0..105.3), y across it (0..68.0)
  * a missing observation is NaN, for players and ball alike
  * slot_team[k] is 0 (home, Barcelona) or 1 (away, Athletic) for track slot k
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from mtah import A, H
from mtah.paths import CACHE_DIR, N_FRAMES, PITCH, games

PARSER_VERSION = 1

FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","


@dataclass
class Tracking:
    """Every tracked object, one row per video frame."""

    players: np.ndarray  # (n_frames, n_slots, 2) float64, metres
    player_speed: np.ndarray  # (n_frames, n_slots) float64, m/s as exported
    ball: np.ndarray  # (n_frames, 2) float64, metres
    ball_extra: np.ndarray  # (n_frames,) float64, the ball's third exported field
    slot_team: np.ndarray  # (n_slots,) int8, H or A
    team_names: tuple  # (home, away)

    @property
    def n_frames(self):
        return self.players.shape[0]

    @property
    def n_slots(self):
        return self.players.shape[1]

    @property
    def ball_seen(self):
        """(n_frames,) bool: the tracking located the ball on this frame."""
        return ~np.isnan(self.ball[:, 0])

    @property
    def player_seen(self):
        """(n_frames, n_slots) bool: this slot holds a tracked object on this frame."""
        return ~np.isnan(self.players[:, :, 0])


def parse_meta(meta_path):
    """(slot_team, (home_name, away_name)) from the ATD metadata XML.

    Same reading as starter/baseline.py: the first <Team> is home, and every <Player> entry is a
    track slot in raw-file order. A slot is a tracklet, not a person.
    """
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(teams)}")
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError(f"{meta_path}: no Player entries")
    slot_team = np.array([H if team_id == home_id else A for team_id in slots], dtype=np.int8)
    names = tuple(team.findtext("Name") for team in teams)
    return slot_team, names


def _floats(text):
    """Floats of a comma-separated group; NaN for anything missing or malformed."""
    out = []
    for value in text.split(VALUE_SEP):
        try:
            out.append(float(value))
        except ValueError:
            out.append(np.nan)
    return out


def parse_raw(raw_path, n_slots, n_frames=N_FRAMES, pitch=PITCH):
    """Parse the raw ATD text into arrays. Slow (a minute or two); use load_tracking()."""
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    flat_size = 3 * n_slots

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            index = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= index < n_frames:
                continue

            flat = parts[1].replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP)
            try:
                if len(flat) != flat_size:
                    raise ValueError
                players[index] = np.array(flat, dtype=np.float64).reshape(n_slots, 3)
            except ValueError:  # an irregular group somewhere: parse group by group
                for slot, group in enumerate(parts[1].split(GROUP_SEP)[:n_slots]):
                    values = _floats(group)
                    if len(values) >= 3:
                        players[index, slot] = values[:3]

            values = _floats(parts[2])
            if len(values) >= 2:
                ball[index, : min(3, len(values))] = values[:3]

    players[:, :, 0] *= length
    players[:, :, 1] *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return players, ball


def _signature(path):
    stat = os.stat(path)
    text = f"{PARSER_VERSION}|{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def load_tracking(
    data_dir=None, source="tactical", use_cache=True, cache_dir=CACHE_DIR, verbose=True
):
    """The tactical tracking as a Tracking object, read from cache when possible."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir, source)
    n_frames = games.source_frames(source, games.GAME)
    slot_team, names = parse_meta(meta_path)
    signature = _signature(raw_path)
    cache_path = os.path.join(str(cache_dir), f"atd_{source}.npz")

    if use_cache and os.path.isfile(cache_path):
        cached = np.load(cache_path)
        if str(cached["signature"]) == signature:
            return Tracking(
                cached["players"],
                cached["player_speed"],
                cached["ball"],
                cached["ball_extra"],
                slot_team,
                names,
            )
        if verbose:
            print("cache is out of date, rebuilding ...")

    if verbose:
        print(f"parsing {os.path.basename(raw_path)} (once; later runs read the cache) ...")
    raw_players, raw_ball = parse_raw(raw_path, len(slot_team), n_frames)
    tracking = Tracking(
        players=np.ascontiguousarray(raw_players[:, :, :2]),
        player_speed=np.ascontiguousarray(raw_players[:, :, 2]),
        ball=np.ascontiguousarray(raw_ball[:, :2]),
        ball_extra=np.ascontiguousarray(raw_ball[:, 2]),
        slot_team=slot_team,
        team_names=names,
    )
    if use_cache:
        os.makedirs(str(cache_dir), exist_ok=True)
        np.savez_compressed(
            cache_path,
            signature=signature,
            players=tracking.players,
            player_speed=tracking.player_speed,
            ball=tracking.ball,
            ball_extra=tracking.ball_extra,
        )
        if verbose:
            print(f"cache written: {cache_path}")
    return tracking


def load_smart_tagging(data_dir=None):
    """Smart Tagging intervals in video frames, or [] if the file is not there.

    Each item: {code, start_frame, end_frame (exclusive), start_s, end_s, tags}. Codes are upper
    case and tag values title case, so compare with .lower(). Treat every tag as evidence, never
    as an answer: it is automatic and has false positives and negatives.
    """
    try:
        import load_smart_tagging as starter_tagging  # from starter/, on the path via mtah.paths

        return starter_tagging.load(games.GAME, data_dir)
    except OSError:
        return []


if __name__ == "__main__":
    tracking = load_tracking()
    seen = tracking.ball_seen
    print(f"teams: H = {tracking.team_names[0]}, A = {tracking.team_names[1]}")
    print(f"frames {tracking.n_frames}, track slots {tracking.n_slots}")
    print(f"ball located on {int(seen.sum())} frames ({100 * seen.mean():.1f}%)")
    print(f"smart tagging instances: {len(load_smart_tagging())}")
