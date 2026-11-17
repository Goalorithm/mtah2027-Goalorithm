"""Metrica Sample Game 3 (FIFA EPTS tracking, JSON events) as a labelled external game.

Same provider and same blank-ball convention as sample games 1 and 2, in the EPTS layout the
competition tracking also uses. The reference comes from metrica_labels.reference_labels:
D where the corrected ball is blank, H/A from the latest owning event. No hand labels.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from metrica_labels import reference_labels

from mtah import A, H
from mtah.load import Tracking
from mtah.paths import FPS, PITCH

FILES = {
    "metadata": "Sample_Game_3_metadata.xml",
    "tracking": "Sample_Game_3_tracking.txt",
    "events": "Sample_Game_3_events.json",
}


def read_tracking(directory):
    """(Tracking, runs) through kloppy's EPTS reader; coordinates rescaled to the model pitch."""
    from kloppy import metrica

    directory = Path(directory)
    dataset = metrica.load_tracking_epts(
        meta_data=str(directory / FILES["metadata"]), raw_data=str(directory / FILES["tracking"])
    )
    if dataset.metadata.frame_rate != FPS:
        raise ValueError("EPTS frame rate must match the model feature units")
    teams = dataset.metadata.teams
    if [str(team.ground).lower().split(".")[-1] for team in teams] != ["home", "away"]:
        raise ValueError("expected a home team followed by an away team")
    slots = [
        (side, player) for side, team in zip((H, A), teams, strict=True) for player in team.players
    ]
    index = {player.player_id: k for k, (_, player) in enumerate(slots)}
    records = dataset.records
    frame_ids = np.array([record.frame_id for record in records])
    if not np.array_equal(frame_ids, np.arange(1, len(records) + 1)):
        raise ValueError("EPTS frame IDs must start at one and be contiguous")
    players = np.full((len(records), len(slots), 2), np.nan)
    ball = np.full((len(records), 2), np.nan)
    periods = np.empty(len(records), dtype=np.int64)
    for row, record in enumerate(records):
        periods[row] = record.period.id
        point = record.ball_coordinates
        if point is not None and point.x is not None and np.isfinite(point.x):
            ball[row] = (point.x, point.y)
        for player, data in record.players_data.items():
            point = data.coordinates
            if point is not None and point.x is not None and np.isfinite(point.x):
                players[row, index[player.player_id]] = (point.x, point.y)
    players *= PITCH
    ball *= PITCH
    edges = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(records)]
    runs = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    speeds = np.full(players.shape[:2], np.nan)
    for start, stop in runs:
        speeds[start + 1 : stop] = (
            np.linalg.norm(np.diff(players[start:stop], axis=0), axis=2) * FPS
        )
    tracking = Tracking(
        players,
        speeds,
        ball,
        np.full(len(records), np.nan),
        np.array([side for side, _ in slots], dtype=np.int8),
        tuple(team.name for team in teams),
    )
    return tracking, runs


def read_events(path):
    """Events as the table metrica_labels expects: Type, Team (Home/Away), Start Frame."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    return pd.DataFrame(
        {
            "Type": [row["type"]["name"] for row in rows],
            "TeamId": [row["team"]["id"] for row in rows],
            "Start Frame": [row["start"]["frame"] for row in rows],
        }
    )


def read_labelled_epts_game(directory, bridge_frames):
    directory = Path(directory)
    tracking, runs = read_tracking(directory)
    events = read_events(directory / FILES["events"])
    root = ET.parse(directory / FILES["metadata"]).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2 or not set(events["TeamId"]) <= set(team_ids):
        raise ValueError("event teams do not match the EPTS metadata teams")
    events["Team"] = events["TeamId"].map({team_ids[0]: "Home", team_ids[1]: "Away"})
    labelled, reference, report = reference_labels(tracking, runs, events, bridge_frames)
    return labelled, reference, runs, report
