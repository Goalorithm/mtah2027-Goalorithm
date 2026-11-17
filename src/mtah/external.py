"""Other-game Metrica sample-data adapter and limited pass-start reference evaluation.

Events are used only as external evaluation references, never as inference features.
Pass starts supply team-control anchors, not full H/A/D possession ground truth.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mtah import A, H, X
from mtah.load import Tracking
from mtah.paths import FPS, PITCH
from mtah.predict import prepare_features, run_pipeline


def _coordinates(table):
    columns = list(table.columns)
    starts = [i for i, name in enumerate(columns) if name.startswith("Player")]
    if not starts or "Ball" not in columns:
        raise ValueError("tracking needs player coordinate pairs and Ball")
    if any(i + 1 >= len(columns) for i in starts):
        raise ValueError("incomplete player coordinate pair")
    players = np.stack([table.iloc[:, i : i + 2].to_numpy(float) for i in starts], axis=1)
    ball_start = columns.index("Ball")
    ball = table.iloc[:, ball_start : ball_start + 2].to_numpy(float)
    if ball.shape[1] != 2 or np.isinf(players).any() or np.isinf(ball).any():
        raise ValueError("invalid tracking coordinates")
    return players, ball


def read_sample_game(directory, game, source_cfg):
    directory = Path(directory)
    prefix = f"Sample_Game_{game}_"
    home = pd.read_csv(directory / f"{prefix}RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(directory / f"{prefix}RawTrackingData_Away_Team.csv", skiprows=2)
    required = ["Period", "Frame", "Time [s]"]
    if not set(required).issubset(home.columns) or not set(required).issubset(away.columns):
        raise ValueError("missing tracking timeline headers")
    if len(home) != len(away) or not home[required].equals(away[required]):
        raise ValueError("home/away tracking timelines differ")
    if not np.array_equal(home["Frame"], np.arange(1, len(home) + 1)):
        raise ValueError("external frame IDs must start at one and be contiguous")
    periods = home["Period"].to_numpy()
    times = home["Time [s]"].to_numpy(float)
    if not np.isfinite(times).all() or not np.isfinite(periods).all():
        raise ValueError("nonfinite timeline metadata")
    boundaries = np.r_[0, np.flatnonzero(periods[1:] != periods[:-1]) + 1, len(home)]
    if (np.diff(periods[boundaries[:-1]]) <= 0).any():
        raise ValueError("periods must form ordered contiguous blocks")
    runs = [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)]
    if source_cfg["fps"] != FPS:
        raise ValueError("source FPS must match model feature units")
    for start, stop in runs:
        if not np.allclose(
            np.diff(times[start:stop]), 1 / FPS, rtol=0, atol=source_cfg["time_tolerance_s"]
        ):
            raise ValueError("tracking time steps do not match reference FPS")
    home_players, home_ball = _coordinates(home)
    away_players, away_ball = _coordinates(away)
    both = np.isfinite(home_ball).all(axis=1) & np.isfinite(away_ball).all(axis=1)
    if not np.allclose(
        home_ball[both], away_ball[both], rtol=0, atol=source_cfg["ball_tolerance_normalized"]
    ):
        raise ValueError("duplicated ball coordinates disagree between team files")
    # Source coordinates are normalized 0..1. Affine-rescale onto the model pitch convention.
    players = np.concatenate((home_players, away_players), axis=1) * PITCH
    ball = np.where(np.isfinite(home_ball).all(axis=1)[:, None], home_ball, away_ball) * PITCH
    speeds = np.full(players.shape[:2], np.nan)
    for start, stop in runs:
        speeds[start + 1 : stop] = (
            np.linalg.norm(np.diff(players[start:stop], axis=0), axis=2) * FPS
        )
    tracking = Tracking(
        players=players,
        player_speed=speeds,
        ball=ball,
        ball_extra=np.full(len(ball), np.nan),
        slot_team=np.r_[np.full(home_players.shape[1], H), np.full(away_players.shape[1], A)],
        team_names=("Home", "Away"),
    )
    events = pd.read_csv(directory / f"{prefix}RawEventsData.csv")
    anchors, anchor_report = pass_start_anchors(
        events, periods, times, source_cfg["time_tolerance_s"]
    )
    metadata = {
        "game": game,
        "frames": len(home),
        "players_home": home_players.shape[1],
        "players_away": away_players.shape[1],
        "periods": runs,
        "source_pitch_m": source_cfg["pitch_m"],
        "model_pitch_m": list(PITCH),
        "coordinate_transform": "normalized source coordinates affine-rescaled onto model pitch",
        "ball_seen_fraction": float(np.isfinite(ball).all(axis=1).mean()),
        "anchors": anchor_report,
    }
    return tracking, runs, anchors, metadata


def pass_start_anchors(events, periods, times, tolerance):
    required = {"Type", "Team", "Period", "Start Frame", "Start Time [s]"}
    if not required.issubset(events.columns):
        raise ValueError("missing event anchor headers")
    events = events[events["Type"].eq("PASS")]
    votes = {}
    teams = {"Home": H, "Away": A}
    for _, row in events.iterrows():
        frame_id = row["Start Frame"]
        if not np.isfinite(frame_id) or int(frame_id) != frame_id:
            raise ValueError("event frames must be integers")
        frame = int(frame_id) - 1
        if not 0 <= frame < len(times) or row["Period"] != periods[frame]:
            raise ValueError("event frame/period is not aligned to tracking")
        if (
            not np.isfinite(row["Start Time [s]"])
            or abs(row["Start Time [s]"] - times[frame]) > tolerance
        ):
            raise ValueError("event time does not match its exact tracking frame")
        if row["Team"] not in teams:
            raise ValueError("unknown event team")
        votes.setdefault(frame, set()).add(teams[row["Team"]])
    codes = np.full(len(times), X, dtype=np.int8)
    conflicts = 0
    for frame, labels in votes.items():
        if len(labels) != 1:
            conflicts += 1
        else:
            codes[frame] = next(iter(labels))
    if not (codes != X).any():
        raise ValueError("no unambiguous pass-start anchors")
    return codes, {
        "pass_events": len(events),
        "unique_pass_start_frames": len(votes),
        "conflicting_frames_excluded": conflicts,
        "evaluated_frames": int(np.sum(codes != X)),
        "frame_time_period_alignment": "exact",
        "coverage_fraction": float(np.mean(codes != X)),
        "reference_scope": "team at pass starts only; no dead-ball or complete possession truth",
    }


def predict_external(config, tracking, runs):
    """Keep all windowing/history/state paths inside the current period."""
    result = np.full(tracking.n_frames, X, dtype=np.int8)
    for start, stop in runs:
        part = Tracking(
            tracking.players[start:stop],
            tracking.player_speed[start:stop],
            tracking.ball[start:stop],
            tracking.ball_extra[start:stop],
            tracking.slot_team,
            tracking.team_names,
        )
        result[start:stop] = run_pipeline(config, part, [])
    return result


def predict_many_external(configs, tracking, runs):
    """Reuse features/numeric tree outputs across models; decode each model/half separately."""
    result = np.full((len(configs), tracking.n_frames), X, dtype=np.int8)
    for start, stop in runs:
        part = Tracking(
            tracking.players[start:stop],
            tracking.player_speed[start:stop],
            tracking.ball[start:stop],
            tracking.ball_extra[start:stop],
            tracking.slot_team,
            tracking.team_names,
        )
        prepared = prepare_features(part)
        for index, config in enumerate(configs):
            result[index, start:stop] = run_pipeline(config, part, [], prepared)
    return result


def anchor_metrics(reference, prediction):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.ndim != 1 or reference.shape != prediction.shape:
        raise ValueError("anchor/prediction shapes differ")
    if not np.isin(reference, [H, A, X]).all() or not np.isin(prediction, range(4)).all():
        raise ValueError("invalid anchor or prediction states")
    mask = reference != X
    if not mask.any():
        raise ValueError("no anchors to evaluate")
    confusion = [
        [int(np.sum(mask & (reference == team) & (prediction == state))) for state in range(4)]
        for team in (H, A)
    ]
    recalls = [row[team] / sum(row) for team, row in enumerate(confusion) if sum(row)]
    return {
        "anchor_count": int(mask.sum()),
        "correct_team_count": int(np.sum(mask & (reference == prediction))),
        "pass_start_team_accuracy": float(np.mean(reference[mask] == prediction[mask])),
        "balanced_team_recall": float(np.mean(recalls)),
        "confusion_rows_H_A_columns_H_A_D_X": confusion,
        "fullmatch_spell_f1": None,
        "fullmatch_accuracy_measured": False,
    }
