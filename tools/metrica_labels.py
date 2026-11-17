"""Complete H/A/D references for the Metrica sample games, derived mechanically.

Metrica is the provider of the competition ground truth, which marks a frame dead where its
corrected tracking marks the ball dead. In the public sample games the corrected ball is blank
exactly there: it is missing five frames after 98-100% of BALL OUT events and five frames
before 97% of SET PIECE events. So the blank ball gives D, and the synchronised events give the
owning team while the ball is live. No frame is labelled by a person and nothing here is read
at inference; the output is a training reference for other-game fitting only.
"""

import numpy as np
import pandas as pd

from mtah import A, D, H, X
from mtah.load import Tracking

OWNING_EVENTS = ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT")


def _runs_of(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def bridge_ball(ball, runs, max_frames):
    """Linear fill of blanks too short to be a stoppage; returns (ball, dead mask)."""
    ball = np.array(ball, dtype=np.float64, copy=True)
    dead = np.zeros(len(ball), dtype=bool)
    for first, last in runs:
        for a, b in _runs_of(~np.isfinite(ball[first:last]).all(axis=1)):
            start, stop = first + a, first + b
            if stop - start <= max_frames and start > first and stop < last:
                alpha = np.arange(1, stop - start + 1)[:, None] / (stop - start + 1)
                ball[start:stop] = ball[start - 1] + alpha * (ball[stop] - ball[start - 1])
            else:
                dead[start:stop] = True
    return ball, dead


def event_owner(events, n_frames):
    """Team of the latest owning event at or before each frame; X before the first one."""
    teams = {"Home": H, "Away": A}
    rows = events[events["Type"].isin(OWNING_EVENTS) & events["Team"].isin(list(teams))]
    frames = rows["Start Frame"].to_numpy(dtype=np.int64) - 1
    if ((frames < 0) | (frames >= n_frames)).any():
        raise ValueError("event frame outside the tracking timeline")
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    sides = rows["Team"].map(teams).to_numpy(dtype=np.int8)[order]
    index = np.searchsorted(frames, np.arange(n_frames), side="right") - 1
    return np.where(index >= 0, sides[np.clip(index, 0, None)], X).astype(np.int8), frames, sides


def reference_labels(tracking, runs, events, bridge_frames):
    """(tracking with bridged ball, H/A/D/X reference, report)."""
    ball, dead = bridge_ball(tracking.ball, runs, bridge_frames)
    owner, frames, sides = event_owner(events, tracking.n_frames)
    reference = np.where(dead, D, owner).astype(np.int8)
    refilled = 0
    for first, last in runs:
        for a, b in _runs_of(~dead[first:last]):
            start, stop = first + a, first + b
            # A live stretch starts at a restart; frames before its first event take that
            # event's team instead of inheriting the owner from before the stoppage.
            inside = np.searchsorted(frames, [start, stop])
            if inside[0] == inside[1]:
                reference[start:stop] = X
                refilled += stop - start
                continue
            head = frames[inside[0]]
            if head > start:
                reference[start:head] = sides[inside[0]]
                refilled += head - start
    if tracking.n_frames and not np.isin(reference, [H, A, D, X]).all():
        raise ValueError("invalid derived reference")
    labelled = Tracking(
        tracking.players,
        tracking.player_speed,
        ball,
        np.full(tracking.n_frames, np.nan),
        tracking.slot_team,
        tracking.team_names,
    )
    report = {
        "label_counts_H_A_D_X": np.bincount(reference, minlength=4).tolist(),
        "frames_set_from_first_event_or_unknown": int(refilled),
        "reference": "D from blank corrected ball; H/A from latest owning event; no hand labels",
    }
    return labelled, reference, report


def read_labelled_game(directory, game, source_cfg, bridge_frames):
    from mtah.external import read_sample_game

    tracking, runs, _, metadata = read_sample_game(directory, game, source_cfg)
    events = pd.read_csv(directory / f"Sample_Game_{game}_RawEventsData.csv")
    labelled, reference, report = reference_labels(tracking, runs, events, bridge_frames)
    return labelled, reference, runs, {**metadata, **report}
