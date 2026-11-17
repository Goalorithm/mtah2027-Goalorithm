"""SoccerTrack v2 adapter: status codes follow clubs, tracking sides swap at half-time."""

import json

import numpy as np
import pytest

from mtah import A, D, H, X
from mtah.paths import PITCH

pytest.importorskip("ijson")  # optional, requirements-validation.txt

from soccertrack_data import (  # noqa: E402
    keep_last_owner,
    parse_game_state,
    reference_from_status,
    status_sides,
    to_model_pitch,
)


def write_events(path, rows):
    actions = [
        {"gameTime": f"{period} - 00:01", "label": label, "frame": frame, "team": team}
        for period, label, frame, team in rows
    ]
    path.write_text(json.dumps({"actions": actions}), encoding="utf-8")
    return path


def test_pitch_centre_and_corners_map_to_the_model_pitch():
    mapped = to_model_pitch([[0.0, 0.0], [-52.5, -34.0], [52.5, 34.0]])
    assert np.allclose(mapped[0], np.asarray(PITCH) / 2)
    assert np.allclose(mapped[1], [0.0, 0.0])
    assert np.allclose(mapped[2], PITCH)


def test_status_side_is_read_per_half(tmp_path):
    status = np.array([1, 1, 1, 2, 2, 2, 0, 3])
    events = write_events(
        tmp_path / "events.json",
        [
            ("1", "PASS", 1, "left"),
            ("1", "DRIVE", 2, "left"),
            ("1", "PASS", 3, "right"),  # a minority event must not flip the mapping
            ("1", "PASS", 4, "right"),
            ("1", "PASS", 5, "right"),
            ("1", "SHOT", 6, "left"),  # not an owning event
            ("1", "PASS", 7, "left"),  # dead-ball frame, no vote
            ("2", "PASS", 1, "right"),
            ("2", "PASS", 2, "right"),
            ("2", "PASS", 4, "left"),
            ("2", "DRIVE", 5, "left"),
        ],
    )
    first, agreement = status_sides(events, "1st", status)
    second, full = status_sides(events, "2nd", status)
    assert first == {1: H, 2: A}
    assert second == {1: A, 2: H}
    assert agreement == pytest.approx(4 / 5)
    assert full == 1.0
    assert reference_from_status(status, first).tolist() == [H, H, H, A, A, A, D, X]
    assert reference_from_status(status, second).tolist() == [A, A, A, H, H, H, D, X]


def test_status_codes_that_do_not_separate_teams_are_rejected(tmp_path):
    status = np.array([1, 1, 2, 2])
    same = write_events(
        tmp_path / "same.json",
        [("1", "PASS", 1, "left"), ("1", "PASS", 3, "left")],
    )
    silent = write_events(tmp_path / "silent.json", [("1", "PASS", 1, "left")])
    for path in (same, silent):
        with pytest.raises(ValueError):
            status_sides(path, "1st", status)


def test_missing_status_is_ungraded():
    reference = reference_from_status(np.array([-1, 0, 3, 2]), {1: H, 2: A})
    assert reference.tolist() == [X, D, X, A]


def test_contested_frames_take_the_previous_owner():
    reference = np.array([X, H, X, X, D, X, A, X], dtype=np.int8)
    assert keep_last_owner(reference).tolist() == [X, H, H, H, D, D, A, A]
    assert reference[2] == X  # the input is left untouched


def test_game_state_keeps_playing_roles_and_feet(tmp_path):
    def record(frame, track, role, team, x, y, box=True):
        return {
            "supercategory": "object",
            "image_id": f"1{frame:06d}",
            "track_id": track,
            "attributes": {"role": role, "team": team},
            "bbox_pitch": {"x_bottom_middle": x, "y_bottom_middle": y},
            "bbox_image": {"x": 100, "y": 200, "w": 20, "h": 40} if box else None,
        }

    content = {
        "images": [{"width": 3840, "height": 1906}],
        "annotations": [
            record(1, 7, "player", "left", 0.0, 0.0),
            record(2, 7, "player", "left", 1.0, 0.0, box=False),
            record(1, 9, "goalkeeper", "right", 52.5, 34.0),
            record(1, 3, "referee", None, 5.0, 5.0),
            record(9, 7, "player", "left", 2.0, 0.0),  # beyond the half
            {"supercategory": "pitch", "image_id": "1000001"},
        ],
    }
    path = tmp_path / "half.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    players, pixels, team, size = parse_game_state(path, 3)
    assert players.shape == (3, 2, 2)
    assert team.tolist() == [H, A]
    assert size == (3840, 1906)
    assert np.allclose(players[0, 0], np.asarray(PITCH) / 2)
    assert np.allclose(players[0, 1], PITCH)
    assert np.allclose(pixels[0, 0], [110.0, 240.0])
    assert np.isfinite(players[1, 0]).all() and np.isnan(pixels[1, 0]).all()
    assert np.isnan(players[2]).all()
