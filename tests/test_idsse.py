"""External labels/units/timelines must be explicit and separated from inference data."""

from datetime import UTC, datetime, timedelta
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from mtah import A, D, H, X
from mtah.idsse import parse_game
from mtah.paths import PITCH


@pytest.fixture
def xml_game(tmp_path):
    meta = ET.Element("PutDataRequest")
    info = ET.SubElement(meta, "MatchInformation")
    ET.SubElement(info, "General", MatchId="OTHER-GAME")
    ET.SubElement(info, "Environment", PitchX="100", PitchY="60")
    teams = ET.SubElement(info, "Teams")
    for side, role in (("H", "home"), ("A", "guest")):
        team = ET.SubElement(teams, "Team", TeamId=side, TeamName=side, Role=role)
        ET.SubElement(ET.SubElement(team, "Players"), "Player", PersonId=side + "1")
    positions = ET.Element("PutDataRequest")
    parent = ET.SubElement(positions, "Positions")
    for k, half in enumerate(("firstHalf", "secondHalf")):
        for side in ("H", "A", "BALL"):
            group = ET.SubElement(
                parent,
                "FrameSet",
                GameSection=half,
                MatchId="OTHER-GAME",
                TeamId=side,
                PersonId=side + "1",
            )
            for frame in range(4):
                time = datetime(2022, 1, 1, tzinfo=UTC) + timedelta(seconds=k * 3600 + frame * 0.04)
                row = {
                    "N": str(100 + 1000 * k + frame),
                    "T": time.isoformat(),
                    "X": "0",
                    "Y": "0",
                    "S": "36",
                }
                if side == "BALL":
                    row.update(
                        BallStatus="0" if frame == 2 else "1",
                        BallPossession=["1", "2", "0", "0"][frame],
                        Z="99",
                    )
                ET.SubElement(group, "Frame", **row)
    meta_path, position_path = tmp_path / "meta.xml", tmp_path / "position.xml"
    ET.ElementTree(meta).write(meta_path, encoding="utf-8")
    ET.ElementTree(positions).write(position_path, encoding="utf-8")
    return meta_path, position_path


def test_labels_units_origin_and_half_separator_are_correct(xml_game):
    tracking, reference, runs, report = parse_game(*xml_game)
    assert runs == [(0, 4), (5, 9)]
    np.testing.assert_array_equal(reference, [H, A, D, X, X, H, A, D, X])
    np.testing.assert_allclose(tracking.ball[0], np.array(PITCH) / 2)
    assert tracking.player_speed[0, 0] == 10  # Export km/h becomes inference m/s.
    assert np.isnan(tracking.ball_extra).all()  # Neither Z nor possession/status is exposed here.
    assert np.isnan(tracking.ball[4]).all()
    assert report["halves"][0]["unlabelled_live_frames"] == 1


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("player_clock", "timestamps differ"),
        ("ball_clock", "25 fps"),
        ("owner", "unknown external ball"),
        ("identity", "does not belong"),
        ("frames", "not ordered and contiguous"),
        ("match", "match IDs differ"),
    ],
)
def test_bad_external_data_cannot_be_silently_realigned(xml_game, mutation, message):
    meta, positions = xml_game
    doc = ET.parse(positions)
    groups = doc.findall(".//FrameSet")
    balls = [g for g in groups if g.get("TeamId") == "BALL"]
    if mutation == "player_clock":
        groups[0][1].set("T", "2022-01-01T00:00:00.080+00:00")
    elif mutation == "ball_clock":
        balls[0][1].set("T", "2022-01-01T00:00:00.080+00:00")
    elif mutation == "owner":
        balls[0][0].set("BallPossession", "9")
    elif mutation == "identity":
        groups[0].set("PersonId", "UNKNOWN")
    elif mutation == "frames":
        balls[0][1].set("N", "103")
    else:
        groups[0].set("MatchId", "PROVIDED-GAME")
    doc.write(positions, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_game(meta, positions)


def test_referee_tracks_are_explicitly_excluded_from_both_teams(xml_game):
    meta, positions = xml_game
    doc = ET.parse(positions)
    official = ET.SubElement(
        doc.find("Positions"),
        "FrameSet",
        TeamId="referee",
        PersonId="REF1",
        GameSection="firstHalf",
        MatchId="OTHER-GAME",
    )
    ET.SubElement(official, "Frame", N="100", T="2022-01-01T00:00:00+00:00", X="0", Y="0", S="0")
    doc.write(positions, encoding="utf-8")
    tracking, _, _, report = parse_game(meta, positions)
    assert tracking.n_slots == 2
    assert report["halves"][0]["excluded_referee_framesets"] == 1
