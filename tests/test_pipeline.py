"""The v0 pipeline must agree with the organisers' starter/baseline.py, frame for frame.

Runs on a small synthetic ATD in the real file format, so it needs no match data and runs in CI.
"""

import random

import baseline as starter_baseline  # starter/baseline.py, on the path via mtah.paths  # noqa: I001
import numpy as np
import pytest

from mtah import D, dead_ball, features, possession
from mtah.load import Tracking, parse_meta, parse_raw
from mtah.paths import PITCH

N_FRAMES, N_SLOTS = 3000, 8


def _group(rng, missing_rate):
    roll = rng.random()
    if roll < missing_rate:
        return "NaN,NaN,NaN"
    if roll < missing_rate * 1.2:
        return "NaN"  # an irregular group: exercises the slow parsing path
    return f"{rng.random():.5f},{rng.random():.5f},{rng.random() * 8:.3f}"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    folder = tmp_path_factory.mktemp("atd")
    meta = folder / "meta.xml"
    players = "".join(
        f'<Player id="P{k}" teamId="{"T1" if k % 2 == 0 else "T2"}"/>' for k in range(N_SLOTS)
    )
    meta.write_text(
        f'<Root><Teams><Team id="T1"><Name>Home FC</Name></Team>'
        f'<Team id="T2"><Name>Away FC</Name></Team></Teams>'
        f"<Players>{players}</Players></Root>"
    )

    rng = random.Random(7)
    lines = []
    for frame_id in range(1, N_FRAMES + 1):
        groups = ";".join(_group(rng, 0.1) for _ in range(N_SLOTS))
        # long stretches without a ball, like the real data
        no_ball = (frame_id // 200) % 3 == 0 or rng.random() < 0.2
        ball = "NaN,NaN,NaN" if no_ball else f"{rng.random():.5f},{rng.random():.5f},0.0"
        lines.append(f"{frame_id}:{groups}:{ball}\n")
    raw = folder / "raw.txt"
    raw.write_text("".join(lines))
    return meta, raw


def _tracking(meta, raw):
    slot_team, names = parse_meta(meta)
    players, ball = parse_raw(raw, len(slot_team), N_FRAMES, PITCH)
    return Tracking(players[:, :, :2], players[:, :, 2], ball[:, :2], ball[:, 2], slot_team, names)


def test_meta_matches_starter(synthetic):
    meta, _ = synthetic
    slot_team, names = parse_meta(meta)
    assert names == ("Home FC", "Away FC")
    assert np.array_equal(slot_team, starter_baseline.team_of_each_track(str(meta)))


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_pipeline_matches_starter_baseline(synthetic, window):
    meta, raw = synthetic
    tracking = _tracking(meta, raw)
    feats = features.build(tracking)

    expected_codes = starter_baseline.nearest_player_labels(
        str(raw), starter_baseline.team_of_each_track(str(meta)), N_FRAMES, PITCH
    )
    assert np.array_equal(feats["nearest_team"], expected_codes)

    config = {
        "dead_ball": {"method": "no_ball_window", "window": window},
        "possession": {"method": "nearest_majority", "window": window},
    }
    dead = dead_ball.detect(tracking, feats, [], config)
    team = possession.assign(tracking, feats, dead, [], config)
    ours = np.array(["H", "A", "D", "X"])[np.where(dead, D, team)]
    theirs = starter_baseline.smooth(expected_codes, window)
    assert np.array_equal(ours, theirs)


def test_unknown_method_is_a_clear_error(synthetic):
    meta, raw = synthetic
    tracking = _tracking(meta, raw)
    feats = features.build(tracking)
    with pytest.raises(KeyError, match="unknown dead_ball.method"):
        dead_ball.detect(tracking, feats, [], {"dead_ball": {"method": "nope"}})
