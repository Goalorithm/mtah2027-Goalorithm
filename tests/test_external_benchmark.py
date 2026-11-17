"""Synthetic other-game alignment, reference isolation, and limited benchmark semantics."""

import hashlib
import io

import benchmark_external
import numpy as np
import pandas as pd
import pytest

from mtah import A, D, H, X, external
from mtah.paths import PITCH


@pytest.fixture
def sample(tmp_path):
    source = {
        "fps": 25,
        "pitch_m": [105, 68],
        "time_tolerance_s": 0.000001,
        "ball_tolerance_normalized": 0.000000001,
    }
    for team, origin in (("Home", 0.1), ("Away", 0.8)):
        lines = ["Synthetic tracking", "Synthetic jerseys", "Period,Frame,Time [s],Player1,,Ball,"]
        for frame in range(1, 9):
            period = 1 if frame <= 4 else 2
            x = origin + 0.01 * frame + (0.2 if period == 2 else 0)
            lines.append(f"{period},{frame},{frame / 25},{x},0.5,0.2,0.5")
        (tmp_path / f"Sample_Game_1_RawTrackingData_{team}_Team.csv").write_text("\n".join(lines))
    events = pd.DataFrame(
        [
            {"Team": "Home", "Type": "PASS", "Period": 1, "Start Frame": 2, "Start Time [s]": 0.08},
            {"Team": "Away", "Type": "PASS", "Period": 2, "Start Frame": 6, "Start Time [s]": 0.24},
        ]
    )
    events.to_csv(tmp_path / "Sample_Game_1_RawEventsData.csv", index=False)
    return tmp_path, source


def test_external_units_frame_alignment_and_speed_reset_at_halftime(sample):
    directory, source = sample
    tracking, runs, anchors, metadata = external.read_sample_game(directory, 1, source)
    assert tracking.n_frames == 8 and tracking.n_slots == 2
    assert runs == [(0, 4), (4, 8)]
    np.testing.assert_allclose(tracking.ball[0], np.array([0.2, 0.5]) * PITCH)
    np.testing.assert_array_equal(tracking.slot_team, [H, A])
    assert np.isnan(tracking.player_speed[[0, 4]]).all()
    assert np.isfinite(tracking.player_speed[[1, 2, 3, 5, 6, 7]]).all()
    np.testing.assert_array_equal(anchors, [X, H, X, X, X, A, X, X])
    assert metadata["anchors"]["evaluated_frames"] == 2


def test_home_away_timeline_mismatch_is_rejected(sample):
    directory, source = sample
    path = directory / "Sample_Game_1_RawTrackingData_Away_Team.csv"
    path.write_text(path.read_text().replace("1,2,0.08", "1,2,0.09"))
    with pytest.raises(ValueError, match="timelines differ"):
        external.read_sample_game(directory, 1, source)


def test_disagreeing_duplicate_ball_positions_are_not_silently_selected(sample):
    directory, source = sample
    path = directory / "Sample_Game_1_RawTrackingData_Away_Team.csv"
    path.write_text(path.read_text().replace(",0.2,0.5", ",0.4,0.5"))
    with pytest.raises(ValueError, match="ball coordinates disagree"):
        external.read_sample_game(directory, 1, source)


def test_duplicate_anchor_events_are_deduplicated_and_conflicts_are_excluded():
    rows = [
        {
            "Team": team,
            "Type": "PASS",
            "Period": 1,
            "Start Frame": frame,
            "Start Time [s]": frame / 25,
        }
        for team, frame in (("Home", 1), ("Home", 1), ("Home", 2), ("Away", 2), ("Away", 3))
    ]
    reference, report = external.pass_start_anchors(
        pd.DataFrame(rows), np.ones(4), np.arange(1, 5) / 25, 0.000001
    )
    np.testing.assert_array_equal(reference, [H, X, A, X])
    assert report["pass_events"] == 5 and report["evaluated_frames"] == 2
    assert report["conflicting_frames_excluded"] == 1


@pytest.mark.parametrize("column,value", [("Start Time [s]", 0.09), ("Period", 2)])
def test_event_alignment_errors_do_not_get_a_guessed_offset(sample, column, value):
    directory, source = sample
    path = directory / "Sample_Game_1_RawEventsData.csv"
    events = pd.read_csv(path)
    events.loc[0, column] = value
    events.to_csv(path, index=False)
    with pytest.raises(ValueError, match="aligned|exact tracking frame"):
        external.read_sample_game(directory, 1, source)


def test_anchor_metrics_count_dead_and_abstaining_predictions_as_misses():
    result = external.anchor_metrics([H, A, H, X], [H, D, X, A])
    assert result["anchor_count"] == 3 and result["correct_team_count"] == 1
    assert result["pass_start_team_accuracy"] == pytest.approx(1 / 3)
    assert result["balanced_team_recall"] == 0.25
    assert result["confusion_rows_H_A_columns_H_A_D_X"] == [[1, 0, 0, 1], [0, 0, 1, 0]]
    assert result["fullmatch_spell_f1"] is None and not result["fullmatch_accuracy_measured"]


def test_external_inference_receives_no_events_and_isolates_each_period(sample, monkeypatch):
    directory, source = sample
    tracking, runs, _, _ = external.read_sample_game(directory, 1, source)
    calls = []

    def pipeline(config, segment, tags):
        assert tags == [] and segment.n_frames == 4
        calls.append(segment)
        return np.full(segment.n_frames, H if len(calls) == 1 else A, dtype=np.int8)

    monkeypatch.setattr(external, "run_pipeline", pipeline)
    np.testing.assert_array_equal(external.predict_external({}, tracking, runs), [H] * 4 + [A] * 4)
    assert len(calls) == 2


def test_pinned_asset_mismatch_preserves_existing_file_and_does_not_download(tmp_path, monkeypatch):
    path = tmp_path / "README.md"
    path.write_bytes(b"existing changed content")

    def no_download(*args, **kwargs):
        raise AssertionError("a changed local source must not be silently replaced")

    monkeypatch.setattr(benchmark_external.urllib.request, "urlopen", no_download)
    with pytest.raises(ValueError, match="changed from pinned"):
        benchmark_external.source_files(
            tmp_path, {"commit": "test", "sha256": {"README.md": "wrong"}}, True
        )
    assert path.read_bytes() == b"existing changed content"


@pytest.mark.parametrize("valid_hash", [True, False])
def test_download_is_verified_before_destination_is_created(tmp_path, monkeypatch, valid_hash):
    payload = b"synthetic public source"
    digest = hashlib.sha256(payload if valid_hash else b"different").hexdigest()
    monkeypatch.setattr(
        benchmark_external.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    config = {"commit": "test", "sha256": {"README.md": digest}}
    if valid_hash:
        result = benchmark_external.source_files(tmp_path, config, True)
        assert result[0]["sha256"] == digest and (tmp_path / "README.md").read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="hash mismatch"):
            benchmark_external.source_files(tmp_path, config, True)
        assert not (tmp_path / "README.md").exists()
    assert not list(tmp_path.glob("*.part"))
