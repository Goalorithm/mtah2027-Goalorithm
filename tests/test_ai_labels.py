"""Pseudo-label provenance, complete interval coverage, and the updated BAR-ATH policy."""

import json

import ai_labels
import labels_to_csv as labels
import numpy as np
import pytest

from mtah import A, D, H, X
from mtah.io import read_codes, sha256
from mtah.paths import N_FRAMES, PUBLIC_GT
from mtah.scoring import ground_truths


def inputs():
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[30000:30003], codes[30003:30006] = H, D
    codes[40000:40006] = A
    baseline = codes.copy()
    baseline[30002:30004] = A
    windows = {
        "PUBLIC": {"start": 11185, "end": 11190, "split": "kalibrasi"},
        "V1": {"start": 30000, "end": 30005, "split": "dev"},
        "V2": {"start": 40000, "end": 40005, "split": "holdout"},
    }
    return codes, baseline, windows


def test_ai_intervals_cover_requested_windows_without_public_or_human_attribution():
    codes, baseline, windows = inputs()
    changes, intervals, summaries, splits = ai_labels.annotation_rows(codes, baseline, windows)
    assert [r["frame_mulai"] for r in changes] == [30000, 30003, 40000]
    assert all(r["anotator_ai"].startswith("algorithm:") for r in changes)
    assert all("pelabel" not in r and "label" not in r for r in changes)
    assert len(summaries) == 2 and summaries[0]["disagreement_frames"] == 2
    for name in ("V1", "V2"):
        window = windows[name]
        reconstructed = np.full(N_FRAMES, X, dtype=np.int8)
        covered = np.zeros(N_FRAMES, dtype=int)
        for row in intervals:
            if row["jendela"] != name:
                continue
            span = slice(row["frame_awal"], row["frame_akhir"] + 1)
            reconstructed[span] = ai_labels.LABELS.index(row["label_ai"])
            covered[span] += 1
        span = slice(window["start"], window["end"] + 1)
        assert np.array_equal(reconstructed[span], codes[span])
        assert (covered[span] == 1).all()
        assert covered.sum() == 6
    assert np.sum(splits["dev"] != X) == 6
    assert np.sum(splits["holdout"] != X) == 6


def test_export_is_deterministic_valid_and_not_accepted_as_human_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_labels, "OUTPUTS_DIR", tmp_path)

    def no_accuracy(*args, **kwargs):
        raise AssertionError("pseudo-label generation must not score model accuracy")

    monkeypatch.setattr(ai_labels.pm, "evaluate", no_accuracy)
    codes, baseline, windows = inputs()
    first, second = tmp_path / "first", tmp_path / "second"
    for target in (first, second):
        result = ai_labels.export_bundle(target, codes, baseline, windows, {"synthetic": True})
        assert not result["is_ground_truth"] and not result["human_reviewed"]
        assert result["selected_final_model"] is None
        assert not result["independent_of_hmm_candidate"]
        assert np.array_equal(read_codes(target / "fullmatch_ai.csv"), codes)
        manifest = json.loads((target / "manifest.json").read_text())
        for filename, digest in manifest["output_sha256"].items():
            assert sha256(target / filename) == digest
        with pytest.raises(labels.LabelError, match="headers"):
            labels.read_changes(target / "perubahan_ai.csv")
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }


def test_export_rejects_destinations_outside_ignored_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_labels, "OUTPUTS_DIR", tmp_path / "outputs")
    with pytest.raises(ValueError, match="ignored outputs"):
        ai_labels.export_bundle(tmp_path / "labels", *inputs(), {})
    assert not (tmp_path / "labels").exists()


def test_abstention_cannot_be_claimed_as_complete_window_annotation():
    codes, baseline, windows = inputs()
    codes[30000] = X
    with pytest.raises(ValueError, match="throughout"):
        ai_labels.annotation_rows(codes, baseline, windows)


def test_trained_pseudo_labels_preserve_their_actual_model_attribution():
    changes, _, _, _ = ai_labels.annotation_rows(
        *inputs(), model_name="algorithm:Gaussian-HMM-Baum-Welch"
    )
    assert all(row["anotator_ai"] == "algorithm:Gaussian-HMM-Baum-Welch" for row in changes)


def test_default_manual_converter_is_retired(capsys):
    assert labels.main([]) == 1
    assert "retired" in capsys.readouterr().out.lower()


def test_scoring_never_discovers_same_game_human_labels_or_pseudo_labels():
    assert ground_truths(False) == {"publik": str(PUBLIC_GT)}
    assert ground_truths(True) == {"publik": str(PUBLIC_GT)}
