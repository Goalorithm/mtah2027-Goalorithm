"""A dev-only selection gate and deliberately unscorable annotation suggestions."""

import csv
import json

import compare_models
import labels_to_csv as labels
import numpy as np
import pytest
import suggest_labels

from mtah import A, D, H, X
from mtah.io import write_submission
from mtah.paths import N_FRAMES


def test_holdout_does_not_choose_between_candidates(tmp_path):
    dev = np.full(N_FRAMES, X, dtype=np.int8)
    dev[30000:30050], dev[30050:30100] = H, A
    holdout = np.full(N_FRAMES, X, dtype=np.int8)
    holdout[40000:40100] = H
    good_dev, good_holdout = dev.copy(), dev.copy()
    good_dev[40000:40100] = A
    good_holdout[30000:30100], good_holdout[40000:40100] = D, H
    first = write_submission(tmp_path / "good_dev.csv", good_dev)
    second = write_submission(tmp_path / "good_holdout.csv", good_holdout)
    result = compare_models.compare([second, first], dev, holdout)
    assert result["selected"] == str(first)
    assert result["candidates"][0]["spell_f1"] == 1
    assert result["holdout"]["spell_f1"] == 0


def test_model_selection_rejects_overlapping_validation(tmp_path):
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[30000:30100] = H
    path = write_submission(tmp_path / "s.csv", codes)
    with pytest.raises(ValueError, match="overlap"):
        compare_models.compare([path], codes, codes)


def test_public_window_cannot_be_used_as_dev(tmp_path):
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[11185:11200] = H
    path = write_submission(tmp_path / "dev.csv", codes)
    with pytest.raises(ValueError, match="public calibration"):
        compare_models.validation_codes(path, "dev")


def test_missing_dev_is_an_explicit_incomplete_result(tmp_path, capsys):
    out = tmp_path / "selection.json"
    assert (
        compare_models.main(
            ["unused.csv", "--dev", str(tmp_path / "missing.csv"), "--out", str(out)]
        )
        == 2
    )
    assert '"selected": null' in out.read_text(encoding="utf-8")
    assert "No final model" in capsys.readouterr().out


def test_annotation_suggestions_are_not_accepted_as_ground_truth(tmp_path):
    codes = np.array([H, H, A, D, D, H], dtype=np.int8)
    windows = {"V1": {"start": 1, "end": 4, "split": "dev", "pelabel": "Budi"}}
    rows = suggest_labels.suggestions(codes, windows, "candidate.csv")
    assert [r["frame_mulai"] for r in rows] == [1, 2, 3]
    assert [r["label_usulan"] for r in rows] == ["H", "A", "D"]
    path = tmp_path / "suggestions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(labels.LabelError, match="missing.*headers"):
        labels.read_changes(path)


def test_partial_dev_or_holdout_frames_cannot_be_scored_as_dev(tmp_path):
    windows = {
        "V1": {"start": 30000, "end": 30099, "split": "dev"},
        "V2": {"start": 40000, "end": 40099, "split": "holdout"},
    }
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[30000:30050] = H
    path = write_submission(tmp_path / "partial.csv", codes)
    with pytest.raises(ValueError, match="exactly match"):
        compare_models.validation_codes(path, "dev", windows)
    codes[30000:30100], codes[40000:40100] = H, A
    write_submission(path, codes)
    with pytest.raises(ValueError, match="exactly match"):
        compare_models.validation_codes(path, "dev", windows)


@pytest.fixture
def selection_inputs(tmp_path):
    windows = tmp_path / "windows.csv"
    windows.write_text(
        "jendela,frame_awal,frame_akhir,split,pelabel\n"
        "V1,30000,30099,dev,Reviewer\nV2,40000,40099,holdout,Reviewer\n"
    )
    dev = np.full(N_FRAMES, X, dtype=np.int8)
    dev[30000:30100] = H
    holdout = np.full(N_FRAMES, X, dtype=np.int8)
    holdout[40000:40100] = A
    candidate = dev.copy()
    candidate[40000:40100] = A
    write_submission(tmp_path / "dev.csv", dev)
    write_submission(tmp_path / "holdout.csv", holdout)
    write_submission(tmp_path / "candidate.csv", candidate)
    common = [
        "--dev",
        str(tmp_path / "dev.csv"),
        "--windows",
        str(windows),
        "--holdout-path",
        str(tmp_path / "holdout.csv"),
        "--out",
        str(tmp_path / "selection.json"),
    ]
    return tmp_path, common


def test_holdout_requires_saved_dev_and_reuses_identical_final_receipt(
    selection_inputs, monkeypatch
):
    directory, common = selection_inputs
    assert compare_models.main([*common, "--holdout"]) == 2
    assert compare_models.main([str(directory / "candidate.csv"), *common]) == 0
    assert compare_models.main([*common, "--holdout"]) == 0
    receipt = (directory / "selection.json").read_bytes()
    report = json.loads(receipt)
    assert report["holdout"]["spell_f1"] == 1

    def no_rescore(*args, **kwargs):
        raise AssertionError("an existing identical final receipt must not be scored again")

    monkeypatch.setattr(compare_models.pm, "evaluate", no_rescore)
    assert compare_models.main([*common, "--holdout"]) == 0
    assert compare_models.main([str(directory / "candidate.csv"), *common]) == 2
    assert (directory / "selection.json").read_bytes() == receipt


@pytest.mark.parametrize("changed", ["dev.csv", "candidate.csv", "windows.csv"])
def test_changed_selection_inputs_cannot_open_holdout(selection_inputs, changed, monkeypatch):
    directory, common = selection_inputs
    assert compare_models.main([str(directory / "candidate.csv"), *common]) == 0
    receipt = (directory / "selection.json").read_bytes()
    with (directory / changed).open("a") as handle:
        handle.write("\n")

    def no_holdout(*args, **kwargs):
        raise AssertionError("changed dev inputs must be rejected before reading holdout labels")

    monkeypatch.setattr(compare_models, "validation_codes", no_holdout)
    assert compare_models.main([*common, "--holdout"]) == 2
    assert (directory / "selection.json").read_bytes() == receipt
