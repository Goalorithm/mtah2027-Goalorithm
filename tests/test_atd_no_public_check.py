"""Competition export can validate repeatability without opening BAR-ATH references."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")

import evaluate_atd as evaluate  # noqa: E402


def test_competition_export_skips_reference_labels_and_repeats_on_cpu(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    model = outputs / "synthetic_model"
    model.mkdir(parents=True)
    (model / "model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(evaluate, "N_FRAMES", 6)
    monkeypatch.setattr(evaluate, "HELPERS", ())
    monkeypatch.setattr(evaluate, "load_model", lambda *_: ({"config": {}}, None))
    monkeypatch.setattr(evaluate, "feature_config", lambda _: {})
    monkeypatch.setattr(
        evaluate, "competition_features", lambda *_: (np.zeros((6, 1)), [(0, 6)], {})
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("BAR-ATH reference labels must not be opened or scored")

    monkeypatch.setattr(evaluate, "read_codes", forbidden)
    monkeypatch.setattr(evaluate.pm, "evaluate", forbidden)
    calls = []

    def predict(_values, _runs, _metadata, _members, device):
        calls.append(device)
        return np.array([0, 0, 0, 1, 1, 2], dtype=np.int8)

    monkeypatch.setattr(evaluate, "predict", predict)
    assert (
        evaluate.main(
            [
                "--model-dir",
                str(model),
                "--mode",
                "competition",
                "--name",
                "candidate",
                "--no-public-check",
            ]
        )
        == 0
    )
    assert calls == ["cpu", "cpu"]
    report = json.loads((model / "competition_report.json").read_text())
    assert report["repeat_identical"] is True
    assert report["bar_ath_reference_labels_read"] is False
    assert report["public_window_correctness_check"] is None
    assert (outputs / "candidate.csv").is_file()
