"""Holdout evaluations must fail if the selected model/config/report changed after freezing."""

import json
from pathlib import Path

import pytest
from benchmark_idsse import verify_freeze

from mtah.io import sha256
from mtah.paths import CONFIG_DIR


@pytest.mark.parametrize("changed", [None, "model", "config", "report"])
def test_pretest_receipt_binds_model_configuration_and_selection_report(tmp_path, changed):
    config = tmp_path / "config.toml"
    config.write_bytes((CONFIG_DIR / "exp_supervised_v1.toml").read_bytes())
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    report = tmp_path / "training_report.json"
    report.write_text(json.dumps({"provenance": {"test_games_read": False}}), encoding="utf-8")
    receipt = {
        "model_sha256": sha256(model),
        "config_sha256": sha256(config),
        "training_report_sha256": sha256(report),
        "test_games": ["J03WMX", "J03WN1"],
        "frozen_before_test_evaluation": True,
    }
    (tmp_path / "test_freeze.json").write_text(json.dumps(receipt), encoding="utf-8")
    if changed:
        path = {"model": model, "config": config, "report": report}[changed]
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write("\n ")  # Even a byte-only drift invalidates the exact receipt.
        with pytest.raises(ValueError, match="pre-test freeze"):
            verify_freeze(config, model)
    else:
        assert verify_freeze(config, model)["model_sha256"] == receipt["model_sha256"]
