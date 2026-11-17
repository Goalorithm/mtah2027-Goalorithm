"""Frozen model composition must reject wrong feature/split or competition-fitting lineage."""

import json

import pytest
import select_joint

from mtah.io import sha256


@pytest.mark.parametrize("change", [None, "features", "split", "bar_ath_fit"])
def test_frozen_component_selection_validates_feature_split_and_fitting_lineage(
    tmp_path, monkeypatch, change
):
    monkeypatch.setattr(select_joint, "ROOT", tmp_path)
    config = tmp_path / "parent.toml"
    config.write_text(
        '[supervised]\nmodel_path="outputs/parent/model.json"\n'
        "[supervised.features]\nsample=1\n"
        '[supervised.training]\ntest_games=["TEST"]\n',
        encoding="utf-8",
    )
    directory = tmp_path / "outputs/parent"
    directory.mkdir(parents=True)
    provenance = {
        "test_games_read": False,
        "bar_ath_data_read": False,
        "bar_ath_labels_read": False,
    }
    model = {"feature_config": {"sample": 1}, "provenance": dict(provenance)}
    report = {
        "training_games": ["TRAIN"],
        "validation_games": ["VALID"],
        "sealed_test_games": ["TEST"],
        "provenance": provenance,
    }
    if change == "features":
        model["feature_config"]["sample"] = 2
    elif change == "split":
        report["training_games"] = ["TEST"]
    elif change == "bar_ath_fit":
        model["provenance"]["bar_ath_data_read"] = True
    checkpoint = directory / "model.json"
    training = directory / "training_report.json"
    checkpoint.write_text(json.dumps(model), encoding="utf-8")
    training.write_text(json.dumps(report), encoding="utf-8")
    receipt = {
        "model_sha256": sha256(checkpoint),
        "config_sha256": sha256(config),
        "training_report_sha256": sha256(training),
        "test_games": ["TEST"],
        "frozen_before_test_evaluation": True,
    }
    (directory / "test_freeze.json").write_text(json.dumps(receipt), encoding="utf-8")
    expected = {"train_games": ["TRAIN"], "validation_games": ["VALID"], "test_games": ["TEST"]}
    if change:
        with pytest.raises(ValueError, match="component has different"):
            select_joint.load_component("parent.toml", {"sample": 1}, expected)
    else:
        assert select_joint.load_component("parent.toml", {"sample": 1}, expected)[0] == model
