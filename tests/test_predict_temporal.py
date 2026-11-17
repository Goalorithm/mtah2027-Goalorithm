"""Staging of the committed TCN checkpoints; no PyTorch or match data needed."""

import predict_temporal
import pytest


def committed(tmp_path, teacher=b"teacher"):
    models = tmp_path / "models"
    (models / "supervised_joint").mkdir(parents=True)
    (models / "supervised_joint" / "model.json").write_bytes(teacher)
    (models / "temporal").mkdir()
    for name in predict_temporal.TEMPORAL_FILES:
        (models / "temporal" / name).write_bytes(name.encode())
    return models


def test_stages_teacher_and_tcn_under_outputs(tmp_path):
    models, outputs = committed(tmp_path), tmp_path / "outputs"
    stage = predict_temporal.stage_checkpoints(models, outputs)
    assert stage == outputs / predict_temporal.STAGE_NAME
    assert (outputs / "supervised_joint" / "model.json").read_bytes() == b"teacher"
    for name in predict_temporal.TEMPORAL_FILES:
        assert (stage / name).read_bytes() == name.encode()
    predict_temporal.stage_checkpoints(models, outputs)  # identical teacher: rerun is fine


def test_refuses_to_overwrite_a_different_local_teacher(tmp_path):
    models, outputs = committed(tmp_path), tmp_path / "outputs"
    (outputs / "supervised_joint").mkdir(parents=True)
    (outputs / "supervised_joint" / "model.json").write_bytes(b"retrained elsewhere")
    with pytest.raises(ValueError, match="differs from the committed teacher"):
        predict_temporal.stage_checkpoints(models, outputs)
    assert (outputs / "supervised_joint" / "model.json").read_bytes() == b"retrained elsewhere"
