"""Released model files are copied only when they are the ones their model files name."""

import json

import pytest
from restore_release import restore

from mtah.io import sha256


def member(directory, weights=b"weights"):
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(weights)
    metadata = {"weights_sha256": sha256(directory / "weights.pt")}
    (directory / "model.json").write_text(json.dumps(metadata))


def bundle(release):
    member(release / "one")
    member(release / "two", b"other")
    ensemble = release / "both"
    ensemble.mkdir()
    members = {name: sha256(release / name / "model.json") for name in ("one", "two")}
    (ensemble / "model.json").write_text(json.dumps({"members": members}))


def test_models_are_copied_and_a_second_run_changes_nothing(tmp_path):
    release, outputs = tmp_path / "release", tmp_path / "outputs"
    bundle(release)
    assert restore(release, outputs) == ["both", "one", "two"]
    assert (outputs / "one" / "weights.pt").read_bytes() == b"weights"
    assert not (outputs / "both" / "weights.pt").exists()
    before = (outputs / "two" / "weights.pt").stat().st_mtime_ns
    restore(release, outputs)
    assert (outputs / "two" / "weights.pt").stat().st_mtime_ns == before


def test_a_different_local_model_is_kept_unless_forced(tmp_path):
    release, outputs = tmp_path / "release", tmp_path / "outputs"
    bundle(release)
    (outputs / "one").mkdir(parents=True)
    (outputs / "one" / "weights.pt").write_bytes(b"local")
    with pytest.raises(FileExistsError):
        restore(release, outputs)
    assert not (outputs / "two").exists()  # nothing is copied when any file would be replaced
    restore(release, outputs, force=True)
    assert (outputs / "one" / "weights.pt").read_bytes() == b"weights"


def test_tampered_bundles_are_rejected(tmp_path):
    release, outputs = tmp_path / "release", tmp_path / "outputs"
    bundle(release)
    (release / "one" / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        restore(release, outputs)
    bundle(tmp_path / "second")
    (tmp_path / "second" / "two" / "model.json").write_text(json.dumps({"weights_sha256": "x"}))
    with pytest.raises(ValueError):
        restore(tmp_path / "second", outputs)
    with pytest.raises(FileNotFoundError):
        (tmp_path / "empty").mkdir()
        restore(tmp_path / "empty", outputs)
