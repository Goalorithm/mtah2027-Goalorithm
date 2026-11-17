"""Artifact checks must reject stale, truncated, reordered or misattributed results."""

import json

import check_release as release
import pytest

from mtah.io import sha256


@pytest.mark.parametrize(
    "body",
    [
        "frame,label\n0,H\n2,D\n1,A\n",  # A tolerant reader might silently sort this.
        "frame,label\n0,H\n1,A\n1,D\n",
        "frame,label\n0,H\n1,A\n",
        "frame,label\n0,H\n1,A\n2,unknown\n",
        "frame,label,extra\n0,H,x\n1,A,x\n2,D,x\n",
    ],
)
def test_malformed_submission_fails_before_scoring(tmp_path, body):
    path = tmp_path / "prediction.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        release.strict_submission(path, 3)


def test_hash_check_detects_missing_and_modified_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"seed":29}', encoding="utf-8")
    hashes = {path.name: sha256(path)}
    release.verify_hashes(tmp_path, hashes)
    path.write_text('{"seed":7}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.verify_hashes(tmp_path, hashes)
    path.unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        release.verify_hashes(tmp_path, hashes)


def test_performance_target_uses_selected_model_and_does_not_confuse_mean_with_minimum():
    games = []
    for selected in [0.7, 0.9]:
        rows = []
        for profile in ["clean", "occluded", "occluded_jittered"]:
            rows += [
                {
                    "config": "config/selected.toml",
                    "profile": profile,
                    "metrics": {"spell_f1_mean": selected},
                },
                {
                    "config": "config/other.toml",
                    "profile": profile,
                    "metrics": {"spell_f1_mean": 0.99},
                },
            ]
        games.append({"models": rows})
    result = release.external_performance({"games": games}, "config/selected.toml")
    assert result["mean_spell_f1_by_profile"]["clean"] == pytest.approx(0.8)
    assert result["mean_target_met_by_profile"]["clean"] is True
    assert result["every_game_target_met_by_profile"]["clean"] is False


def test_manifest_cannot_redirect_verification_outside_bundle(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    for name in ("../outside.json", str(outside.resolve())):
        with pytest.raises(ValueError, match="escapes"):
            release.verify_hashes(tmp_path / "bundle", {name: sha256(outside)})


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "N_FRAMES", 3)
    for name in (
        "tools/ai_labels.py",
        "labels/windows.csv",
        "config/default.toml",
        "config/trained.toml",
        "src/mtah/example.py",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture\n", encoding="utf-8")
    directory = tmp_path / "outputs/bundle"
    directory.mkdir(parents=True)
    for name in ("fullmatch_ai.csv", "pseudolabel_dev.csv", "pseudolabel_holdout.csv"):
        (directory / name).write_text("frame,label\n0,H\n1,A\n2,D\n", encoding="utf-8")
    for name in ("perubahan_ai.csv", "intervals_ai.csv"):
        (directory / name).write_text("synthetic\n", encoding="utf-8")
    manifest = {
        "annotation_origin": "algorithmic_pseudo_labels",
        "human_reviewed": False,
        "is_ground_truth": False,
        "accuracy_measured": False,
        "independent_of_hmm_candidate": False,
        "selected_final_model": None,
        "output_sha256": {p.name: sha256(p) for p in directory.iterdir()},
        "provenance": {
            "config_sha256": sha256(tmp_path / "config/trained.toml"),
            "trained_checkpoint_sha256": "synthetic-checkpoint",
            "data_sha256": {"tracking": "synthetic-data"},
            "reference_labels_read": False,
            "video_uploaded": False,
            "public_gt_used_for_training_or_selection": False,
            "source_sha256": {"src/mtah/example.py": sha256(tmp_path / "src/mtah/example.py")},
            **{
                key: sha256(tmp_path / name)
                for key, name in (
                    ("exporter_sha256", "tools/ai_labels.py"),
                    ("windows_sha256", "labels/windows.csv"),
                    ("baseline_config_sha256", "config/default.toml"),
                )
            },
        },
    }
    return directory, tmp_path / "config/trained.toml", manifest


def verify_fixture(bundle):
    directory, config, manifest = bundle
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return release.verify_bundle(
        directory, config, "synthetic-checkpoint", {"data_sha256": {"tracking": "synthetic-data"}}
    )


def test_valid_bundle_and_changed_source_are_distinguished(bundle):
    assert verify_fixture(bundle)["is_ground_truth"] is False
    (release.ROOT / "src/mtah/example.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_fixture(bundle)


@pytest.mark.parametrize("field", ["is_ground_truth", "accuracy_measured", "human_reviewed"])
def test_pseudo_labels_cannot_be_presented_as_reviewed_ground_truth(bundle, field):
    bundle[2][field] = True
    with pytest.raises(ValueError, match="unexpected pseudo-label"):
        verify_fixture(bundle)


def test_pseudo_labels_from_different_checkpoint_are_rejected(bundle):
    bundle[2]["provenance"]["trained_checkpoint_sha256"] = "other-model"
    with pytest.raises(ValueError, match="model changed"):
        verify_fixture(bundle)


def test_force_added_match_data_is_detected_even_if_normally_ignored(monkeypatch):
    def git(*args):
        return "src/mtah/hmm.py\0outputs/model.json\0" if args == ("ls-files", "-z") else ""

    monkeypatch.setattr(release, "git_output", git)
    with pytest.raises(ValueError, match="private/generated asset is tracked"):
        release.verify_repository()
