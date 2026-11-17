"""The publication guard inspects tracked files, including normally ignored ones."""

from check_repository_content import excluded_paths, violations


def test_manifest_rejects_removed_code_and_entire_model_directory(tmp_path):
    manifest = tmp_path / "excluded.txt"
    manifest.write_text("# retained locally\ntools/removed.py\nrelease/atd/\n\n", encoding="utf-8")
    tracked = [
        "tools/removed.py",
        "release/atd/one/weights.pt",
        "release/atd/one/model.json",
        "tools/allowed.py",
        "release/atd_notes.md",
        "",
    ]
    assert violations(tracked, excluded_paths(manifest)) == sorted(tracked[:3])


def test_force_added_outputs_and_match_data_are_rejected():
    private = ["outputs/model.json", "cache/tracking.bin", "data/tracking.xml", "clip.MP4"]
    assert violations([*private, "data/README.md", "config/default.toml"], ()) == sorted(private)
