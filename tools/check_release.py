"""Check repository quality, with optional verification of external-only model artifacts."""

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np

from mtah.config import load_config
from mtah.io import sha256, write_submission
from mtah.load import load_smart_tagging, load_tracking
from mtah.paths import N_FRAMES, OUTPUTS_DIR, ROOT, games
from mtah.predict import run_pipeline
from mtah.review import atomic_write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_path(directory, name):
    """Manifest paths must stay within their declared directory, including through symlinks."""
    directory = Path(directory).resolve()
    path = (directory / name).resolve()
    require(path.is_relative_to(directory), f"manifest path escapes its directory: {name}")
    return path


def verify_hashes(directory, mapping):
    require(isinstance(mapping, dict) and bool(mapping), "empty or invalid hash manifest")
    for name, expected in sorted(mapping.items()):
        path = local_path(directory, name)
        require(path.is_file(), f"missing artifact: {name}")
        require(sha256(path) == expected, f"SHA-256 mismatch: {name}")


def strict_submission(path, n_frames=None):
    """Reject reordered, missing, repeated or unknown rows before any tolerant scorer reads them."""
    n_frames = N_FRAMES if n_frames is None else n_frames
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        require(next(reader, None) == ["frame", "label"], "expected frame,label header")
        count = 0
        for index, row in enumerate(reader):
            require(
                len(row) == 2 and row[0] == str(index) and row[1] in ("H", "A", "D", "X"),
                f"invalid or nonsequential prediction row {index}",
            )
            count += 1
        require(count == n_frames, f"expected {n_frames} frames, found {count}")


def verify_bundle(directory, config_path, checkpoint_digest, training_provenance):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest["annotation_origin"] == "algorithmic_pseudo_labels", "unknown label origin")
    for key in (
        "human_reviewed",
        "is_ground_truth",
        "accuracy_measured",
        "independent_of_hmm_candidate",
    ):
        require(manifest[key] is False, f"unexpected pseudo-label declaration: {key}")
    require(manifest["selected_final_model"] is None, "pseudo-labels must not choose a winner")
    expected = {
        "fullmatch_ai.csv",
        "perubahan_ai.csv",
        "intervals_ai.csv",
        "pseudolabel_dev.csv",
        "pseudolabel_holdout.csv",
    }
    require(set(manifest["output_sha256"]) == expected, "incomplete pseudo-label bundle")
    verify_hashes(directory, manifest["output_sha256"])
    provenance = manifest["provenance"]
    require(provenance["config_sha256"] == sha256(config_path), "pseudo-label config changed")
    require(
        provenance["trained_checkpoint_sha256"] == checkpoint_digest, "pseudo-label model changed"
    )
    require(provenance["data_sha256"] == training_provenance["data_sha256"], "label data differs")
    for key in (
        "reference_labels_read",
        "video_uploaded",
        "public_gt_used_for_training_or_selection",
    ):
        require(provenance[key] is False, f"unexpected exporter declaration: {key}")
    verify_hashes(ROOT, provenance["source_sha256"])
    verify_hashes(
        ROOT,
        {
            "tools/ai_labels.py": provenance["exporter_sha256"],
            "labels/windows.csv": provenance["windows_sha256"],
            "config/default.toml": provenance["baseline_config_sha256"],
        },
    )
    for name in ("fullmatch_ai.csv", "pseudolabel_dev.csv", "pseudolabel_holdout.csv"):
        strict_submission(directory / name)
    return manifest


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return result.stdout.decode("utf-8")


def verify_external():
    """Check frozen benchmark inputs/results without fitting or reranking candidates."""
    config_path = ROOT / "config/external_benchmark.toml"
    config = load_config(config_path)
    directory = OUTPUTS_DIR / "external_benchmark"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["benchmark_config_sha256"] == sha256(config_path), "benchmark config changed")
    require(
        report["source_commit"] == config["source"]["commit"], "benchmark source commit differs"
    )
    require(report["selected_final_model"] is None, "partial benchmark must not choose final model")
    for key in (
        "models_fitted_or_tuned",
        "bar_ath_reference_labels_read",
        "fullmatch_accuracy_measured",
    ):
        require(report[key] is False, f"unexpected benchmark declaration: {key}")
    verify_hashes(ROOT, report["source_code_sha256"])
    verify_hashes(ROOT, {"tools/benchmark_external.py": report["benchmark_script_sha256"]})
    source_hashes = {row["file"]: row["sha256"] for row in report["assets"]}
    require(source_hashes == config["source"]["sha256"], "benchmark asset manifest differs")
    verify_hashes(ROOT / "data/external/metrica_sample", source_hashes)
    require(
        [g["metadata"]["game"] for g in report["games"]] == config["source"]["games"],
        "benchmark game coverage changed",
    )
    anchors = 0
    for game in report["games"]:
        number = game["metadata"]["game"]
        anchors += game["metadata"]["anchors"]["evaluated_frames"]
        require(
            [row["config"] for row in game["models"]] == config["benchmark"]["model_configs"],
            "benchmark model coverage changed",
        )
        for row in game["models"]:
            model_path = local_path(ROOT, row["config"])
            require(sha256(model_path) == row["config_sha256"], "benchmark model config changed")
            model = load_config(model_path)
            method = model["dead_ball"]["method"]
            if method == "supervised":
                require(
                    sha256(local_path(ROOT, model[method]["model_path"]))
                    == row["checkpoint_sha256"],
                    "benchmark checkpoint changed",
                )
            path = directory / f"sample{number}_{model_path.stem}.csv"
            require(sha256(path) == row["prediction_sha256"], "benchmark prediction changed")
            strict_submission(path, game["metadata"]["frames"])
    return {
        "report_sha256": sha256(report_path),
        "source_commit": report["source_commit"],
        "pass_start_anchors": anchors,
        "fullmatch_accuracy_measured": False,
    }


def verify_repository():
    protected = ["evaluation", "starter", "games.py", "config/default.toml"]
    require(
        not git_output("diff", "HEAD", "--", *protected),
        "protected organiser/baseline file changed",
    )
    require(
        not git_output("ls-files", "--others", "--exclude-standard", "--", *protected),
        "unexpected untracked file under protected paths",
    )
    tracked = git_output("ls-files", "-z").split("\0")
    for name in filter(None, tracked):
        lower = name.lower()
        forbidden = (
            name.startswith(("outputs/", "cache/"))
            or (name.startswith("data/") and name != "data/README.md")
            or lower.endswith((".mp4", ".npz", ".parquet", ".gzip", ".pl3container"))
        )
        require(not forbidden, f"private/generated asset is tracked: {name}")
    return {"tracked_files": len(list(filter(None, tracked))), "protected_files_unchanged": True}


def verify_team_experiments():
    plan_path = ROOT / "config/team_experiments.toml"
    plan = load_config(plan_path)
    directory = OUTPUTS_DIR / "team_experiments"
    path = directory / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report["plan_sha256"] == sha256(plan_path), "experiment plan changed")
    verify_hashes(ROOT, {"tools/run_experiments.py": report["script_sha256"]})
    verify_hashes(ROOT, report["source_sha256"])
    assets = games.asset_paths(source="tactical")
    for key, digest in report["data_sha256"].items():
        require(sha256(assets[key]) == digest, f"experiment data changed: {key}")
    for key in ("reference_labels_read", "accuracy_measured", "public_labels_used_for_tuning"):
        require(report[key] is False, f"unexpected experiment declaration: {key}")
    require(report["selected_final_model"] is None, "unlabelled diagnostics must not rank models")
    expected = [(Path(name).stem, name, []) for name in plan["configs"]]
    expected += [(row["name"], row["config"], row["overrides"]) for row in plan.get("sweep", [])]
    actual = [(row["name"], row["config"], row["overrides"]) for row in report["experiments"]]
    require(actual == expected, "experiment/sweep coverage differs from frozen plan")
    for row in report["experiments"]:
        config_path = local_path(ROOT, row["config"])
        require(sha256(config_path) == row["config_sha256"], "experiment config changed")
        config = load_config(config_path, row["overrides"])
        method = config["dead_ball"]["method"]
        if method == "supervised":
            require(
                sha256(local_path(ROOT, config[method]["model_path"])) == row["checkpoint_sha256"],
                "experiment checkpoint changed",
            )
        prediction = local_path(directory, f"{row['name']}.csv")
        require(row["repeat_identical"] is True, "experiment repeat was not identical")
        require(sha256(prediction) == row["prediction_sha256"], "experiment prediction changed")
        strict_submission(prediction)
    return {
        "report_sha256": sha256(path),
        "deterministic_candidates_and_sweeps": len(expected),
        "reference_labels_read": False,
        "selected_final_model": None,
    }


def external_performance(benchmark, candidate):
    """Keep the user's 0.8 requirement separate from code/artifact verification."""
    candidate = local_path(ROOT, candidate)
    means, minimums = {}, {}
    for profile in ("clean", "occluded", "occluded_jittered"):
        values = [
            row["metrics"]["spell_f1_mean"]
            for game in benchmark["games"]
            for row in game["models"]
            if row["profile"] == profile and local_path(ROOT, row["config"]) == candidate
        ]
        require(
            len(values) == len(benchmark["games"]) and bool(values),
            "active candidate missing from benchmark",
        )
        require(all(np.isfinite(v) and 0 <= v <= 1 for v in values), "invalid benchmark spellF1")
        means[profile], minimums[profile] = float(np.mean(values)), min(values)
    return {
        "target_spell_f1": 0.8,
        "mean_spell_f1_by_profile": means,
        "minimum_game_spell_f1_by_profile": minimums,
        "mean_target_met_by_profile": {p: score >= 0.8 for p, score in means.items()},
        "every_game_target_met_by_profile": {p: score >= 0.8 for p, score in minimums.items()},
    }


def verify_supervised_candidate(
    out,
    config_name="config/exp_supervised_joint.toml",
    plan_name="config/idsse_benchmark_joint.toml",
    benchmark_directory="outputs/idsse_benchmark_joint",
):
    from benchmark_idsse import verify_freeze
    from train_supervised import validate_split

    from mtah.supervised import feature_names, validate_classifier

    config_path = local_path(ROOT, config_name)
    config = load_config(config_path)
    cfg = config["supervised"]
    validate_split(cfg["training"])
    model_path = local_path(ROOT, cfg["model_path"])
    model = json.loads(model_path.read_text(encoding="utf-8"))
    validate_classifier(model)
    require(
        model["feature_config"] == cfg["features"]
        and model["feature_names"] == feature_names(cfg["features"]),
        "supervised feature configuration changed",
    )
    receipt = verify_freeze(config_path, model_path)
    provenance = model["provenance"]
    require(provenance["config_sha256"] == sha256(config_path), "supervised config hash changed")
    require(
        {"src/mtah/idsse.py", "src/mtah/supervised.py"} <= provenance["source_sha256"].keys(),
        "supervised source provenance is incomplete",
    )
    verify_hashes(ROOT, provenance["source_sha256"])
    verify_hashes(
        ROOT,
        {
            provenance.get("trainer_path", "tools/train_supervised.py"): provenance[
                "trainer_sha256"
            ],
            "config/idsse_sources.json": provenance["source_manifest_sha256"],
        },
    )
    if "helper_sha256" in provenance:
        verify_hashes(ROOT, provenance["helper_sha256"])
    for key in ("bar_ath_labels_read", "bar_ath_data_read", "test_games_read"):
        require(provenance[key] is False, f"unexpected supervised training declaration: {key}")
    for name, version in provenance["packages"].items():
        require(importlib.metadata.version(name) == version, f"supervised package changed: {name}")
    train_report_path = model_path.parent / "training_report.json"
    training = json.loads(train_report_path.read_text(encoding="utf-8"))
    require(training["provenance"] == provenance, "supervised model/training provenance differs")
    expected_games = set(cfg["training"]["train_games"] + cfg["training"]["validation_games"])
    require(
        set(provenance["external_game_sha256"]) == expected_games,
        "supervised fitting/selection read an unexpected game",
    )
    source = json.loads((ROOT / "config/idsse_sources.json").read_text(encoding="utf-8"))
    verify_hashes(
        ROOT / "data/external/idsse", {row["name"]: row["sha256"] for row in source["files"]}
    )
    selection = training["decoder_selection"]
    best = min(selection, key=lambda row: (-row["mean_spell_f1"], row["decoder_scale"]))
    require(
        model["decoder_scale"] == training["selected_decoder_scale"] == best["decoder_scale"],
        "supervised decoder differs from validation selection",
    )
    if "boundary_classifier" in model:
        validate_classifier(model["boundary_classifier"])
        require(
            model["boundary_weight"]
            == training["selected_boundary_weight"]
            == best["boundary_weight"],
            "boundary decoder differs from validation selection",
        )
        parent = model_path.parent / "frozen_parent"
        verify_hashes(
            parent,
            {
                "model.json": provenance["parent_model_sha256"],
                "config.toml": provenance["parent_config_sha256"],
                "training_report.json": provenance["parent_report_sha256"],
            },
        )
        frozen_model = json.loads((parent / "model.json").read_text(encoding="utf-8"))
        if "handover_log_odds_bias" in model:
            from calibrate_handover import calibrated_transition

            require(
                model["handover_log_odds_bias"]
                == training["selected_handover_log_odds_bias"]
                == best["handover_log_odds_bias"],
                "handover transition bias differs from validation selection",
            )
            frozen_model["transition"] = calibrated_transition(
                frozen_model["transition"], model["handover_log_odds_bias"]
            ).tolist()
        if "ensemble_weight" in model:
            from select_ensemble import fuse_classifiers

            verify_hashes(
                model_path.parent / "frozen_secondary", provenance["secondary_component_sha256"]
            )
            secondary = json.loads(
                (model_path.parent / "frozen_secondary/model.json").read_text(encoding="utf-8")
            )
            require(
                model["ensemble_weight"]
                == training["selected_ensemble_weight"]
                == best["ensemble_weight"],
                "ensemble mixture differs from validation selection",
            )
            frozen_model = fuse_classifiers(frozen_model, secondary, model["ensemble_weight"])
        if model.get("refitted_on_train_and_validation"):
            from refit_external import validated_refit_games

            fit_games = validated_refit_games(cfg["training"], cfg["refit"]["fit_games"])
            require(
                training["training_games"] == fit_games and training["validation_games"] == [],
                "refit fitting/validation roles are incorrect",
            )
            require(provenance["refit_games"] == fit_games, "refit source-game declaration changed")
            require(
                model["decoder_scale"] == frozen_model["decoder_scale"]
                and model["boundary_weight"] == frozen_model["boundary_weight"],
                "refit changed previously selected hyperparameters",
            )
            require(
                training["validation"] == [],
                "refit cannot claim an independent validation score on fitting games",
            )
            verify_hashes(model_path.parent, provenance["refit_parameter_files_sha256"])
        else:
            require(
                all(
                    model.get(key) == value
                    for key, value in frozen_model.items()
                    if key != "provenance"
                ),
                "frozen state-classifier parameters changed in joint model",
            )
        if "event_component_sha256" in provenance:
            verify_hashes(model_path.parent / "frozen_events", provenance["event_component_sha256"])
            frozen_events = json.loads(
                (model_path.parent / "frozen_events/model.json").read_text(encoding="utf-8")
            )
            require(
                model["boundary_classifier"] == frozen_events["boundary_classifier"]
                and model["boundary_prior"] == frozen_events["boundary_prior"],
                "frozen event-classifier parameters changed in joint model",
            )
    benchmark_path = local_path(ROOT, benchmark_directory) / "report.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    require(
        benchmark["split"] == "test" and benchmark["pre_test_freeze"] == receipt,
        "full-state benchmark was not bound to the pre-test model receipt",
    )
    verify_hashes(ROOT, benchmark["source_code_sha256"])
    verify_hashes(
        ROOT,
        {
            "tools/benchmark_idsse.py": benchmark["benchmark_script_sha256"],
            plan_name: benchmark["benchmark_plan_sha256"],
        },
    )
    require(
        benchmark["bar_ath_labels_read"] is False and benchmark["models_fitted_or_tuned"] is False,
        "test evaluation must not fit models or read BAR-ATH reference labels",
    )
    require(
        benchmark["competition_accuracy_measured"] is False
        and benchmark["selected_final_competition_model"] is None,
        "external evaluation cannot establish the final competition winner",
    )
    require(
        [row["game"] for row in benchmark["games"]] == cfg["training"]["test_games"],
        "full-state test-game coverage changed",
    )
    plan = load_config(local_path(ROOT, plan_name))
    expected_rows = [
        (profile, name)
        for profile in ("clean", "occluded", "occluded_jittered")
        for name in plan["model_configs"]
    ]
    for game in benchmark["games"]:
        require(
            [(row["profile"], row["config"]) for row in game["models"]] == expected_rows,
            "full-state test model/profile coverage changed",
        )
        for row in game["models"]:
            path = local_path(ROOT, row["config"])
            require(sha256(path) == row["config_sha256"], "full-state benchmark config changed")
            candidate = load_config(path)
            method = candidate["dead_ball"]["method"]
            if method == "supervised":
                require(
                    sha256(local_path(ROOT, candidate[method]["model_path"]))
                    == row["checkpoint_sha256"],
                    "full-state benchmark checkpoint changed",
                )
            prediction = benchmark_path.parent / f"{game['game']}_{row['profile']}_{path.stem}.csv"
            require(
                sha256(prediction) == row["prediction_sha256"],
                "full-state benchmark output changed",
            )
            strict_submission(prediction, game["metadata"]["frames_with_separator"])
    prediction = OUTPUTS_DIR / f"{config_path.stem}.csv"
    strict_submission(prediction)
    with tempfile.TemporaryDirectory(prefix="supervised-repeat-", dir=out) as temporary:
        repeated = Path(temporary) / "prediction.csv"
        codes = run_pipeline(config, load_tracking(), load_smart_tagging())
        write_submission(repeated, codes)
        require(
            sha256(repeated) == sha256(prediction), "supervised prediction regeneration differs"
        )
    return {
        "model_sha256": sha256(model_path),
        "prediction_sha256": sha256(prediction),
        "prediction_repeat_identical": True,
        "external_full_state_test_games": len(benchmark["games"]),
        "benchmark_sha256": sha256(benchmark_path),
        "competition_accuracy_measured": False,
        "performance": external_performance(benchmark, config_name),
    }


def run_command(arguments, log_path):
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    atomic_write(log_path, (result.stdout + result.stderr).encode())
    require(result.returncode == 0, f"command failed; see {log_path.relative_to(ROOT)}")
    return {"log": str(log_path.relative_to(ROOT)).replace("\\", "/")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/release_check")
    parser.add_argument(
        "--with-artifacts",
        action="store_true",
        help="also verify regenerated external-model benchmarks and experiment artifacts",
    )
    parser.add_argument("--supervised-config", default="config/exp_supervised_joint.toml")
    parser.add_argument("--supervised-plan", default="config/idsse_benchmark_joint.toml")
    parser.add_argument("--supervised-benchmark-dir", default="outputs/idsse_benchmark_joint")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: verification reports must stay under ignored outputs/")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "passed": True, "detail": detail})
            print(f"PASS {name}", flush=True)
            return detail
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            importlib.metadata.PackageNotFoundError,
            subprocess.CalledProcessError,
        ) as error:
            checks.append({"name": name, "passed": False, "error": str(error)})
            print(f"FAIL {name}: {error}", flush=True)
            return None

    check("repository", verify_repository)
    for name, command in (
        ("excluded_paths", ["tools/check_repository_content.py"]),
        ("lint", ["-m", "ruff", "check", "src", "tools", "tests", "scripts"]),
        ("format", ["-m", "ruff", "format", "--check", "src", "tools", "tests", "scripts"]),
        ("tests", ["-m", "pytest", "-q"]),
        ("dependencies", ["-m", "pip", "check"]),
    ):
        check(
            name,
            lambda command=command, name=name: run_command(
                [sys.executable, *command], out / f"{name}.log"
            ),
        )
    supervised = None
    if args.with_artifacts:
        check("frozen_external_benchmark", verify_external)
        check("all_tech_experiments", verify_team_experiments)
        supervised = check(
            "supervised_candidate_and_full_state_test",
            lambda: verify_supervised_candidate(
                out, args.supervised_config, args.supervised_plan, args.supervised_benchmark_dir
            ),
        )
    report = {
        "version": 2,
        "verifier_sha256": sha256(__file__),
        "with_artifacts": args.with_artifacts,
        "performance": supervised["performance"] if supervised else None,
        "checks": checks,
        "local_checks_passed": all(row["passed"] for row in checks),
        "fullmatch_accuracy_measured": False,
        "selected_final_model": None,
        "published_or_submitted": False,
    }
    atomic_write(
        out / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    status = "passed" if report["local_checks_passed"] else "FAILED"
    print(f"Local checks {status}; report: {out / 'report.json'}")
    print("Code checks do not validate historical checkpoints or establish a competition score.")
    return 0 if report["local_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
