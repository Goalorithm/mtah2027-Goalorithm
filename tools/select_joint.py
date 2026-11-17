"""Select frozen state/handover components on external validation games, without new fitting."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from benchmark_idsse import verify_freeze
from train_boundaries import profiles
from train_supervised import json_bytes, metric_summary, validate_split

from mtah.boundary import decode_boundaries
from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import feature_matrix


def load_component(path, features, training):
    config = load_config(ROOT / path)
    checkpoint = ROOT / config["supervised"]["model_path"]
    verify_freeze(ROOT / path, checkpoint)
    model = json.loads(checkpoint.read_text(encoding="utf-8"))
    report_path = checkpoint.parent / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        model["feature_config"] != features
        or config["supervised"]["features"] != features
        or report["training_games"] != training["train_games"]
        or report["validation_games"] != training["validation_games"]
        or report["sealed_test_games"] != training["test_games"]
        or any(
            model["provenance"][key] is not False
            for key in ("test_games_read", "bar_ath_labels_read", "bar_ath_data_read")
        )
    ):
        raise ValueError(
            "component has different features, game splits or test-fitting declaration"
        )
    return model, report, checkpoint, report_path


def copy_component(directory, config_path, component):
    model, report, checkpoint, report_path = component
    for path, name in (
        (checkpoint, "model.json"),
        (ROOT / config_path, "config.toml"),
        (report_path, "training_report.json"),
    ):
        atomic_write(directory / name, path.read_bytes())
    return {
        name: sha256(directory / name)
        for name in ("model.json", "config.toml", "training_report.json")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_joint.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised_joint")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("joint selection output must remain under ignored outputs/")
    config = load_config(args.config)
    cfg = config["supervised"]["training"]
    features = config["supervised"]["features"]
    extra = config["supervised"]["joint_selection"]
    validate_split(cfg)
    weights = extra["potential_weights"]
    if (
        not weights
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or len(set(weights)) != len(weights)
    ):
        raise ValueError("joint potential weights must be finite, nonnegative and unique")
    bodies = {p: load_component(p, features, cfg) for p in extra["body_configs"]}
    event_component = load_component(extra["events_config"], features, cfg)
    event_model = event_component[0]
    rows = []
    for index, game in enumerate(cfg["validation_games"]):
        print(f"Selecting frozen components on validation game {game}", flush=True)
        tracking, reference, runs, _ = read_game(args.data_dir, game)
        for profile_name, profile in profiles(tracking, runs, cfg, cfg["seed"] + 100 + index):
            values = feature_matrix(profile, features, runs)
            events = compiled_symmetric_probabilities(values, event_model["boundary_classifier"])
            for body_path, component in bodies.items():
                candidate = copy.deepcopy(component[0])
                candidate["boundary_prior"] = event_model["boundary_prior"]
                probabilities = compiled_symmetric_probabilities(values, candidate)
                for weight in weights:
                    candidate["boundary_weight"] = weight
                    metrics = metric_summary(
                        reference, decode_boundaries(probabilities, events, candidate, runs)
                    )
                    rows.append(
                        {
                            "game": game,
                            "profile": profile_name,
                            "body_config": body_path,
                            "decoder_scale": candidate["decoder_scale"],
                            "boundary_weight": weight,
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"validation {game}/{profile_name}/{Path(body_path).stem} weight={weight}: "
                        f"spellF1 {metrics['spell_f1_mean']:.4f}",
                        flush=True,
                    )
    selection = [
        {
            "body_config": path,
            "boundary_weight": weight,
            "decoder_scale": bodies[path][0]["decoder_scale"],
            "mean_spell_f1": float(
                np.mean(
                    [
                        r["metrics"]["spell_f1_mean"]
                        for r in rows
                        if r["body_config"] == path and r["boundary_weight"] == weight
                    ]
                )
            ),
        }
        for path in bodies
        for weight in weights
    ]
    selection.sort(key=lambda r: (-r["mean_spell_f1"], r["body_config"], r["boundary_weight"]))
    best = selection[0]
    component = bodies[best["body_config"]]
    model = copy.deepcopy(component[0])
    model["boundary_classifier"] = event_model["boundary_classifier"]
    model["boundary_prior"] = event_model["boundary_prior"]
    model["boundary_semantics"] = event_model["boundary_semantics"]
    model["boundary_weight"] = best["boundary_weight"]
    parent_hashes = copy_component(out / "frozen_parent", best["body_config"], component)
    event_hashes = copy_component(out / "frozen_events", extra["events_config"], event_component)
    sources = {
        game: {key: sha256(path) for key, path in game_assets(args.data_dir, game).items()}
        for game in cfg["train_games"] + cfg["validation_games"]
    }
    provenance = {
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "src/mtah").glob("*.py"))
        },
        "trainer_path": "tools/select_joint.py",
        "trainer_sha256": sha256(__file__),
        "helper_sha256": {
            p: sha256(ROOT / p) for p in ("tools/train_supervised.py", "tools/train_boundaries.py")
        },
        "external_game_sha256": sources,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
        "fitting_stage": "validation selection of two frozen components; no additional fitting",
        "parent_model_sha256": parent_hashes["model.json"],
        "parent_config_sha256": parent_hashes["config.toml"],
        "parent_report_sha256": parent_hashes["training_report.json"],
        "event_component_sha256": event_hashes,
        "label_source": "DFL/Sportec provider labels; validation-only component selection",
    }
    model["provenance"] = provenance
    report = {
        "training_games": cfg["train_games"],
        "validation_games": cfg["validation_games"],
        "sealed_test_games": cfg["test_games"],
        "train_rows_with_augmentation": component[1]["train_rows_with_augmentation"],
        "additional_training_rows": 0,
        "features": model["feature_names"],
        "validation": rows,
        "decoder_selection": selection,
        "selected_decoder_scale": model["decoder_scale"],
        "selected_boundary_weight": model["boundary_weight"],
        "selected_body_config": best["body_config"],
        "provenance": provenance,
        "parent_original_provenance": component[0]["provenance"],
        "event_original_provenance": event_model["provenance"],
        "selection_basis": (
            "mean spellF1 on two validation games, three input profiles; no new fitting"
        ),
        "competition_accuracy_measured": False,
        "selected_final_competition_model": None,
        "test_set_previous_exposure": cfg["test_set_previous_exposure"],
    }
    atomic_write(out / "model.json", json_bytes(model))
    report["model_sha256"] = sha256(out / "model.json")
    atomic_write(out / "training_report.json", json_bytes(report))
    atomic_write(
        out / "test_freeze.json",
        json_bytes(
            {
                "model_sha256": report["model_sha256"],
                "config_sha256": sha256(args.config),
                "training_report_sha256": sha256(out / "training_report.json"),
                "test_games": cfg["test_games"],
                "frozen_before_test_evaluation": True,
            }
        ),
    )
    print("SELECTED:", best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
