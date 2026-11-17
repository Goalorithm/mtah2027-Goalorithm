"""Fit directed handovers on three external games, retaining a frozen state classifier."""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from train_supervised import json_bytes, metric_summary, validate_split

from mtah import A, H, X
from mtah.boundary import boundary_targets, decode_boundaries
from mtah.compiled_trees import compiled_probabilities, compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import (
    export_classifier,
    feature_matrix,
    feature_names,
    hide_ball,
    perturb_ball,
    swap_features,
)


def profiles(tracking, runs, cfg, seed):
    hidden = hide_ball(tracking, runs, cfg["dropout_fraction"], cfg["dropout_mean_frames"], seed)
    return [
        ("clean", tracking),
        ("occluded", hidden),
        ("occluded_jittered", perturb_ball(hidden, cfg["coordinate_noise_std_m"], seed + 1000)),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_boundary.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised_boundary")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("training output must remain under ignored outputs/")
    config = load_config(args.config)
    cfg, features = config["supervised"]["training"], config["supervised"]["features"]
    extra = config["supervised"]["boundary_training"]
    validate_split(cfg)
    weights = extra["potential_weights"]
    if (
        not weights
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or len(set(weights)) != len(weights)
    ):
        raise ValueError("boundary weights must be finite, nonnegative, distinct")
    parent_config_path = ROOT / extra["parent_config"]
    parent_config = load_config(parent_config_path)
    parent_path = ROOT / parent_config["supervised"]["model_path"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_report_path = parent_path.parent / "training_report.json"
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    if (
        parent_report["model_sha256"] != sha256(parent_path)
        or parent["feature_config"] != features
        or parent["feature_names"] != feature_names(features)
        or parent_report["training_games"] != cfg["train_games"]
        or parent_report["validation_games"] != cfg["validation_games"]
        or parent_report["sealed_test_games"] != cfg["test_games"]
        or parent["provenance"]["test_games_read"] is not False
        or cfg["decoder_scales"] != [parent["decoder_scale"]]
    ):
        raise ValueError("frozen parent has a different split, features, fitted weights or decoder")
    frozen = out / "frozen_parent"
    for source, name in (
        (parent_path, "model.json"),
        (parent_config_path, "config.toml"),
        (parent_report_path, "training_report.json"),
    ):
        atomic_write(frozen / name, source.read_bytes())
    arrays, targets, sources, counts = [], [], {}, {}
    names = feature_names(features)
    for index, game in enumerate(cfg["train_games"]):
        print(f"Preparing handover training game {game}", flush=True)
        tracking, reference, runs, _ = read_game(args.data_dir, game)
        target = boundary_targets(reference, runs, extra["target_radius_frames"])
        sources[game] = {
            key: sha256(path) for key, path in game_assets(args.data_dir, game).items()
        }
        counts[game] = np.bincount(target[target != X], minlength=3).tolist()
        for _, profile in profiles(tracking, runs, cfg, cfg["seed"] + index):
            selected = np.arange(0, len(target), cfg["stride"])
            selected = selected[target[selected] != X]
            values = feature_matrix(profile, features, runs)[selected]
            labels = target[selected]
            arrays += [values, swap_features(values, names)]
            targets += [labels, np.where(labels == H, A, np.where(labels == A, H, labels))]
    train_x, train_y = np.concatenate(arrays), np.concatenate(targets)
    del arrays, targets, values, tracking, profile
    frequency = np.bincount(train_y, minlength=3).astype(float)
    if (frequency == 0).any():
        raise ValueError("handover training must contain both directions and negative examples")
    class_weights = np.sqrt(frequency.max() / frequency)
    sample_weight = class_weights[train_y]
    prior = frequency * class_weights
    prior /= prior.sum()
    classifier = HistGradientBoostingClassifier(
        max_iter=cfg["max_iter"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        learning_rate=cfg["learning_rate"],
        l2_regularization=cfg["l2_regularization"],
        min_samples_leaf=cfg["min_samples_leaf"],
        early_stopping=False,
        random_state=cfg["seed"],
        verbose=1,
    )
    with threadpool_limits(limits=1):
        print(
            f"Training directed handovers on {len(train_y)} rows / {len(names)} features",
            flush=True,
        )
        classifier.fit(train_x, train_y, sample_weight=sample_weight)
        event_model = export_classifier(classifier, names)
        sample = train_x[:: max(1, len(train_x) // 1000)]
        error = float(
            np.max(
                np.abs(
                    compiled_probabilities(sample, event_model) - classifier.predict_proba(sample)
                )
            )
        )
        if error > 1e-12:
            raise ValueError("compiled portable boundary inference differs from fitted library")
        model = copy.deepcopy(parent)
        model.pop("provenance")
        model["boundary_classifier"] = event_model
        model["boundary_prior"] = prior.tolist()
        model["boundary_semantics"] = {
            "H": "handover to home",
            "A": "handover to away",
            "D": "no direct team handover",
        }
        del train_x, sample, sample_weight
        rows = []
        for index, game in enumerate(cfg["validation_games"]):
            print(f"Selecting handover potential on validation game {game}", flush=True)
            tracking, reference, runs, _ = read_game(args.data_dir, game)
            sources[game] = {
                key: sha256(path) for key, path in game_assets(args.data_dir, game).items()
            }
            for profile_name, profile in profiles(tracking, runs, cfg, cfg["seed"] + 100 + index):
                values = feature_matrix(profile, features, runs)
                probabilities = compiled_symmetric_probabilities(values, parent)
                events = (
                    classifier.predict_proba(values)
                    + classifier.predict_proba(swap_features(values, names))[:, [A, H, 2]]
                ) / 2
                for weight in weights:
                    model["boundary_weight"] = weight
                    prediction = decode_boundaries(probabilities, events, model, runs)
                    metrics = metric_summary(reference, prediction)
                    rows.append(
                        {
                            "game": game,
                            "profile": profile_name,
                            "boundary_weight": weight,
                            "decoder_scale": model["decoder_scale"],
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"validation {game}/{profile_name} weight={weight}: "
                        f"spellF1 {metrics['spell_f1_mean']:.4f}",
                        flush=True,
                    )
    selection = [
        {
            "boundary_weight": weight,
            "decoder_scale": model["decoder_scale"],
            "mean_spell_f1": float(
                np.mean(
                    [r["metrics"]["spell_f1_mean"] for r in rows if r["boundary_weight"] == weight]
                )
            ),
        }
        for weight in weights
    ]
    selection.sort(key=lambda r: (-r["mean_spell_f1"], r["boundary_weight"]))
    model["boundary_weight"] = selection[0]["boundary_weight"]
    provenance = {
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "src/mtah").glob("*.py"))
        },
        "trainer_path": "tools/train_boundaries.py",
        "trainer_sha256": sha256(__file__),
        "external_game_sha256": sources,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
        "fitting_stage": "frozen parent state classifier plus directed handover classifier",
        "parent_model_sha256": sha256(frozen / "model.json"),
        "parent_config_sha256": sha256(frozen / "config.toml"),
        "parent_report_sha256": sha256(frozen / "training_report.json"),
        "label_source": "DFL/Sportec provider BallStatus/Possession on training games only",
    }
    model["provenance"] = provenance
    report = {
        "training_games": cfg["train_games"],
        "validation_games": cfg["validation_games"],
        "sealed_test_games": cfg["test_games"],
        "train_rows_with_augmentation": len(train_y),
        "features": names,
        "boundary_counts_H_A_none": counts,
        "boundary_training_class_weights": class_weights.tolist(),
        "validation": rows,
        "decoder_selection": selection,
        "selected_decoder_scale": model["decoder_scale"],
        "selected_boundary_weight": model["boundary_weight"],
        "portable_probability_max_error": error,
        "provenance": provenance,
        "parent_original_provenance": parent["provenance"],
        "selection_basis": (
            "mean external validation spellF1 on original/missing/noisy input; "
            "zero-weight ablation included"
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
    print(
        f"TRAINED: selected boundary weight {model['boundary_weight']}; no test-game fitting",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
