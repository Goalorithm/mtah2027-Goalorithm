"""Refit frozen-selection state/handover hyperparameters on train+validation games, never test."""

import argparse
import importlib.metadata
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from select_joint import copy_component, load_component
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from train_boundaries import profiles
from train_supervised import json_bytes, training_weights, transition_parameters, validate_split

from mtah import A, H, X
from mtah.boundary import boundary_targets
from mtah.compiled_trees import compiled_probabilities
from mtah.config import load_config
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import export_classifier, feature_matrix, feature_names, swap_features


def validated_refit_games(selection, fit_games):
    validate_split(selection)
    expected = selection["train_games"] + selection["validation_games"]
    if (
        not isinstance(fit_games, list)
        or len(set(fit_games)) != len(fit_games)
        or set(fit_games) != set(expected)
        or set(fit_games) & set(selection["test_games"])
    ):
        raise ValueError(
            "refit must use exactly train+validation games and exclude every test game"
        )
    return fit_games


def fit_classifier(values, labels, weights, cfg, names, stage):
    classifier = HistGradientBoostingClassifier(
        **{
            key: cfg[key]
            for key in (
                "max_iter",
                "max_leaf_nodes",
                "learning_rate",
                "l2_regularization",
                "min_samples_leaf",
            )
        },
        early_stopping=False,
        random_state=cfg["seed"],
        verbose=1,
    )
    print(f"Refitting {stage} on {len(labels)} augmented rows", flush=True)
    classifier.fit(values, labels, sample_weight=weights)
    model = export_classifier(classifier, names)
    sample = values[:: max(1, len(values) // 1000)]
    error = float(
        np.max(np.abs(compiled_probabilities(sample, model) - classifier.predict_proba(sample)))
    )
    if error > 1e-12:
        raise ValueError("portable refit inference differs from training library")
    return model, error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_refit.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised_refit")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("refit output must remain under ignored outputs/")
    config = load_config(args.config)
    cfg, features = config["supervised"]["training"], config["supervised"]["features"]
    extra = config["supervised"]["refit"]
    fit_games = validated_refit_games(cfg, extra["fit_games"])
    component = load_component(extra["parent_config"], features, cfg)
    parent = component[0]
    parent_dir = component[2].parent
    state_path, event_path = (
        parent_dir / "frozen_parent/config.toml",
        parent_dir / "frozen_events/config.toml",
    )
    state_config, event_config = load_config(state_path), load_config(event_path)
    state_cfg, event_cfg = (
        state_config["supervised"]["training"],
        event_config["supervised"]["training"],
    )
    for old in (state_config, event_config):
        if old["supervised"]["features"] != features:
            raise ValueError("refit parent feature configurations differ")
    for key in (
        "seed",
        "stride",
        "dropout_fraction",
        "dropout_mean_frames",
        "coordinate_noise_std_m",
    ):
        if state_cfg[key] != event_cfg[key] or state_cfg[key] != cfg[key]:
            raise ValueError("refit augmentation must preserve the chosen parent parameters")
    names = feature_names(features)
    arrays, state_labels, event_labels, state_weights, references = [], [], [], [], []
    sources, counts = {}, {}
    for index, game in enumerate(fit_games):
        print(f"Preparing final-refit game {game}", flush=True)
        tracking, reference, runs, _ = read_game(args.data_dir, game)
        references.append((reference, runs))
        target = boundary_targets(
            reference, runs, event_config["supervised"]["boundary_training"]["target_radius_frames"]
        )
        weights = training_weights(
            reference,
            runs,
            state_cfg["short_spell_weight_exponent"],
            state_cfg["boundary_weight"],
            state_cfg["boundary_frames"],
        )
        sources[game] = {
            key: sha256(path) for key, path in game_assets(args.data_dir, game).items()
        }
        counts[game] = np.bincount(reference[reference != X], minlength=3).tolist()
        selected = np.arange(0, len(reference), cfg["stride"])
        selected = selected[reference[selected] != X]
        for _, profile in profiles(tracking, runs, cfg, cfg["seed"] + index):
            values = feature_matrix(profile, features, runs)[selected]
            arrays += [values, swap_features(values, names)]
            labels, events = reference[selected], target[selected]
            state_labels += [labels, np.where(labels == H, A, np.where(labels == A, H, labels))]
            event_labels += [events, np.where(events == H, A, np.where(events == A, H, events))]
            state_weights += [weights[selected], weights[selected]]
    values, labels, events = (
        np.concatenate(arrays),
        np.concatenate(state_labels),
        np.concatenate(event_labels),
    )
    weights = np.concatenate(state_weights)
    del arrays, state_labels, event_labels, state_weights, tracking, profile
    frequency = np.bincount(events, minlength=3).astype(float)
    if (frequency == 0).any() or (events == X).any():
        raise ValueError("refit event labels must include H/A/none only")
    class_weights = np.sqrt(frequency.max() / frequency)
    event_prior = frequency * class_weights
    event_prior /= event_prior.sum()
    with threadpool_limits(limits=1):
        model, state_error = fit_classifier(
            values, labels, weights, state_cfg, names, "state classifier"
        )
        event_model, event_error = fit_classifier(
            values, events, class_weights[events], event_cfg, names, "handover classifier"
        )
    model.update(transition_parameters(references, state_cfg["transition_pseudocount"]))
    model.update(
        {
            "feature_config": features,
            "decoder_scale": parent["decoder_scale"],
            "boundary_classifier": event_model,
            "boundary_prior": event_prior.tolist(),
            "boundary_weight": parent["boundary_weight"],
            "boundary_semantics": parent["boundary_semantics"],
            "refitted_on_train_and_validation": True,
        }
    )
    frozen = copy_component(out / "frozen_parent", extra["parent_config"], component)
    atomic_write(out / "frozen_state_config.toml", state_path.read_bytes())
    atomic_write(out / "frozen_event_config.toml", event_path.read_bytes())
    provenance = {
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "src/mtah").glob("*.py"))
        },
        "trainer_path": "tools/refit_external.py",
        "trainer_sha256": sha256(__file__),
        "helper_sha256": {
            p: sha256(ROOT / p)
            for p in (
                "tools/select_joint.py",
                "tools/train_boundaries.py",
                "tools/train_supervised.py",
            )
        },
        "external_game_sha256": sources,
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
        "refit_games": fit_games,
        "fitting_stage": (
            "final refit on train+validation after parameter selection; no new validation score"
        ),
        "parent_model_sha256": frozen["model.json"],
        "parent_config_sha256": frozen["config.toml"],
        "parent_report_sha256": frozen["training_report.json"],
        "refit_parameter_files_sha256": {
            name: sha256(out / name)
            for name in ("frozen_state_config.toml", "frozen_event_config.toml")
        },
    }
    model["provenance"] = provenance
    original_best = component[1]["decoder_selection"][0]
    report = {
        "training_games": fit_games,
        "validation_games": [],
        "sealed_test_games": cfg["test_games"],
        "selection_training_games": cfg["train_games"],
        "selection_validation_games": cfg["validation_games"],
        "train_rows_with_augmentation": len(labels),
        "original_fit_frames": sum(sum(v) for v in counts.values()),
        "features": names,
        "training_counts_H_A_D": counts,
        "validation": [],
        "decoder_selection": [
            {
                "decoder_scale": parent["decoder_scale"],
                "boundary_weight": parent["boundary_weight"],
                "mean_spell_f1": original_best["mean_spell_f1"],
                "metric_scope": "frozen parent before refit, not the refitted model",
            }
        ],
        "selected_decoder_scale": parent["decoder_scale"],
        "selected_boundary_weight": parent["boundary_weight"],
        "provenance": provenance,
        "selection_basis": "inherited frozen parent hyperparameters; no post-refit validation",
        "portable_probability_max_errors": {"state": state_error, "handover": event_error},
        "state_parameters_changed_from_parent": model["trees"] != parent["trees"]
        or model["baseline"] != parent["baseline"],
        "handover_parameters_changed_from_parent": event_model["trees"]
        != parent["boundary_classifier"]["trees"]
        or event_model["baseline"] != parent["boundary_classifier"]["baseline"],
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
        f"REFITTED on {sum(sum(v) for v in counts.values())} original frames / "
        f"{len(labels)} augmented rows; test games not fitted",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
