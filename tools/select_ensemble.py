"""Validate a finite-JSON geometric mixture of two frozen state classifiers and handovers."""

import argparse
import copy
import importlib.metadata
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from select_joint import copy_component, load_component
from train_boundaries import profiles
from train_supervised import json_bytes, metric_summary, validate_split

from mtah.boundary import decode_boundaries
from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.hmm import _probabilities
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import feature_matrix, validate_classifier


def blend_probabilities(first, second, prior_first, prior_second, weight):
    if isinstance(weight, bool) or not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("ensemble weight must be between zero and one")
    for probabilities in (first, second):
        _probabilities(probabilities, (len(first), 3), "component probabilities")
    for prior in (prior_first, prior_second):
        _probabilities(prior, (3,), "component prior")
        if (np.asarray(prior) <= 0).any():
            raise ValueError("ensemble priors must be strictly positive")
    if weight == 0:
        return first
    tiny = np.finfo(float).tiny
    scores = (
        (1 - weight) * (np.log(np.maximum(first, tiny)) - np.log(prior_first))
        + weight * (np.log(np.maximum(second, tiny)) - np.log(prior_second))
        + np.log(prior_first)
    )
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fuse_classifiers(first, second, weight):
    """Realise a prior-corrected geometric mixture using the existing portable tree schema."""
    validate_classifier(first)
    validate_classifier(second)
    if (
        first["feature_names"] != second["feature_names"]
        or first["decoder_scale"] != second["decoder_scale"]
    ):
        raise ValueError("ensemble components need identical feature layout and emission scale")
    priors = [np.asarray(m["class_prior"]) for m in (first, second)]
    blend_probabilities(np.ones((1, 3)) / 3, np.ones((1, 3)) / 3, *priors, weight)
    result = copy.deepcopy(first)
    if weight == 0:
        return result
    result["baseline"] = (
        (1 - weight) * np.asarray(first["baseline"])
        + weight * np.asarray(second["baseline"])
        + weight * (np.log(priors[0]) - np.log(priors[1]))
    ).tolist()
    result["trees"] = []
    for component, factor in ((first, 1 - weight), (second, weight)):
        if factor == 0:
            continue
        for tree in component["trees"]:
            scaled = copy.deepcopy(tree)
            scaled["value"] = (np.asarray(tree["value"]) * factor).tolist()
            result["trees"].append(scaled)
    validate_classifier(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_ensemble.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised_ensemble")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("ensemble output must remain under ignored outputs/")
    config = load_config(args.config)
    cfg, features = config["supervised"]["training"], config["supervised"]["features"]
    extra = config["supervised"]["ensemble_selection"]
    validate_split(cfg)
    first = load_component(extra["parent_config"], features, cfg)
    second = load_component(extra["secondary_config"], features, cfg)
    event_component = load_component(extra["events_config"], features, cfg)
    events_model = event_component[0]
    candidates = {}
    for weight in extra["ensemble_weights"]:
        candidate = fuse_classifiers(first[0], second[0], weight)
        candidate["boundary_classifier"] = events_model["boundary_classifier"]
        candidate["boundary_prior"] = events_model["boundary_prior"]
        candidate["boundary_semantics"] = events_model["boundary_semantics"]
        candidates[weight] = candidate
    rows, max_error = [], 0.0
    for index, game in enumerate(cfg["validation_games"]):
        print(f"Selecting frozen ensemble on validation game {game}", flush=True)
        tracking, reference, runs, _ = read_game(args.data_dir, game)
        for profile_name, profile in profiles(tracking, runs, cfg, cfg["seed"] + 100 + index):
            values = feature_matrix(profile, features, runs)
            p1 = compiled_symmetric_probabilities(values, first[0])
            # Team symmetry is applied to each component before fusing likelihoods.
            # The exported forest instead fuses logits before symmetry. Validate the exact
            # exported representation below; use it for scores rather than claiming identity.
            events = compiled_symmetric_probabilities(values, events_model["boundary_classifier"])
            for mixture, candidate in candidates.items():
                probabilities = (
                    p1 if mixture == 0 else compiled_symmetric_probabilities(values, candidate)
                )
                if mixture == 0:
                    max_error = max(max_error, float(np.max(np.abs(probabilities - p1))))
                for potential in extra["potential_weights"]:
                    candidate["boundary_weight"] = potential
                    metrics = metric_summary(
                        reference, decode_boundaries(probabilities, events, candidate, runs)
                    )
                    rows.append(
                        {
                            "game": game,
                            "profile": profile_name,
                            "ensemble_weight": mixture,
                            "boundary_weight": potential,
                            "decoder_scale": candidate["decoder_scale"],
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"validation {game}/{profile_name} mix={mixture} handover={potential}: "
                        f"spellF1 {metrics['spell_f1_mean']:.4f}",
                        flush=True,
                    )
    selection = [
        {
            "ensemble_weight": mixture,
            "boundary_weight": potential,
            "decoder_scale": first[0]["decoder_scale"],
            "mean_spell_f1": float(
                np.mean(
                    [
                        r["metrics"]["spell_f1_mean"]
                        for r in rows
                        if r["ensemble_weight"] == mixture and r["boundary_weight"] == potential
                    ]
                )
            ),
        }
        for mixture in candidates
        for potential in extra["potential_weights"]
    ]
    selection.sort(key=lambda r: (-r["mean_spell_f1"], r["ensemble_weight"], r["boundary_weight"]))
    best = selection[0]
    model = candidates[best["ensemble_weight"]]
    model["boundary_weight"] = best["boundary_weight"]
    model["ensemble_weight"] = best["ensemble_weight"]
    frozen = copy_component(out / "frozen_parent", extra["parent_config"], first)
    secondary = copy_component(out / "frozen_secondary", extra["secondary_config"], second)
    event = copy_component(out / "frozen_events", extra["events_config"], event_component)
    provenance = {
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "src/mtah").glob("*.py"))
        },
        "trainer_path": "tools/select_ensemble.py",
        "trainer_sha256": sha256(__file__),
        "helper_sha256": {
            p: sha256(ROOT / p)
            for p in (
                "tools/select_joint.py",
                "tools/train_boundaries.py",
                "tools/train_supervised.py",
            )
        },
        "external_game_sha256": {
            g: {k: sha256(p) for k, p in game_assets(args.data_dir, g).items()}
            for g in cfg["train_games"] + cfg["validation_games"]
        },
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
        "fitting_stage": (
            "validation selection of frozen weighted forests and handover model; no new fitting"
        ),
        "parent_model_sha256": frozen["model.json"],
        "parent_config_sha256": frozen["config.toml"],
        "parent_report_sha256": frozen["training_report.json"],
        "secondary_component_sha256": secondary,
        "event_component_sha256": event,
    }
    model["provenance"] = provenance
    report = {
        "training_games": cfg["train_games"],
        "validation_games": cfg["validation_games"],
        "sealed_test_games": cfg["test_games"],
        "train_rows_with_augmentation": first[1]["train_rows_with_augmentation"],
        "additional_training_rows": 0,
        "features": model["feature_names"],
        "validation": rows,
        "decoder_selection": selection,
        "selected_decoder_scale": model["decoder_scale"],
        "selected_boundary_weight": model["boundary_weight"],
        "selected_ensemble_weight": model["ensemble_weight"],
        "provenance": provenance,
        "zero_mix_probability_max_error": max_error,
        "selection_basis": (
            "mean validation spellF1 on two games and three profiles; "
            "exported forests evaluated directly"
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
    print("SELECTED ENSEMBLE:", best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
