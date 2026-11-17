"""Train on three IDSSE games; select decoder on two different validation games, never BAR-ATH."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from mtah import A, H, X
from mtah.config import load_config
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT, pm
from mtah.review import atomic_write
from mtah.supervised import (
    decode_probabilities,
    export_classifier,
    feature_matrix,
    feature_names,
    hide_ball,
    perturb_ball,
    raw_probabilities,
    swap_features,
)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_split(cfg):
    groups = [cfg[key] for key in ("train_games", "validation_games", "test_games")]
    if any(
        not isinstance(group, list) or not group or len(set(group)) != len(group)
        for group in groups
    ):
        raise ValueError("game splits must be nonempty and internally unique")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training, validation and test games must be disjoint")
    if type(cfg["stride"]) is not int or cfg["stride"] < 1:
        raise ValueError("training stride must be a positive integer")
    scales = cfg["decoder_scales"]
    if (
        not scales
        or not np.isfinite(scales).all()
        or min(scales) <= 0
        or len(set(scales)) != len(scales)
    ):
        raise ValueError("decoder scales must be finite, positive and distinct")


def transition_parameters(references, pseudocount):
    if not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("transition pseudocount must be positive")
    counts = np.zeros((3, 3))
    starts = np.zeros(3)
    prior = np.zeros(3)
    for reference, runs in references:
        prior += np.bincount(reference[reference != X], minlength=3)
        for a, b in runs:
            valid = reference[a:b] != X
            if valid.any():
                starts[reference[a:b][np.flatnonzero(valid)[0]]] += 1
            first, second = reference[a : b - 1], reference[a + 1 : b]
            keep = (first != X) & (second != X)
            np.add.at(counts, (first[keep], second[keep]), 1)
    swap = [A, H, 2]
    counts = (counts + counts[np.ix_(swap, swap)]) / 2 + pseudocount
    starts = (starts + starts[swap]) / 2 + pseudocount
    prior = (prior + prior[swap]) / 2
    if (prior == 0).any():
        raise ValueError("external training data must contain all H/A/D states")
    return {
        "transition": (counts / counts.sum(axis=1, keepdims=True)).tolist(),
        "initial": (starts / starts.sum()).tolist(),
        "class_prior": (prior / prior.sum()).tolist(),
    }


def metric_summary(reference, prediction):
    result = pm.evaluate(reference, prediction, bootstrap_n=0)
    return {
        key: result[key]
        for key in (
            "spell_f1_mean",
            "macro_f1",
            "alive_dead_f1",
            "per_class",
            "n_gt_spells",
            "n_pred_spells",
            "n_graded",
        )
    }


def training_weights(reference, runs, exponent=0.0, boundary_weight=1.0, boundary_frames=0):
    """Weight short, labelled training spells; no validation/test labels enter fitting."""
    if (
        not np.isfinite(exponent)
        or not 0 <= exponent <= 1
        or not np.isfinite(boundary_weight)
        or boundary_weight < 1
        or type(boundary_frames) is not int
        or boundary_frames < 0
    ):
        raise ValueError("invalid short-spell or boundary training weights")
    weights = np.ones(len(reference), dtype=float)
    for start, stop in runs:
        for a, b, side in pm.segments(reference, [(start, stop)]):
            if side in (H, A):
                weights[a:b] = np.clip((125 / (b - a)) ** exponent, 0.5, 5.0)
        changes = (
            np.flatnonzero(
                (reference[start + 1 : stop] != reference[start : stop - 1])
                & (reference[start + 1 : stop] != X)
                & (reference[start : stop - 1] != X)
            )
            + start
            + 1
        )
        for frame in changes:
            weights[max(start, frame - boundary_frames) : min(stop, frame + boundary_frames)] *= (
                boundary_weight
            )
    return weights


def possession_balanced_samples(reference, runs, cfg):
    """Give every live training possession equal total weight, retaining all short spells."""
    stride = cfg["stride"]
    short = cfg.get("short_spell_frames", 50)
    band = cfg.get("boundary_sample_frames", 5)
    mass = cfg.get("possession_mass", 125.0)
    dead_weight = cfg.get("dead_sample_weight", 1.0)
    if (
        type(short) is not int
        or short < 1
        or type(band) is not int
        or band < 0
        or not np.isfinite(mass)
        or mass <= 0
        or not np.isfinite(dead_weight)
        or dead_weight <= 0
    ):
        raise ValueError("invalid possession-balanced sampling parameters")
    keep = np.zeros(len(reference), dtype=bool)
    keep[::stride] = True
    segments = []
    for start, stop in runs:
        for a, b, side in pm.segments(reference, [(start, stop)]):
            segments.append((a, b, side))
            if side in (H, A):
                if b - a <= short:
                    keep[a:b] = True
                else:
                    keep[a : min(b, a + band)] = True
                    keep[max(a, b - band) : b] = True
    valid = np.zeros(len(reference), dtype=bool)
    for start, stop in runs:
        valid[start:stop] = reference[start:stop] != X
    keep &= valid
    weights = np.zeros(len(reference))
    for a, b, side in segments:
        count = int(keep[a:b].sum())
        if count:
            weights[a:b] = mass / count if side in (H, A) else dead_weight
    selected = np.flatnonzero(keep)
    return selected, weights[selected]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_v1.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: supervised checkpoints must stay under ignored outputs/")
        return 1
    try:
        config = load_config(args.config)
        cfg, feature_cfg = config["supervised"]["training"], config["supervised"]["features"]
        validate_split(cfg)
        names = feature_names(feature_cfg)
        arrays, labels, references, sources, training_counts = [], [], [], {}, {}
        sample_weights = []
        for index, game in enumerate(cfg["train_games"]):
            print(f"Preparing external training game {game}", flush=True)
            tracking, reference, runs, _ = read_game(args.data_dir, game)
            references.append((reference, runs))
            weights = training_weights(
                reference,
                runs,
                cfg.get("short_spell_weight_exponent", 0.0),
                cfg.get("boundary_weight", 1.0),
                cfg.get("boundary_frames", 0),
            )
            if cfg.get("sampling") == "possession_balanced":
                chosen, chosen_weights = possession_balanced_samples(reference, runs, cfg)
            else:
                chosen = np.arange(0, len(reference), cfg["stride"])
                chosen = chosen[reference[chosen] != X]
                chosen_weights = weights[chosen]
            sources[game] = {
                key: sha256(path) for key, path in game_assets(args.data_dir, game).items()
            }
            profiles = [
                tracking,
                hide_ball(
                    tracking,
                    runs,
                    cfg["dropout_fraction"],
                    cfg["dropout_mean_frames"],
                    cfg["seed"] + index,
                ),
            ]
            if cfg.get("coordinate_noise_std_m", 0) > 0:
                profiles.append(
                    perturb_ball(
                        profiles[-1], cfg["coordinate_noise_std_m"], cfg["seed"] + 1000 + index
                    )
                )
            for profile in profiles:
                values = feature_matrix(profile, feature_cfg, runs)
                selected = chosen
                values, target = values[selected], reference[selected]
                arrays += [values, swap_features(values, names)]
                labels += [target, np.where(target == H, A, np.where(target == A, H, target))]
                sample_weights += [chosen_weights, chosen_weights]
            training_counts[game] = {
                "frames": int(np.sum(reference != X)),
                "class_counts_H_A_D": np.bincount(reference[reference != X], minlength=3).tolist(),
            }
        train_x, train_y = np.concatenate(arrays), np.concatenate(labels)
        del arrays, labels, profiles, values, tracking
        classifier = HistGradientBoostingClassifier(
            max_iter=cfg["max_iter"],
            max_leaf_nodes=cfg["max_leaf_nodes"],
            learning_rate=cfg["learning_rate"],
            l2_regularization=cfg["l2_regularization"],
            min_samples_leaf=cfg["min_samples_leaf"],
            early_stopping=False,
            random_state=cfg["seed"],
            verbose=cfg.get("verbose", 0),
        )
        with threadpool_limits(limits=1):
            print(
                f"Training boosted trees on {len(train_y)} feature rows; {len(names)} features",
                flush=True,
            )
            fit_weights = np.concatenate(sample_weights)
            weighted = (
                cfg.get("sampling") == "possession_balanced"
                or cfg.get("short_spell_weight_exponent", 0) > 0
                or cfg.get("boundary_weight", 1) > 1
            )
            classifier.fit(train_x, train_y, sample_weight=fit_weights if weighted else None)
            model = export_classifier(classifier, names)
            # Validate custom JSON inference against the fitted library, including NaN routes.
            sample = train_x[:: max(1, len(train_x) // 1000)]
            error = float(
                np.max(np.abs(raw_probabilities(sample, model) - classifier.predict_proba(sample)))
            )
            if error > 1e-12:
                raise ValueError(f"portable tree inference differs from sklearn: {error}")
            model.update(transition_parameters(references, cfg["transition_pseudocount"]))
            if cfg.get("sampling") == "possession_balanced":
                class_mass = np.bincount(train_y, weights=fit_weights, minlength=3)
                model["class_prior"] = (class_mass / class_mass.sum()).tolist()
            model["feature_config"] = feature_cfg
            rows = []
            for index, game in enumerate(cfg["validation_games"]):
                print(f"Evaluating decoder on external validation game {game}", flush=True)
                tracking, reference, runs, _ = read_game(args.data_dir, game)
                sources[game] = {
                    key: sha256(path) for key, path in game_assets(args.data_dir, game).items()
                }
                profiles = [
                    ("clean", tracking),
                    (
                        "occluded",
                        hide_ball(
                            tracking,
                            runs,
                            cfg["dropout_fraction"],
                            cfg["dropout_mean_frames"],
                            cfg["seed"] + 100 + index,
                        ),
                    ),
                ]
                if cfg.get("coordinate_noise_std_m", 0) > 0:
                    profiles.append(
                        (
                            "occluded_jittered",
                            perturb_ball(
                                profiles[-1][1],
                                cfg["coordinate_noise_std_m"],
                                cfg["seed"] + 1100 + index,
                            ),
                        )
                    )
                for profile_name, profile in profiles:
                    values = feature_matrix(profile, feature_cfg, runs)
                    probabilities = (
                        classifier.predict_proba(values)
                        + classifier.predict_proba(swap_features(values, names))[:, [A, H, 2]]
                    ) / 2
                    for scale in cfg["decoder_scales"]:
                        model["decoder_scale"] = scale
                        prediction = decode_probabilities(probabilities, model, runs)
                        metrics = metric_summary(reference, prediction)
                        rows.append(
                            {
                                "game": game,
                                "profile": profile_name,
                                "decoder_scale": scale,
                                "metrics": metrics,
                            }
                        )
                        print(
                            f"validation {game}/{profile_name} scale={scale}: "
                            f"spellF1 {metrics['spell_f1_mean']:.4f}, "
                            f"macroF1 {metrics['macro_f1']:.4f}",
                            flush=True,
                        )
        candidates = [
            {
                "decoder_scale": scale,
                "mean_spell_f1": float(
                    np.mean(
                        [
                            row["metrics"]["spell_f1_mean"]
                            for row in rows
                            if row["decoder_scale"] == scale
                        ]
                    )
                ),
            }
            for scale in cfg["decoder_scales"]
        ]
        candidates.sort(key=lambda row: (-row["mean_spell_f1"], row["decoder_scale"]))
        model["decoder_scale"] = candidates[0]["decoder_scale"]
        provenance = {
            "config_sha256": sha256(args.config),
            "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
            "source_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                for p in sorted((ROOT / "src/mtah").glob("*.py"))
            },
            "trainer_sha256": sha256(__file__),
            "external_game_sha256": sources,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scikit-learn", "threadpoolctl")
            },
            "bar_ath_labels_read": False,
            "bar_ath_data_read": False,
            "test_games_read": False,
            "label_source": (
                "DFL/Sportec provider BallStatus and BallPossession, Bassek et al. 2025, CC-BY-4.0"
            ),
        }
        model["provenance"] = provenance
        report = {
            "training_games": cfg["train_games"],
            "validation_games": cfg["validation_games"],
            "sealed_test_games": cfg["test_games"],
            "train_rows_with_augmentation": len(train_y),
            "features": names,
            "training_counts": training_counts,
            "validation": rows,
            "decoder_selection": candidates,
            "selected_decoder_scale": model["decoder_scale"],
            "selection_basis": (
                "mean spellF1 on separate validation games, clean and synthetic occlusion profiles"
            ),
            "portable_probability_max_error": error,
            "provenance": provenance,
            "competition_accuracy_measured": False,
            "test_set_previous_exposure": cfg.get(
                "test_set_previous_exposure", "none_at_original_freeze"
            ),
            "selected_final_competition_model": None,
        }
        atomic_write(out / "model.json", json_bytes(model))
        report["model_sha256"] = sha256(out / "model.json")
        atomic_write(out / "training_report.json", json_bytes(report))
        receipt = {
            "model_sha256": report["model_sha256"],
            "config_sha256": sha256(args.config),
            "training_report_sha256": sha256(out / "training_report.json"),
            "test_games": cfg["test_games"],
            "frozen_before_test_evaluation": True,
        }
        atomic_write(out / "test_freeze.json", json_bytes(receipt))
        print(
            f"TRAINED: selected scale {model['decoder_scale']}; sealed test games unread; "
            f"{out / 'model.json'}"
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
