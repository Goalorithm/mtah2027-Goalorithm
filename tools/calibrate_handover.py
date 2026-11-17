"""Select external-validation H/A transition offsets with frozen classifier weights."""

import argparse
import copy
import importlib.metadata
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from select_joint import copy_component, load_component
from train_boundaries import profiles
from train_supervised import json_bytes, metric_summary, validate_split

from mtah import A, H
from mtah.boundary import decode_boundaries
from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.hmm import _probabilities
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import feature_matrix


def calibrated_transition(transition, bias):
    matrix = _probabilities(transition, (3, 3), "transition").copy()
    if isinstance(bias, bool) or not np.isfinite(bias):
        raise ValueError("handover log-odds bias must be finite")
    if bias == 0:
        return matrix
    for side, other in ((H, A), (A, H)):
        live = matrix[side, side] + matrix[side, other]
        if live == 0:
            continue
        with np.errstate(divide="ignore"):
            odds = np.log(matrix[side, other]) - np.log(matrix[side, side]) + bias
        switch = np.exp(-np.logaddexp(0, -odds))
        matrix[side, other] = live * switch
        matrix[side, side] = live * (1 - switch)
    return _probabilities(matrix, (3, 3), "calibrated transition")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/exp_supervised_calibrated.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/supervised_calibrated")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("calibration output must remain under ignored outputs/")
    config = load_config(args.config)
    cfg, features = config["supervised"]["training"], config["supervised"]["features"]
    extra = config["supervised"]["handover_calibration"]
    validate_split(cfg)
    component = load_component(extra["parent_config"], features, cfg)
    parent = component[0]
    candidates = {}
    for bias in extra["log_odds_biases"]:
        model = copy.deepcopy(parent)
        model["transition"] = calibrated_transition(parent["transition"], bias).tolist()
        candidates[bias] = model
    if not candidates:
        raise ValueError("calibration needs at least one candidate bias")
    rows = []
    for index, game in enumerate(cfg["validation_games"]):
        print(f"Calibrating frozen transitions on validation game {game}", flush=True)
        tracking, reference, runs, _ = read_game(args.data_dir, game)
        for profile_name, profile in profiles(tracking, runs, cfg, cfg["seed"] + 100 + index):
            values = feature_matrix(profile, features, runs)
            probabilities = compiled_symmetric_probabilities(values, parent)
            events = compiled_symmetric_probabilities(values, parent["boundary_classifier"])
            for bias, model in candidates.items():
                metrics = metric_summary(
                    reference, decode_boundaries(probabilities, events, model, runs)
                )
                rows.append(
                    {
                        "game": game,
                        "profile": profile_name,
                        "handover_log_odds_bias": bias,
                        "boundary_weight": model["boundary_weight"],
                        "decoder_scale": model["decoder_scale"],
                        "metrics": metrics,
                    }
                )
                print(
                    f"validation {game}/{profile_name} handover_bias={bias}: "
                    f"spellF1 {metrics['spell_f1_mean']:.4f}",
                    flush=True,
                )
    selection = [
        {
            "handover_log_odds_bias": bias,
            "boundary_weight": parent["boundary_weight"],
            "decoder_scale": parent["decoder_scale"],
            "mean_spell_f1": float(
                np.mean(
                    [
                        r["metrics"]["spell_f1_mean"]
                        for r in rows
                        if r["handover_log_odds_bias"] == bias
                    ]
                )
            ),
        }
        for bias in candidates
    ]
    selection.sort(
        key=lambda r: (
            -r["mean_spell_f1"],
            abs(r["handover_log_odds_bias"]),
            r["handover_log_odds_bias"],
        )
    )
    best = selection[0]
    model = candidates[best["handover_log_odds_bias"]]
    model["handover_log_odds_bias"] = best["handover_log_odds_bias"]
    frozen = copy_component(out / "frozen_parent", extra["parent_config"], component)
    provenance = {
        "config_sha256": sha256(args.config),
        "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "src/mtah").glob("*.py"))
        },
        "trainer_path": "tools/calibrate_handover.py",
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
        "fitting_stage": ("validation-only H/A transition calibration; classifier weights frozen"),
        "parent_model_sha256": frozen["model.json"],
        "parent_config_sha256": frozen["config.toml"],
        "parent_report_sha256": frozen["training_report.json"],
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
        "selected_handover_log_odds_bias": model["handover_log_odds_bias"],
        "provenance": provenance,
        "selection_basis": (
            "mean validation spellF1 on two games and three profiles; D rates unchanged"
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
    print("CALIBRATED:", best, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
