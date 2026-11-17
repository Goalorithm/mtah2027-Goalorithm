"""Select a multi-context neural ensemble on validation only; do not fit/read test references."""

import json
import shutil
from itertools import product

import _src_path  # noqa: F401
import numpy as np
from evaluate_temporal import load_model
from temporal_model import blend
from train_supervised import json_bytes, metric_summary

from mtah.boundary import decode_boundaries
from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def main():
    policy_path = ROOT / "config/temporal_ensemble.toml"
    cfg = load_config(policy_path)
    directories = [ROOT / name for name in cfg["component_dirs"]]
    components = [load_model(path, "cpu") for path in directories]
    metadata, parent, _, _ = components[0]
    for part, _, _, _ in components:
        if part["features"] != metadata["features"] or any(
            part["config"][key] != cfg[key]
            for key in ("train_games", "validation_games", "test_games")
        ):
            raise ValueError("ensemble components must share feature definitions and game roles")
    validation = []
    for game in cfg["validation_games"]:
        for profile in ("clean", "occluded", "occluded_jittered"):
            neural = []
            for path in directories:
                with np.load(path / f"validation_{game}_{profile}.npz", allow_pickle=False) as data:
                    neural.append(data["state"])
            with np.load(
                OUTPUTS_DIR / f"temporal_cache/{game}_validation_{profile}.npz", allow_pickle=False
            ) as data:
                validation.append(
                    {
                        "game": game,
                        "profile": profile,
                        "neural": neural,
                        "teacher": data["state"],
                        "event": data["event"],
                        "reference": data["reference"],
                        "runs": data["runs"],
                    }
                )
    rows = []
    for component_weight, state_weight in product(cfg["component_weights"], cfg["state_weights"]):
        details = []
        for item in validation:
            neural = blend(*item["neural"], component_weight)
            pred = decode_boundaries(
                blend(item["teacher"], neural, state_weight),
                blend(item["event"], item["event"], 0),
                parent,
                item["runs"],
            )
            details.append(
                {
                    "game": item["game"],
                    "profile": item["profile"],
                    "metrics": metric_summary(item["reference"], pred),
                }
            )
        mean = float(np.mean([r["metrics"]["spell_f1_mean"] for r in details]))
        rows.append(
            {
                "component_weight": component_weight,
                "state_weight": state_weight,
                "mean_spell_f1": mean,
                "validation": details,
            }
        )
        print(
            f"Ensemble second={component_weight},neural={state_weight}: val {mean:.5f}", flush=True
        )
    best = min(rows, key=lambda r: (-r["mean_spell_f1"], r["component_weight"], r["state_weight"]))
    component_scores = [
        json.loads((p / "training_report.json").read_text())["selected"]["mean_spell_f1"]
        for p in directories
    ]
    baseline = max(component_scores)
    adopted = best["mean_spell_f1"] >= baseline + cfg["minimum_gain_over_best_component"]
    out = OUTPUTS_DIR / "temporal_ensemble"
    out.mkdir(exist_ok=True)
    snapshots = []
    for index, path in enumerate(directories):
        subdir = f"components/model{index}"
        snapshots.append(subdir)
        (out / subdir).mkdir(parents=True, exist_ok=True)
        for name in (
            "model.json",
            "weights.pt",
            "parent.json",
            "training_report.json",
            "test_freeze.json",
        ):
            shutil.copy2(path / name, out / subdir / name)
    atomic_write(out / "parent.json", json_bytes(parent))
    model = {
        "config": cfg,
        "features": metadata["features"],
        "components": snapshots,
        "component_weight": best["component_weight"],
        "state_weight": best["state_weight"],
        "decoder_scale": parent["decoder_scale"],
        "adopted_over_best_component": adopted,
        "helper_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "tools/temporal_ensemble.py",
                "tools/select_temporal_ensemble.py",
                "tools/evaluate_temporal_ensemble.py",
                "tools/evaluate_temporal.py",
                "config/temporal_ensemble.toml",
            )
        },
    }
    atomic_write(out / "model.json", json_bytes(model))
    report = {
        "selection": rows,
        "selected": best,
        "best_component_validation": baseline,
        "minimum_required_gain": cfg["minimum_gain_over_best_component"],
        "adopted_over_best_component": adopted,
        "component_validation_scores": component_scores,
        "classifiers_refitted": False,
        "test_games_read": False,
        "bar_ath_labels_read": False,
        "competition_accuracy_measured": False,
        "selected_final_competition_model": None,
    }
    atomic_write(out / "training_report.json", json_bytes(report))
    names = ["model.json", "parent.json", "training_report.json"]
    names += [
        f"{subdir}/{name}"
        for subdir in snapshots
        for name in (
            "model.json",
            "weights.pt",
            "parent.json",
            "training_report.json",
            "test_freeze.json",
        )
    ]
    mapping = {name: sha256(out / name) for name in names}
    atomic_write(
        out / "test_freeze.json",
        json_bytes(
            {
                "files_sha256": mapping,
                "frozen_before_test_evaluation": True,
                "test_previous_exposure": cfg["test_previous_exposure"],
            }
        ),
    )
    print(f"Ensemble frozen; validation {best['mean_spell_f1']:.5f}, adopted={adopted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
