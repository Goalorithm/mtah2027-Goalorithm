"""Frozen models on a held-out broadcast game with partial algorithmic H/A references."""

import argparse
import json
from pathlib import Path

import _src_path  # noqa: F401
import torch
from evaluate_temporal import load_model, predict
from skillcorner_data import load_profiles, read_game
from temporal_ensemble import load_ensemble, predict_ensemble
from train_supervised import json_bytes

from mtah.config import load_config
from mtah.external import anchor_metrics, predict_external
from mtah.io import sha256, write_submission
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def partial_metrics(reference, prediction):
    original = anchor_metrics(reference, prediction)
    return {
        "known_owner_accuracy": original["pass_start_team_accuracy"],
        "balanced_team_recall": original["balanced_team_recall"],
        "known_reference_count": original["anchor_count"],
        "correct_team_count": original["correct_team_count"],
        "confusion_rows_H_A_columns_H_A_D_X": original["confusion_rows_H_A_columns_H_A_D_X"],
        "complete_HAD_spell_f1": None,
        "D_accuracy": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--out-dir", default="outputs/skillcorner_benchmark")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("benchmark output must stay under ignored outputs/")
    cfg = load_config("config/skillcorner.toml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    models = []
    # Load and verify every receipt before any test-frame reference is parsed.
    for name in (
        "temporal",
        "temporal_v2",
        "temporal_domain",
        "temporal_domain_v1",
        "temporal_refit",
    ):
        path = OUTPUTS_DIR / name
        metadata, parent, net, receipt = load_model(path, device)
        models.append((name, metadata, parent, net, receipt, predict))
    ensemble_path = OUTPUTS_DIR / "temporal_domain_ensemble"
    if (ensemble_path / "model.json").is_file():
        candidate = json.loads((ensemble_path / "model.json").read_text())
        if candidate["adopted_over_best_component"]:
            metadata, parent, net, receipt = load_ensemble(ensemble_path, device)
            models.append((ensemble_path.name, metadata, parent, net, receipt, predict_ensemble))
    out.mkdir(exist_ok=True)
    report = {
        "source_config_sha256": sha256(ROOT / "config/skillcorner.toml"),
        "source_manifest_sha256": sha256(ROOT / "data/external/skillcorner_new/manifest.json"),
        "script_sha256": sha256(__file__),
        "parser_sha256": sha256(ROOT / "tools/skillcorner_data.py"),
        "split": args.split,
        "rows": [],
        "models_fitted_or_tuned": False,
        "reference_scope": cfg["reference"],
        "complete_HAD_reference": False,
        "reference_origin": (
            "SkillCorner Game Intelligence, algorithmic silver labels; not manual ground truth"
        ),
        "bar_ath_accuracy_measured": False,
    }
    for game in cfg[f"{args.split}_games"]:
        items = load_profiles(game, args.split, models[0][2])
        for item in items:
            for name, metadata, parent, net, receipt, predictor in models:
                result = predictor(item, net, metadata, parent, device)
                path = out / f"{game}_{item['profile']}_{name}.csv"
                write_submission(path, result)
                metrics = partial_metrics(item["reference"], result)
                report["rows"].append(
                    {
                        "game": game,
                        "profile": str(item["profile"]),
                        "model": name,
                        "receipt": receipt,
                        "prediction_sha256": sha256(path),
                        "metrics": metrics,
                    }
                )
                print(
                    f"Partial SkillCorner {game}/{item['profile']}/{name}: "
                    f"owner accuracy {metrics['known_owner_accuracy']:.4%}",
                    flush=True,
                )
        tracking, reference, runs, _ = read_game(game)
        prediction = predict_external(load_config("config/exp_hmm_v2.toml"), tracking, runs)
        path = out / f"{game}_clean_hmm.csv"
        write_submission(path, prediction)
        report["rows"].append(
            {
                "game": game,
                "profile": "clean",
                "model": "configured_HMM",
                "prediction_sha256": sha256(path),
                "metrics": partial_metrics(reference, prediction),
            }
        )
    atomic_write(out / "report.json", json_bytes(report))
    print(
        "Partial references cannot establish complete spellF1, "
        "dead-ball accuracy or competition score",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
