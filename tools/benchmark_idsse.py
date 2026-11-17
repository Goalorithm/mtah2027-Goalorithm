"""Evaluate frozen models on complete external H/A/D references after training is frozen."""

import argparse
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401

from mtah.config import load_config
from mtah.external import predict_many_external
from mtah.idsse import game_assets, read_game
from mtah.io import sha256, write_submission
from mtah.paths import OUTPUTS_DIR, ROOT, pm
from mtah.review import atomic_write
from mtah.supervised import hide_ball, perturb_ball


def verify_freeze(config_path, model_path):
    """Run before loading any final-game reference; a changed model invalidates the receipt."""
    config = load_config(config_path)
    receipt_path = model_path.parent / "test_freeze.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report_path = model_path.parent / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        receipt["model_sha256"] != sha256(model_path)
        or receipt["config_sha256"] != sha256(config_path)
        or receipt["training_report_sha256"] != sha256(report_path)
        or receipt["test_games"] != config["supervised"]["training"]["test_games"]
        or receipt["frozen_before_test_evaluation"] is not True
        or report["provenance"]["test_games_read"] is not False
    ):
        raise ValueError("model/config/training report changed since the pre-test freeze")
    return {
        "receipt_sha256": sha256(receipt_path),
        "model_sha256": sha256(model_path),
        "training_report_sha256": sha256(report_path),
    }


def markdown(report):
    lines = [
        "# IDSSE full-state external benchmark",
        "",
        "Reference: DFL/Sportec provider BallStatus and BallPossession, CC-BY-4.0.",
        "These are other-game results, not the hidden BAR-ATH competition score.",
        f"Split: {report['split']}. Models are frozen before loading the final-game references.",
        f"Previous test-set exposure: {report['test_set_previous_exposure']}.",
        "",
        "| Game | Profile | Model | spellF1 | macroF1 | D F1 | Pred/true spells |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for game in report["games"]:
        for row in game["models"]:
            s = row["metrics"]
            lines.append(
                f"| {game['game']} | {row['profile']} | {Path(row['config']).stem} | "
                f"{s['spell_f1_mean']:.4f} | {s['macro_f1']:.4f} | "
                f"{s['per_class']['D']['f1']:.4f} | {s['n_pred_spells']}/{s['n_gt_spells']} |"
            )
    lines += [
        "",
        "Occluded profiles hide ball observations with a fixed label-independent Markov mask.",
        "These masks test missing-ball robustness; automatic tracking has other errors too.",
        "Provider ownership/stoppage semantics and corrected tracking differ from the competition.",
        "No BAR-ATH reference labels are used; no competition winner is selected.",
        "",
        "Source: https://doi.org/10.1038/s41597-025-04505-y",
        "Mirror: https://huggingface.co/datasets/pysport/idsse-data",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="config/idsse_benchmark.toml")
    parser.add_argument("--config", default="config/exp_supervised_v1.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/idsse_benchmark")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: benchmark reports must stay under ignored outputs/")
        return 1
    try:
        config = load_config(args.config)
        plan = load_config(args.plan)
        checkpoint = ROOT / config["supervised"]["model_path"]
        freeze = verify_freeze(args.config, checkpoint)
        source = json.loads((ROOT / "config/idsse_sources.json").read_text(encoding="utf-8"))
        expected_hashes = {row["name"]: row["sha256"] for row in source["files"]}
        models = []
        for name in plan["model_configs"]:
            candidate = load_config(ROOT / name)
            info = {"config": name, "config_sha256": sha256(ROOT / name)}
            method = candidate["dead_ball"]["method"]
            if method == "supervised":
                info["checkpoint_sha256"] = sha256(ROOT / candidate[method]["model_path"])
            models.append((candidate, info))
        report = {
            "split": args.split,
            "games": [],
            "pre_test_freeze": freeze,
            "benchmark_plan_sha256": sha256(args.plan),
            "benchmark_script_sha256": sha256(__file__),
            "source_manifest_sha256": sha256(ROOT / "config/idsse_sources.json"),
            "source_code_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                for p in sorted((ROOT / "src/mtah").glob("*.py"))
            },
            "provider_reference": "DFL/Sportec BallStatus and BallPossession",
            "models_fitted_or_tuned": False,
            "bar_ath_labels_read": False,
            "competition_accuracy_measured": False,
            "selected_final_competition_model": None,
            "test_set_previous_exposure": config["supervised"]["training"].get(
                "test_set_previous_exposure", "none_at_original_freeze"
            ),
        }
        games = config["supervised"]["training"][f"{args.split}_games"]
        for index, game in enumerate(games):
            print(f"Opening frozen {args.split} game {game}", flush=True)
            assets = game_assets(args.data_dir, game)
            digests = {key: sha256(path) for key, path in assets.items()}
            if any(digests[key] != expected_hashes[path.name] for key, path in assets.items()):
                raise ValueError("external game differs from pinned provider assets")
            tracking, reference, runs, metadata = read_game(args.data_dir, game)
            rows = []
            profiles = [
                ("clean", tracking),
                (
                    "occluded",
                    hide_ball(
                        tracking,
                        runs,
                        plan["dropout_fraction"],
                        plan["dropout_mean_frames"],
                        plan["seed"] + 200 + index,
                    ),
                ),
            ]
            if plan.get("coordinate_noise_std_m", 0) > 0:
                profiles.append(
                    (
                        "occluded_jittered",
                        perturb_ball(
                            profiles[-1][1],
                            plan["coordinate_noise_std_m"],
                            plan["seed"] + 300 + index,
                        ),
                    )
                )
            for profile_name, profile in profiles:
                predictions = predict_many_external([config for config, _ in models], profile, runs)
                for prediction, (_, info) in zip(predictions, models, strict=True):
                    path = out / f"{game}_{profile_name}_{Path(info['config']).stem}.csv"
                    write_submission(path, prediction)
                    result = pm.evaluate(reference, prediction, bootstrap_n=plan["bootstrap_draws"])
                    metrics = {
                        key: result[key]
                        for key in (
                            "spell_f1_mean",
                            "macro_f1",
                            "alive_dead_f1",
                            "per_class",
                            "n_pred_spells",
                            "n_gt_spells",
                            "n_graded",
                            "spell_bootstrap_low",
                            "spell_bootstrap_high",
                            "bootstrap_low",
                            "bootstrap_high",
                        )
                    }
                    metrics["transition_f1_12_frames"] = result["transitions"][12]["f1"]
                    metrics["transition_f1_25_frames"] = result["transitions"][25]["f1"]
                    rows.append(
                        {
                            **info,
                            "profile": profile_name,
                            "prediction_sha256": sha256(path),
                            "metrics": metrics,
                        }
                    )
                    print(
                        f"{game}/{profile_name}/{Path(info['config']).stem}: "
                        f"spellF1 {metrics['spell_f1_mean']:.4f}, "
                        f"macroF1 {metrics['macro_f1']:.4f}",
                        flush=True,
                    )
            report["games"].append(
                {"game": game, "metadata": metadata, "asset_sha256": digests, "models": rows}
            )
        atomic_write(
            out / "report.json",
            (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        atomic_write(out / "report.md", markdown(report).encode())
        print(f"-> {out / 'report.md'}; no final competition model selected")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
