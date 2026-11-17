"""Post-hoc noise stress test of frozen models; never fit or select on these outcomes."""

import argparse
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
from benchmark_idsse import verify_freeze
from train_supervised import metric_summary

from mtah.config import load_config
from mtah.external import predict_external
from mtah.idsse import read_game
from mtah.io import sha256, write_submission
from mtah.load import Tracking
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import hide_ball


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="config/idsse_stress.toml")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out-dir", default="outputs/idsse_stress")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: stress reports must stay under ignored outputs/")
        return 1
    try:
        plan = load_config(args.plan)
        candidate_path = ROOT / "config/exp_supervised_v1.toml"
        candidate = load_config(candidate_path)
        receipt = verify_freeze(candidate_path, ROOT / candidate["supervised"]["model_path"])
        frozen = []
        for name in plan["model_configs"]:
            config = load_config(ROOT / name)
            info = {"config": name, "config_sha256": sha256(ROOT / name)}
            method = config["dead_ball"]["method"]
            if method == "supervised":
                info["checkpoint_sha256"] = sha256(ROOT / config[method]["model_path"])
            frozen.append((config, info))
        report = {
            "post_hoc_stress_test": True,
            "models_fitted_or_selected": False,
            "pre_test_freeze": receipt,
            "plan_sha256": sha256(args.plan),
            "script_sha256": sha256(__file__),
            "bar_ath_reference_labels_read": False,
            "warning": (
                "Independent 0.15m frame noise is a stress test; "
                "it does not estimate competition noise."
            ),
            "games": [],
        }
        for index, game in enumerate(candidate["supervised"]["training"]["test_games"]):
            tracking, reference, runs, metadata = read_game(args.data_dir, game)
            hidden = hide_ball(
                tracking,
                runs,
                plan["dropout_fraction"],
                plan["dropout_mean_frames"],
                plan["seed"] + index,
            )
            noise = np.random.default_rng(plan["seed"] + 100 + index).normal(
                0, plan["ball_noise_std_m"], tracking.ball.shape
            )
            stressed = Tracking(
                hidden.players,
                hidden.player_speed,
                hidden.ball + noise,
                hidden.ball_extra,
                hidden.slot_team,
                hidden.team_names,
            )
            rows = []
            for config, info in frozen:
                prediction = predict_external(config, stressed, runs)
                path = out / f"{game}_{Path(info['config']).stem}.csv"
                write_submission(path, prediction)
                metric = metric_summary(reference, prediction)
                rows.append({**info, "metrics": metric, "prediction_sha256": sha256(path)})
                print(
                    f"stress {game}/{Path(info['config']).stem}: "
                    f"spellF1 {metric['spell_f1_mean']:.4f}, macroF1 {metric['macro_f1']:.4f}",
                    flush=True,
                )
            report["games"].append({"game": game, "metadata": metadata, "models": rows})
        atomic_write(
            out / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        )
        print(f"-> {out / 'report.json'}; no model selected or changed")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
