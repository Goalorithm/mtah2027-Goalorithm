"""Run frozen Tech 1/2/3 candidates twice and log unlabelled diagnostics, without ranking them."""

import argparse
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np

from mtah.config import load_config
from mtah.io import sha256, write_submission
from mtah.load import load_smart_tagging, load_tracking
from mtah.paths import N_FRAMES, OUTPUTS_DIR, ROOT, games, match_runs, pm
from mtah.predict import prepare_features, run_pipeline
from mtah.review import atomic_write
from mtah.scoring import match_stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/team_experiments.toml")
    parser.add_argument("--out-dir", default="outputs/team_experiments")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: experiment results must stay under ignored outputs/")
        return 1
    try:
        plan = load_config(args.config)
        variants = [
            {"name": Path(path).stem, "config": path, "overrides": []} for path in plan["configs"]
        ] + plan.get("sweep", [])
        if len({row["name"] for row in variants}) != len(variants):
            raise ValueError("experiment names must be unique")
        frozen = []
        for variant in variants:
            if Path(variant["name"]).name != variant["name"] or not variant["name"]:
                raise ValueError("experiment names must be plain filenames")
            config = load_config(ROOT / variant["config"], variant["overrides"])
            info = {**variant, "config_sha256": sha256(ROOT / variant["config"])}
            method = config["dead_ball"]["method"]
            if method == "supervised":
                info["checkpoint_sha256"] = sha256(ROOT / config[method]["model_path"])
            frozen.append((config, info))
        assets = games.asset_paths(source="tactical")
        report = {
            "plan_sha256": sha256(args.config),
            "script_sha256": sha256(__file__),
            "source_sha256": {
                str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                for p in sorted((ROOT / "src/mtah").glob("*.py"))
            },
            "data_sha256": {
                key: sha256(assets[key]) for key in ("atd_meta", "atd_raw", "smart_tagging")
            },
            "reference_labels_read": False,
            "accuracy_measured": False,
            "public_labels_used_for_tuning": False,
            "selected_final_model": None,
            "warning": "Fewer/longer possessions and tag agreement do not establish accuracy.",
            "experiments": [],
        }
        tracking, tags = load_tracking(), load_smart_tagging()
        prepared_first, prepared_second = prepare_features(tracking), prepare_features(tracking)
        out.mkdir(parents=True, exist_ok=True)
        for config, info in frozen:
            first = run_pipeline(config, tracking, tags, prepared_first)
            second = run_pipeline(config, tracking, tags, prepared_second)
            if not np.array_equal(first, second):
                raise ValueError(f"non-deterministic prediction: {info['name']}")
            path = out / f"{info['name']}.csv"
            write_submission(path, first)
            reason = pm.validate_submission(pm.read_label_csv(str(path)), N_FRAMES)
            if reason:
                raise ValueError(reason)
            with tempfile.TemporaryDirectory(prefix="repeat-", dir=out) as temporary:
                repeated = Path(temporary) / "repeat.csv"
                write_submission(repeated, second)
                if sha256(path) != sha256(repeated):
                    raise ValueError(f"prediction bytes differ: {info['name']}")
            stats = match_stats(first)
            pieces, hits = 0, 0
            slack = plan["set_piece_slack_frames"]
            for item in tags:
                if str(item.get("code", "")).casefold() != "set pieces":
                    continue
                for start, stop in match_runs():
                    a, b = (
                        max(start, item["start_frame"] - slack),
                        min(stop, item["end_frame"] + slack),
                    )
                    if a < b:
                        pieces += 1
                        hits += bool((first[a:b] == 2).any())
            stats["automatic_set_pieces"] = pieces
            stats["set_pieces_near_predicted_D"] = hits
            row = {
                **info,
                "prediction_sha256": sha256(path),
                "repeat_identical": True,
                "statistics": stats,
            }
            report["experiments"].append(row)
            print(
                f"{info['name']}: {stats['n_possessions']} possessions; "
                f"{stats['short_under_2s']} under 2s; D {stats['dead_share']:.2%}; "
                f"X {stats['abstain_share']:.2%}; repeated bytes identical",
                flush=True,
            )
        atomic_write(
            out / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "experiment",
                "config",
                "overrides",
                "possessions",
                "under_2s",
                "mean_s",
                "dead_share",
                "x_share",
                "set_pieces_near_D",
                "sha256",
                "accuracy",
                "final_selected",
            ]
        )
        for row in report["experiments"]:
            stats = row["statistics"]
            writer.writerow(
                [
                    row["name"],
                    row["config"],
                    " ".join(row["overrides"]),
                    stats["n_possessions"],
                    stats["short_under_2s"],
                    stats["mean_s"],
                    stats["dead_share"],
                    stats["abstain_share"],
                    stats["set_pieces_near_predicted_D"],
                    row["prediction_sha256"],
                    "not_measured",
                    "no",
                ]
            )
        atomic_write(out / "log_eksperimen.tsv", buffer.getvalue().encode())
        print(f"-> {out / 'report.json'}; no model selected")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
