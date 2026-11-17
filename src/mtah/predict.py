"""One command: data in, validated submission out, scored and summarised.

    python -m mtah.predict                                   # default config, final file name
    python -m mtah.predict --name exp_window75 --set possession.window=75
    python -m mtah.predict --config config/exp_x.toml --name exp_x
    python -m mtah.predict --holdout                         # final decision only

What it does, in order:
  1. loads the tracking (from cache/ after the first run) and the Smart Tagging
  2. builds features, runs the dead-ball detector and the possession assigner from the config
  3. writes outputs/<name>.csv and a copy in evaluation/results/ for run_evaluation.py
  4. validates the file with the organisers' own checks, and stops if it fails
  5. prints full-match statistics, scores on every local ground truth, the file's sha256, and
     one tab-separated line to paste into the team's experiment log
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys

import numpy as np

from mtah import LABELS, D, dead_ball, features, possession
from mtah.config import load_config
from mtah.io import sha256, write_submission
from mtah.load import load_smart_tagging, load_tracking
from mtah.paths import EVALUATION_DIR, N_FRAMES, OUTPUTS_DIR, ROOT, pm
from mtah.scoring import ground_truths, match_stats, score


def prepare_features(tracking):
    """An explicit per-tracking cache for offline multi-model evaluations."""
    result = features.build(tracking)
    result["_tracking_id"] = id(tracking)
    return result


def run_pipeline(config, tracking, tagging, prepared_features=None):
    """Codes (0..3) for every video frame. Pure: same inputs, same output."""
    feats = prepare_features(tracking) if prepared_features is None else prepared_features
    if feats.get("_tracking_id") != id(tracking):
        raise ValueError("prepared features belong to a different tracking input")
    dead = dead_ball.detect(tracking, feats, tagging, config)
    team = possession.assign(tracking, feats, dead, tagging, config)
    return np.where(dead, D, team).astype(np.int8)


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="TOML config (default: config/default.toml)")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value; repeatable",
    )
    parser.add_argument("--name", help="output name (default: team_name from the config)")
    parser.add_argument("--data-dir", help="data bundle root (default: ./data or MTAH_DATA_DIR)")
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR))
    parser.add_argument(
        "--holdout", action="store_true", help="also score the holdout labels (final decision only)"
    )
    parser.add_argument("--no-score", action="store_true", help="write and validate only")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.set)
    name = args.name or config["team_name"]

    tracking = load_tracking(args.data_dir)
    tagging = load_smart_tagging(args.data_dir)
    codes = run_pipeline(config, tracking, tagging)

    path = os.path.join(args.out_dir, f"{name}.csv")
    write_submission(path, codes)
    reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
    if reason:
        print(f"FAIL  {path}\n      {reason}")
        return 1
    results_copy = EVALUATION_DIR / "results" / f"{name}.csv"
    shutil.copyfile(path, results_copy)

    counts = "  ".join(f"{label} {int(np.sum(codes == k)):>6}" for k, label in enumerate(LABELS))
    stats = match_stats(codes)
    digest = sha256(path)
    print(f"\nOK    {os.path.relpath(path, ROOT)}   (copy in evaluation/results/)")
    print(f"      {counts}")
    print(f"      sha256 {digest}")
    print("\nfull match (no ground truth)")
    print(
        f"      possessions {stats['n_possessions']}  (H {stats['n_home']}, A {stats['n_away']})"
        f"   mean {stats['mean_s']:.1f} s   median {stats['median_s']:.1f} s"
        f"   under 2 s: {stats['short_under_2s']}"
    )
    print(
        f"      dead ball {100 * stats['dead_share']:.1f}%   "
        f"abstain (X) {100 * stats['abstain_share']:.1f}%"
    )

    scores = {}
    if not args.no_score:
        print("\nscores        spellF1  macroF1   poss (gt)    mean s (gt)")
        for gt_name, gt_path in ground_truths(args.holdout).items():
            s = score(codes, gt_path)
            scores[gt_name] = s
            print(
                f"  {gt_name:<10}  {s['spell_f1']:.4f}   {s['macro_f1']:.4f}   "
                f"{s['n_pred']:>4} ({s['n_gt']:>3})   {s['mean_s']:5.1f} ({s['gt_mean_s']:5.1f})"
            )
        if "dev" not in scores:
            print("  (human labels on this game are prohibited; see docs/RULE_UPDATE.md)")
        print(
            "\n  The public window has 17 possessions: use it to check correctness, not to choose"
            "\n  between versions. AI pseudo-label agreement is not accuracy."
            "\n  See docs/RULE_UPDATE.md."
        )

    def fmt(gt_name, key):
        return f"{scores[gt_name][key]:.4f}" if gt_name in scores else ""

    row = [
        datetime.date.today().isoformat(),
        name,
        _git("rev-parse", "--abbrev-ref", "HEAD"),
        _git("rev-parse", "--short", "HEAD"),
        " ".join(args.set) or os.path.relpath(config["_path"], ROOT),
        fmt("publik", "spell_f1"),
        fmt("dev", "spell_f1"),
        fmt("dev", "macro_f1"),
        str(stats["n_possessions"]),
        f"{stats['mean_s']:.1f}",
    ]
    print("\nexperiment log row (paste into the Sheet):")
    print("\t".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
