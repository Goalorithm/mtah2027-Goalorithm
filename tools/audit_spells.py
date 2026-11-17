"""Diagnose missed short spells on an external IDSSE benchmark; never train or tune a model."""

import argparse
import json
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np

from mtah.idsse import read_game
from mtah.io import read_codes, sha256
from mtah.paths import OUTPUTS_DIR, pm
from mtah.review import atomic_write


def duration_diagnostics(reference, prediction, threshold=0.5):
    runs = pm.valid_runs(reference)
    truth = pm.possession_spells(reference, runs)
    filled = pm.fill_abstentions(prediction, runs)
    guessed = pm.possession_spells(filled, runs)
    overlap = pm.spell_overlaps(truth, guessed)
    matched = pm.matched_gt_flags(overlap, len(truth), threshold)
    lengths = np.array([b - a for a, b, _ in truth]) / 25
    bands = {}
    for name, mask in (
        ("under_2s", lengths < 2),
        ("2_to_5s", (lengths >= 2) & (lengths < 5)),
        ("5s_and_longer", lengths >= 5),
    ):
        bands[name] = {
            "true_spells": int(mask.sum()),
            "matched": int(matched[mask].sum()),
            "recall": float(matched[mask].mean()) if mask.any() else None,
        }
    total = len(truth) + len(guessed)
    return {
        "n_true_spells": len(truth),
        "n_pred_spells": len(guessed),
        "matching_iou": threshold,
        "duration_recall": bands,
        "count_only_f1_upper_bound_with_fixed_counts": 2 * min(len(truth), len(guessed)) / total
        if total
        else 0.0,
        "spell_scores": pm.spell_scores(reference, filled, runs),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="outputs/idsse_benchmark_v3/report.json")
    parser.add_argument("--data-dir", default="data/external/idsse")
    parser.add_argument("--out", default="outputs/idsse_benchmark_v3/duration_audit.json")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("audit must remain under ignored outputs/")
    source = Path(args.benchmark)
    benchmark = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for game in benchmark["games"]:
        _, reference, _, _ = read_game(args.data_dir, game["game"])
        for model in game["models"]:
            name = Path(model["config"]).stem
            path = source.parent / f"{game['game']}_{model['profile']}_{name}.csv"
            if sha256(path) != model["prediction_sha256"]:
                raise ValueError("prediction no longer matches frozen benchmark")
            rows.append(
                {
                    "game": game["game"],
                    "profile": model["profile"],
                    "model": name,
                    **duration_diagnostics(reference, read_codes(path)),
                }
            )
    result = {
        "purpose": "post-hoc external error diagnosis; no fitting or model selection",
        "benchmark_sha256": sha256(source),
        "bar_ath_labels_read": False,
        "target_spell_f1": 0.8,
        "rows": rows,
    }
    atomic_write(out, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
