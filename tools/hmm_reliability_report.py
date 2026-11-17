"""Compare HMM evidence expiry and prediction changes without opening reference labels.

This is a regression/behaviour report, not an accuracy evaluation or candidate selector.
"""

import argparse
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np

from mtah import LABELS, D
from mtah.config import load_config
from mtah.hmm import ball_features, outside_signal, sequence_runs
from mtah.io import read_codes, sha256
from mtah.load import load_tracking
from mtah.paths import CONFIG_DIR, N_FRAMES, OUTPUTS_DIR, pm
from mtah.review import atomic_write
from mtah.scoring import match_stats


def summarize(before, after, history, before_cfg, after_cfg, runs=None):
    before, after = np.asarray(before), np.asarray(after)
    if before.ndim != 1 or before.shape != after.shape or len(before) != len(history["ball_gap"]):
        raise ValueError("prediction and history lengths must match")
    if (
        not np.isin(before, range(len(LABELS))).all()
        or not np.isin(after, range(len(LABELS))).all()
    ):
        raise ValueError("unknown prediction labels")
    horizon = after_cfg["outside_memory_frames"]
    old_signal = outside_signal(
        history, before_cfg["margin_m"], before_cfg.get("outside_memory_frames")
    )
    new_signal = outside_signal(history, after_cfg["margin_m"], horizon)
    graded = np.zeros(len(before), dtype=bool)
    for start, stop in sequence_runs(len(before), runs):
        graded[start:stop] = True
    age = history["ball_gap"]
    expired = graded & (old_signal > 0) & (age >= horizon) & (age > 0)
    changed = graded & (before != after)
    return {
        "purpose": "evidence reliability and regression diagnostics; not accuracy",
        "accuracy_measured": False,
        "selected_final_model": None,
        "outside_memory_frames": horizon,
        "graded_frames": int(graded.sum()),
        "changed_prediction_frames": int(changed.sum()),
        "ungraded_predictions_unchanged": bool(np.array_equal(before[~graded], after[~graded])),
        "changed_label_pairs": {
            f"{a}_to_{b}": int(np.sum(changed & (before == i) & (after == j)))
            for i, a in enumerate(LABELS)
            for j, b in enumerate(LABELS)
            if np.any(changed & (before == i) & (after == j))
        },
        "expired_outside_observation_frames": int(expired.sum()),
        "outside_evidence_remaining_on_expired_frames": int(np.count_nonzero(new_signal[expired])),
        "dead_predictions_on_expired_frames_before": int(np.sum(expired & (before == D))),
        "dead_predictions_on_expired_frames_after": int(np.sum(expired & (after == D))),
        "warning": (
            "Other evidence and temporal context may still support D after geometry expires."
        ),
        "changed_hmm_parameters": {
            key: {"before": before_cfg.get(key), "after": after_cfg.get(key)}
            for key in sorted(before_cfg.keys() | after_cfg.keys())
            if before_cfg.get(key) != after_cfg.get(key)
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", default=str(OUTPUTS_DIR / "exp_hmm_v1_compat.csv"))
    parser.add_argument("--after", default=str(OUTPUTS_DIR / "exp_hmm_v2.csv"))
    parser.add_argument("--before-config", default=str(CONFIG_DIR / "exp_hmm_v1.toml"))
    parser.add_argument("--after-config", default=str(CONFIG_DIR / "exp_hmm_v2.toml"))
    parser.add_argument(
        "--out", default=str(OUTPUTS_DIR / "tech1_review" / "hmm_v2_reliability.json")
    )
    args = parser.parse_args(argv)
    try:
        codes = []
        for path in (args.before, args.after):
            reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
            if reason:
                raise ValueError(f"{path}: {reason}")
            codes.append(read_codes(path))
        before_cfg = load_config(args.before_config)["hmm"]
        after_cfg = load_config(args.after_config)["hmm"]
        tracking = load_tracking(verbose=False)
        history = ball_features(
            tracking.ball, after_cfg["max_gap"], after_cfg["max_interp_speed_mps"]
        )
        report = summarize(*codes, history, before_cfg, after_cfg)
        report["prediction_statistics_before"] = match_stats(codes[0])
        report["prediction_statistics_after"] = match_stats(codes[1])
        report["input_sha256"] = {
            key: sha256(value)
            for key, value in {
                "before_prediction": args.before,
                "after_prediction": args.after,
                "before_config": args.before_config,
                "after_config": args.after_config,
            }.items()
        }
        atomic_write(args.out, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps(report, indent=2))
        print(f"-> {Path(args.out)}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
