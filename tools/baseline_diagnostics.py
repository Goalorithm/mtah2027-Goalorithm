"""Record five baseline mismatch intervals per task on the official public correctness example.

Not a training/tuning input: run only after candidate configurations are frozen. Never reads video
or creates human labels. The report is local and is not imported by any model/selection tool.
"""

import argparse
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np

from mtah import A, D, H, X
from mtah.io import read_codes, sha256
from mtah.paths import OUTPUTS_DIR, PUBLIC_GT, ROOT
from mtah.review import atomic_write
from mtah.rule_support import segments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/baseline_diagnostics")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: diagnostics must stay under ignored outputs/")
        return 1
    reference = read_codes(PUBLIC_GT)
    baseline = read_codes(ROOT / "evaluation/results/baseline_smooth50.csv")
    covered = reference != X
    masks = {
        "tech2_dead_ball": covered & ((reference == D) != (baseline == D)),
        "tech3_possession": np.isin(reference, [H, A])
        & np.isin(baseline, [H, A])
        & (reference != baseline),
    }
    report = {
        "scope": "official public example; correctness diagnostics only",
        "used_for_training_or_tuning": False,
        "configs_frozen_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in sorted((ROOT / "config").glob("exp_*.toml"))
        },
        "reference_sha256": sha256(PUBLIC_GT),
        "diagnostics": {},
    }
    for task, mask in masks.items():
        runs = segments(mask)
        report["diagnostics"][task] = {
            "mismatch_frames": int(mask.sum()),
            "mismatch_intervals": len(runs),
            "first_five_intervals": [
                {"start": int(a), "stop_exclusive": int(b), "frames": int(b - a)}
                for a, b in runs[:5]
            ],
            "interpretation": (
                "Missing tracking is not a referee decision; outside/restart evidence is required."
                if task.startswith("tech2")
                else "Nearest-player voting does not establish controlled possession; "
                "brief touches and loose balls need persistent ownership."
            ),
        }
        print(f"{task}: {len(runs[:5])} public correctness mismatch intervals recorded")
    atomic_write(
        out / "report.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
