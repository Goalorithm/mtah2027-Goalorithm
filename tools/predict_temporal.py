"""Regenerate the final TCN submission from the committed checkpoints, on CPU only.

    python tools/predict_temporal.py          # -> outputs/Goalorithm.csv

The TCN and its tree teacher were fitted on external IDSSE matches only (models/README.md).
evaluate_temporal.py loads frozen models from outputs/, so this script stages the committed
copies there, hides any GPU (CUDA/TF32 arithmetic can differ from CPU), runs the competition
inference and copies the result to the team's submission name.
"""

import os
import shutil
import sys
from pathlib import Path

import _src_path  # noqa: F401

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import N_FRAMES, OUTPUTS_DIR, ROOT, pm

MODELS = ROOT / "models"
TEMPORAL_FILES = (
    "model.json",
    "parent.json",
    "test_freeze.json",
    "training_report.json",
    "weights.pt",
)
STAGE_NAME = "temporal_final"


def stage_checkpoints(models=MODELS, outputs=OUTPUTS_DIR):
    """Copy the committed teacher and TCN into outputs/; never overwrite a different teacher."""
    teacher_source = Path(models) / "supervised_joint" / "model.json"
    teacher_target = Path(outputs) / "supervised_joint" / "model.json"
    if teacher_target.exists() and sha256(teacher_target) != sha256(teacher_source):
        raise ValueError(
            f"{teacher_target} differs from the committed teacher; move it away and rerun"
        )
    teacher_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(teacher_source, teacher_target)
    stage = Path(outputs) / STAGE_NAME
    stage.mkdir(parents=True, exist_ok=True)
    for name in TEMPORAL_FILES:
        shutil.copyfile(Path(models) / "temporal" / name, stage / name)
    return stage


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported
    stage = stage_checkpoints()
    import evaluate_temporal

    status = evaluate_temporal.main(["--model-dir", str(stage), "--mode", "competition"])
    if status:
        return status
    produced = OUTPUTS_DIR / f"exp_{STAGE_NAME}.csv"
    final = OUTPUTS_DIR / f"{load_config()['team_name']}.csv"
    shutil.copyfile(produced, final)
    reason = pm.validate_submission(pm.read_label_csv(final), N_FRAMES)
    if reason:
        print(f"FAIL  {final}\n      {reason}")
        return 1
    print(f"OK    {os.path.relpath(final, ROOT)}\n      sha256 {sha256(final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
