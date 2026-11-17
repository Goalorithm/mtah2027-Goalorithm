"""Check that this laptop is ready: data in place, cache built, pipeline matches the organisers.

  python tools/cek_setup.py

Run it once after setup, and again whenever something feels off. Every line should say OK.
The key check: with config/default.toml our pipeline must reproduce the labels of the organisers'
published baseline_smooth50.csv on every frame. If it does, the loader, the frame alignment and
the features are correct, and every later difference is a real change of method.
"""

import os
import sys
import tempfile
import time

import _src_path  # noqa: F401  (finds src/ without pip install -e .)
import numpy as np

from mtah.config import load_config
from mtah.io import read_codes, sha256, write_submission
from mtah.paths import EVALUATION_DIR, PUBLIC_GT, games
from mtah.scoring import score

BASELINE = EVALUATION_DIR / "results" / "baseline_smooth50.csv"
EXPECTED_PUBLIC_SPELL_F1 = 0.3256


def check(ok, message, fix=""):
    print(f"{'OK  ' if ok else 'FAIL'}  {message}")
    if not ok and fix:
        print(f"      -> {fix}")
    return ok


def main():
    print(f"python {sys.version.split()[0]}")
    if not check(
        sys.version_info[:2] == (3, 11),
        "Python 3.11",
        "install Python 3.11 and recreate .venv (docs/SETUP.md, step 1)",
    ):
        return 1

    try:
        games.require(["atd_meta", "atd_raw", "smart_tagging"])
        check(True, f"match data found in {games.DATA_DIR}")
    except OSError as error:
        check(False, "match data", str(error).splitlines()[0] + " (docs/SETUP.md, step 4)")
        return 1

    from mtah.load import load_smart_tagging, load_tracking
    from mtah.predict import run_pipeline

    began = time.time()
    tracking = load_tracking()
    check(
        True,
        f"tracking loaded in {time.time() - began:.0f} s "
        f"({tracking.n_frames} frames, {tracking.n_slots} slots, "
        f"ball seen {100 * tracking.ball_seen.mean():.1f}%)",
    )
    tagging = load_smart_tagging()
    check(len(tagging) > 0, f"smart tagging: {len(tagging)} instances")

    codes = run_pipeline(load_config(), tracking, tagging)
    expected = read_codes(BASELINE)
    differ = int(np.sum(codes != expected))
    same = check(
        differ == 0,
        f"default config reproduces baseline_smooth50 ({differ} frames differ)",
        "the loader or features disagree with starter/baseline.py; tell Tech 1",
    )

    with tempfile.TemporaryDirectory() as folder:
        first = write_submission(os.path.join(folder, "a.csv"), codes)
        second = write_submission(
            os.path.join(folder, "b.csv"), run_pipeline(load_config(), tracking, tagging)
        )
        check(sha256(first) == sha256(second), "two runs give byte-identical files")

    public = score(codes, PUBLIC_GT)["spell_f1"]
    check(
        abs(public - EXPECTED_PUBLIC_SPELL_F1) < 1e-4,
        f"public spellF1 {public:.4f} (expected {EXPECTED_PUBLIC_SPELL_F1})",
    )
    print("\nReady." if same else "\nNot ready: fix the FAIL lines above.")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
