"""Where things live, and access to the organisers' modules.

The organisers' code (games.py, evaluation/, starter/) is kept byte-for-byte as upstream ships it,
so `git pull upstream main` never conflicts. This module puts it on the import path once, so the
rest of the package can simply `from mtah.paths import games, pm`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "evaluation"
STARTER_DIR = ROOT / "starter"
CONFIG_DIR = ROOT / "config"
CACHE_DIR = ROOT / "cache"
LABELS_DIR = ROOT / "labels"
OUTPUTS_DIR = ROOT / "outputs"
PUBLIC_GT = EVALUATION_DIR / "gt" / "BAR-ATH_possession_gt_public.csv"

for _path in (ROOT, EVALUATION_DIR, STARTER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


import possession_metrics as pm  # noqa: E402, F401  (re-exported: from mtah.paths import pm)

import games  # noqa: E402

N_FRAMES = games.GAMES[games.GAME]["n_video_frames"]
FPS = games.GAMES[games.GAME]["fps"]
PITCH = games.GAMES[games.GAME]["pitch"]  # (length, width) in metres


def match_runs():
    """Half-open [start, stop) video-frame spans of the two halves: the frames that are graded.

    The same spans the scorer derives from the ground truth, built here from games.py so the
    full-match statistics can be computed without any ground truth at all.
    """
    return [(first, last + 1) for _, first, last in games.half_video_blocks(games.GAME)]
