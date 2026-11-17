"""Draw what the tracking sees on given frames: players by team, the ball, the nearest player.

  python tools/frame_view.py 15880 15890 15900 15910
  python tools/frame_view.py --from 15880 --to 15980 --step 20 --pred baseline_smooth50

Each panel's title shows the frame, the video time, and the ground-truth / predicted label when
--gt / --pred are given. Output: outputs/frames_<first>_<last>.png (or --out).
Drawn from tracking only. Do not check predictions against the video: the organisers'
rules clarification of 3 November 2026 prohibits it (see docs/RULE_UPDATE.md).
"""

import argparse
import math
import os
import sys

import _src_path  # noqa: F401  (finds src/ without pip install -e .)
import numpy as np

from mtah import LABELS
from mtah import features as feats_module
from mtah.io import read_codes
from mtah.load import load_tracking
from mtah.paths import FPS, OUTPUTS_DIR, PITCH, PUBLIC_GT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timeline import resolve  # noqa: E402

TEAM_COLOURS = ("#c0392b", "#2471a3")  # home, away


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", nargs="*", type=int)
    parser.add_argument("--from", dest="first", type=int)
    parser.add_argument("--to", dest="last", type=int)
    parser.add_argument("--step", type=int, default=FPS)
    parser.add_argument("--gt", default=str(PUBLIC_GT))
    parser.add_argument("--pred", help="result name or path")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    frames = list(args.frames)
    if args.first is not None and args.last is not None:
        frames += list(range(args.first, args.last + 1, args.step))
    if not frames:
        parser.error("give frames, or --from/--to")
    if len(frames) > 24:
        parser.error("at most 24 panels; widen --step")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import utils as starter_utils  # starter/utils.py, on the path via mtah.paths

    tracking = load_tracking(verbose=False)
    feats = feats_module.build(tracking)
    gt = read_codes(args.gt) if args.gt else None
    pred = read_codes(resolve(args.pred)) if args.pred else None

    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    length, width = PITCH
    for ax, frame in zip(axes.flat, frames, strict=False):
        starter_utils.draw_pitch(ax, length, width)
        xy = tracking.players[frame]
        for team in (0, 1):
            mask = (tracking.slot_team == team) & ~np.isnan(xy[:, 0])
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=28,
                color=TEAM_COLOURS[team],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        if feats["ball_seen"][frame]:
            bx, by = tracking.ball[frame]
            ax.scatter([bx], [by], s=22, color="white", edgecolor="black", zorder=4)
            ax.add_patch(
                plt.Circle(
                    (bx, by),
                    float(np.nan_to_num(feats["nearest_dist"][frame])),
                    fill=False,
                    linestyle=":",
                    color="white",
                    zorder=2,
                )
            )
        seconds = frame / FPS
        title = f"{frame}  ({int(seconds // 60)}:{seconds % 60:04.1f})"
        if gt is not None:
            title += f"  gt {LABELS[gt[frame]]}"
        if pred is not None:
            title += f"  pred {LABELS[pred[frame]]}"
        if not feats["ball_seen"][frame]:
            title += "  [no ball]"
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    for ax in list(axes.flat)[len(frames) :]:
        ax.set_axis_off()
    fig.tight_layout()

    out = args.out or os.path.join(str(OUTPUTS_DIR), f"frames_{frames[0]}_{frames[-1]}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
