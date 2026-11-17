"""Draw ground truth, predictions and the raw signals side by side over a range of frames.

The fastest way to see WHERE a version goes wrong: a missed stoppage, a possession split in two,
a hand-over that comes late.

  python tools/timeline.py --start 15500 --end 18684 --pred baseline_smooth50 nama_tim
  python tools/timeline.py --window publik --pred baseline_smooth50
  python tools/timeline.py --start 30000 --end 37499 --gt labels/validasi_dev.csv --pred exp_x

--pred takes names in evaluation/results/ (without .csv) or paths. Rows for the ball signal and
the Smart Tagging are added automatically when the match data is available.
Output: outputs/timeline_<start>_<end>.png (or --out).
"""

import argparse
import os
import sys

import _src_path  # noqa: F401  (finds src/ without pip install -e .)
import numpy as np

from mtah import A, D, H, X
from mtah.io import read_codes
from mtah.paths import EVALUATION_DIR, FPS, OUTPUTS_DIR, PUBLIC_GT

COLOURS = {H: "#c0392b", A: "#2471a3", D: "#7f8c8d", X: "#f4f6f7"}
NAMED_WINDOWS = {"publik": (11185, 18684)}


def resolve(name_or_path):
    if os.path.isfile(name_or_path):
        return name_or_path
    candidate = EVALUATION_DIR / "results" / f"{name_or_path}.csv"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"no such file or result name: {name_or_path}")


def spans(codes, start, end):
    """[(first_frame, length, code)] constant runs of codes[start:end+1]."""
    window = codes[start : end + 1]
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    return [
        (start + int(a), int(b - a), int(window[a]))
        for a, b in zip(bounds[:-1], bounds[1:], strict=False)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int, help="inclusive")
    parser.add_argument("--window", choices=sorted(NAMED_WINDOWS))
    parser.add_argument("--gt", default=str(PUBLIC_GT), help="ground truth CSV")
    parser.add_argument("--pred", nargs="*", default=[], help="result names or paths")
    parser.add_argument("--no-data", action="store_true", help="skip the tracking rows")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    if args.window:
        args.start, args.end = NAMED_WINDOWS[args.window]
    if args.start is None or args.end is None or args.end <= args.start:
        parser.error("give --start and --end (end > start), or --window")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = [("ground truth", read_codes(args.gt))]
    rows += [(name, read_codes(resolve(name))) for name in args.pred]

    signals = []
    if not args.no_data:
        try:
            from mtah import features as feats_module
            from mtah.load import load_smart_tagging, load_tracking

            tracking = load_tracking(verbose=False)
            feats = feats_module.build(tracking)
            signals.append(("ball seen", feats["ball_seen"].astype(np.int8)))
            signals.append(("nearest team", feats["nearest_team"]))
            tagging = load_smart_tagging()
        except OSError:
            tagging = []
            print("match data not found: drawing labels only")
    else:
        tagging = []

    n_rows = len(rows) + len(signals) + (1 if tagging else 0)
    fig, ax = plt.subplots(figsize=(16, 0.6 * n_rows + 1.6))
    y = n_rows - 1
    labels = []
    for name, codes in rows:
        for first, length, code in spans(codes, args.start, args.end):
            ax.broken_barh([(first, length)], (y + 0.1, 0.8), color=COLOURS[code], linewidth=0)
        labels.append((y + 0.5, name))
        y -= 1
    for name, values in signals:
        window = values[args.start : args.end + 1]
        frames = np.arange(args.start, args.end + 1)
        if name == "ball seen":
            ax.fill_between(
                frames, y + 0.1, y + 0.1 + 0.8 * window, color="#27ae60", step="mid", linewidth=0
            )
        else:
            for code, colour in ((H, COLOURS[H]), (A, COLOURS[A])):
                hit = frames[window == code]
                ax.scatter(hit, np.full(len(hit), y + 0.5), s=2, color=colour, marker="|")
        labels.append((y + 0.5, name))
        y -= 1
    if tagging:
        for item in tagging:
            if item["end_frame"] < args.start or item["start_frame"] > args.end:
                continue
            ax.broken_barh(
                [(item["start_frame"], item["end_frame"] - item["start_frame"])],
                (y + 0.1, 0.8),
                facecolor="none",
                edgecolor="#8e44ad",
                linewidth=0.8,
            )
            ax.text(
                max(item["start_frame"], args.start),
                y + 0.5,
                item["code"][:14].lower(),
                fontsize=6,
                va="center",
                clip_on=True,
            )
        labels.append((y + 0.5, "smart tagging"))

    ax.set_yticks([p for p, _ in labels], [n for _, n in labels])
    ax.set_xlim(args.start, args.end + 1)
    ax.set_ylim(0, n_rows)
    ticks = ax.get_xticks()
    ax.set_xticks(
        ticks, [f"{int(t)}\n{int(t) // FPS // 60}:{int(t) // FPS % 60:02d}" for t in ticks]
    )
    ax.set_xlim(args.start, args.end + 1)
    ax.set_xlabel("video frame / video time (mm:ss)")
    ax.legend(
        handles=[
            Patch(color=COLOURS[k], label=n)
            for k, n in ((H, "H home"), (A, "A away"), (D, "D dead"), (X, "X none"))
        ],
        loc="upper right",
        bbox_to_anchor=(1, 1.15),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()

    out = args.out or os.path.join(str(OUTPUTS_DIR), f"timeline_{args.start}_{args.end}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
