"""Full-match sanity report for one or more results, no ground truth needed.

  python tools/match_report.py baseline_smooth50 exp_x

For each result: number of possessions, their durations, the share of dead ball and of X, and
(when the match data is available) how many Smart Tagging set pieces sit next to a predicted
stoppage. Also writes outputs/durations.png, a histogram of possession durations per result.

Read it as an alarm, not a score: if a change triples the possessions or erases half the
stoppages, something is wrong, whatever the public 5 minutes say.
"""

import argparse
import os
import sys

import _src_path  # noqa: F401  (finds src/ without pip install -e .)
import numpy as np

from mtah import D
from mtah.io import read_codes
from mtah.paths import FPS, OUTPUTS_DIR, match_runs, pm
from mtah.scoring import match_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timeline import resolve  # noqa: E402

SET_PIECE_SLACK = 2 * FPS  # a restart within 2 s of a predicted stoppage counts as explained


def set_piece_agreement(codes, tagging):
    """(n_set_pieces, n_with_a_stoppage_nearby) over the graded halves."""
    pieces = [t for t in tagging if (t["code"] or "").lower() == "set pieces"]
    dead = codes == D
    hits = 0
    for piece in pieces:
        low = max(0, piece["start_frame"] - SET_PIECE_SLACK)
        high = min(len(codes), piece["end_frame"] + SET_PIECE_SLACK)
        hits += bool(dead[low:high].any())
    return len(pieces), hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="result names in evaluation/results/ or paths")
    args = parser.parse_args(argv)

    try:
        from mtah.load import load_smart_tagging

        tagging = load_smart_tagging()
    except OSError:
        tagging = []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.arange(0, 61, 2)
    header = (
        f"{'result':<26}{'poss':>6}{'H':>5}{'A':>5}{'mean s':>8}{'median':>8}"
        f"{'<2 s':>6}{'D %':>7}{'X %':>7}"
    )
    if tagging:
        header += f"{'set pieces near D':>19}"
    print(header)
    for name in args.results:
        codes = read_codes(resolve(name))
        stats = match_stats(codes)
        line = (
            f"{name[:25]:<26}{stats['n_possessions']:>6}{stats['n_home']:>5}"
            f"{stats['n_away']:>5}{stats['mean_s']:>8.1f}{stats['median_s']:>8.1f}"
            f"{stats['short_under_2s']:>6}{100 * stats['dead_share']:>7.1f}"
            f"{100 * stats['abstain_share']:>7.1f}"
        )
        if tagging:
            total, hits = set_piece_agreement(codes, tagging)
            line += f"{hits:>12} / {total:<4}"
        print(line)
        runs = match_runs()
        spells = pm.possession_spells(pm.fill_abstentions(codes, runs), runs)
        durations = np.clip([(b - a) / FPS for a, b, _ in spells], 0, 60)
        ax.hist(durations, bins=bins, histtype="step", linewidth=1.5, label=name)

    ax.set_xlabel("possession duration (s, clipped at 60)")
    ax.set_ylabel("possessions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(str(OUTPUTS_DIR), "durations.png")
    os.makedirs(str(OUTPUTS_DIR), exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
