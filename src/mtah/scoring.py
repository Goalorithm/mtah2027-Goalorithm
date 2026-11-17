"""Scores and sanity statistics, all computed with the organisers' metric code.

Two kinds of numbers:
  * score(): against a ground truth -- the public 5 minutes, or the team's validation labels
  * match_stats(): on the full match, with no ground truth at all. Not a score, an alarm:
    a change that suddenly doubles the number of possessions is wrong, whatever it does on
    5 minutes of public ground truth.
"""

import numpy as np

from mtah import A, D, H, X
from mtah.io import read_codes
from mtah.paths import FPS, PUBLIC_GT, match_runs, pm


def score(pred_codes, gt_path):
    """The headline metrics of one prediction against one ground truth file."""
    gt_codes = read_codes(gt_path)
    result = pm.evaluate(gt_codes, np.asarray(pred_codes, dtype=np.int8), bootstrap_n=0)
    return {
        "spell_f1": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "n_pred": result["n_pred_spells"],
        "n_gt": result["n_gt_spells"],
        "mean_s": result["mean_spell_duration_s"],
        "gt_mean_s": result["gt_mean_spell_duration_s"],
    }


def ground_truths(include_holdout=False):
    """Only the official public correctness example is a permitted BAR-ATH reference.

    Upstream rules at f8f2629 prohibit human labels on this game for validation too.
    Keep the parameter for caller compatibility; it never unlocks prohibited local labels.
    Pseudo-labels are not ground truth and must not be discovered as validation references.
    """
    return {"publik": str(PUBLIC_GT)}


def match_stats(codes):
    """Possession structure over both halves of the full match, no ground truth needed."""
    codes = np.asarray(codes, dtype=np.int8)
    runs = match_runs()
    in_match = np.zeros(len(codes), dtype=bool)
    for start, stop in runs:
        in_match[start:stop] = True
    filled = pm.fill_abstentions(codes, runs)
    spells = pm.possession_spells(filled, runs)
    lengths = np.array([b - a for a, b, _ in spells], dtype=float) / FPS
    graded = int(in_match.sum())
    return {
        "n_possessions": len(spells),
        "n_home": sum(1 for *_, k in spells if k == H),
        "n_away": sum(1 for *_, k in spells if k == A),
        "mean_s": float(lengths.mean()) if len(lengths) else 0.0,
        "median_s": float(np.median(lengths)) if len(lengths) else 0.0,
        "short_under_2s": int(np.sum(lengths < 2.0)),
        "dead_share": float(np.sum(codes[in_match] == D)) / graded,
        "abstain_share": float(np.sum(codes[in_match] == X)) / graded,
    }
