"""Error audits must expose hidden short possessions rather than reward merging them."""

import numpy as np
from audit_spells import duration_diagnostics

from mtah import A, D, H, X


def test_merged_short_possessions_have_low_recall_despite_a_matched_long_interval():
    reference = np.array([H] * 20 + [A] * 10 + [H] * 150 + [D] * 10)
    prediction = np.array([H] * 180 + [D] * 10)
    result = duration_diagnostics(reference, prediction)
    assert result["n_true_spells"] == 3
    assert result["n_pred_spells"] == 1
    assert result["count_only_f1_upper_bound_with_fixed_counts"] == 0.5
    assert result["duration_recall"]["under_2s"] == {"true_spells": 2, "matched": 0, "recall": 0.0}
    assert result["duration_recall"]["5s_and_longer"]["recall"] == 1


def test_perfect_predictions_keep_unknown_gaps_and_match_each_half_separately():
    reference = np.array([H] * 30 + [X] + [H] * 30 + [A] * 30)
    result = duration_diagnostics(reference, reference)
    assert result["n_true_spells"] == result["n_pred_spells"] == 3
    assert result["duration_recall"]["under_2s"]["matched"] == 3
    assert result["spell_scores"]["mean_f1"] == 1
