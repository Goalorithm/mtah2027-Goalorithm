"""Diagnostic counts must respect graded spans and never equate expiry with a live label."""

import hmm_reliability_report as report
import numpy as np

from mtah import D, H, X
from mtah.hmm import ball_features


def test_report_separates_expired_evidence_from_decoded_states_and_ungraded_frames():
    before = np.full(50, X, dtype=np.int8)
    before[10:30] = D
    after = before.copy()
    after[20:30] = H
    ball = np.full((50, 2), np.nan)
    ball[10] = [-3, 30]
    history = ball_features(ball, 0, 45, runs=[(10, 30)])
    result = report.summarize(
        before,
        after,
        history,
        {"margin_m": 0.5},
        {"margin_m": 0.5, "outside_memory_frames": 5},
        runs=[(10, 30)],
    )
    assert result["graded_frames"] == 20
    assert result["expired_outside_observation_frames"] == 15
    assert result["outside_evidence_remaining_on_expired_frames"] == 0
    assert result["dead_predictions_on_expired_frames_before"] == 15
    assert result["dead_predictions_on_expired_frames_after"] == 5
    assert result["changed_label_pairs"] == {"D_to_H": 10}
    assert result["ungraded_predictions_unchanged"]
    assert not result["accuracy_measured"] and result["selected_final_model"] is None
