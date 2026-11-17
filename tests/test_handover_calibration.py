"""Calibration must change only legal H/A odds and preserve exact zero-bias behavior."""

import numpy as np
import pytest
from calibrate_handover import calibrated_transition

from mtah import A, D, H


def transition():
    return np.array([[0.995, 0.002, 0.003], [0.002, 0.995, 0.003], [0.003, 0.003, 0.994]])


@pytest.mark.parametrize("bias", [-2.0, 2.0])
def test_calibration_preserves_dead_rates_and_symmetry_and_shifts_only_live_odds(bias):
    original = transition()
    result = calibrated_transition(original, bias)
    np.testing.assert_array_equal(original, transition())
    np.testing.assert_array_equal(result[:, D], original[:, D])
    np.testing.assert_array_equal(result[D], original[D])
    np.testing.assert_allclose(result.sum(axis=1), 1)
    np.testing.assert_array_equal(result, result[np.ix_([A, H, D], [A, H, D])])
    assert result[H, A] / result[H, H] == pytest.approx(
        original[H, A] / original[H, H] * np.exp(bias)
    )


def test_zero_bias_is_exact_and_does_not_modify_input():
    original = transition()
    np.testing.assert_array_equal(calibrated_transition(original, 0), original)


def test_forbidden_live_transitions_remain_forbidden():
    np.testing.assert_array_equal(calibrated_transition(np.eye(3), 3), np.eye(3))


def test_nonfinite_bias_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        calibrated_transition(transition(), np.nan)
