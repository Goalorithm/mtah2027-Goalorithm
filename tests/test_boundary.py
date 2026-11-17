"""Handover models must respect run boundaries, exact ablation, symmetry and forbidden paths."""

import numpy as np

from mtah import A, D, H, X
from mtah.boundary import boundary_targets, decode_boundaries
from mtah.supervised import decode_probabilities


def model():
    return {
        "class_prior": [1 / 3] * 3,
        "decoder_scale": 0.3,
        "initial": [0.5, 0.5, 0],
        "transition": [[0.999, 0.001, 0], [0.001, 0.999, 0], [0.001, 0.001, 0.998]],
        "boundary_prior": [0.005, 0.005, 0.99],
        "boundary_weight": 2.0,
    }


def test_boundary_labels_only_mark_direct_team_changes_and_reset_at_unknown_gaps():
    y = np.array([H] * 10 + [A] * 5 + [D] * 5 + [X] * 2 + [H] * 10 + [A] * 5)
    result = boundary_targets(y, [(0, 20), (22, 37)], radius=2)
    assert (result[8:13] == A).all()
    assert (result[15:20] == D).all()
    assert (result[20:22] == X).all()
    assert (result[22:25] == D).all()
    swapped = np.where(y == H, A, np.where(y == A, H, y))
    expected = np.where(result == H, A, np.where(result == A, H, result))
    np.testing.assert_array_equal(boundary_targets(swapped, [(0, 20), (22, 37)], 2), expected)


def test_zero_boundary_weight_exactly_recovers_existing_viterbi():
    rng = np.random.default_rng(8)
    probabilities = rng.dirichlet([4, 3, 1], 120)
    events = rng.dirichlet([1, 1, 10], 120)
    cfg = model()
    cfg["boundary_weight"] = 0
    runs = [(5, 50), (60, 115)]
    np.testing.assert_array_equal(
        decode_boundaries(probabilities, events, cfg, runs),
        decode_probabilities(probabilities, cfg, runs),
    )


def test_directed_event_evidence_can_recover_a_short_weak_emission_possession():
    probabilities = np.tile([0.7, 0.29, 0.01], (100, 1))
    probabilities[45:55] = [0.44, 0.55, 0.01]
    events = np.tile([0.001, 0.001, 0.998], (100, 1))
    events[45] = [0.001, 0.998, 0.001]
    events[55] = [0.998, 0.001, 0.001]
    cfg = model()
    assert (decode_probabilities(probabilities, cfg, [(0, 100)]) == H).all()
    result = decode_boundaries(probabilities, events, cfg, [(0, 100)])
    assert (result[:45] == H).all() and (result[45:55] == A).all() and (result[55:] == H).all()
    swapped = decode_boundaries(probabilities[:, [A, H, D]], events[:, [A, H, D]], cfg, [(0, 100)])
    np.testing.assert_array_equal(swapped, 1 - result)


def test_event_model_cannot_cross_a_forbidden_transition():
    probabilities = np.tile([0.5, 0.49, 0.01], (50, 1))
    events = np.tile([0.001, 0.998, 0.001], (50, 1))
    cfg = model()
    cfg["transition"] = np.eye(3).tolist()
    cfg["initial"] = [1, 0, 0]
    assert (decode_boundaries(probabilities, events, cfg, [(0, 50)]) == H).all()


def test_dense_handover_sampling_keeps_positive_frames_missed_by_a_regular_stride():
    from train_dense_boundaries import dense_sample

    target = np.full(100, D)
    target[[3, 4, 5]] = A
    target[[17, 18, 19]] = H
    target[40:50] = X
    selected = dense_sample(target, 10)
    assert {3, 4, 5, 17, 18, 19} <= set(selected)
    assert not np.isin(selected, np.arange(40, 50)).any()
    assert all(i % 10 == 0 for i in selected if target[i] == D)
