"""Boundary loss must favor correct transitions and ignore padding; dropout must flag missing."""

import pytest

torch = pytest.importorskip("torch")

from temporal_losses import augment_features, transition_loss  # noqa: E402


def test_correct_transition_scores_better_than_constant_prediction():
    labels = torch.tensor([[0, 0, 1, 1]])
    correct = torch.tensor([[[8.0, 8, -8, -8], [-8.0, -8, 8, 8], [-8.0] * 4]])
    constant = torch.tensor([[[8.0] * 4, [-8.0] * 4, [-8.0] * 4]])
    assert transition_loss(correct, labels) < transition_loss(constant, labels)


def test_unknown_separator_has_no_cross_period_boundary_gradient():
    labels = torch.tensor([[0, 0, 3, 1, 1]])
    first = torch.randn((1, 3, 5), requires_grad=True)
    second = first.detach().clone()
    second[:, :, 2] = 1000
    torch.testing.assert_close(transition_loss(first, labels), transition_loss(second, labels))
    transition_loss(first, labels).backward()
    assert torch.isfinite(first.grad).all()
    assert (first.grad[:, :, 2] == 0).all()


def test_all_padding_returns_finite_zero_loss():
    logits = torch.randn((2, 3, 5), requires_grad=True)
    result = transition_loss(logits, torch.full((2, 5), 3))
    assert result.item() == 0
    result.backward()
    assert (logits.grad == 0).all()


def test_feature_dropout_updates_masks_without_mutating_input():
    values = torch.zeros((2, 6, 8))
    values[:, :3] = 4
    original = values.clone()
    cfg = {"feature_dropout": 1.0, "motion_group_dropout": 0.0, "feature_noise_std": 0.0}
    result = augment_features(values, ["H_distance", "A_distance", "ball_speed"], cfg)
    assert (result[:, :3] == 0).all() and (result[:, 3:] == 1).all()
    torch.testing.assert_close(values, original)
