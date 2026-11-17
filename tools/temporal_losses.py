"""Training-only boundary consistency and feature dropout; no changes to evaluation inputs."""

import torch
from torch.nn import functional as F


def transition_loss(logits, labels, positive_weight=12.0):
    """Probability of adjacent state changes, excluding every unknown/padding boundary."""
    valid = (labels[:, 1:] != 3) & (labels[:, :-1] != 3)
    probabilities = logits.float().softmax(dim=1)
    change = 1 - (probabilities[:, :, 1:] * probabilities[:, :, :-1]).sum(dim=1)
    target = (labels[:, 1:] != labels[:, :-1]).float()
    weights = torch.where(target > 0, positive_weight, 1.0) * valid
    error = F.binary_cross_entropy(change.clamp(1e-5, 1 - 1e-5), target, reduction="none")
    return (error * weights).sum() / weights.sum().clamp_min(1)


def augment_features(values, names, cfg):
    """Drop channels for a whole training crop; encoded mask channels follow dropped values."""
    n = len(names)
    result = values.clone()
    shape = (len(values), n, 1)
    dropped = torch.rand(shape, device=values.device) < cfg["feature_dropout"]
    motion = torch.tensor(
        ["ball" in name or name in ("ball_turn", "ball_acceleration") for name in names],
        device=values.device,
    )[None, :, None]
    group = torch.rand((len(values), 1, 1), device=values.device) < cfg["motion_group_dropout"]
    dropped = dropped | (group & motion)
    observed = values[:, n:] == 0
    noise = torch.randn_like(values[:, :n]) * cfg["feature_noise_std"]
    result[:, :n] = torch.where(dropped, 0.0, values[:, :n] + noise * observed)
    result[:, n:] = torch.where(dropped, 1.0, values[:, n:])
    return result
