"""Optional local noncausal TCN; independent of the frozen src inference inventory."""

import numpy as np
import torch
from torch import nn

from mtah.hmm import sequence_runs
from mtah.supervised import swap_features


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values):
        return values + self.layers(values)


class TemporalNet(nn.Module):
    def __init__(self, features, channels=64, dilations=(1, 2, 4, 8, 16, 32), dropout=0.15):
        super().__init__()
        self.halo = 2 * sum(dilations)
        self.input = nn.Conv1d(features * 2, channels, 1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, dilation, dropout) for dilation in dilations]
        )
        self.state = nn.Conv1d(channels, 3, 1)
        self.event = nn.Conv1d(channels, 3, 1)

    def forward(self, values):
        hidden = self.blocks(self.input(values))
        return self.state(hidden), self.event(hidden)


def normalization(arrays, names):
    """Train-only moments, tied under H/A swapping; missing indicators stay explicit."""
    count = np.zeros(len(names))
    total, squares = np.zeros(len(names)), np.zeros(len(names))
    for values in arrays:
        for current in (values, swap_features(values, names)):
            finite = np.isfinite(current)
            safe = np.where(finite, current, 0).astype(np.float64)
            count += finite.sum(axis=0)
            total += safe.sum(axis=0)
            squares += (safe * safe).sum(axis=0)
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(squares / np.maximum(count, 1) - mean * mean, 0.01))
    return mean.astype(np.float32), scale.astype(np.float32)


def encode(values, mean, scale):
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    normalized = np.where(finite, (values - mean) / scale, 0)
    return np.concatenate([np.clip(normalized, -8, 8), (~finite).astype(np.float32)], axis=-1)


@torch.inference_mode()
def infer(net, values, runs, mean, scale, device, chunk=4096):
    """Overlap by the exact receptive radius; never bridge a period or unknown separator."""
    net.eval()
    state = np.full((len(values), 3), 1 / 3, dtype=np.float32)
    event = state.copy()
    for a, b in sequence_runs(len(values), runs):
        for start in range(a, b, chunk):
            stop = min(b, start + chunk)
            lo, hi = max(a, start - net.halo), min(b, stop + net.halo)
            encoded = encode(values[lo:hi], mean, scale)
            tensor = torch.from_numpy(encoded.T.copy())[None].to(device)
            first, second = net(tensor)
            sl = slice(start - lo, stop - lo)
            state[start:stop] = first.softmax(1)[0, :, sl].T.cpu().numpy()
            event[start:stop] = second.softmax(1)[0, :, sl].T.cpu().numpy()
    return state, event


def symmetric_infer(net, values, runs, names, mean, scale, device, chunk=4096):
    first, second = infer(net, values, runs, mean, scale, device, chunk)
    swapped_state, swapped_event = infer(
        net, swap_features(values, names), runs, mean, scale, device, chunk
    )
    return (first + swapped_state[:, [1, 0, 2]]) / 2, (second + swapped_event[:, [1, 0, 2]]) / 2


def blend(first, second, weight):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("mixture weight must be between zero and one")
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    result = (1 - weight) * first + weight * second
    if (
        first.shape != second.shape
        or result.ndim != 2
        or result.shape[1] != 3
        or not np.isfinite(result).all()
        or (result < 0).any()
        or (result.sum(axis=1) <= 0).any()
    ):
        raise ValueError("invalid state/event mixture probabilities")
    # Float32 softmax/cache rows may differ from one by ~1e-7. The existing scorer-facing
    # decoder checks normalized float64 distributions, so restore that contract explicitly.
    return result / result.sum(axis=1, keepdims=True)
