"""Optional TCN invariants: period isolation, exact receptive overlap and team symmetry."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from temporal_model import (  # noqa: E402
    TemporalNet,
    blend,
    encode,
    infer,
    normalization,
    symmetric_infer,
)
from train_temporal import validate_games  # noqa: E402


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(81, 3)).astype(np.float32)
    values[5:8, 0] = np.nan
    names = ["H_distance", "A_distance", "speed"]
    mean, scale = normalization([values[:40]], names)
    net = TemporalNet(3, channels=8, dilations=[1, 2], dropout=0).eval()
    return values, names, mean, scale, net


def test_overlap_chunks_match_complete_run(fixture):
    values, _, mean, scale, net = fixture
    full = infer(net, values, [(0, 81)], mean, scale, "cpu", chunk=200)
    chunked = infer(net, values, [(0, 81)], mean, scale, "cpu", chunk=13)
    for first, second in zip(full, chunked, strict=True):
        np.testing.assert_allclose(first, second, atol=1e-7)


def test_periods_do_not_share_context(fixture):
    values, _, mean, scale, net = fixture
    before = infer(net, values, [(0, 40), (41, 81)], mean, scale, "cpu")
    changed = values.copy()
    changed[41:] = 100
    after = infer(net, changed, [(0, 40), (41, 81)], mean, scale, "cpu")
    for first, second in zip(before, after, strict=True):
        np.testing.assert_array_equal(first[:40], second[:40])
        np.testing.assert_allclose(first[40], np.full(3, 1 / 3))


def test_symmetric_inference_swaps_both_heads(fixture):
    values, names, mean, scale, net = fixture
    before = symmetric_infer(net, values, [(0, 81)], names, mean, scale, "cpu")
    after = symmetric_infer(net, values[:, [1, 0, 2]], [(0, 81)], names, mean, scale, "cpu")
    for first, second in zip(before, after, strict=True):
        np.testing.assert_allclose(first, second[:, [1, 0, 2]], atol=1e-7)


def test_normalization_is_tied_and_missing_stays_visible(fixture):
    values, _, mean, scale, _ = fixture
    assert mean[0] == mean[1] and scale[0] == scale[1]
    encoded = encode(values, mean, scale)
    assert np.isfinite(encoded).all()
    assert encoded[5, 0] == 0 and encoded[5, 3] == 1


@pytest.mark.parametrize("key", ["train_games", "validation_games", "test_games"])
def test_rejects_game_role_overlap(key):
    cfg = {"train_games": ["TRAIN"], "validation_games": ["VAL"], "test_games": ["TEST"]}
    cfg[key] = ["TRAIN", "VAL"]
    with pytest.raises(ValueError, match="overlap"):
        validate_games(cfg)


def test_weights_only_roundtrip_preserves_predictions(fixture, tmp_path):
    values, _, mean, scale, net = fixture
    path = tmp_path / "weights.pt"
    torch.save(net.state_dict(), path)
    other = TemporalNet(3, channels=8, dilations=[1, 2], dropout=0)
    other.load_state_dict(torch.load(path, weights_only=True))
    for first, second in zip(
        infer(net, values, [(0, 81)], mean, scale, "cpu"),
        infer(other, values, [(0, 81)], mean, scale, "cpu"),
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)


def test_float32_probability_rounding_satisfies_existing_decoder_contract():
    from mtah.hmm import _probabilities

    rng = np.random.default_rng(7)
    values = rng.random((1000, 3))
    values = (values / values.sum(axis=1, keepdims=True)).astype(np.float32)
    mixed = blend(values, values[:, [1, 0, 2]], 0.25)
    _probabilities(mixed, (1000, 3), "neural mixture")
    np.testing.assert_allclose(mixed.sum(axis=1), 1, atol=1e-15)
