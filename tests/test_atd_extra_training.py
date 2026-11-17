"""Extra fitting profiles from another provider: fixed plan, reserved matches, sampling share."""

import numpy as np
import pytest

from mtah import A, D, H, X
from mtah.config import load_config
from mtah.paths import ROOT

pytest.importorskip("ijson")  # optional, requirements-validation.txt
pytest.importorskip("torch")

from prepare_atd_soccertrack import profile_plan  # noqa: E402
from train_atd import sample_batch  # noqa: E402


def configs():
    cfg = load_config(ROOT / "config/atd.toml")
    return cfg, load_config(ROOT / cfg["soccertrack_train"]["config"])


def test_plan_is_fixed_and_never_touches_the_check_matches():
    cfg, source = configs()
    plan = profile_plan(cfg, source)
    again = profile_plan(cfg, source)
    assert [row[:4] for row in plan] == [row[:4] for row in again]
    assert plan[0][4] == again[0][4]
    assert len(plan) == len(source["train_matches"]) * 2 * cfg["soccertrack_train"]["profiles"]
    assert len({row[3] for row in plan}) == len(plan)  # one seed per profile
    assert not {row[0] for row in plan} & set(source["matches"])
    with pytest.raises(ValueError):
        profile_plan(cfg, {**source, "train_matches": [source["matches"][0]]})


def test_extra_seeds_do_not_collide_with_the_other_fitting_profiles():
    from prepare_atd import profile_plan as base_plan

    cfg, source = configs()
    base = {row[3] for row in base_plan(cfg)}
    assert not base & {row[3] for row in profile_plan(cfg, source)}


def item(label, frames=4000, columns=3):
    return {
        "values": np.zeros((frames, columns), dtype=np.float16),
        "reference": np.full(frames, label, dtype=np.int8),
        "target": np.full(frames, D, dtype=np.int8),
        "weights": np.ones(frames, dtype=np.float32),
        "runs": np.asarray([(0, frames)]),
        "changes": np.asarray([], dtype=np.int64),
    }


def draw(share, extra, seed=3):
    cfg = {"batch_size": 64, "core_frames": 32, "extra_train_share": share}
    names = ["ball_seen", "interpolated", "gap_past"]
    rng = np.random.default_rng(seed)
    mean, scale = np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
    _, labels, _, weights = sample_batch([item(D)], cfg, names, mean, scale, 4, rng, extra)
    return labels.numpy(), weights.numpy()


def test_share_of_samples_comes_from_the_extra_pool():
    labels, _ = draw(0.5, [item(X)])
    from_extra = (labels == X).all(axis=1).mean()
    assert 0.25 < from_extra < 0.75
    # A window can run past the end of a half, so a base sample is one with any D frame.
    assert ((labels == D).any(axis=1) | (labels == X).all(axis=1)).all()


def test_no_extra_pool_or_zero_share_leaves_sampling_unchanged():
    plain, _ = draw(0.5, [])
    zero, _ = draw(0.0, [item(X)])
    assert (plain == D).any(axis=1).all() and (zero == D).any(axis=1).all()


def test_ungraded_frames_of_an_extra_profile_carry_no_weight():
    mixed = item(H)
    mixed["reference"][::2] = X
    labels, weights = draw(1.0, [mixed])
    assert np.isin(labels, [H, A, X]).all()  # H may be swapped to A, never to D
    assert (weights[labels == X] == 0).all() and (weights[labels != X] > 0).all()


def test_contested_frames_follow_the_competition_rule_within_each_half():
    from train_atd import contested_to_last_owner

    cfg = {"short_spell_exponent": 0.5, "boundary_weight": 1.5, "boundary_frames": 2}
    reference = np.array([X, H, X, X, A, X, D, X, X, A, X, H], dtype=np.int8)
    entry = {"reference": reference.copy(), "runs": np.asarray([(0, 8), (8, 12)])}
    contested_to_last_owner(entry, cfg)
    # The second half starts ungraded again: no owner is carried across the interval.
    assert entry["reference"].tolist() == [X, H, H, H, A, A, D, D, X, A, A, H]
    assert entry["target"].shape == reference.shape and entry["weights"].shape == reference.shape
    assert entry["weights"].dtype == np.float32 and (entry["weights"] > 0).all()
    assert entry["target"][4] == A and entry["target"][11] == H  # handovers are event targets


def test_decoder_score_weights_label_sources_equally(monkeypatch):
    import ensemble_atd

    def fake_score(items, probabilities, decoder, scale, weight):
        return float(np.mean(items)), [{"n": len(items)}]

    monkeypatch.setattr(ensemble_atd, "score", fake_score)
    groups = {"idsse": [0.6, 0.8, 0.7, 0.7], "metrica": [0.8], "soccertrack": [0.9, 0.9]}
    value, rows, means = ensemble_atd.balanced_score(groups, groups, None, 0.2, 0.5)
    assert value == pytest.approx((0.7 + 0.8 + 0.9) / 3)  # not the mean over the seven profiles
    assert means == {"idsse": pytest.approx(0.7), "metrica": 0.8, "soccertrack": 0.9}
    assert len(rows) == 3
