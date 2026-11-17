"""Portable supervised inference, label isolation, symmetry, sequences and data separation."""

import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from train_supervised import transition_parameters, validate_split

from mtah import A, D, H, X
from mtah.compiled_trees import compiled_probabilities
from mtah.config import load_config
from mtah.load import Tracking
from mtah.paths import CONFIG_DIR
from mtah.predict import prepare_features, run_pipeline
from mtah.supervised import (
    decode_probabilities,
    export_classifier,
    feature_matrix,
    feature_names,
    hide_ball,
    perturb_ball,
    raw_probabilities,
    swap_features,
    symmetric_probabilities,
)


def config():
    return load_config(CONFIG_DIR / "exp_supervised_v1.toml")["supervised"]


def tracking(n=300):
    players = np.tile([[20.0, 30.0], [40.0, 30.0]], (n, 1, 1))
    return Tracking(
        players,
        np.full((n, 2), 2.5),
        np.tile([20.2, 30.0], (n, 1)),
        np.zeros(n),
        np.array([H, A]),
        ("Home", "Away"),
    )


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(41)
    values = rng.normal(size=(600, 4))
    values[::7, 0] = np.nan
    labels = np.where(values[:, 1] > 0.4, D, np.where(values[:, 2] > values[:, 3], H, A))
    with threadpool_limits(limits=1):
        classifier = HistGradientBoostingClassifier(
            max_iter=12, max_leaf_nodes=7, early_stopping=False, random_state=3
        ).fit(values, labels)
    model = export_classifier(classifier, ["seen", "pace", "H_vote", "A_vote"])
    return classifier, values, model


def test_portable_json_matches_sklearn_on_missing_and_observed_values(fitted):
    classifier, values, model = fitted
    restored = json.loads(json.dumps(model))
    np.testing.assert_allclose(
        raw_probabilities(values, restored),
        classifier.predict_proba(values),
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_array_equal(
        compiled_probabilities(values, restored), raw_probabilities(values, restored)
    )


def test_missing_only_sklearn_splits_export_to_finite_json_with_identical_probabilities():
    values = np.ones((300, 2))
    values[:100, 1] = 0
    values[100:200, 1] = 1
    values[200:, 0] = np.nan
    labels = np.repeat([H, A, D], 100)
    with threadpool_limits(limits=1):
        classifier = HistGradientBoostingClassifier(
            max_iter=3,
            max_leaf_nodes=3,
            min_samples_leaf=5,
            early_stopping=False,
        ).fit(values, labels)
    assert any(
        np.isinf(tree.nodes["num_threshold"]).any()
        for stage in classifier._predictors
        for tree in stage
    )
    model = json.loads(
        json.dumps(export_classifier(classifier, ["ball_seen", "distance"]), allow_nan=False)
    )
    assert any("finite_route" in tree for tree in model["trees"])
    np.testing.assert_allclose(
        raw_probabilities(values, model), classifier.predict_proba(values), atol=1e-14
    )
    np.testing.assert_array_equal(
        compiled_probabilities(values, model), raw_probabilities(values, model)
    )


def test_symmetric_prediction_is_exactly_equivariant_to_team_swap(fitted):
    _, values, model = fitted
    probabilities = symmetric_probabilities(values, model)
    swapped = symmetric_probabilities(swap_features(values, model["feature_names"]), model)
    np.testing.assert_allclose(probabilities, swapped[:, [A, H, D]], atol=1e-14)


@pytest.mark.parametrize("mutation", ["cycle", "feature", "nan"])
def test_invalid_tree_cannot_execute_unbounded_or_silent_wrong_inference(fitted, mutation):
    model = copy.deepcopy(fitted[2])
    tree = next(t for t in model["trees"] if not t["leaf"][0])
    if mutation == "cycle":
        tree["left"][0] = 0
    elif mutation == "feature":
        tree["feature"][0] = 999
    else:
        tree["value"][0] = float("nan")
    with pytest.raises(ValueError):
        raw_probabilities(fitted[1], model)


def test_tracking_features_do_not_read_identity_strings_or_ball_extra():
    tr, cfg = tracking(), config()["features"]
    tr.ball[80:90] = np.nan
    first = feature_matrix(tr, cfg)
    tr.ball_extra[:] = np.arange(tr.n_frames) % 3  # Could contain an answer key: must be ignored.
    tr.team_names = ("SECRET LABELS", "ANOTHER TEAM")
    np.testing.assert_array_equal(feature_matrix(tr, cfg), first)
    tr.slot_team = 1 - tr.slot_team
    np.testing.assert_array_equal(feature_matrix(tr, cfg), swap_features(first, feature_names(cfg)))


@pytest.mark.parametrize("lag", [1, 5])
def test_future_and_past_features_never_cross_half_or_ungraded_boundaries(lag):
    tr, cfg = tracking(), config()["features"]
    cfg["motion_lag_frames"] = lag
    tr.ball[150:] = [40.2, 30]
    runs = [(10, 140), (160, 290)]
    full = feature_matrix(tr, cfg, runs)
    assert np.isnan(full[:10]).all() and np.isnan(full[140:160]).all()
    for a, b in runs:
        part = Tracking(
            tr.players[a:b],
            tr.player_speed[a:b],
            tr.ball[a:b],
            tr.ball_extra[a:b],
            tr.slot_team,
            tr.team_names,
        )
        np.testing.assert_array_equal(full[a:b], feature_matrix(part, cfg))


def test_occlusion_augmentation_is_reproducible_label_independent_and_non_mutating():
    tr = tracking(10000)
    original = tr.ball.copy()
    first = hide_ball(tr, [(0, 5000), (5000, 10000)], 0.4, 25, 7)
    second = hide_ball(tr, [(0, 5000), (5000, 10000)], 0.4, 25, 7)
    np.testing.assert_array_equal(first.ball, second.ball)
    np.testing.assert_array_equal(tr.ball, original)
    assert 0.3 < np.isnan(first.ball[:, 0]).mean() < 0.5


def test_temporal_decoder_keeps_half_reset_and_suppresses_isolated_flip():
    probabilities = np.tile([0.99, 0.005, 0.005], (300, 1))
    probabilities[100] = [0.005, 0.99, 0.005]
    probabilities[200:] = [0.005, 0.99, 0.005]
    model = {
        "class_prior": [1 / 3] * 3,
        "decoder_scale": 0.1,
        "initial": [1 / 3] * 3,
        "transition": (np.eye(3) * 0.9985 + 0.0005).tolist(),
    }
    result = decode_probabilities(probabilities, model, [(10, 180), (200, 290)])
    assert (result[10:180] == H).all() and (result[200:290] == A).all()
    assert (result[:10] == X).all() and (result[180:200] == X).all()


def test_training_and_selection_cannot_share_games_with_sealed_test():
    cfg = config()["training"]
    validate_split(cfg)
    cfg["test_games"].append(cfg["train_games"][0])
    with pytest.raises(ValueError, match="disjoint"):
        validate_split(cfg)


def test_transition_estimation_does_not_join_halves_or_unknown_reference_gaps():
    references = np.array([H, H, X, A, A, D, D], dtype=np.int8)
    model = transition_parameters([(references, [(0, 2), (3, 7)])], 1.0)
    matrix = np.array(model["transition"])
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_allclose(matrix, matrix[np.ix_([A, H, D], [A, H, D])])
    # Without the X mask the H->A event would inflate both cross-team counts above pseudocount.
    assert matrix[H, A] < matrix[H, H]


def test_net_motion_is_less_sensitive_to_stationary_coordinate_jitter():
    tr = tracking(3000)
    original = tr.ball.copy()
    noisy = perturb_ball(tr, 0.15, 9)
    np.testing.assert_array_equal(tr.ball, original)
    cfg = config()["features"]
    assert len(feature_names(cfg)) == 26  # Preserve the v1 feature layout.
    cfg["motion_lag_frames"] = 5
    matrix = feature_matrix(noisy, cfg)
    names = feature_names(cfg)
    raw = matrix[10:-10, names.index("ball_speed")]
    robust = matrix[10:-10, names.index("net_ball_speed")]
    assert np.nanmedian(robust) < 0.3 * np.nanmedian(raw)
    assert len(names) == 29


def test_coordinate_noise_preserves_missing_frames_and_is_reproducible():
    tr = tracking()
    tr.ball[30:40] = np.nan
    first, second = perturb_ball(tr, 0.15, 8), perturb_ball(tr, 0.15, 8)
    np.testing.assert_array_equal(first.ball, second.ball)
    assert np.isnan(first.ball[30:40]).all()


def test_multimodel_cache_reuses_numeric_probabilities_but_invalidates_changed_weights(
    fitted, tmp_path, monkeypatch
):
    import mtah.compiled_trees as compiled

    tr, cfg = tracking(), config()["features"]
    model = copy.deepcopy(fitted[2])
    model.update(
        {
            "feature_names": feature_names(cfg),
            "feature_config": cfg,
            "class_prior": [1 / 3] * 3,
            "decoder_scale": 0.3,
            "initial": [1 / 3] * 3,
            "transition": (np.eye(3) * 0.9985 + 0.0005).tolist(),
        }
    )
    calls = []
    original = compiled.compiled_symmetric_probabilities

    def record(values, checkpoint):
        calls.append(1)
        return original(values, checkpoint)

    monkeypatch.setattr(compiled, "compiled_symmetric_probabilities", record)
    path1, path2 = tmp_path / "first.json", tmp_path / "second.json"
    path1.write_text(json.dumps(model), encoding="utf-8")
    model["decoder_scale"] = 1.0
    path2.write_text(json.dumps(model), encoding="utf-8")

    def candidate(path):
        return {
            "dead_ball": {"method": "supervised"},
            "possession": {"method": "supervised"},
            "supervised": {"model_path": str(path), "features": cfg},
        }

    prepared = prepare_features(tr)
    run_pipeline(candidate(path1), tr, [], prepared)
    cached = run_pipeline(candidate(path2), tr, [], prepared)
    assert len(calls) == 1
    np.testing.assert_array_equal(cached, run_pipeline(candidate(path2), tr, []))
    assert len(calls) == 2
    model["baseline"][H] += 0.1
    path2.write_text(json.dumps(model), encoding="utf-8")
    run_pipeline(candidate(path2), tr, [], prepared)
    assert len(calls) == 3


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_frozen_forest_fusion_matches_prior_corrected_geometric_probabilities(fitted, weight):
    from select_ensemble import blend_probabilities, fuse_classifiers

    first, second = copy.deepcopy(fitted[2]), copy.deepcopy(fitted[2])
    first.update({"class_prior": [0.2, 0.3, 0.5], "decoder_scale": 0.3})
    second.update({"class_prior": [0.4, 0.35, 0.25], "decoder_scale": 0.3})
    second["baseline"][H] += 0.5
    second["trees"][0]["value"] = (np.asarray(second["trees"][0]["value"]) * 1.1).tolist()
    values = fitted[1]
    expected = blend_probabilities(
        raw_probabilities(values, first),
        raw_probabilities(values, second),
        first["class_prior"],
        second["class_prior"],
        weight,
    )
    fused = fuse_classifiers(first, second, weight)
    np.testing.assert_allclose(raw_probabilities(values, fused), expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(
        compiled_probabilities(values, fused), raw_probabilities(values, fused)
    )


def test_frozen_forest_fusion_rejects_invalid_weight(fitted):
    from select_ensemble import fuse_classifiers

    model = copy.deepcopy(fitted[2])
    model.update({"class_prior": [1 / 3] * 3, "decoder_scale": 0.3})
    with pytest.raises(ValueError, match="ensemble weight"):
        fuse_classifiers(model, model, -0.1)
