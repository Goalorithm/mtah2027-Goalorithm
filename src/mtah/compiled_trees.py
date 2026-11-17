"""Fast evaluation of the same finite JSON trees using the pinned sklearn CPU predictor."""

import numpy as np

from mtah import A, H
from mtah.supervised import swap_features, validate_classifier


def compiled_probabilities(values, model):
    from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor

    validate_classifier(model)
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(model["feature_names"]) or np.isinf(values).any():
        raise ValueError("invalid supervised feature matrix")
    scores = np.tile(model["baseline"], (len(values), 1))
    empty = np.zeros((0, 8), dtype=np.uint32)
    feature_map = np.zeros(values.shape[1], dtype=np.uint32)
    for tree in model["trees"]:
        nodes = np.zeros(len(tree["leaf"]), dtype=PREDICTOR_RECORD_DTYPE)
        for field, key in (
            ("value", "value"),
            ("feature_idx", "feature"),
            ("num_threshold", "threshold"),
            ("missing_go_to_left", "missing_left"),
            ("left", "left"),
            ("right", "right"),
            ("is_leaf", "leaf"),
        ):
            nodes[field] = tree[key]
        if "finite_route" in tree:
            route = np.asarray(tree["finite_route"])
            nodes["num_threshold"][route > 0] = np.inf
            nodes["num_threshold"][route < 0] = -np.inf
        predictor = TreePredictor(nodes, empty, empty)
        scores[:, tree["class"]] += predictor.predict(values, empty, feature_map, n_threads=1)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def compiled_symmetric_probabilities(values, model):
    first = compiled_probabilities(values, model)
    second = compiled_probabilities(swap_features(values, model["feature_names"]), model)[
        :, [A, H, 2]
    ]
    return (first + second) / 2
