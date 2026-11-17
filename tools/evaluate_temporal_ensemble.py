"""Reuse the same external evaluation/output checks with a frozen ensemble predictor."""

import sys

import _src_path  # noqa: F401
import evaluate_temporal
from temporal_ensemble import load_ensemble, predict_ensemble

if __name__ == "__main__":
    evaluate_temporal.load_model = load_ensemble
    evaluate_temporal.predict = predict_ensemble
    raise SystemExit(
        evaluate_temporal.main(["--model-dir", "outputs/temporal_ensemble", *sys.argv[1:]])
    )
