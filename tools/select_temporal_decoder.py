"""Refine decoder strengths on validation only, keeping all neural/tree weights frozen."""

import copy
import json
import shutil
from itertools import product

import _src_path  # noqa: F401
import numpy as np
from evaluate_temporal import load_model
from temporal_model import blend
from train_supervised import json_bytes, metric_summary

from mtah.boundary import decode_boundaries
from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def main():
    policy_path = ROOT / "config/temporal_decoder.toml"
    policy = load_config(policy_path)
    directory = ROOT / policy["parent_dir"]
    out = OUTPUTS_DIR / "temporal_decoder"
    metadata, parent, _, original_receipt = load_model(directory, "cpu")
    cfg = metadata["config"]
    prepared = []
    for game in cfg["validation_games"]:
        for profile in ("clean", "occluded", "occluded_jittered"):
            with np.load(
                directory / f"validation_{game}_{profile}.npz", allow_pickle=False
            ) as data:
                state, event = data["state"], data["event"]
            with np.load(
                OUTPUTS_DIR / f"temporal_cache/{game}_validation_{profile}.npz", allow_pickle=False
            ) as data:
                prepared.append(
                    {
                        "game": game,
                        "profile": profile,
                        "state": blend(data["state"], state, metadata["state_weight"]),
                        "event": blend(data["event"], event, metadata["event_weight"]),
                        "reference": data["reference"],
                        "runs": data["runs"],
                    }
                )
    rows = []
    for boundary, scale in product(policy["boundary_weights"], policy["decoder_scales"]):
        decoder = {**parent, "boundary_weight": boundary, "decoder_scale": scale}
        details = []
        for item in prepared:
            pred = decode_boundaries(item["state"], item["event"], decoder, item["runs"])
            details.append(
                {
                    "game": item["game"],
                    "profile": item["profile"],
                    "metrics": metric_summary(item["reference"], pred),
                }
            )
        mean = float(np.mean([d["metrics"]["spell_f1_mean"] for d in details]))
        rows.append(
            {
                "boundary_weight": boundary,
                "decoder_scale": scale,
                "mean_spell_f1": mean,
                "validation": details,
            }
        )
        print(f"Validation boundary={boundary},scale={scale}: {mean:.5f}", flush=True)
    best = min(rows, key=lambda r: (-r["mean_spell_f1"], r["decoder_scale"], r["boundary_weight"]))
    out.mkdir(exist_ok=True)
    shutil.copy2(directory / "weights.pt", out / "weights.pt")
    assert sha256(out / "weights.pt") == metadata["weights_sha256"]
    refined = {**parent, "boundary_weight": best["boundary_weight"]}
    atomic_write(out / "parent.json", json_bytes(refined))
    metadata = copy.deepcopy(metadata)
    metadata["decoder_scale"] = best["decoder_scale"]
    metadata["frozen_parent_sha256"] = sha256(out / "parent.json")
    metadata["provenance"]["helper_sha256"].update(
        {
            "tools/select_temporal_decoder.py": sha256(__file__),
            "config/temporal_decoder.toml": sha256(policy_path),
        }
    )
    metadata["provenance"]["decoder_refinement"] = {
        "parent_receipt": original_receipt,
        "fitting": "no classifier fitting; validation-only strengths",
        "classifier_weights_unchanged": True,
        "test_games_read_for_selection": False,
    }
    atomic_write(out / "model.json", json_bytes(metadata))
    report = json.loads((directory / "training_report.json").read_text())
    report["parent_selected"] = report["selected"]
    report["selected"] = {
        **best,
        "state_weight": metadata["state_weight"],
        "event_weight": metadata["event_weight"],
    }
    report["decoder_refinement_selection"] = rows
    report["model_sha256"] = sha256(out / "model.json")
    report["provenance"] = metadata["provenance"]
    atomic_write(out / "training_report.json", json_bytes(report))
    atomic_write(
        out / "test_freeze.json",
        json_bytes(
            {
                "model_sha256": sha256(out / "model.json"),
                "weights_sha256": sha256(out / "weights.pt"),
                "report_sha256": sha256(out / "training_report.json"),
                "frozen_before_test_evaluation": True,
                "test_previous_exposure": "also evaluated by parent TCN; not an untouched holdout",
            }
        ),
    )
    print(f"Frozen refined decoder: {best['mean_spell_f1']:.5f}; weights unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
