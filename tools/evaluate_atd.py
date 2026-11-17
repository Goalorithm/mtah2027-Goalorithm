"""Evaluate the ATD-realistic TCN: ATD-like external test, BAR-ATH candidate, public check.

The public five-minute window is printed as a pipeline correctness check only. It is never
used to choose a model, epoch or decoder; those come from ATD-like external validation.
"""

import argparse
import json
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
import torch
from atd_features import atd_feature_names
from atd_inputs import competition_features
from check_release import strict_submission
from prepare_atd import feature_config, load_split
from temporal_model import TemporalNet, symmetric_infer
from train_atd import HELPERS, decode
from train_supervised import json_bytes, metric_summary

from mtah import A, D, H
from mtah.io import read_codes, sha256, write_submission
from mtah.paths import N_FRAMES, OUTPUTS_DIR, PUBLIC_GT, ROOT, pm
from mtah.review import atomic_write


def load_member(directory, device):
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text())
    if sha256(directory / "weights.pt") != metadata["weights_sha256"]:
        raise ValueError("weights changed since training")
    for name, expected in metadata["provenance"]["helper_sha256"].items():
        if name != "tools/train_atd.py" and sha256(ROOT / name) != expected:
            raise ValueError(f"model dependency changed since training: {name}")
    cfg = metadata["config"]
    net = TemporalNet(
        len(metadata["features"]), cfg["channels"], cfg["dilations"], cfg["dropout"]
    ).to(device)
    net.load_state_dict(
        torch.load(directory / "weights.pt", map_location=device, weights_only=True)
    )
    return metadata, net


def load_model(directory, device):
    """(metadata, members). An ensemble manifest lists member directories under outputs/."""
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text())
    if "members" not in metadata:
        return metadata, [load_member(directory, device)]
    members = []
    for name, expected in metadata["members"].items():
        if sha256(OUTPUTS_DIR / name / "model.json") != expected:
            raise ValueError(f"ensemble member changed: {name}")
        members.append(load_member(OUTPUTS_DIR / name, device))
    if any(member[0]["features"] != metadata["features"] for member in members):
        raise ValueError("ensemble members use different features")
    return metadata, members


def member_columns(values, member):
    """The columns a member was fitted on, out of the complete feature matrix."""
    if values.shape[1] == len(member["features"]):
        return values
    every = atd_feature_names(feature_config(member["config"]))
    if values.shape[1] != len(every):
        raise ValueError("feature matrix does not match the model's feature definition")
    position = {name: index for index, name in enumerate(every)}
    return np.ascontiguousarray(values[:, [position[name] for name in member["features"]]])


def probabilities(values, runs, metadata, members, device):
    """Mean state/event probabilities over the members, each with its own normalisation."""
    runs = [tuple(int(v) for v in run) for run in runs]
    total = None
    for member, net in members:
        part = symmetric_infer(
            net,
            member_columns(values, member),
            runs,
            member["features"],
            np.asarray(member["mean"], dtype=np.float32),
            np.asarray(member["scale"], dtype=np.float32),
            device,
        )
        total = part if total is None else tuple(a + b for a, b in zip(total, part, strict=True))
    return tuple(part / len(members) for part in total)


def predict(values, runs, metadata, net, device):
    state, event = probabilities(values, runs, metadata, net, device)
    return decode(
        state,
        event,
        metadata["decoder"],
        runs,
        metadata["decoder_scale"],
        metadata["boundary_weight"],
    )


def structure(prediction, runs):
    """Possession structure of a prediction; descriptive, not an accuracy."""
    segments = pm.segments(prediction, runs)
    spells = [b - a for a, b, side in segments if side in (H, A)]
    graded = np.concatenate([prediction[a:b] for a, b in runs])
    return {
        "possessions": len(spells),
        "possessions_under_2s": int(np.sum(np.asarray(spells) < 50)),
        "mean_possession_s": float(np.mean(spells) / 25) if spells else 0.0,
        "share_H_A_D": [float(np.mean(graded == side)) for side in (H, A, D)],
    }


def legacy_predictor(directory, device):
    """The earlier frozen TCN+tree model, scored on the same degraded input for comparison."""
    from evaluate_temporal import load_model as load_legacy
    from evaluate_temporal import predict as predict_legacy

    from mtah.compiled_trees import compiled_symmetric_probabilities

    metadata, parent, net, _ = load_legacy(directory, device)
    width = len(metadata["features"])

    def run(item):
        values = item["values"][:, :width]
        legacy = {
            "values": values,
            "runs": [tuple(int(v) for v in run) for run in item["runs"]],
            "state": compiled_symmetric_probabilities(values.astype(np.float64), parent),
            "event": compiled_symmetric_probabilities(
                values.astype(np.float64), parent["boundary_classifier"]
            ),
        }
        return predict_legacy(legacy, net, metadata, parent, device)

    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/atd")
    parser.add_argument("--mode", choices=["test", "competition"], default="test")
    parser.add_argument("--legacy-dir", default="outputs/temporal")
    parser.add_argument("--name", default=None, help="competition CSV name under outputs/")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="default: cpu for the competition file, so it never depends on the machine's GPU; "
        "auto (GPU when there is one) for the test games",
    )
    args = parser.parse_args(argv)
    out = Path(args.model_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("model must remain under ignored outputs/")
    # Float results differ in the last digits between a GPU and a CPU, and a decoder decision
    # that sits on the edge can then flip. The submission is therefore always computed on CPU.
    choice = args.device or ("cpu" if args.mode == "competition" else "auto")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    device = choice
    torch.set_num_threads(4)
    metadata, net = load_model(out, device)
    cfg = metadata["config"]
    report = {
        "mode": args.mode,
        "model_sha256": sha256(out / "model.json"),
        "helper_sha256": {name: sha256(ROOT / name) for name in HELPERS},
        "rows": [],
        "competition_accuracy_measured": False,
    }
    if args.mode == "test":
        legacy = legacy_predictor(ROOT / args.legacy_dir, device) if args.legacy_dir else None
        for item in load_split(cfg, "test"):
            runs = [tuple(int(v) for v in run) for run in item["runs"]]
            prediction = predict(item["values"], runs, metadata, net, device)
            complete = pm.evaluate(item["reference"], prediction, bootstrap_n=500)
            metrics = metric_summary(item["reference"], prediction)
            metrics.update(
                {key: complete[key] for key in ("spell_bootstrap_low", "spell_bootstrap_high")}
            )
            row = {
                "game": str(item["game"]),
                "profile": str(item["profile"]),
                "ball_seen": float(item["ball_seen"].mean()),
                "metrics": metrics,
            }
            line = f"{row['game']}/{row['profile']}: spellF1 {metrics['spell_f1_mean']:.4f}"
            if legacy is not None:
                row["legacy_metrics"] = metric_summary(item["reference"], legacy(item))
                line += f" (earlier model {row['legacy_metrics']['spell_f1_mean']:.4f})"
            report["rows"].append(row)
            print(line, flush=True)
        report["mean_spell_f1"] = float(
            np.mean([row["metrics"]["spell_f1_mean"] for row in report["rows"]])
        )
        if legacy is not None:
            report["legacy_mean_spell_f1"] = float(
                np.mean([row["legacy_metrics"]["spell_f1_mean"] for row in report["rows"]])
            )
        report["test_previous_exposure"] = cfg["test_previous_exposure"]
        print(json.dumps({k: v for k, v in report.items() if k.endswith("mean_spell_f1")}))
    else:
        values, runs, report["inputs"] = competition_features(cfg, feature_config(cfg))
        print(json.dumps(report["inputs"]), flush=True)
        first = predict(values, runs, metadata, net, device)
        second = predict(values, runs, metadata, net, device)
        if not np.array_equal(first, second):
            raise ValueError("fullmatch predictions do not repeat exactly")
        path = OUTPUTS_DIR / f"{args.name or 'exp_' + out.name}.csv"
        write_submission(path, first)
        strict_submission(path, N_FRAMES)
        public = pm.evaluate(read_codes(PUBLIC_GT), first, bootstrap_n=0)
        report.update(
            prediction=str(path.relative_to(ROOT)).replace("\\", "/"),
            prediction_sha256=sha256(path),
            repeat_identical=True,
            device=device,
            torch_version=str(torch.__version__),
            frames=N_FRAMES,
            structure=structure(first, runs),
            public_window_correctness_check={
                "note": "17 possessions; never used to choose a model, epoch or decoder",
                "spell_f1_mean": public["spell_f1_mean"],
                "macro_f1": public["macro_f1"],
                "per_class": public["per_class"],
            },
        )
        print(json.dumps({k: report[k] for k in ("structure", "prediction_sha256")}))
        check = report["public_window_correctness_check"]
        print(
            f"Public window check: spellF1 {check['spell_f1_mean']:.4f}, "
            f"macroF1 {check['macro_f1']:.4f}",
            flush=True,
        )
    atomic_write(out / f"{args.mode}_report.json", json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
