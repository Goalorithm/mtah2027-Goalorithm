"""Evaluate a frozen local TCN on external test/anchor data or produce a BAR-ATH candidate."""

import argparse
import json
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
import torch
from audit_spells import duration_diagnostics
from check_release import strict_submission
from prepare_temporal import load_profiles
from temporal_model import TemporalNet, blend, symmetric_infer
from train_supervised import json_bytes, metric_summary
from train_temporal import validate_games

from mtah.boundary import decode_boundaries
from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.external import anchor_metrics, read_sample_game
from mtah.io import sha256, write_submission
from mtah.load import load_tracking
from mtah.paths import N_FRAMES, OUTPUTS_DIR, ROOT, games, match_runs, pm
from mtah.review import atomic_write
from mtah.supervised import feature_matrix


def load_model(directory, device):
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text())
    receipt = json.loads((directory / "test_freeze.json").read_text())
    for name, expected in (
        ("model.json", receipt["model_sha256"]),
        ("training_report.json", receipt["report_sha256"]),
        ("weights.pt", receipt["weights_sha256"]),
        ("parent.json", metadata["frozen_parent_sha256"]),
    ):
        if sha256(directory / name) != expected:
            raise ValueError(f"frozen artifact changed: {name}")
    provenance = metadata["provenance"]
    if sha256(ROOT / metadata["config_path"]) != provenance["config_sha256"]:
        raise ValueError("temporal configuration changed")
    validate_games(metadata["config"])
    parent_config_path = ROOT / metadata["config"]["parent_config"]
    parent_config = load_config(parent_config_path)
    if (
        sha256(parent_config_path) != provenance["parent_config_sha256"]
        or sha256(ROOT / parent_config["supervised"]["model_path"]) != provenance["parent_sha256"]
    ):
        raise ValueError("teacher source changed since temporal fitting")
    for mapping in (provenance["source_sha256"], provenance["helper_sha256"]):
        for name, expected in mapping.items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"model dependency changed: {name}")
    if any(
        provenance[k] is not False
        for k in ("test_games_read", "bar_ath_labels_read", "bar_ath_data_read")
    ):
        raise ValueError("fitting-role declarations are invalid")
    cfg = metadata["config"]
    net = TemporalNet(
        len(metadata["features"]), cfg["channels"], cfg["dilations"], cfg["dropout"]
    ).to(device)
    net.load_state_dict(
        torch.load(directory / "weights.pt", map_location=device, weights_only=True)
    )
    parent = json.loads((directory / "parent.json").read_text())
    return metadata, parent, net, receipt


def predict(item, net, metadata, parent, device):
    state, event = symmetric_infer(
        net,
        item["values"],
        item["runs"],
        metadata["features"],
        np.asarray(metadata["mean"], dtype=np.float32),
        np.asarray(metadata["scale"], dtype=np.float32),
        device,
    )
    return decode_boundaries(
        blend(item["state"], state, metadata["state_weight"]),
        blend(item["event"], event, metadata["event_weight"]),
        {**parent, "decoder_scale": metadata["decoder_scale"]},
        item["runs"],
    )


def tracking_item(tracking, runs, parent):
    values = feature_matrix(tracking, parent["feature_config"], runs)
    return {
        "values": values.astype(np.float32),
        "runs": runs,
        "state": compiled_symmetric_probabilities(values, parent),
        "event": compiled_symmetric_probabilities(values, parent["boundary_classifier"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/temporal")
    parser.add_argument("--mode", choices=["test", "competition", "anchors"], default="test")
    args = parser.parse_args(argv)
    out = Path(args.model_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("model must remain under ignored outputs/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    metadata, parent, net, receipt = load_model(out, device)
    cfg = metadata["config"]
    report = {
        "mode": args.mode,
        "freeze": receipt,
        "rows": [],
        "source_sha256": sha256(__file__),
        "competition_accuracy_measured": False,
        "selected_final_competition_model": None,
    }
    if args.mode == "test":
        for index, game in enumerate(cfg["test_games"]):
            for item in load_profiles(game, index, "test", cfg):
                prediction = predict(item, net, metadata, parent, device)
                teacher = decode_boundaries(
                    blend(item["state"], item["state"], 0),
                    blend(item["event"], item["event"], 0),
                    parent,
                    item["runs"],
                )
                path = out / f"test_{game}_{item['profile']}.csv"
                write_submission(path, prediction)
                metrics = metric_summary(item["reference"], prediction)
                complete = pm.evaluate(item["reference"], prediction, bootstrap_n=500)
                metrics.update(
                    {key: complete[key] for key in ("spell_bootstrap_low", "spell_bootstrap_high")}
                )
                report["rows"].append(
                    {
                        "game": game,
                        "profile": str(item["profile"]),
                        "metrics": metrics,
                        "teacher_metrics": metric_summary(item["reference"], teacher),
                        "prediction_sha256": sha256(path),
                        "duration_audit": duration_diagnostics(item["reference"], prediction),
                    }
                )
                print(
                    f"{game}/{item['profile']}: spellF1 {metrics['spell_f1_mean']:.5f}", flush=True
                )
        means = {
            profile: float(
                np.mean(
                    [
                        row["metrics"]["spell_f1_mean"]
                        for row in report["rows"]
                        if row["profile"] == profile
                    ]
                )
            )
            for profile in ("clean", "occluded", "occluded_jittered")
        }
        report["mean_spell_f1"] = means
        report["target_met"] = {key: value >= 0.8 for key, value in means.items()}
        minimums = {
            profile: min(
                row["metrics"]["spell_f1_mean"]
                for row in report["rows"]
                if row["profile"] == profile
            )
            for profile in means
        }
        report["minimum_game_spell_f1"] = minimums
        report["every_game_target_met"] = {key: value >= 0.8 for key, value in minimums.items()}
        report["test_previous_exposure"] = cfg["test_previous_exposure"]
    elif args.mode == "competition":
        tracking = load_tracking()
        runs = match_runs()
        item = tracking_item(tracking, runs, parent)
        first = predict(item, net, metadata, parent, device)
        second = predict(item, net, metadata, parent, device)
        if not np.array_equal(first, second):
            raise ValueError("TCN fullmatch predictions do not repeat exactly")
        path = OUTPUTS_DIR / f"exp_{out.name}.csv"
        write_submission(path, first)
        strict_submission(path, N_FRAMES)
        report.update(prediction_sha256=sha256(path), repeat_identical=True, frames=N_FRAMES)
        assets = games.asset_paths(source="tactical")
        report["input_sha256"] = {key: sha256(assets[key]) for key in ("atd_meta", "atd_raw")}
    else:
        plan = load_config("config/external_benchmark.toml")
        for game in (1, 2):
            tracking, runs, reference, _ = read_sample_game(
                ROOT / "data/external/metrica_sample", game, plan["source"]
            )
            item = tracking_item(tracking, runs, parent)
            prediction = predict(item, net, metadata, parent, device)
            metrics = anchor_metrics(reference, prediction)
            report["rows"].append({"game": game, "metrics": metrics})
            print(f"Metrica {game}: {metrics['pass_start_team_accuracy']:.4%}", flush=True)
    atomic_write(out / f"{args.mode}_report.json", json_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
