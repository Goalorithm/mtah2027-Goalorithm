"""Average several ATD-realistic TCNs and choose the shared decoder on ATD-like validation.

Members differ only in seed or one training setting. Their state and event probabilities are
averaged before decoding. The decoder scale and boundary weight of the ensemble are selected
on the same external validation profiles the members used; no test game or BAR-ATH frame is
read. The result is a manifest that tools/evaluate_atd.py loads like a single model.

With --soccertrack-validation the decoder is chosen on three label sources weighted equally:
the IDSSE validation games, the Metrica validation game, and SoccerTrack halves labelled by the
competition's possession rule (tools/soccertrack_validation.py). IDSSE counts possession more
finely than the competition describes it, so choosing on IDSSE profiles alone favours a decoder
that changes team too readily.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import _src_path  # noqa: F401
import torch
from evaluate_atd import load_member, probabilities
from prepare_atd import load_split
from train_atd import HELPERS, score
from train_supervised import json_bytes

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def balanced_score(groups, averaged, decoder, scale, weight):
    """Mean over label sources of the mean spellF1 of each source's profiles."""
    means, rows = {}, []
    for name, items in groups.items():
        means[name], part = score(items, averaged[name], decoder, scale, weight)
        rows += part
    return float(sum(means.values()) / len(means)), rows, means


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("members", nargs="+", help="member directory names under outputs/")
    parser.add_argument(
        "--soccertrack-validation",
        action="store_true",
        help="choose the decoder on IDSSE, Metrica and SoccerTrack profiles, weighted equally",
    )
    parser.add_argument("--out-dir", default="outputs/atd_ensemble")
    parser.add_argument(
        "--decoder-from",
        default=None,
        help="reuse this validation-selected decoder (refit members have no validation games)",
    )
    args = parser.parse_args(argv)
    out = Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()) or len(set(args.members)) < 2:
        parser.error("need two or more distinct members and an output under outputs/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    members = [load_member(OUTPUTS_DIR / name, device) for name in args.members]
    first = members[0][0]
    cfg = first["config"]
    for metadata, _ in members[1:]:
        same = all(
            metadata["config"][key] == cfg[key]
            for key in ("validation_games", "metrica_validation_games", "test_games")
        )
        if not same or metadata["features"] != first["features"]:
            raise ValueError("ensemble members must share features and held-out games")
    single, selections = [], []
    if args.decoder_from:
        source = json.loads((Path(args.decoder_from) / "model.json").read_text())
        best = {
            "decoder_scale": source["decoder_scale"],
            "boundary_weight": source["boundary_weight"],
            "mean_spell_f1": None,
            "decoder_from": Path(args.decoder_from).name,
        }
    else:
        valid = load_split(cfg, "validation")
        averaged = [
            probabilities(item["values"], item["runs"], first, members, device) for item in valid
        ]
        for name, member in zip(args.members, members, strict=True):
            own = [
                probabilities(item["values"], item["runs"], first, [member], device)
                for item in valid
            ]
            value, _ = score(
                valid,
                own,
                member[0]["decoder"],
                member[0]["decoder_scale"],
                member[0]["boundary_weight"],
            )
            single.append({"member": name, "mean_spell_f1": value})
            print(f"Member {name}: validation {value:.5f}", flush=True)
        scales = cfg["decoder_scales"]
        groups, grouped = {"validation": valid}, {"validation": averaged}
        if args.soccertrack_validation:
            from soccertrack_validation import validation_items

            current = load_config(ROOT / first["config_path"])
            scales = current["soccertrack_train"]["ensemble_decoder_scales"]
            metrica = {str(game) for game in cfg["metrica_validation_games"]}
            groups = {
                "idsse": [item for item in valid if str(item["game"]) not in metrica],
                "metrica": [item for item in valid if str(item["game"]) in metrica],
                "soccertrack": validation_items(current),
            }
            grouped = {
                name: [
                    probabilities(item["values"], item["runs"], first, members, device)
                    for item in items
                ]
                for name, items in groups.items()
            }
        for scale, weight in product(scales, cfg["boundary_weights"]):
            value, rows, means = balanced_score(groups, grouped, first["decoder"], scale, weight)
            selections.append(
                {
                    "decoder_scale": scale,
                    "boundary_weight": weight,
                    "mean_spell_f1": value,
                    "by_source": means,
                    "validation": rows,
                }
            )
            shown = ", ".join(f"{name} {mean:.4f}" for name, mean in means.items())
            print(
                f"Ensemble scale={scale}, boundary={weight}: val {value:.5f} ({shown})", flush=True
            )
        best = min(
            selections,
            key=lambda r: (-r["mean_spell_f1"], r["decoder_scale"], r["boundary_weight"]),
        )
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "members": {name: sha256(OUTPUTS_DIR / name / "model.json") for name in args.members},
        "config_path": first["config_path"],
        "config": cfg,
        "features": first["features"],
        "decoder": first["decoder"],
        "decoder_scale": best["decoder_scale"],
        "boundary_weight": best["boundary_weight"],
        "provenance": {
            "helper_sha256": {name: sha256(ROOT / name) for name in HELPERS},
            "bar_ath_labels_read": False,
            "bar_ath_data_read": False,
            "test_games_read": False,
        },
    }
    atomic_write(out / "model.json", json_bytes(manifest))
    report = {
        "members": single,
        "selection": selections,
        "selected": best,
        "model_sha256": sha256(out / "model.json"),
        "competition_accuracy_measured": False,
    }
    atomic_write(out / "training_report.json", json_bytes(report))
    print(json.dumps({k: best[k] for k in ("decoder_scale", "boundary_weight", "mean_spell_f1")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
