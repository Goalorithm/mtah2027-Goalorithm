"""Score ATD-realistic models on SoccerTrack v2: unseen games, league, provider and labels.

Every earlier external number comes from two providers (IDSSE, Metrica) whose games also fed
the fit. SoccerTrack v2 is a third: Japanese university matches tracked by another provider,
whose own per-frame possession state is the reference. No model has seen these games.

What is real here: the players' tracking and the possession labels. What is still simulated:
the measurement of the ball. The provider's ball is a reconstruction between annotated events,
so, exactly as for the IDSSE and Metrica test profiles, it is passed through tools/atd_sim.py
to become an automatic-like ball, a video-like ball and a cutaway signal. The panoramic video
is not used: it shows a yellow ball at night on frosted turf, which the white-ball detector
built for the competition footage does not cover.

The provider marks contested frames (status 3). The competition ground truth has no such state
and keeps the last owner, so both readings are scored: contested frames left ungraded, and
contested frames given to the previous owner.
"""

import argparse
import json
from pathlib import Path

import _src_path  # noqa: F401
import atd_sim
import numpy as np
import torch
from atd_features import atd_feature_matrix
from broadcast_cut import simulate_cutaways
from evaluate_atd import legacy_predictor, load_model, predict
from prepare_atd import feature_config
from soccertrack_data import keep_last_owner, read_half, with_provider_ball
from train_supervised import json_bytes, metric_summary

from mtah.config import load_config
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/atd_refit_ensemble")
    parser.add_argument("--legacy-dir", default="outputs/temporal")
    parser.add_argument("--config", default="config/soccertrack.toml")
    args = parser.parse_args(argv)
    out = Path(args.model_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("model must remain under ignored outputs/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metadata, members = load_model(out, device)
    cfg = metadata["config"]
    legacy = legacy_predictor(ROOT / args.legacy_dir, device) if args.legacy_dir else None
    source = load_config(ROOT / args.config)
    params = dict(cfg["simulation"]["benchmark"])
    rows = []
    for match in source["matches"]:
        for half in ("1st", "2nd"):
            item = read_half(match, half, source)
            clean, reference, runs = with_provider_ball(item), item["reference"], item["runs"]
            owner = keep_last_owner(reference)
            for profile in range(source["profiles"]):
                seed = source["seed"] + 1000 * len(rows) + profile
                degraded, video = atd_sim.simulate_game(
                    clean, owner, runs, params, cfg["simulation"], seed
                )
                cut = simulate_cutaways(owner, runs, params, cfg["simulation"], seed)
                values = atd_feature_matrix(
                    degraded, feature_config(cfg), cfg["team_features"], runs, video, cut
                ).astype(np.float16)
                prediction = predict(values, runs, metadata, members, device)
                earlier = (
                    legacy({"values": values, "runs": np.asarray(runs, dtype=np.int64)})
                    if legacy
                    else None
                )
                row = {
                    "match": match,
                    "half": half,
                    "profile": profile,
                    "contested_ungraded": metric_summary(reference, prediction),
                    "contested_to_last_owner": metric_summary(owner, prediction),
                }
                if earlier is not None:
                    row["earlier_contested_ungraded"] = metric_summary(reference, earlier)
                    row["earlier_contested_to_last_owner"] = metric_summary(owner, earlier)
                rows.append(row)
                print(
                    f"{match} {half} profile {profile}: spellF1 "
                    f"{row['contested_ungraded']['spell_f1_mean']:.4f} ungraded / "
                    f"{row['contested_to_last_owner']['spell_f1_mean']:.4f} last owner",
                    flush=True,
                )
    means = {
        key: {
            "spell_f1_mean": float(np.mean([row[key]["spell_f1_mean"] for row in rows])),
            "macro_f1": float(np.mean([row[key]["macro_f1"] for row in rows])),
        }
        for key in rows[0]
        if isinstance(rows[0][key], dict)
    }
    report = {
        "rows": rows,
        "means": means,
        "real": "player tracking and the provider's possession state",
        "simulated": "the ball measurement (tools/atd_sim.py, benchmark parameters) and cutaways",
        "games_seen_by_any_model": False,
        "bar_ath_accuracy_measured": False,
    }
    atomic_write(out / "soccertrack_report.json", json_bytes(report))
    print(json.dumps(means, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
