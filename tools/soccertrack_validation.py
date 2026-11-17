"""SoccerTrack v2 validation profiles labelled by the competition's possession rule.

CHALLENGE.md defines possession by control: a touch without control changes nothing, and "in
phases where the ball is contested or loose, possession stays with the last team that had it".
IDSSE's possession channel is finer than that (7-11 possessions per live minute against 4-6
here), so a decoder chosen on IDSSE and Metrica games alone is chosen for another convention.
SoccerTrack's provider marks team control and contested phases separately; giving contested
frames to the last owner yields references that follow the competition's wording.

The profiles are ATD-like (benchmark parameters) halves of `validation_matches` in
config/soccertrack.toml. Those matches are never fitted on.
"""

import hashlib
import json

import _src_path  # noqa: F401
import atd_sim
import numpy as np
from atd_features import atd_feature_matrix
from broadcast_cut import simulate_cutaways
from prepare_atd import SOURCES, feature_config
from soccertrack_data import keep_last_owner, read_half, with_provider_ball

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import ROOT

OWN_SOURCES = ("tools/soccertrack_data.py", "tools/soccertrack_validation.py")
HALVES = ("1st", "2nd")


def validation_items(cfg):
    """Cached items shaped like prepare_atd.load_split's: values, reference, runs, game, profile."""
    settings = cfg["soccertrack_train"]
    source = load_config(ROOT / settings["config"])
    cache = ROOT / "outputs" / settings["validation_cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    params = dict(cfg["simulation"]["benchmark"])
    fixed = {k: v for k, v in cfg["simulation"].items() if k not in ("train", "benchmark")}
    items = []
    for position, match in enumerate(source["validation_matches"]):
        if match in source["train_matches"]:
            raise ValueError(f"match {match} is a fitting match")
        for part, half in enumerate(HALVES):
            clean = None
            for index in range(settings["validation_profiles"]):
                seed = cfg["seed"] + settings["validation_seed_offset"]
                seed += 1000 * (2 * position + part) + index
                payload = {
                    "sources": {name: sha256(ROOT / name) for name in (*SOURCES, *OWN_SOURCES)},
                    "features": feature_config(cfg),
                    "team_features": cfg["team_features"],
                    "simulation": fixed,
                    "params": params,
                    "revision": source["revision"],
                    "key": [match, half, index, seed],
                }
                signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                path = cache / f"{match}_{half}_validation{index}.npz"
                item = None
                if path.is_file():
                    with np.load(path, allow_pickle=False) as data:
                        if str(data["signature"]) == signature:
                            item = {key: data[key] for key in data.files if key != "signature"}
                if item is None:
                    clean = clean or read_half(match, half, source)
                    owner, runs = keep_last_owner(clean["reference"]), clean["runs"]
                    degraded, video = atd_sim.simulate_game(
                        with_provider_ball(clean), owner, runs, params, cfg["simulation"], seed
                    )
                    cut = simulate_cutaways(owner, runs, params, cfg["simulation"], seed)
                    values = atd_feature_matrix(
                        degraded, feature_config(cfg), cfg["team_features"], runs, video, cut
                    )
                    item = {
                        "values": values.astype(np.float16),
                        "reference": np.asarray(owner, dtype=np.int8),
                        "runs": np.asarray(runs, dtype=np.int64),
                        "game": np.asarray(f"{match}_{half}"),
                        "profile": np.asarray(f"validation{index}"),
                    }
                    np.savez_compressed(path, **item, signature=signature)
                    print(
                        f"Cached SoccerTrack validation profile {match}_{half}/{index}", flush=True
                    )
                items.append(item)
    return items


if __name__ == "__main__":
    loaded = validation_items(load_config(ROOT / "config/atd.toml"))
    print(f"{len(loaded)} SoccerTrack validation profiles")
