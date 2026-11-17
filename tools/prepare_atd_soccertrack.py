"""ATD-like fitting profiles of SoccerTrack v2 halves: a third provider joins the fit.

IDSSE and Metrica give five fitting games. SoccerTrack v2 (CC BY 4.0) adds eight matches from
another league and provider, with the provider's own per-frame possession state as reference.
Each half is degraded by tools/atd_sim.py exactly like the other fitting games. The provider's
contested frames carry no label here: they are simulated with the last owner (the simulator
needs to know where stoppages are) but are left ungraded in the fitting targets, so the model is
never told which team owns a contested ball.

The two matches of `matches` in config/soccertrack.toml are never read here; they remain the
check of tools/evaluate_atd_soccertrack.py.
"""

import argparse
import hashlib
import json

import _src_path  # noqa: F401
import atd_sim
import numpy as np
from prepare_atd import SOURCES, build_item, feature_config
from soccertrack_data import keep_last_owner, read_half, with_provider_ball
from train_supervised import training_weights

from mtah.boundary import boundary_targets
from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import ROOT

OWN_SOURCES = ("tools/soccertrack_data.py", "tools/prepare_atd_soccertrack.py")
HALVES = ("1st", "2nd")


def profile_plan(cfg, source):
    """Every (match, half, index, seed, parameters), in a fixed order."""
    plan = []
    settings = cfg["soccertrack_train"]
    for position, match in enumerate(source["train_matches"]):
        if match in source["matches"]:
            raise ValueError(f"match {match} is reserved for the independent check")
        for part, half in enumerate(HALVES):
            for index in range(settings["profiles"]):
                seed = cfg["seed"] + settings["seed_offset"] + 1000 * (2 * position + part) + index
                params = atd_sim.sample_parameters(
                    cfg["simulation"]["train"], np.random.default_rng(seed)
                )
                plan.append((match, half, index, seed, params))
    return plan


def _signature(cfg, source, match, half, index, seed, params):
    fixed = {k: v for k, v in cfg["simulation"].items() if k not in ("train", "benchmark")}
    payload = {
        "sources": {name: sha256(ROOT / name) for name in (*SOURCES, *OWN_SOURCES)},
        "features": feature_config(cfg),
        "team_features": cfg["team_features"],
        "simulation": fixed,
        "params": params,
        "weights": [cfg["short_spell_exponent"], cfg["boundary_weight"], cfg["boundary_frames"]],
        "revision": source["revision"],
        "key": [match, half, index, seed],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build(item, cfg, name, index, seed, params):
    reference, runs = item["reference"], item["runs"]
    owner = keep_last_owner(reference)
    built = build_item(
        with_provider_ball(item), owner, runs, None, cfg, name, "extra", index, seed, params
    )
    # Contested frames were simulated with the last owner; they are not fitting targets.
    built["reference"] = np.asarray(reference, dtype=np.int8)
    built["target"] = boundary_targets(reference, runs, 2)
    built["weights"] = training_weights(
        reference,
        runs,
        cfg["short_spell_exponent"],
        cfg["boundary_weight"],
        cfg["boundary_frames"],
    ).astype(np.float32)
    return built


def load_extra(cfg, matches=None):
    """Cached SoccerTrack fitting profiles; a half is parsed only when a profile is missing."""
    settings = cfg["soccertrack_train"]
    source = load_config(ROOT / settings["config"])
    cache = ROOT / "outputs" / settings["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    items, clean = [], {}
    for match, half, index, seed, params in profile_plan(cfg, source):
        if matches and match not in matches:
            continue
        name = f"{match}_{half}"
        signature = _signature(cfg, source, match, half, index, seed, params)
        path = cache / f"{name}_extra{index}.npz"
        item = None
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    item = {key: data[key] for key in data.files if key != "signature"}
        if item is None:
            if name not in clean:
                clean.clear()  # one parsed half in memory at a time
                clean[name] = read_half(match, half, source)
            item = build(clean[name], cfg, name, index, seed, params)
            np.savez_compressed(path, **item, signature=signature)
            print(f"Cached ATD-like SoccerTrack profile {name}/{index}", flush=True)
        item["params"] = params
        items.append(item)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/atd.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--matches", nargs="*", default=None, help="only these matches")
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.overrides)
    loaded = load_extra(config, arguments.matches)
    seen = np.mean([item["ball_seen"].mean() for item in loaded])
    graded = np.mean([(item["reference"] != 3).mean() for item in loaded])
    print(f"{len(loaded)} SoccerTrack profiles, ball seen {seen:.3f}, graded {graded:.3f}")
