"""Cache ATD-like profiles of labelled external games: features plus separate references."""

import hashlib
import json

import _src_path  # noqa: F401
import atd_sim
import numpy as np
from atd_features import atd_feature_matrix, atd_feature_names
from broadcast_cut import simulate_cutaways
from metrica_epts import FILES, read_labelled_epts_game
from metrica_labels import read_labelled_game
from train_supervised import training_weights

from mtah.boundary import boundary_targets
from mtah.config import load_config
from mtah.idsse import read_game
from mtah.io import sha256
from mtah.paths import ROOT

SOURCES = (
    "tools/atd_sim.py",
    "tools/atd_features.py",
    "tools/broadcast_cut.py",
    "tools/touch_features.py",
    "tools/metrica_labels.py",
    "tools/metrica_epts.py",
    "tools/prepare_atd.py",
    "src/mtah/supervised.py",
    "src/mtah/interaction.py",
    "src/mtah/hmm.py",
    "src/mtah/features.py",
    "src/mtah/idsse.py",
    "src/mtah/external.py",
)


def cache_directory(cfg):
    """Profiles of a what-if experiment live in their own folder so runs never collide."""
    return ROOT / "outputs" / cfg.get("cache_dir", "atd_cache")


def feature_config(cfg):
    return load_config(ROOT / cfg["parent_config"])["supervised"]["features"]


def load_game(game, cfg):
    """(tracking, reference, runs, ball height or None); integers are Metrica games."""
    if game == cfg["metrica_epts"]["game"]:
        directory = ROOT / cfg["metrica_epts"]["directory"]
        for key, name in FILES.items():
            if sha256(directory / name) != cfg["metrica_epts"]["sha256"][key]:
                raise ValueError(f"Metrica EPTS sample file differs from the pinned source: {name}")
        tracking, reference, runs, _ = read_labelled_epts_game(directory, cfg["metrica_bridge"])
        return tracking, reference, runs, None
    if isinstance(game, int):
        plan = load_config(ROOT / "config/external_benchmark.toml")
        tracking, reference, runs, _ = read_labelled_game(
            ROOT / "data/external/metrica_sample", game, plan["source"], cfg["metrica_bridge"]
        )
        return tracking, reference, runs, None
    tracking, reference, runs, _ = read_game(ROOT / "data/external/idsse", game)
    return tracking, reference, runs, None


def profile_plan(cfg):
    """Every (split, game, index, seed, parameters) the workflow uses, in a fixed order."""
    plan = []
    groups = (
        ("train", [*cfg["train_games"], *cfg["metrica_games"]], cfg["train_profiles"]),
        (
            "validation",
            [*cfg["validation_games"], *cfg["metrica_validation_games"]],
            cfg["validation_profiles"],
        ),
        ("test", cfg["test_games"], cfg["test_profiles"]),
        # Final refit only: the validation games with training-style random parameters.
        (
            "refit",
            [*cfg["validation_games"], *cfg["metrica_validation_games"]],
            cfg["train_profiles"],
        ),
    )
    for offset, (split, games, count) in enumerate(groups):
        for position, game in enumerate(games):
            for index in range(count):
                seed = cfg["seed"] + 100000 * offset + 1000 * position + index
                if split in ("train", "refit"):
                    params = atd_sim.sample_parameters(
                        cfg["simulation"]["train"], np.random.default_rng(seed)
                    )
                else:
                    params = dict(cfg["simulation"]["benchmark"])
                plan.append((split, game, index, seed, params))
    return plan


def _signature(cfg, split, game, index, seed, params):
    fixed = {k: v for k, v in cfg["simulation"].items() if k not in ("train", "benchmark")}
    payload = {
        "sources": {name: sha256(ROOT / name) for name in SOURCES},
        "features": feature_config(cfg),
        "team_features": cfg["team_features"],
        "simulation": fixed,
        "params": params,
        "weights": [cfg["short_spell_exponent"], cfg["boundary_weight"], cfg["boundary_frames"]],
        "metrica_bridge": cfg["metrica_bridge"],
        "key": [split, str(game), index, seed],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_item(tracking, reference, runs, height, cfg, game, split, index, seed, params):
    degraded, video = atd_sim.simulate_game(
        tracking, reference, runs, params, cfg["simulation"], seed
    )
    cut = simulate_cutaways(reference, runs, params, cfg["simulation"], seed)
    values = atd_feature_matrix(
        degraded, feature_config(cfg), cfg["team_features"], runs, video, cut
    )
    return {
        # Features are bounded ratios; half precision keeps many profiles in 16 GB of RAM.
        "values": values.astype(np.float16),
        "reference": np.asarray(reference, dtype=np.int8),
        "runs": np.asarray(runs, dtype=np.int64),
        "target": boundary_targets(reference, runs, 2),
        "weights": training_weights(
            reference,
            runs,
            cfg["short_spell_exponent"],
            cfg["boundary_weight"],
            cfg["boundary_frames"],
        ).astype(np.float32),
        "game": np.asarray(str(game)),
        "profile": np.asarray(f"{split}{index}"),
        "ball_seen": np.isfinite(degraded.ball).all(axis=1),
        "video_seen": np.isfinite(video).all(axis=1),
        "cut": cut > 0.5,
    }


def load_split(cfg, split):
    """Cached items of one split; a clean game is read only when a profile is missing."""
    cache = cache_directory(cfg)
    cache.mkdir(parents=True, exist_ok=True)
    items, clean = [], {}
    for part, game, index, seed, params in profile_plan(cfg):
        if part != split:
            continue
        signature = _signature(cfg, split, game, index, seed, params)
        path = cache / f"{game}_{split}{index}.npz"
        item = None
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                if str(data["signature"]) == signature:
                    item = {key: data[key] for key in data.files if key != "signature"}
        if item is None:
            if game not in clean:
                clean.clear()  # keep one clean game in memory at a time
                clean[game] = load_game(game, cfg)
            item = build_item(*clean[game], cfg, game, split, index, seed, params)
            np.savez_compressed(path, **item, signature=signature)
            print(f"Cached ATD-like {split} profile {game}/{index}", flush=True)
        item["params"] = params
        items.append(item)
    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/atd.toml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test", "refit"])
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.overrides)
    names = atd_feature_names(feature_config(config))
    for name in arguments.splits:
        loaded = load_split(config, name)
        seen = np.mean([item["ball_seen"].mean() for item in loaded])
        print(f"{name}: {len(loaded)} profiles, {len(names)} features, ball seen {seen:.3f}")
