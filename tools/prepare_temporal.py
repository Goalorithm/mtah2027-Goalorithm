"""Cache raw interaction features and frozen teacher outputs, separating reference labels."""

import json

import _src_path  # noqa: F401
import numpy as np
from train_boundaries import profiles
from train_supervised import training_weights

from mtah.boundary import boundary_targets
from mtah.compiled_trees import compiled_symmetric_probabilities
from mtah.config import load_config
from mtah.idsse import game_assets, read_game
from mtah.io import sha256
from mtah.paths import ROOT
from mtah.supervised import feature_matrix


def load_parent(cfg):
    config = load_config(ROOT / cfg["parent_config"])
    path = ROOT / config["supervised"]["model_path"]
    return config, json.loads(path.read_text(encoding="utf-8")), path


def load_profiles(game, index, split, cfg):
    config, parent, path = load_parent(cfg)
    features = config["supervised"]["features"]
    training = config["supervised"]["training"]
    directory = ROOT / "outputs/temporal_cache"
    directory.mkdir(exist_ok=True)
    assets = game_assets(ROOT / "data/external/idsse", game)
    signature = {
        "assets": {k: sha256(v) for k, v in assets.items()},
        "sources": {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / "src/mtah").glob("*.py")},
        "preparer": sha256(__file__),
        "parent": sha256(path),
        "split": split,
        "index": index,
        "seed": training["seed"],
    }
    outputs = []
    if split == "train":
        seed = training["seed"] + index
    elif split == "validation":
        seed = training["seed"] + 100 + index
    else:
        seed = training["seed"] + 200 + index
    # Test profiles use precisely the benchmark's separate jitter seed, not the training seed.
    paths = [
        directory / f"{game}_{split}_{name}.npz"
        for name in ("clean", "occluded", "occluded_jittered")
    ]
    try:
        for p in paths:
            with np.load(p, allow_pickle=False) as data:
                if json.loads(str(data["signature"])) != signature:
                    raise ValueError("cache signature changed")
                outputs.append({key: data[key] for key in data.files if key != "signature"})
        return outputs
    except (OSError, ValueError, KeyError):
        outputs = []
    print(f"Preparing temporal {split} features: {game}", flush=True)
    tracking, reference, runs, _ = read_game(ROOT / "data/external/idsse", game)
    items = profiles(tracking, runs, training, seed)
    if split == "test":
        from mtah.supervised import perturb_ball

        items[-1] = ("occluded_jittered", perturb_ball(items[1][1], 0.15, seed + 100))
    for (profile, current), p in zip(items, paths, strict=True):
        original = feature_matrix(current, features, runs)
        state = compiled_symmetric_probabilities(original, parent)
        event = compiled_symmetric_probabilities(original, parent["boundary_classifier"])
        values = original.astype(np.float32)
        item = {
            "values": values,
            "state": state.astype(np.float32),
            "event": event.astype(np.float32),
            "reference": reference,
            "runs": np.asarray(runs),
            "target": boundary_targets(reference, runs, 2),
            "weights": training_weights(reference, runs, 0.5, 1.5, 8).astype(np.float32),
            "game": np.asarray(game),
            "profile": np.asarray(profile),
        }
        np.savez_compressed(p, **item, signature=json.dumps(signature, sort_keys=True))
        outputs.append(item)
        print(f"Cached {game}/{profile}", flush=True)
    return outputs


if __name__ == "__main__":
    cfg = load_config("config/temporal.toml")
    for split, key in (("train", "train_games"), ("validation", "validation_games")):
        for index, game in enumerate(cfg[key]):
            load_profiles(game, index, split, cfg)
