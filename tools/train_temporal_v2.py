"""Train a local GPU TCN on external training games; select epochs/mixtures on validation."""

import argparse
import copy
import random
from itertools import product
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
import torch
from benchmark_idsse import verify_freeze
from prepare_temporal import load_parent, load_profiles
from temporal_losses import augment_features, transition_loss
from temporal_model import TemporalNet, blend, encode, normalization, symmetric_infer
from torch.nn import functional as F
from train_supervised import json_bytes, metric_summary, training_weights

from mtah.boundary import decode_boundaries
from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import swap_features


def validate_games(cfg):
    groups = [cfg[k] for k in ("train_games", "validation_games", "test_games")]
    if any(not group or len(set(group)) != len(group) for group in groups):
        raise ValueError("game groups must be nonempty and unique")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("training/validation/test games overlap")


def sample_batch(items, cfg, names, mean, scale, halo, rng):
    size, core = cfg["batch_size"], cfg["core_frames"]
    width = core + 2 * halo
    inputs, labels, events, weights = [], [], [], []
    for _ in range(size):
        item = items[int(rng.integers(len(items)))]
        runs = item["runs"]
        lengths = runs[:, 1] - runs[:, 0]
        a, b = runs[rng.choice(len(runs), p=lengths / lengths.sum())]
        candidates = item["changes"]
        candidates = candidates[(candidates >= a) & (candidates < b)]
        if len(candidates) and rng.random() < 0.5:
            center = int(rng.choice(candidates)) + int(rng.integers(-core // 3, core // 3 + 1))
        else:
            center = int(rng.integers(a, b))
        start = center - core // 2
        lo, hi = start - halo, start + core + halo
        values = np.full((width, len(names)), np.nan, dtype=np.float32)
        y, e = np.full(core, 3, dtype=np.int64), np.full(core, 3, dtype=np.int64)
        w = np.zeros(core, dtype=np.float32)
        left, right = max(a, lo), min(b, hi)
        values[left - lo : right - lo] = item["values"][left:right]
        left, right = max(a, start), min(b, start + core)
        if left < right:
            sl = slice(left - start, right - start)
            y[sl], e[sl], w[sl] = (
                item["reference"][left:right],
                item["target"][left:right],
                item["weights"][left:right],
            )
        if rng.random() < 0.5:
            values = swap_features(values, names)
            y, e = np.array([1, 0, 2, 3])[y], np.array([1, 0, 2, 3])[e]
        inputs.append(encode(values, mean, scale).T)
        labels.append(y)
        events.append(e)
        weights.append(w)
    return tuple(torch.from_numpy(np.stack(part)) for part in (inputs, labels, events, weights))


def validation(net, items, parent, names, mean, scale, device, state_weight=0.5, event_weight=0):
    rows, probabilities = [], []
    for item in items:
        state, event = symmetric_infer(
            net, item["values"], item["runs"], names, mean, scale, device
        )
        prediction = decode_boundaries(
            blend(item["state"], state, state_weight),
            blend(item["event"], event, event_weight),
            parent,
            item["runs"],
        )
        metrics = metric_summary(item["reference"], prediction)
        rows.append(
            {"game": str(item["game"]), "profile": str(item["profile"]), "metrics": metrics}
        )
        probabilities.append((state, event))
    return rows, probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/temporal_v2.toml")
    parser.add_argument("--out-dir", default="outputs/temporal_v2")
    args = parser.parse_args(argv)
    cfg, out = load_config(args.config), Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("output must stay under ignored outputs/")
    validate_games(cfg)
    parent_config, parent, parent_path = load_parent(cfg)
    verify_freeze(ROOT / cfg["parent_config"], parent_path)
    expected = parent_config["supervised"]["training"]
    if any(cfg[k] != expected[k] for k in ("train_games", "validation_games", "test_games")):
        raise ValueError("temporal model split must match frozen teacher")
    out.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Temporal device: {device}; torch {torch.__version__}", flush=True)
    train = [
        item
        for i, game in enumerate(cfg["train_games"])
        for item in load_profiles(game, i, "train", cfg)
    ]
    valid = [
        item
        for i, game in enumerate(cfg["validation_games"])
        for item in load_profiles(game, i, "validation", cfg)
    ]
    for item in train:
        item["weights"] = training_weights(
            item["reference"], item["runs"], cfg["short_spell_weight_exponent"], 1.5, 8
        ).astype(np.float32)
        item["changes"] = np.flatnonzero(np.r_[False, np.diff(item["reference"]) != 0])
    names = parent["feature_names"]
    mean, scale = normalization([item["values"] for item in train], names)
    net = TemporalNet(len(names), cfg["channels"], cfg["dilations"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    # Rare directed events have bounded weights; no-event remains class 2.
    event_weight = torch.tensor([8.0, 8.0, 1.0], device=device)
    rng, history, best_score, best_epoch = np.random.default_rng(seed), [], -1.0, 0
    learning_curve = []
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        total = 0.0
        for _ in range(cfg["steps_per_epoch"]):
            values, labels, target, weights = [
                x.to(device) for x in sample_batch(train, cfg, names, mean, scale, net.halo, rng)
            ]
            values = augment_features(values, names, cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                state, events = net(values)
                sl = slice(net.halo, net.halo + cfg["core_frames"])
                loss_state = F.cross_entropy(
                    state[:, :, sl], labels, ignore_index=3, reduction="none"
                )
                loss = (loss_state * weights).sum() / weights.sum().clamp_min(1)
                loss += cfg["boundary_loss_weight"] * F.cross_entropy(
                    events[:, :, sl], target, weight=event_weight, ignore_index=3
                )
            loss += cfg["transition_loss_weight"] * transition_loss(state[:, :, sl], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        learning_curve.append({"epoch": epoch, "training_loss": total / cfg["steps_per_epoch"]})
        print(
            f"Epoch {epoch}/{cfg['epochs']}: train loss {total / cfg['steps_per_epoch']:.5f}",
            flush=True,
        )
        if epoch % cfg["validation_every"] == 0 or epoch == cfg["epochs"]:
            rows, _ = validation(net, valid, parent, names, mean, scale, device)
            score = float(np.mean([row["metrics"]["spell_f1_mean"] for row in rows]))
            history.append({"epoch": epoch, "mean_spell_f1": score, "validation": rows})
            print(f"Epoch {epoch}: validation spellF1 {score:.5f}", flush=True)
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save(
                    {k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "weights.pt"
                )
    net.load_state_dict(torch.load(out / "weights.pt", map_location=device, weights_only=True))
    _, probabilities = validation(net, valid, parent, names, mean, scale, device)
    selections = []
    for weight, event_mix, decoder_scale in product(
        cfg["mixture_weights"], cfg["event_mixture_weights"], cfg["decoder_scales"]
    ):
        scores, rows = [], []
        decoder = {**parent, "decoder_scale": decoder_scale}
        for item, (state, event) in zip(valid, probabilities, strict=True):
            prediction = decode_boundaries(
                blend(item["state"], state, weight),
                blend(item["event"], event, event_mix),
                decoder,
                item["runs"],
            )
            metrics = metric_summary(item["reference"], prediction)
            scores.append(metrics["spell_f1_mean"])
            rows.append(
                {"game": str(item["game"]), "profile": str(item["profile"]), "metrics": metrics}
            )
        entry = {
            "state_weight": weight,
            "event_weight": event_mix,
            "decoder_scale": decoder_scale,
            "mean_spell_f1": float(np.mean(scores)),
            "validation": rows,
        }
        selections.append(entry)
        print(
            f"Mix state={weight},event={event_mix},scale={decoder_scale}: "
            f"val {entry['mean_spell_f1']:.5f}",
            flush=True,
        )
    best = min(
        selections,
        key=lambda r: (
            -r["mean_spell_f1"],
            r["state_weight"],
            r["event_weight"],
            r["decoder_scale"],
        ),
    )
    # Save probabilities for reproducible decoder research without fitting on validation labels.
    for item, (state, event) in zip(valid, probabilities, strict=True):
        np.savez_compressed(
            out / f"validation_{item['game']}_{item['profile']}.npz", state=state, event=event
        )
    frozen_parent = copy.deepcopy(parent)
    atomic_write(out / "parent.json", json_bytes(frozen_parent))
    provenance = {
        "config_sha256": sha256(args.config),
        "parent_sha256": sha256(parent_path),
        "parent_config_sha256": sha256(ROOT / cfg["parent_config"]),
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in (ROOT / "src/mtah").glob("*.py")
        },
        "helper_sha256": {
            p: sha256(ROOT / p)
            for p in (
                "tools/temporal_model.py",
                "tools/prepare_temporal.py",
                "tools/train_temporal.py",
                "tools/train_temporal_v2.py",
                "tools/temporal_losses.py",
                "tools/train_boundaries.py",
                "tools/train_supervised.py",
                "tools/benchmark_idsse.py",
            )
        },
        "torch_version": str(torch.__version__),
        "device": device,
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
    }
    metadata = {
        "config_path": args.config,
        "config": cfg,
        "features": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "state_weight": best["state_weight"],
        "event_weight": best["event_weight"],
        "decoder_scale": best["decoder_scale"],
        "selected_epoch": best_epoch,
        "weights_sha256": sha256(out / "weights.pt"),
        "frozen_parent_sha256": sha256(out / "parent.json"),
        "provenance": provenance,
    }
    atomic_write(out / "model.json", json_bytes(metadata))
    report = {
        "selection": selections,
        "epoch_history": history,
        "learning_curve": learning_curve,
        "learned_parameters": sum(p.numel() for p in net.parameters()),
        "selected": best,
        "model_sha256": sha256(out / "model.json"),
        "provenance": provenance,
        "competition_accuracy_measured": False,
        "selected_final_competition_model": None,
    }
    atomic_write(out / "training_report.json", json_bytes(report))
    atomic_write(
        out / "test_freeze.json",
        json_bytes(
            {
                "model_sha256": sha256(out / "model.json"),
                "weights_sha256": sha256(out / "weights.pt"),
                "report_sha256": sha256(out / "training_report.json"),
                "frozen_before_test_evaluation": True,
                "test_previous_exposure": cfg["test_previous_exposure"],
            }
        ),
    )
    print(
        f"Frozen epoch {best_epoch}, validation {best['mean_spell_f1']:.5f}; no test games fitted",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
