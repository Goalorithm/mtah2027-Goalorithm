"""Final fixed-schedule refit on train+old validation games, never claiming new validation."""

import copy

import _src_path  # noqa: F401
import numpy as np
import torch
from evaluate_temporal import load_model
from prepare_temporal import load_profiles
from refit_external import validated_refit_games
from skillcorner_data import load_profiles as source_profiles
from temporal_losses import augment_features, transition_loss
from temporal_model import TemporalNet
from torch.nn import functional as F
from train_supervised import json_bytes, training_weights
from train_temporal import sample_batch

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def main():
    cfg = load_config("config/temporal_refit.toml")
    out = OUTPUTS_DIR / "temporal_refit"
    out.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    warm_path = ROOT / cfg["warm_start_dir"]
    warm, parent, _, _ = load_model(warm_path, "cpu")
    selection = load_config(cfg["parent_config"])["supervised"]["training"]
    fit_games = validated_refit_games(selection, cfg["actual_fit_games_idsse"])
    strong, counts = [], {}
    for split, games in (("train", cfg["train_games"]), ("validation", cfg["validation_games"])):
        for index, game in enumerate(games):
            items = load_profiles(game, index, split, cfg)
            counts[game] = int(np.sum(items[0]["reference"] != 3))
            for item in items:
                item["weights"] = training_weights(
                    item["reference"], item["runs"], cfg["short_spell_weight_exponent"], 1.5, 8
                ).astype(np.float32)
                item["changes"] = np.flatnonzero(np.r_[False, np.diff(item["reference"]) != 0])
            strong.extend(items)
    if set(counts) != set(fit_games):
        raise ValueError("loaded strong-reference games differ from declared refit set")
    source_cfg = load_config(cfg["source_config"])
    weak = [
        item
        for game in source_cfg["train_games"]
        for item in source_profiles(game, "train", parent)
    ]
    for item in weak:
        item["weights"] *= 4
    items = strong * 2 + weak
    mean, scale = np.asarray(warm["mean"], np.float32), np.asarray(warm["scale"], np.float32)
    net = TemporalNet(len(warm["features"]), cfg["channels"], cfg["dilations"], cfg["dropout"]).to(
        device
    )
    net.load_state_dict(
        torch.load(warm_path / "weights.pt", map_location=device, weights_only=True)
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    event_weight = torch.tensor([8.0, 8.0, 1.0], device=device)
    rng, history = np.random.default_rng(cfg["seed"]), []
    print(
        f"Refitting {sum(counts.values())} strong frames "
        f"plus partial source training games on {device}",
        flush=True,
    )
    net.train()
    for epoch in range(1, cfg["epochs"] + 1):
        total = 0.0
        for _ in range(cfg["steps_per_epoch"]):
            values, labels, target, weights = [
                v.to(device)
                for v in sample_batch(items, cfg, warm["features"], mean, scale, net.halo, rng)
            ]
            values = augment_features(values, warm["features"], cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                state, event = net(values)
                sl = slice(net.halo, net.halo + cfg["core_frames"])
                error = F.cross_entropy(state[:, :, sl], labels, ignore_index=3, reduction="none")
                loss = (error * weights).sum() / weights.sum().clamp_min(1)
                if (target != 3).any():
                    loss += cfg["boundary_loss_weight"] * F.cross_entropy(
                        event[:, :, sl], target, weight=event_weight, ignore_index=3
                    )
            loss += cfg["transition_loss_weight"] * transition_loss(state[:, :, sl], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        history.append({"epoch": epoch, "training_loss": total / cfg["steps_per_epoch"]})
        print(
            f"Fixed refit epoch {epoch}/{cfg['epochs']}: loss {history[-1]['training_loss']:.5f}",
            flush=True,
        )
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "weights.pt")
    atomic_write(out / "parent.json", json_bytes(parent))
    provenance = {
        "config_sha256": sha256(ROOT / "config/temporal_refit.toml"),
        "parent_sha256": warm["provenance"]["parent_sha256"],
        "parent_config_sha256": warm["provenance"]["parent_config_sha256"],
        "source_sha256": warm["provenance"]["source_sha256"],
        "helper_sha256": {
            name: sha256(ROOT / name)
            for name in (
                "tools/refit_temporal.py",
                "tools/refit_external.py",
                "tools/prepare_temporal.py",
                "tools/skillcorner_data.py",
                "tools/temporal_model.py",
                "tools/temporal_losses.py",
                "tools/train_temporal.py",
                "tools/train_supervised.py",
                "tools/train_boundaries.py",
                "tools/evaluate_temporal.py",
                "config/skillcorner.toml",
                "data/external/skillcorner_new/manifest.json",
            )
        },
        "actual_fit_games_idsse": fit_games,
        "additional_training_games_skillcorner": source_cfg["train_games"],
        "consumed_former_validation_games": cfg["validation_games"],
        "warm_start_model_sha256": sha256(warm_path / "model.json"),
        "torch_version": str(torch.__version__),
        "device": device,
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
    }
    metadata = copy.deepcopy(warm)
    metadata.update(
        config=cfg,
        config_path="config/temporal_refit.toml",
        selected_epoch=cfg["epochs"],
        weights_sha256=sha256(out / "weights.pt"),
        frozen_parent_sha256=sha256(out / "parent.json"),
        provenance=provenance,
        actual_fit_games_idsse=fit_games,
        refitted_on_train_and_validation=True,
    )
    atomic_write(out / "model.json", json_bytes(metadata))
    report = {
        "actual_fit_games_idsse": fit_games,
        "original_strong_frames": sum(counts.values()),
        "strong_frames_by_game": counts,
        "additional_training_games_skillcorner": source_cfg["train_games"],
        "validation_games": [],
        "validation": [],
        "learning_curve": history,
        "selected_epoch": cfg["epochs"],
        "selection_basis": "fixed last epoch, specified before fitting",
        "warm_start_model_sha256": provenance["warm_start_model_sha256"],
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
                "test_previous_exposure": "reused after earlier TCN/domain models; not untouched",
            }
        ),
    )
    print(
        "Final refit frozen; former validation games are now fitting data, no new validation score",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
