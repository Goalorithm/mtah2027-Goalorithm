"""Train the standalone ATD-realistic TCN; pick epoch and decoder on ATD-like validation.

Every fitting and validation profile is an external game degraded by tools/atd_sim.py.
No BAR-ATH frame or label is read here, and the IDSSE test games stay out of the process.

With --refit-from the validation games join the fit: the epoch count and decoder come from
the earlier validation-selected model and nothing is selected again.
"""

import argparse
import json
import random
from itertools import product
from pathlib import Path

import _src_path  # noqa: F401
import numpy as np
import torch
from atd_features import atd_feature_names, group_columns
from prepare_atd import feature_config, load_split
from temporal_model import TemporalNet, blend, encode, normalization, symmetric_infer
from torch.nn import functional as F
from train_supervised import (
    json_bytes,
    metric_summary,
    training_weights,
    transition_parameters,
)

from mtah import A, D, H, X
from mtah.boundary import boundary_targets, decode_boundaries
from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT
from mtah.review import atomic_write
from mtah.supervised import swap_features

HELPERS = (
    "tools/atd_sim.py",
    "tools/atd_features.py",
    "tools/broadcast_cut.py",
    "tools/touch_features.py",
    "tools/metrica_labels.py",
    "tools/metrica_epts.py",
    "tools/prepare_atd.py",
    "tools/train_atd.py",
    "tools/temporal_model.py",
)


def validate_games(cfg):
    groups = [
        [*cfg["train_games"], *cfg["metrica_games"]],
        [*cfg["validation_games"], *cfg["metrica_validation_games"]],
        cfg["test_games"],
    ]
    if any(not group or len(set(group)) != len(group) for group in groups):
        raise ValueError("game groups must be nonempty and unique")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i)):
        raise ValueError("training/validation/test games overlap")


def refit_plan(directory):
    """Epoch count and decoder of a validation-selected model or ensemble under outputs/."""
    directory = directory.resolve()
    if not directory.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("the selected model must be under outputs/")
    manifest = json.loads((directory / "model.json").read_text())
    names = list(manifest.get("members", {directory.name: None}))
    chosen = [
        json.loads((OUTPUTS_DIR / name / "model.json").read_text())["selected_epoch"]
        for name in names
    ]
    return {
        "source": directory.name,
        "source_sha256": sha256(directory / "model.json"),
        "member_epochs": chosen,
        "epochs": int(np.median(chosen)),
        "decoder_scale": manifest["decoder_scale"],
        "boundary_weight": manifest["boundary_weight"],
    }


def decoder_parameters(items, pseudocount):
    """Transition, initial and prior terms from clean training references, one per game."""
    first = {}
    for item in items:
        first.setdefault(str(item["game"]), item)
    references = [(item["reference"], [tuple(r) for r in item["runs"]]) for item in first.values()]
    result = transition_parameters(references, pseudocount)
    counts = np.zeros(3)
    for item in first.values():
        counts += np.bincount(item["target"][item["target"] != X], minlength=3)
    counts[[H, A]] = counts[[H, A]].mean()
    result["boundary_prior"] = (counts / counts.sum()).tolist()
    return result


def decode(state, event, decoder, runs, scale, weight):
    return decode_boundaries(
        blend(state, state, 0),
        blend(event, event, 0),
        {**decoder, "decoder_scale": scale, "boundary_weight": weight},
        [tuple(int(v) for v in run) for run in runs],
    )


def contested_to_last_owner(item, cfg):
    """Label a provider's contested frames the way the competition defines them.

    CHALLENGE.md: "In phases where the ball is contested or loose, possession stays with the
    last team that had it." Frames before the first owner of a half stay ungraded.
    """
    runs = [tuple(int(v) for v in run) for run in item["runs"]]
    reference = np.array(item["reference"], copy=True)
    for start, stop in runs:
        last = X
        for frame in range(start, stop):
            if reference[frame] == X:
                reference[frame] = last
            else:
                last = reference[frame]
    item["reference"] = reference
    item["target"] = boundary_targets(reference, runs, 2)
    item["weights"] = training_weights(
        reference,
        runs,
        cfg["short_spell_exponent"],
        cfg["boundary_weight"],
        cfg["boundary_frames"],
    ).astype(np.float32)


def sample_batch(items, cfg, names, mean, scale, halo, rng, extra=()):
    size, core = cfg["batch_size"], cfg["core_frames"]
    share = cfg.get("extra_train_share", 0.0) if len(extra) else 0.0
    width = core + 2 * halo
    swap = np.array([A, H, D, X])
    inputs, labels, events, weights = [], [], [], []
    for _ in range(size):
        pool = extra if share and rng.random() < share else items
        item = pool[int(rng.integers(len(pool)))]
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
        y, e = np.full(core, X, dtype=np.int64), np.full(core, X, dtype=np.int64)
        w = np.zeros(core, dtype=np.float32)
        left, right = max(a, lo), min(b, hi)
        values[left - lo : right - lo] = item["values"][left:right]
        left, right = max(a, start), min(b, start + core)
        if left < right:
            sl = slice(left - start, right - start)
            y[sl] = item["reference"][left:right]
            e[sl] = item["target"][left:right]
            w[sl] = item["weights"][left:right]
        w[y == X] = 0
        if rng.random() < 0.5:
            values = swap_features(values, names)
            y, e = swap[y], swap[e]
        inputs.append(encode(values, mean, scale).T)
        labels.append(y)
        events.append(e)
        weights.append(w)
    return tuple(torch.from_numpy(np.stack(part)) for part in (inputs, labels, events, weights))


def predict_probabilities(net, items, names, mean, scale, device):
    return [
        symmetric_infer(
            net,
            item["values"],
            [tuple(int(v) for v in run) for run in item["runs"]],
            names,
            mean,
            scale,
            device,
        )
        for item in items
    ]


def score(items, probabilities, decoder, scale, weight):
    rows = []
    for item, (state, event) in zip(items, probabilities, strict=True):
        prediction = decode(state, event, decoder, item["runs"], scale, weight)
        rows.append(
            {
                "game": str(item["game"]),
                "profile": str(item["profile"]),
                "metrics": metric_summary(item["reference"], prediction),
            }
        )
    return float(np.mean([row["metrics"]["spell_f1_mean"] for row in rows])), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/atd.toml")
    parser.add_argument("--out-dir", default="outputs/atd")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--refit-from",
        default=None,
        help="validation-selected model or ensemble under outputs/ whose epochs/decoder to reuse",
    )
    args = parser.parse_args(argv)
    cfg, out = load_config(args.config, args.overrides), Path(args.out_dir).resolve()
    if not out.is_relative_to(OUTPUTS_DIR.resolve()):
        parser.error("output must stay under ignored outputs/")
    validate_games(cfg)
    out.mkdir(parents=True, exist_ok=True)
    selected = refit_plan(Path(args.refit_from)) if args.refit_from else None
    epochs = selected["epochs"] if selected else cfg["epochs"]
    seed = cfg["training_seed"]  # cfg["seed"] fixes the simulated profiles instead
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ATD workflow device: {device}; torch {torch.__version__}", flush=True)
    train = load_split(cfg, "train")
    if selected:
        train, valid = train + load_split(cfg, "refit"), []
        limit = cfg["refit_profiles"]
        train = [item for item in train if int(str(item["profile"])[-1]) < limit]
    else:
        valid = load_split(cfg, "validation")
    extra = []
    if cfg.get("extra_train_share", 0.0) > 0:
        from prepare_atd_soccertrack import load_extra

        extra = load_extra(cfg)  # another provider's games; never validation or test games
        contested = cfg["soccertrack_train"]["contested"]
        if contested == "last_owner":
            for item in extra:
                contested_to_last_owner(item, cfg)
        elif contested != "ungraded":
            raise ValueError("soccertrack_train.contested must be last_owner or ungraded")
        print(f"Extra fitting profiles: {len(extra)}, contested frames {contested}", flush=True)
    for item in (*train, *extra):
        item["changes"] = np.flatnonzero(np.r_[False, np.diff(item["reference"]) != 0])
    every = atd_feature_names(feature_config(cfg))
    keep = group_columns(every, cfg["feature_groups"])
    names = [every[i] for i in keep]
    if len(keep) < len(every):
        for item in (*train, *extra, *valid):  # one profile at a time: memory never doubles
            item["values"] = np.ascontiguousarray(item["values"][:, keep])
    mean, scale = normalization([item["values"] for item in (*train, *extra)], names)
    decoder = decoder_parameters(train, cfg["transition_pseudocount"])
    net = TemporalNet(len(names), cfg["channels"], cfg["dilations"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    event_weight = torch.tensor([8.0, 8.0, 1.0], device=device)
    rng, history, curve = np.random.default_rng(seed), [], []
    best_score, best_epoch = -1.0, 0
    monitor_scale, monitor_weight = cfg["decoder_scales"][1], cfg["boundary_weights"][1]
    for epoch in range(1, epochs + 1):
        net.train()
        total = 0.0
        for _ in range(cfg["steps_per_epoch"]):
            values, labels, target, weights = [
                x.to(device)
                for x in sample_batch(train, cfg, names, mean, scale, net.halo, rng, extra)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                state, events = net(values)
                sl = slice(net.halo, net.halo + cfg["core_frames"])
                loss_state = F.cross_entropy(
                    state[:, :, sl].float(), labels, ignore_index=X, reduction="none"
                )
                loss = (loss_state * weights).sum() / weights.sum().clamp_min(1)
                loss = loss + cfg["boundary_loss_weight"] * F.cross_entropy(
                    events[:, :, sl].float(), target, weight=event_weight, ignore_index=X
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        curve.append({"epoch": epoch, "training_loss": total / cfg["steps_per_epoch"]})
        print(f"Epoch {epoch}/{epochs}: train loss {curve[-1]['training_loss']:.5f}", flush=True)
        if valid and (epoch % cfg["validation_every"] == 0 or epoch == epochs):
            probabilities = predict_probabilities(net, valid, names, mean, scale, device)
            value, rows = score(valid, probabilities, decoder, monitor_scale, monitor_weight)
            history.append({"epoch": epoch, "mean_spell_f1": value, "validation": rows})
            print(f"Epoch {epoch}: validation spellF1 {value:.5f}", flush=True)
            if value > best_score:
                best_score, best_epoch = value, epoch
                torch.save(
                    {k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "weights.pt"
                )
    selections = []
    if selected:
        best_epoch = epochs
        best = {
            "decoder_scale": selected["decoder_scale"],
            "boundary_weight": selected["boundary_weight"],
            "mean_spell_f1": None,
        }
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "weights.pt")
    else:
        net.load_state_dict(torch.load(out / "weights.pt", map_location=device, weights_only=True))
        probabilities = predict_probabilities(net, valid, names, mean, scale, device)
        for decoder_scale, weight in product(cfg["decoder_scales"], cfg["boundary_weights"]):
            value, rows = score(valid, probabilities, decoder, decoder_scale, weight)
            selections.append(
                {
                    "decoder_scale": decoder_scale,
                    "boundary_weight": weight,
                    "mean_spell_f1": value,
                    "validation": rows,
                }
            )
            print(f"Decoder scale={decoder_scale}, boundary={weight}: val {value:.5f}", flush=True)
        best = min(
            selections,
            key=lambda r: (-r["mean_spell_f1"], r["decoder_scale"], r["boundary_weight"]),
        )
    provenance = {
        "config_sha256": sha256(args.config),
        "overrides": args.overrides,
        "refit_from": selected,
        "helper_sha256": {name: sha256(ROOT / name) for name in HELPERS},
        "torch_version": str(torch.__version__),
        "device": device,
        "bar_ath_labels_read": False,
        "bar_ath_data_read": False,
        "test_games_read": False,
    }
    metadata = {
        "config_path": args.config,
        "config": {k: v for k, v in cfg.items() if k != "_path"},
        "features": names,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "decoder": decoder,
        "decoder_scale": best["decoder_scale"],
        "boundary_weight": best["boundary_weight"],
        "selected_epoch": best_epoch,
        "weights_sha256": sha256(out / "weights.pt"),
        "provenance": provenance,
    }
    atomic_write(out / "model.json", json_bytes(metadata))
    report = {
        "selection": selections,
        "epoch_history": history,
        "learning_curve": curve,
        "learned_parameters": sum(p.numel() for p in net.parameters()),
        "selected": best,
        "train_profiles": [
            {"game": str(i["game"]), "profile": str(i["profile"]), "params": i["params"]}
            for i in (*train, *extra)
        ],
        "model_sha256": sha256(out / "model.json"),
        "competition_accuracy_measured": False,
    }
    atomic_write(out / "training_report.json", json_bytes(report))
    shown = "refit, no validation" if selected else f"validation {best['mean_spell_f1']:.5f}"
    print(
        f"Selected epoch {best_epoch}, scale {best['decoder_scale']}, boundary "
        f"{best['boundary_weight']}: {shown}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
