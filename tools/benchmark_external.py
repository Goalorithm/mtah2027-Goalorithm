"""Frozen other-game evaluation on Metrica pass-start anchors; not full possession accuracy."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import _src_path  # noqa: F401

from mtah.config import load_config
from mtah.external import anchor_metrics, predict_many_external, read_sample_game
from mtah.io import sha256, write_submission
from mtah.paths import CONFIG_DIR, OUTPUTS_DIR, ROOT
from mtah.review import atomic_write


def source_files(directory, source, download=False):
    """Fetch exact pinned files or verify an existing local copy; never upload anything."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, digest in sorted(source["sha256"].items()):
        if Path(name).name != name:
            raise ValueError("source filenames must not contain directories")
        path = directory / name
        if name == "README.md":
            relative = name
        else:
            game = name.split("_")[2]
            relative = f"data/Sample_Game_{game}/{name}"
        url = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{source['commit']}/{relative}"
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"missing external asset {name}; use --download")
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                print(f"Downloading external sample: {name}", flush=True)
                request = urllib.request.Request(
                    url, headers={"User-Agent": "Goalorithm-evaluation"}
                )
                with (
                    urllib.request.urlopen(request, timeout=60) as response,
                    temporary.open("wb") as out,
                ):
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                if sha256(temporary) != digest:
                    raise ValueError(f"download hash mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if sha256(path) != digest:
            raise ValueError(f"external asset changed from pinned source: {name}")
        manifest.append({"file": name, "url": url, "sha256": digest, "bytes": path.stat().st_size})
    return manifest


def markdown_report(report):
    lines = [
        "# External pass-start benchmark",
        "",
        "Reference: Metrica Sports sample-data, two separate anonymised sample games.",
        "This measures team accuracy only at PASS start frames. It does not measure full H/A/D",
        "accuracy, dead-ball accuracy, transition timing, or spellF1. No model is fitted here.",
        "",
        "| Game | Config | Anchors | Correct team | Pass-start accuracy | Balanced recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for game in report["games"]:
        for row in game["models"]:
            metric = row["metrics"]
            lines.append(
                f"| {game['metadata']['game']} | {row['config']} | {metric['anchor_count']} | "
                f"{metric['correct_team_count']} | {metric['pass_start_team_accuracy']:.2%} | "
                f"{metric['balanced_team_recall']:.2%} |"
            )
    lines += [
        "",
        "The same frozen configs are evaluated on both games. Events are evaluation references",
        "only and never enter inference. No BAR-ATH reference labels are read.",
        "Both periods are decoded separately. Coordinates are affine-rescaled to the model pitch.",
        "These sample games have no Smart Tagging features; transfer involves a domain shift.",
        "A high pass-start score cannot establish the best model for the competition ranking.",
        "",
        "Source acknowledgement: https://github.com/metrica-sports/sample-data",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG_DIR / "external_benchmark.toml"))
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "external" / "metrica_sample"))
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "external_benchmark"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.out_dir).resolve()
    if not directory.is_relative_to(OUTPUTS_DIR.resolve()):
        print("ERROR: benchmark outputs must remain under ignored outputs/")
        return 1
    try:
        config = load_config(args.config)
        models = []
        for name in config["benchmark"]["model_configs"]:
            path = ROOT / name
            candidate = load_config(path)
            metadata = {"config": name, "config_sha256": sha256(path)}
            method = candidate["dead_ball"]["method"]
            if method == "supervised":
                metadata["checkpoint_sha256"] = sha256(ROOT / candidate[method]["model_path"])
            models.append((candidate, metadata))
        assets = source_files(args.data_dir, config["source"], args.download)
        report = {
            "source": config["source"]["repository"],
            "source_commit": config["source"]["commit"],
            "source_acknowledgement": "Metrica Sports sample-data; see source README Legal stuff",
            "assets": assets,
            "source_code_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in sorted((ROOT / "src" / "mtah").glob("*.py"))
            },
            "benchmark_config_sha256": sha256(args.config),
            "benchmark_script_sha256": sha256(__file__),
            "models_fitted_or_tuned": False,
            "bar_ath_reference_labels_read": False,
            "fullmatch_accuracy_measured": False,
            "selected_final_model": None,
            "games": [],
        }
        for game in config["source"]["games"]:
            tracking, runs, anchors, metadata = read_sample_game(
                args.data_dir, game, config["source"]
            )
            rows = []
            predictions = predict_many_external(
                [candidate for candidate, _ in models], tracking, runs
            )
            for prediction, (_, model_metadata) in zip(predictions, models, strict=True):
                output = directory / f"sample{game}_{Path(model_metadata['config']).stem}.csv"
                write_submission(output, prediction)
                metrics = anchor_metrics(anchors, prediction)
                rows.append(
                    {**model_metadata, "metrics": metrics, "prediction_sha256": sha256(output)}
                )
                print(
                    f"Sample {game} / {Path(model_metadata['config']).stem}: "
                    f"{metrics['correct_team_count']}/{metrics['anchor_count']} "
                    "pass starts correct "
                    f"({metrics['pass_start_team_accuracy']:.2%}); not fullmatch accuracy",
                    flush=True,
                )
            report["games"].append({"metadata": metadata, "models": rows})
        atomic_write(
            directory / "report.json",
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        )
        atomic_write(directory / "report.md", markdown_report(report).encode())
        print(f"-> {directory / 'report.md'}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
