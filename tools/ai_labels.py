"""Generate reproducible local AI pseudo-labels, never human labels or independent ground truth.

Runs the HMM and baseline from tracking/tagging, without reading predictions or reference labels.
Exports the six previously planned windows for diagnostics. Disagreement is an uncertainty flag,
not an estimated error probability. No video or dataset is sent to an external service.
"""

import argparse
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import _src_path  # noqa: F401
import labels_to_csv as plans
import numpy as np

from mtah import LABELS, X
from mtah.config import load_config
from mtah.io import sha256, write_submission
from mtah.load import load_smart_tagging, load_tracking
from mtah.paths import CONFIG_DIR, LABELS_DIR, N_FRAMES, OUTPUTS_DIR, ROOT, games, pm
from mtah.predict import run_pipeline
from mtah.review import atomic_write
from mtah.scoring import match_stats

MODEL_NAME = "algorithm:HMM-Viterbi"


def csv_data(headers, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def annotation_rows(codes, baseline, windows, model_name=MODEL_NAME):
    """Inclusive intervals cover every window; no prediction is decided by a person."""
    codes, baseline = np.asarray(codes), np.asarray(baseline)
    if codes.ndim != 1 or codes.shape != baseline.shape:
        raise ValueError("predictions must be equal-length one-dimensional arrays")
    if (
        not np.isin(codes, range(len(LABELS))).all()
        or not np.isin(baseline, range(len(LABELS))).all()
    ):
        raise ValueError("predictions contain unknown label codes")
    changes, intervals, summaries, split_codes = [], [], [], {}
    for name, window in windows.items():
        split = window["split"]
        if split == "kalibrasi":
            continue  # Public labels are neither used nor exported by this workflow.
        start, stop = window["start"], window["end"] + 1
        if not 0 <= start < stop <= len(codes):
            raise ValueError("window is outside prediction frame range")
        segment, reference = codes[start:stop], baseline[start:stop]
        if (segment == X).any():
            raise ValueError("HMM must provide H/A/D throughout each requested window")
        painted = split_codes.setdefault(split, np.full(len(codes), X, dtype=np.int8))
        painted[start:stop] = segment
        bounds = np.r_[0, np.flatnonzero(segment[1:] != segment[:-1]) + 1, len(segment)]
        for first in bounds[:-1]:
            changes.append(
                {
                    "jendela": name,
                    "frame_mulai": start + int(first),
                    "label_ai": LABELS[int(segment[first])],
                    "anotator_ai": model_name,
                    "asal": "algorithmic_pseudo_label",
                }
            )
        # Split diagnostic intervals when either method changes, preserving exact boundaries.
        boundaries = np.r_[
            0,
            np.flatnonzero((segment[1:] != segment[:-1]) | (reference[1:] != reference[:-1])) + 1,
            len(segment),
        ]
        for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
            agrees = bool(segment[first] == reference[first])
            intervals.append(
                {
                    "jendela": name,
                    "frame_awal": start + int(first),
                    "frame_akhir": start + int(last) - 1,
                    "label_ai": LABELS[int(segment[first])],
                    "label_baseline": LABELS[int(reference[first])],
                    "metode_setuju": int(agrees),
                    "ketidakpastian": "tidak_terkalibrasi" if agrees else "metode_berbeda",
                }
            )
        summaries.append(
            {
                "window": name,
                "legacy_split_name": split,
                "frames": stop - start,
                "change_points": len(bounds) - 1,
                "disagreement_frames": int(np.sum(segment != reference)),
                "agreement_fraction": float(np.mean(segment == reference)),
                "label_counts": {
                    label: int(np.sum(segment == k)) for k, label in enumerate(LABELS)
                },
            }
        )
    if not summaries:
        raise ValueError("no non-calibration windows to annotate")
    return changes, intervals, summaries, split_codes


def export_bundle(directory, codes, baseline, windows, provenance, model_name=MODEL_NAME):
    """Keep generated pseudo-labels under ignored outputs, separate from human-label tools."""
    directory = Path(directory).resolve()
    if not directory.is_relative_to(OUTPUTS_DIR.resolve()):
        raise ValueError("AI annotation bundles must stay under the ignored outputs directory")
    changes, intervals, summaries, splits = annotation_rows(codes, baseline, windows, model_name)
    files = {
        "perubahan_ai.csv": csv_data(tuple(changes[0]), changes),
        "intervals_ai.csv": csv_data(tuple(intervals[0]), intervals),
    }
    with tempfile.TemporaryDirectory(prefix="goalorithm-ai-") as folder:
        stage = Path(folder)
        for name, values in {
            "fullmatch_ai": codes,
            **{f"pseudolabel_{s}": v for s, v in splits.items()},
        }.items():
            path = stage / f"{name}.csv"
            write_submission(path, values)
            reason = pm.validate_submission(pm.read_label_csv(path), N_FRAMES)
            if reason:
                raise ValueError(reason)
            files[path.name] = path.read_bytes()
        for name, data in files.items():
            (stage / name).write_bytes(data)
        manifest = {
            "version": 1,
            "annotation_origin": "algorithmic_pseudo_labels",
            "model": model_name,
            "human_reviewed": False,
            "is_ground_truth": False,
            "independent_of_hmm_candidate": False,
            "selected_final_model": None,
            "accuracy_measured": False,
            "uncertainty_basis": (
                "HMM/baseline disagreement; agreement is not calibrated confidence"
            ),
            "legacy_dev_holdout_names": "window identifiers only; not independent validation sets",
            "warning": (
                "Do not evaluate HMM accuracy or select a winner against its own pseudo-labels."
            ),
            "provenance": provenance,
            "windows": summaries,
            "fullmatch_prediction_statistics": match_stats(codes),
            "output_sha256": {name: sha256(stage / name) for name in files},
        }
        files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    # Manifest is written last; consumers can verify every output hash before using the bundle.
    for name, data in files.items():
        atomic_write(directory / name, data)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG_DIR / "exp_hmm_v1.toml"))
    parser.add_argument("--windows", default=str(LABELS_DIR / "windows.csv"))
    parser.add_argument("--out-dir", default=str(OUTPUTS_DIR / "ai_labels"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        method = config.get("dead_ball", {}).get("method")
        if method != "hmm" or config.get("possession", {}).get("method") != method:
            raise ValueError("AI annotation requires the same HMM method in both adapters")
        windows = plans.read_windows(args.windows)
        tracking, tagging = load_tracking(), load_smart_tagging()
        codes = run_pipeline(config, tracking, tagging)
        baseline_config = load_config()
        baseline = run_pipeline(baseline_config, tracking, tagging)
        assets = games.asset_paths(source="tactical")
        provenance = {
            "data_sha256": {
                key: sha256(assets[key]) for key in ("atd_meta", "atd_raw", "smart_tagging")
            },
            "config_sha256": sha256(args.config),
            "baseline_config_sha256": sha256(CONFIG_DIR / "default.toml"),
            "windows_sha256": sha256(args.windows),
            "source_sha256": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                for path in sorted((ROOT / "src" / "mtah").glob("*.py"))
            },
            "exporter_sha256": sha256(__file__),
            "reference_labels_read": False,
            "video_uploaded": False,
            "public_gt_used_for_training_or_selection": False,
        }
        model_name = MODEL_NAME
        manifest = export_bundle(args.out_dir, codes, baseline, windows, provenance, model_name)
        print("AI PSEUDO-LABELS GENERATED LOCALLY (not human labels or ground truth)")
        for row in manifest["windows"]:
            print(
                f"{row['window']}: {row['frames']} frames, {row['change_points']} changes, "
                f"{row['disagreement_frames']} frames disagree with baseline"
            )
        print(f"-> {args.out_dir}")
        return 0
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
