"""Select candidates on reviewed dev labels; optionally inspect only the selected model on holdout.

Never selects on public ground truth, possession counts, or an unreviewed suggestion file.
Missing dev/holdout labels are an explicit incomplete result (exit code 2).
"""

import argparse
import json
import sys
from pathlib import Path

import _src_path  # noqa: F401
import labels_to_csv as labels
import numpy as np
from timeline import resolve

from mtah import X
from mtah.io import sha256
from mtah.paths import LABELS_DIR, N_FRAMES, OUTPUTS_DIR, match_runs, pm
from mtah.review import atomic_write


def read_valid_codes(path):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, N_FRAMES)
    if reason:
        raise ValueError(f"{path}: {reason}")
    return pm.label_codes(table)


def validation_codes(path, split, windows=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing reviewed {split} labels: {path}")
    codes = read_valid_codes(path)
    graded = codes != X
    permitted = np.zeros(N_FRAMES, dtype=bool)
    for start, stop in match_runs():
        permitted[start:stop] = True
    if not graded.any():
        raise ValueError(f"{split} has no annotated frames")
    if graded[11185:18685].any():
        raise ValueError(f"{split} must exclude the public calibration window")
    if (graded & ~permitted).any():
        raise ValueError(f"{split} contains annotations outside the graded halves")
    if windows is not None:
        expected = np.zeros(N_FRAMES, dtype=bool)
        for window in windows.values():
            if window["split"] == split:
                expected[window["start"] : window["end"] + 1] = True
        if not np.array_equal(graded, expected):
            raise ValueError(f"{split} coverage must exactly match its designated windows")
    return codes


def compare(paths, dev_codes, holdout_codes=None):
    """Pick once on dev. Holdout scores only that fixed selection, never all candidates."""
    candidates = []
    for path in paths:
        codes = read_valid_codes(path)
        result = pm.evaluate(dev_codes, codes)
        candidates.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "spell_f1": result["spell_f1_mean"],
                "macro_f1": result["macro_f1"],
            }
        )
    if not candidates:
        raise ValueError("at least one candidate is required")
    candidates.sort(key=lambda row: (-row["spell_f1"], -row["macro_f1"], row["path"]))
    report = {
        "status": "dev_selected",
        "candidates": candidates,
        "selected": candidates[0]["path"],
        "selection_basis": "reviewed_dev_only",
    }
    if holdout_codes is not None:
        if ((dev_codes != X) & (holdout_codes != X)).any():
            raise ValueError("dev and holdout annotations overlap")
        result = pm.evaluate(holdout_codes, read_valid_codes(report["selected"]))
        report["holdout"] = {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"]}
        report["status"] = "selected_model_checked_on_holdout"
    return report


def check_holdout(report, dev_path, windows_path, holdout_path, results=()):
    """Verify the saved dev decision before opening holdout; never re-rank candidates here."""
    if report.get("status") not in ("dev_selected", "selected_model_checked_on_holdout"):
        raise ValueError("run dev selection successfully before requesting holdout")
    if report.get("validation") != {
        "dev_sha256": sha256(dev_path),
        "windows_sha256": sha256(windows_path),
    }:
        raise ValueError("dev labels or window definitions changed since selection")
    candidates = report["candidates"]
    if results and {str(Path(p).resolve()) for p in results} != {r["path"] for r in candidates}:
        raise ValueError("candidate list differs from the saved dev selection")
    for row in candidates:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("candidate file changed since dev selection")
    if report["selected"] != candidates[0]["path"]:
        raise ValueError("saved selection is inconsistent")
    windows = labels.read_windows(windows_path)
    holdout = validation_codes(holdout_path, "holdout", windows)
    digest = sha256(holdout_path)
    if "holdout" in report:
        if report["holdout_sha256"] != digest:
            raise ValueError("holdout labels changed after the final check")
        return report  # Reprinting the same receipt does not open another model-selection round.
    result = pm.evaluate(holdout, read_valid_codes(report["selected"]))
    return {
        **report,
        "status": "selected_model_checked_on_holdout",
        "holdout_sha256": digest,
        "holdout": {"spell_f1": result["spell_f1_mean"], "macro_f1": result["macro_f1"]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="*", help="result names or paths; optional for holdout")
    parser.add_argument("--dev", default=str(LABELS_DIR / "validasi_dev.csv"))
    parser.add_argument("--windows", default=str(LABELS_DIR / "windows.csv"))
    parser.add_argument("--holdout", action="store_true", help="final decision only")
    parser.add_argument("--holdout-path", default=str(LABELS_DIR / "validasi_holdout.csv"))
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "tech1_review" / "selection.json"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    previous = {}
    try:
        previous = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
        if Path(args.windows).resolve() == (LABELS_DIR / "windows.csv").resolve():
            raise ValueError(
                "BAR-ATH human-label validation is retired under updated rules; "
                "use tools/ai_labels.py for pseudo-label diagnostics, not accuracy selection. "
                "See docs/RULE_UPDATE.md"
            )
        paths = [str(Path(resolve(name)).resolve()) for name in args.results]
        if args.holdout:
            report = check_holdout(previous, args.dev, args.windows, args.holdout_path, paths)
        else:
            if "holdout" in previous:
                raise ValueError(
                    "final holdout check already recorded; saved decision is preserved"
                )
            dev = validation_codes(args.dev, "dev", labels.read_windows(args.windows))
            report = compare(paths, dev)
            report["validation"] = {
                "dev_sha256": sha256(args.dev),
                "windows_sha256": sha256(args.windows),
            }
    except (OSError, ValueError) as error:
        if not previous.get("selected") and (not out.exists() or previous):
            report = {"status": "incomplete", "selected": None, "reason": str(error)}
            atomic_write(out, (json.dumps(report, indent=2) + "\n").encode())
        print(f"INCOMPLETE: {error}")
        print("No final model was released; any saved dev selection is preserved.")
        return 2
    atomic_write(out, (json.dumps(report, indent=2) + "\n").encode())
    for row in report["candidates"]:
        print(
            f"{Path(row['path']).name}: dev spellF1 {row['spell_f1']:.4f} "
            f"macroF1 {row['macro_f1']:.4f}"
        )
    print(f"Dev selection: {report['selected']}")
    if "holdout" in report:
        print(f"Selected-model holdout: {report['holdout']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
