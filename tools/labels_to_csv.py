"""Turn the team's hand labels (exported from Google Sheets) into scorable ground-truth files.

Hand labels are for VALIDATION ONLY. They are never copied into a submission (RULES.md).

Inputs, both CSV exports of the "Label validasi" Google Sheet (File > Download > CSV):

  labels/windows.csv   one row per labelled window
      jendela,frame_awal,frame_akhir,split,pelabel
      V1,30000,37499,dev,Budi
      (frame_akhir is inclusive; split is dev, holdout or kalibrasi; pelabel = whose labels count)

  labels/perubahan.csv one row per change of state inside a window
      jendela,frame_mulai,label,pelabel,catatan
      V1,30000,H,Budi,awal jendela
      V1,30212,A,Budi,intersep
      (the first row of a window must start exactly at frame_awal; label is H, A or D)

Usage:
  python tools/labels_to_csv.py
      -> labels/validasi_dev.csv, labels/validasi_holdout.csv (199319 rows, X outside windows)

  python tools/labels_to_csv.py --labeler Budi --out labels/pelabel_budi.csv
      -> everything Budi labelled, in one file. Used for calibration on the public window
         (copy it into evaluation/results/ and run evaluation/run_evaluation.py) and for
         measuring agreement between two labellers on the same window.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import _src_path  # noqa: F401  (finds src/ without pip install -e .)
import numpy as np

from mtah import LABELS, X
from mtah.io import write_submission
from mtah.paths import LABELS_DIR, N_FRAMES, match_runs

PUBLIC_WINDOW = (11185, 18684)
STATES = {"H": LABELS.index("H"), "A": LABELS.index("A"), "D": LABELS.index("D")}


class LabelError(ValueError):
    pass


def read_csv(path, required=()):
    """[(line_number, row)] with lower-cased keys, skipping fully empty rows.

    Google Sheets exports pre-formatted empty rows as ",,,,", so blank rows are expected and
    ignored; line numbers still match the spreadsheet row, so error messages point at the
    right row.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        missing = set(required) - set(headers)
        if missing or len(headers) != len(set(headers)):
            raise LabelError(f"{path}: missing or duplicate headers; missing {sorted(missing)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row:
                raise LabelError(f"{path}:{line}: more cells than column headers")
            rows.append((line, {k.strip().lower(): (v or "").strip() for k, v in row.items()}))
    return [(line, row) for line, row in rows if any(row.values())]


def read_windows(path):
    windows = {}
    for line, row in read_csv(path, ("jendela", "frame_awal", "frame_akhir", "split", "pelabel")):
        name = row.get("jendela", "")
        try:
            start, end = int(row["frame_awal"]), int(row["frame_akhir"])
        except (KeyError, ValueError) as error:
            raise LabelError(f"{path}:{line}: frame_awal/frame_akhir must be integers") from error
        if not name or name in windows:
            raise LabelError(f"{path}:{line}: window name missing or repeated: {name!r}")
        if not 0 <= start <= end < N_FRAMES:
            raise LabelError(
                f"{path}:{line}: {name} frames {start}-{end} outside 0..{N_FRAMES - 1}"
            )
        split = row.get("split", "dev").lower() or "dev"
        if split not in ("dev", "holdout", "kalibrasi"):
            raise LabelError(f"{path}:{line}: split must be dev, holdout or kalibrasi")
        if split != "kalibrasi" and not row.get("pelabel"):
            raise LabelError(f"{path}:{line}: {name} needs a designated pelabel")
        if not any(first <= start <= end < stop for first, stop in match_runs()):
            raise LabelError(f"{path}:{line}: {name} must lie within a single graded half")
        if split != "kalibrasi" and start <= PUBLIC_WINDOW[1] and end >= PUBLIC_WINDOW[0]:
            raise LabelError(
                f"{path}:{line}: {name} overlaps the public window "
                f"{PUBLIC_WINDOW}; use split=kalibrasi for that one"
            )
        windows[name] = {
            "start": start,
            "end": end,
            "split": split,
            "pelabel": row.get("pelabel", ""),
        }

    if not windows:
        raise LabelError(f"{path}: no validation windows")
    spans = sorted((w["start"], w["end"], n) for n, w in windows.items())
    for (_, end_a, a), (start_b, _, b) in zip(spans, spans[1:], strict=False):
        if start_b <= end_a:
            raise LabelError(f"{path}: windows {a} and {b} overlap")
    return windows


def read_changes(path):
    changes = defaultdict(list)  # (window, labeller) -> [(frame, code, line)]
    for line, row in read_csv(path, ("jendela", "frame_mulai", "label", "pelabel")):
        label = row.get("label", "").upper()
        if label not in STATES:
            raise LabelError(f"{path}:{line}: label must be H, A or D, got {label!r}")
        try:
            frame = int(row["frame_mulai"])
        except (KeyError, ValueError) as error:
            raise LabelError(f"{path}:{line}: frame_mulai must be an integer") from error
        if not row.get("jendela") or not row.get("pelabel"):
            raise LabelError(f"{path}:{line}: jendela and pelabel must not be blank")
        changes[(row["jendela"], row["pelabel"])].append((frame, STATES[label], line))
    return changes


def validate_rows(window_name, window, rows, path):
    """Validate one labeller's change points before any output is written."""
    rows = sorted(rows)
    if not rows:
        raise LabelError(f"{path}: window {window_name} has no rows for {window['pelabel']!r}")
    if rows[0][0] != window["start"]:
        raise LabelError(
            f"{path}:{rows[0][2]}: first row of {window_name} must start at "
            f"frame_awal {window['start']}, got {rows[0][0]}"
        )
    frames = [frame for frame, _, _ in rows]
    if len(set(frames)) != len(frames):
        raise LabelError(f"{path}: window {window_name} repeats a frame_mulai")
    for frame, _, line in rows:
        if not window["start"] <= frame <= window["end"]:
            raise LabelError(
                f"{path}:{line}: frame {frame} outside window {window_name} "
                f"({window['start']}-{window['end']})"
            )
    return rows


def validate_changes(windows, changes, path):
    """Validate all supplied annotations, including alternative labellers and calibration."""
    for (name, _), rows in changes.items():
        if name not in windows:
            raise LabelError(f"{path}: unknown window {name!r}")
        validate_rows(name, windows[name], rows, path)


def missing_windows(windows, changes):
    """Unfinished designated annotations; calibration is optional for split export."""
    return [
        f"{name} ({window['pelabel']})"
        for name, window in windows.items()
        if window["split"] != "kalibrasi" and not changes.get((name, window["pelabel"]))
    ]


def paint_window(codes, window_name, window, rows, path):
    """Fill codes[start..end] from validated change points."""
    rows = validate_rows(window_name, window, rows, path)
    bounds = [frame for frame, _, _ in rows[1:]] + [window["end"] + 1]
    for (frame, code, _), stop in zip(rows, bounds, strict=True):
        codes[frame:stop] = code


def build(windows, changes, path, split=None, labeler=None):
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    used = 0
    for name, window in windows.items():
        who = labeler or window["pelabel"]
        if split is not None and window["split"] != split:
            continue
        rows = changes.get((name, who), [])
        if labeler and not rows:
            continue
        paint_window(codes, name, window, rows, path)
        used += 1
    return codes, used


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--windows", default=os.path.join(str(LABELS_DIR), "windows.csv"))
    parser.add_argument("--changes", default=os.path.join(str(LABELS_DIR), "perubahan.csv"))
    parser.add_argument("--labeler", help="one labeller's windows only (calibration/agreement)")
    parser.add_argument("--out", help="output path, required with --labeler")
    parser.add_argument("--out-dir", help="split output directory (default: labels/)")
    parser.add_argument("--check-only", action="store_true", help="validate without writing files")
    args = parser.parse_args(argv)

    try:
        if not args.check_only and os.path.abspath(args.windows) == os.path.abspath(
            LABELS_DIR / "windows.csv"
        ):
            raise LabelError(
                "manual BAR-ATH validation export is retired under updated rules; "
                "run tools/ai_labels.py and see docs/RULE_UPDATE.md"
            )
        windows = read_windows(args.windows)
        changes = read_changes(args.changes)
        validate_changes(windows, changes, args.changes)
        if args.labeler:
            if not args.out:
                parser.error("--labeler needs --out")
            codes, used = build(windows, changes, args.changes, labeler=args.labeler)
            if not used:
                raise LabelError(f"no annotated windows for {args.labeler!r}")
            if not args.check_only:
                write_submission(args.out, codes)
            print(f"{args.labeler}: {used} window(s) -> {args.out}")
            return 0

        unfinished = missing_windows(windows, changes)
        if unfinished:
            raise LabelError("missing annotations for " + ", ".join(unfinished))
        splits = sorted({w["split"] for w in windows.values()} - {"kalibrasi"})
        if not splits:
            raise LabelError("no dev or holdout windows; add labelled validation windows first")
        outputs = []
        for split in splits:
            codes, used = build(windows, changes, args.changes, split=split)
            out = os.path.join(args.out_dir or str(LABELS_DIR), f"validasi_{split}.csv")
            outputs.append((split, codes, used, out))
        for split, codes, used, out in outputs:
            if not args.check_only:
                write_submission(out, codes)
            graded = int(np.sum(codes != X))
            print(
                f"{split:<8} {used} window(s), {graded} frames ({graded / 25 / 60:.1f} min) "
                f"{'checked only' if args.check_only else '-> ' + out}"
            )
    except LabelError as error:
        print(f"ERROR  {error}")
        return 1
    except FileNotFoundError as error:
        print(f"ERROR  file not found: {error.filename}")
        print("       export the 'windows' and 'perubahan' tabs as CSV into labels/ first")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
