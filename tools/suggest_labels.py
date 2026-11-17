"""Export model change-point suggestions for local review, never as validation ground truth.

The output deliberately uses label_usulan and pelabel_tujuan, so labels_to_csv.py cannot
mistake it for reviewed annotations. Review against the tactical video before entering any
accepted change points into the team's perubahan tab.
"""

import argparse
import csv
import sys
from pathlib import Path

import _src_path  # noqa: F401
import labels_to_csv as labels
import numpy as np
from timeline import resolve

from mtah import LABELS, X
from mtah.paths import LABELS_DIR, N_FRAMES, OUTPUTS_DIR, pm


def suggestions(codes, windows, source):
    rows = []
    for name, window in windows.items():
        if window["split"] == "kalibrasi":
            continue
        start, stop = window["start"], window["end"] + 1
        values = codes[start:stop]
        changes = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        for offset in changes:
            code = values[offset]
            rows.append(
                {
                    "jendela": name,
                    "frame_mulai": start + int(offset),
                    "label_usulan": LABELS[code],
                    "pelabel_tujuan": window["pelabel"],
                    "status": "abstain" if code == X else "belum_diverifikasi",
                    "sumber": source,
                }
            )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("prediction", help="result name or frame,label CSV path")
    parser.add_argument("--windows", default=str(LABELS_DIR / "windows.csv"))
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "tech1_review" / "perubahan_usulan.csv"))
    args = parser.parse_args(argv)
    path = resolve(args.prediction)
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, N_FRAMES)
    if reason:
        parser.error(reason)
    rows = suggestions(pm.label_codes(table), labels.read_windows(args.windows), Path(path).name)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "jendela",
                "frame_mulai",
                "label_usulan",
                "pelabel_tujuan",
                "status",
                "sumber",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} unreviewed model suggestions -> {destination}")
    print("These are not ground truth. The validation converter rejects this suggestion schema.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
