"""Reading and writing frame,label files, byte-for-byte reproducibly."""

import hashlib
import os

import numpy as np

from mtah import LABELS
from mtah.paths import pm


def write_submission(path, codes):
    """Write codes (H/A/D/X as 0..3) as a frame,label CSV.

    newline="\\n" on purpose: on Windows the default would write \\r\\n, and the same labels would
    then hash differently on different laptops. The rules require the repository to regenerate
    the submitted file exactly, so the bytes must not depend on the operating system.
    """
    codes = np.asarray(codes)
    letters = np.array(LABELS)[codes]
    folder = os.path.dirname(str(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(letters))
    return path


def read_codes(path):
    """A frame,label CSV as codes 0..3, using the organisers' own reader."""
    return pm.label_codes(pm.read_label_csv(str(path)))


def sha256(path):
    """Hex digest of a file; equal digests mean identical submissions."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
