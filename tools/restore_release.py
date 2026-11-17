"""Copy the released model files from release/atd/ into outputs/, where the tools read them.

The submitted labels come from an ensemble fitted on other games. Fitting on a GPU is repeatable
on one machine but not bit-for-bit across GPUs, so the fitted weights are released with the
code: restoring them and running `tools/evaluate_atd.py --mode competition` regenerates the
submission without refitting. Nothing here is match data; the weights were fitted on IDSSE and
Metrica sample games only.
"""

import argparse
import json
import shutil

import _src_path  # noqa: F401

from mtah.io import sha256
from mtah.paths import OUTPUTS_DIR, ROOT

RELEASE = ROOT / "release" / "atd"
FILES = ("model.json", "weights.pt")


def check(directory):
    """A member's weights must be the ones its model file was written for."""
    metadata = json.loads((directory / "model.json").read_text())
    if "weights_sha256" in metadata:
        if sha256(directory / "weights.pt") != metadata["weights_sha256"]:
            raise ValueError(f"{directory.name}: weights do not match model.json")
    return metadata


def restore(release=RELEASE, outputs=OUTPUTS_DIR, force=False):
    """Copy every released model directory; refuse to replace a different local model."""
    models = sorted(path for path in release.iterdir() if (path / "model.json").is_file())
    if not models:
        raise FileNotFoundError(f"no released models under {release}")
    plan = []
    for source in models:
        metadata = check(source)
        for name, expected in metadata.get("members", {}).items():
            if sha256(release / name / "model.json") != expected:
                raise ValueError(f"{source.name}: member {name} differs from the released one")
        for name in FILES:
            if not (source / name).is_file():
                continue
            target = outputs / source.name / name
            if target.is_file() and sha256(target) != sha256(source / name) and not force:
                raise FileExistsError(
                    f"{target} differs from the released file; move it away or pass --force"
                )
            plan.append((source / name, target))
    for source, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha256(target) != sha256(source):
            shutil.copyfile(source, target)
    return [source.name for source in models]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace differing local models")
    args = parser.parse_args(argv)
    names = restore(force=args.force)
    print(f"restored {len(names)} models into {OUTPUTS_DIR}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
