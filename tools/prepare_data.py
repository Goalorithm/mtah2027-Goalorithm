"""Link split downloads into the organisers' data layout without copying large videos.

Sources are preserved. Existing different files and conflicting duplicate downloads are
rejected before any link is created. Hard links require the source and data folder to be on
the same filesystem. Nothing is uploaded or added to Git.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import _src_path  # noqa: F401

from mtah.paths import OUTPUTS_DIR, ROOT, games


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(source_root, excluded=ROOT):
    """Discover files without traversing the checkout, generated outputs, or virtualenvs."""
    found = []
    excluded = Path(excluded).resolve()
    for folder, dirs, files in os.walk(source_root, followlinks=False):
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", ".venv", "__pycache__", "cache", "outputs"}
            and (Path(folder) / d).resolve() != excluded
        ]
        found.extend(Path(folder) / name for name in sorted(files))
    return found


def make_plan(source_root, data_dir):
    files = discover(source_root)
    plan = []
    for source in ("tactical", "broadcast"):
        for asset, destination in games.asset_paths(data_dir=str(data_dir), source=source).items():
            destination = Path(destination)
            if asset == "video":
                pattern = re.compile(rf"BAR-ATH_{source}(?:-\d+)?\.mp4", re.IGNORECASE)
                candidates = [p for p in files if pattern.fullmatch(p.name)]
            else:
                candidates = [p for p in files if p.name == destination.name]
            if not candidates:
                raise ValueError(f"missing download: {destination.name}")
            hashes = {str(p): file_hash(p) for p in candidates}
            if len(set(hashes.values())) != 1:
                raise ValueError(
                    f"conflicting downloads for {destination.name}; choose the source explicitly"
                )
            origin = sorted(candidates, key=lambda p: str(p).casefold())[0]
            digest = hashes[str(origin)]
            if destination.exists() and file_hash(destination) != digest:
                raise ValueError(f"existing asset differs; preserved: {destination}")
            plan.append(
                {
                    "source": str(origin.resolve()),
                    "destination": str(destination.absolute()),
                    "asset": asset,
                    "recording": source,
                    "bytes": origin.stat().st_size,
                    "sha256": digest,
                    "duplicates": len(candidates),
                }
            )
            print(
                f"Verified {source}/{destination.name} ({origin.stat().st_size:,} bytes)",
                flush=True,
            )
    return plan


def apply_plan(plan):
    for item in plan:
        destination = Path(item["destination"])
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item["source"], destination)
        except OSError as error:
            raise OSError(
                f"cannot hard-link {destination}; sources are unchanged: {error}"
            ) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", default=str(ROOT.parent))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--manifest", default=str(OUTPUTS_DIR / "data_audit" / "assets.json"))
    args = parser.parse_args(argv)
    try:
        plan = make_plan(args.source_root, args.data_dir)
        if not args.check_only:
            apply_plan(plan)
        manifest = Path(args.manifest)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps({"linked": not args.check_only, "assets": plan}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{'Checked' if args.check_only else 'Prepared'} {len(plan)} assets -> {manifest}")
        return 0
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
