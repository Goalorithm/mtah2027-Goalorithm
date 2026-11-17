"""Download pinned CC-BY-4.0 IDSSE external assets; never upload competition data."""

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _src_path  # noqa: F401

from mtah.io import sha256
from mtah.paths import ROOT


def fetch_file(row, directory):
    path = directory / row["name"]
    if Path(row["name"]).name != row["name"]:
        raise ValueError("asset names must be plain filenames")
    if path.exists():
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"existing external asset changed: {path.name}")
        return path.name
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        request = urllib.request.Request(
            row["url"] + "?download=true", headers={"User-Agent": "Goalorithm-research"}
        )
        print(f"Downloading {path.name}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if temporary.stat().st_size != row["bytes"] or sha256(temporary) != row["sha256"]:
            raise ValueError(f"download integrity failure: {path.name}")
        temporary.replace(path)
        print(f"Verified {path.name}", flush=True)
        return path.name
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="config/idsse_sources.json")
    parser.add_argument("--data-dir", default="data/external/idsse")
    args = parser.parse_args(argv)
    directory = Path(args.data_dir).resolve()
    if not directory.is_relative_to((ROOT / "data/external").resolve()):
        print("ERROR: external assets must stay under ignored data/external/")
        return 1
    directory.mkdir(parents=True, exist_ok=True)
    try:
        source = json.loads(Path(args.sources).read_text(encoding="utf-8"))
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda row: fetch_file(row, directory), source["files"]))
        print(
            f"Verified all {len(source['files'])} external assets "
            f"at pinned revision {source['revision']}"
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
