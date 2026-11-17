"""Reject excluded research and private/generated assets in the Git index."""

import subprocess
from pathlib import Path


def excluded_paths(manifest):
    return tuple(
        line.strip()
        for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def violations(tracked, excluded):
    rejected = []
    for name in tracked:
        if not name:
            continue
        private = (
            name.startswith(("outputs/", "cache/"))
            or (name.startswith("data/") and name != "data/README.md")
            or name.lower().endswith((".mp4", ".npz", ".parquet", ".gzip", ".pl3container"))
        )
        removed = any(
            name == path or (path.endswith("/") and name.startswith(path)) for path in excluded
        )
        if private or removed:
            rejected.append(name)
    return sorted(rejected)


def main():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    rejected = violations(tracked, excluded_paths(root / ".github/excluded-paths.txt"))
    if rejected:
        print("FAIL: excluded or private assets are tracked:")
        print("\n".join(rejected))
        return 1
    print("PASS: Git index contains no excluded research or private/generated assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
