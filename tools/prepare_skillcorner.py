"""Download public pinned SkillCorner files, verifying tracking against Git LFS SHA-256."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

import _src_path  # noqa: F401

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import ROOT
from mtah.review import atomic_write


def main():
    cfg = load_config("config/skillcorner.toml")
    directory = ROOT / "data/external/skillcorner_new"
    directory.mkdir(parents=True, exist_ok=True)
    commit = cfg["commit"]
    games = cfg["train_games"] + cfg["validation_games"] + cfg["test_games"]
    files = []
    for game in games:
        prefix = f"data/matches/{game}/{game}"
        metadata = urlopen(
            f"https://raw.githubusercontent.com/SkillCorner/opendata/{commit}/{prefix}_match.json",
            timeout=120,
        ).read()
        path = directory / f"{game}_match.json"
        atomic_write(path, metadata)
        phases = urlopen(
            f"https://raw.githubusercontent.com/SkillCorner/opendata/{commit}/{prefix}_phases_of_play.csv",
            timeout=120,
        ).read()
        phase_path = directory / f"{game}_phases_of_play.csv"
        atomic_write(phase_path, phases)
        pointer = (
            urlopen(
                f"https://raw.githubusercontent.com/SkillCorner/opendata/{commit}/{prefix}_tracking_extrapolated.jsonl",
                timeout=120,
            )
            .read()
            .decode()
        )
        digest = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
        size = re.search(r"size (\d+)", pointer)
        if digest is None or size is None:
            raise ValueError("tracking source is not a valid pinned LFS pointer")
        files.append(
            {
                "game": game,
                "metadata_sha256": sha256(path),
                "phases_sha256": sha256(phase_path),
                "tracking_sha256": digest[1],
                "tracking_bytes": int(size[1]),
            }
        )
    manifest = {
        "commit": commit,
        "files": files,
        "roles": {key: cfg[key] for key in ("train_games", "validation_games", "test_games")},
        "reference_scope": cfg["reference"],
        "bar_ath_data_read": False,
    }
    atomic_write(directory / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())

    def download(row):
        game = row["game"]
        path = directory / f"{game}_tracking_extrapolated.jsonl"
        if path.is_file() and sha256(path) == row["tracking_sha256"]:
            return
        url = (
            f"https://media.githubusercontent.com/media/SkillCorner/opendata/{commit}/"
            f"data/matches/{game}/{game}_tracking_extrapolated.jsonl"
        )
        temporary = path.with_suffix(".part")
        digest = hashlib.sha256()
        with urlopen(url, timeout=300) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                digest.update(chunk)
        if (
            digest.hexdigest() != row["tracking_sha256"]
            or temporary.stat().st_size != row["tracking_bytes"]
        ):
            raise ValueError(f"download failed LFS integrity: {game}")
        temporary.replace(path)
        print(f"Downloaded and verified SkillCorner {game}", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    print(
        "Four other-game sources ready; no possession labels parsed for selection yet", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
