"""Download the pinned SoccerTrack v2 files used for real-footage validation.

The dataset is open (CC BY 4.0) but the Hugging Face repository asks every account to accept
its terms once, so a logged-in account is needed: accept the terms on the dataset page, then
run `hf auth login` in a terminal. The token stays in the local Hugging Face cache; this tool
never prints or stores it.
"""

import argparse

import _src_path  # noqa: F401
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalTokenNotFoundError

from mtah.config import load_config
from mtah.paths import ROOT


def patterns(cfg):
    wanted = ["README.md", "CHANGES_*.md", "mot/README.md"]
    for match in cfg["matches"]:
        wanted += [
            f"gsr/{match}/*",
            f"bas/{match}/*",
            f"ball/{match}_*",
            f"raw/{match}/{match}_homography.npy",
            f"raw/{match}/{match}_camera_intrinsics.npz",
            f"raw/{match}/{match}_keypoints.json",
            f"raw/{match}/{match}_padding_info.csv",
            f"raw/{match}/{match}_mapx.npy",
            f"raw/{match}/{match}_mapy.npy",
        ]
    for match in cfg.get("train_matches", []):
        wanted += [f"gsr/{match}/*", f"bas/{match}/*", f"ball/{match}_*"]
    for half in cfg["video_halves"]:
        match, part = half.split("_")
        wanted.append(f"videos/{match}/{match}_panorama_{part}_half.mp4")
    return wanted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/soccertrack.toml")
    parser.add_argument("--no-video", action="store_true", help="skip the panoramic videos")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    wanted = patterns(cfg)
    if args.no_video:
        wanted = [pattern for pattern in wanted if not pattern.startswith("videos/")]
    target = ROOT / cfg["directory"]
    try:
        snapshot_download(
            repo_id=cfg["repository"],
            repo_type="dataset",
            revision=cfg["revision"],
            local_dir=target,
            allow_patterns=wanted,
        )
    except (GatedRepoError, LocalTokenNotFoundError) as error:
        print(
            "Access is not set up yet. Accept the terms at "
            f"https://huggingface.co/datasets/{cfg['repository']} and run `hf auth login`."
        )
        print(f"({type(error).__name__})")
        return 1
    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB under {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
