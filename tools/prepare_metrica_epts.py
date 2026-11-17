"""Download Metrica Sample Game 3 at the pinned public commit and verify its SHA-256."""

import urllib.request

import _src_path  # noqa: F401
from metrica_epts import FILES

from mtah.config import load_config
from mtah.io import sha256
from mtah.paths import ROOT


def main():
    cfg = load_config("config/atd.toml")["metrica_epts"]
    directory = ROOT / cfg["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    base = cfg["source"].replace("github.com", "raw.githubusercontent.com").replace("/tree/", "/")
    for key, name in FILES.items():
        path = directory / name
        if not path.is_file() or sha256(path) != cfg["sha256"][key]:
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{base}/{name}", path)  # noqa: S310 (pinned https URL)
        if sha256(path) != cfg["sha256"][key]:
            raise ValueError(f"{name} does not match the pinned SHA-256")
    print("Metrica Sample Game 3 is present and matches the pinned source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
