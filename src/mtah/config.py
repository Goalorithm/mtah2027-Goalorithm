"""Load a TOML config, with optional command-line overrides.

Every number that can be tuned lives in config/*.toml, never inside the code, so an experiment is
"copy the config, change a value, run". Overrides make one-off tries even quicker:

    python -m mtah.predict --set possession.window=75 --set dead_ball.window=40
"""

import copy
import tomllib
from pathlib import Path

from mtah.paths import CONFIG_DIR

DEFAULT_CONFIG = CONFIG_DIR / "default.toml"


def _parse_value(text):
    """A TOML value from the command line: 75 -> int, 0.5 -> float, true -> bool, else string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def apply_override(config, assignment):
    """Apply one 'section.key=value' override in place."""
    if "=" not in assignment:
        raise ValueError(f"override must look like section.key=value, got {assignment!r}")
    dotted, raw = assignment.split("=", 1)
    keys = dotted.strip().split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = _parse_value(raw.strip())


def load_config(path=None, overrides=()):
    """The config as a dict. `path` defaults to config/default.toml."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    config = copy.deepcopy(config)
    for assignment in overrides:
        apply_override(config, assignment)
    config["_path"] = str(path)
    return config
