import sys
from pathlib import Path

import mtah.paths  # noqa: F401  (puts games.py, evaluation/ and starter/ on the path)

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
