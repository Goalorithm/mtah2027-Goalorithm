"""Persist explicit human video review separately from algorithm-generated suggestions."""

import datetime
import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ReviewStore:
    """A window becomes complete only after explicit, attributed full-window review."""

    def __init__(self, windows, path, video_sha256):
        self.windows, self.path = windows, Path(path)
        spans = {name: [w["start"], w["end"], w["split"]] for name, w in windows.items()}
        window_sha256 = hashlib.sha256(json.dumps(spans, sort_keys=True).encode()).hexdigest()
        identity = {"video_sha256": video_sha256, "window_sha256": window_sha256}
        self.data = {"version": 1, "identity": identity, "windows": {}}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if loaded.get("version") != 1 or loaded.get("identity") != identity:
                raise ValueError("saved review belongs to a different video or window definition")
            self.data = loaded

    def get(self, name):
        if name not in self.windows:
            raise ValueError(f"unknown window {name}")
        return self.data["windows"].get(
            name, {"reviewer": "", "points": [], "complete": False, "revision": 0}
        )

    def save(self, name, reviewer, points, complete=False, confirmed=False, revision=0):
        if self.path.exists():
            latest = json.loads(self.path.read_text(encoding="utf-8"))
            if latest.get("identity") != self.data["identity"]:
                raise ValueError("saved review identity changed; reopen the review tool")
            self.data = latest
        window = self.windows.get(name)
        if window is None:
            raise ValueError("unknown window")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("enter the actual reviewer's name")
        if revision != self.get(name)["revision"]:
            raise ValueError("review changed since it was loaded; reopen this window first")
        normalized = []
        frames = set()
        for point in points:
            frame, label = point.get("frame"), point.get("label")
            if type(frame) is not int or not window["start"] <= frame <= window["end"]:
                raise ValueError("change-point frame lies outside the selected window")
            if label not in ("H", "A", "D") or frame in frames:
                raise ValueError("labels must be H/A/D and frame numbers must be unique")
            frames.add(frame)
            normalized.append({"frame": frame, "label": label, "note": str(point.get("note", ""))})
        normalized.sort(key=lambda point: point["frame"])
        if complete and (
            not confirmed or not normalized or normalized[0]["frame"] != window["start"]
        ):
            raise ValueError(
                "completion requires a first-frame label and explicit full-window review"
            )
        record = {
            "reviewer": reviewer.strip(),
            "points": normalized,
            "complete": bool(complete),
            "basis": "human_video_review",
            "revision": revision + 1,
            "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        updated = {**self.data, "windows": {**self.data["windows"], name: record}}
        atomic_write(self.path, (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode())
        self.data = updated
        return record

    def export_rows(self):
        required = [name for name, w in self.windows.items() if w["split"] != "kalibrasi"]
        missing = [name for name in required if not self.get(name)["complete"]]
        if missing:
            raise ValueError("full-window review is still missing: " + ", ".join(missing))
        windows, changes = [], []
        for name, window in self.windows.items():
            record = self.get(name)
            reviewer = record["reviewer"] if record["complete"] else window["pelabel"]
            windows.append(
                {
                    "jendela": name,
                    "frame_awal": window["start"],
                    "frame_akhir": window["end"],
                    "split": window["split"],
                    "pelabel": reviewer,
                }
            )
            if record["complete"]:
                for point in record["points"]:
                    changes.append(
                        {
                            "jendela": name,
                            "frame_mulai": point["frame"],
                            "label": point["label"],
                            "pelabel": reviewer,
                            "catatan": point["note"],
                        }
                    )
        return windows, changes
