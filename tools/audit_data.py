"""Audit local tracking, homographies, tags, and both video timelines without exposing frames.

--full-decode decodes every video frame and checks its presentation timestamp. It can take
several minutes. Only counts, metadata and validation results are written to the report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import _src_path  # noqa: F401
import av
import numpy as np
import pandas as pd

from mtah.load import parse_meta
from mtah.paths import FPS, OUTPUTS_DIR, games


def audit_tracking(meta, raw, expected):
    slot_team, _ = parse_meta(meta)
    count, malformed, bad_ids = 0, 0, 0
    with open(raw) as handle:
        for count, line in enumerate(handle, start=1):
            fields = line.rstrip("\r\n").split(":")
            if len(fields) != 3:
                malformed += 1
                continue
            try:
                bad_ids += int(fields[0]) != count
            except ValueError:
                bad_ids += 1
    return {
        "records": count,
        "expected": expected,
        "slots": len(slot_team),
        "malformed_records": malformed,
        "noncontiguous_ids": bad_ids,
        "ok": count == expected and not malformed and not bad_ids,
    }


def audit_homography(path, expected):
    table = pd.read_csv(path)
    matrix = [f"h{i}{j}" for i in range(3) for j in range(3)]
    contiguous = np.array_equal(table["frame"].to_numpy(), np.arange(expected))
    flags = table["valid"].to_numpy()
    valid = flags == 1
    sane_flags = bool(np.isin(flags, [0, 1]).all())
    finite = bool(np.isfinite(table.loc[valid, matrix].to_numpy()).all())
    return {
        "rows": len(table),
        "expected": expected,
        "valid_rows": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "contiguous": bool(contiguous),
        "finite_valid_matrices": finite,
        "ok": bool(contiguous and sane_flags and finite),
    }


def audit_video(path, expected, full_decode=False, progress=None):
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected exactly one video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        result = {
            "frames_in_header": stream.frames,
            "expected_frames": expected,
            "fps": fps,
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "start_pts": stream.start_time,
            "time_base": str(stream.time_base),
            "duration_seconds": float(stream.duration * stream.time_base),
            "full_decode": bool(full_decode),
        }
        if full_decode:
            stream.thread_type = "AUTO"
            count, bad_timestamps, corrupt = 0, 0, 0
            began, last_progress = time.monotonic(), time.monotonic()
            for count, frame in enumerate(container.decode(stream), start=1):
                expected_time = (count - 1) / FPS
                bad_timestamps += frame.pts is None or abs(frame.time - expected_time) > 1e-6
                corrupt += bool(frame.is_corrupt)
                if progress and time.monotonic() - last_progress >= 20:
                    progress(count, time.monotonic() - began)
                    last_progress = time.monotonic()
            result.update(
                decoded_frames=count,
                bad_timestamps=bad_timestamps,
                corrupt_frames=corrupt,
                decode_seconds=round(time.monotonic() - began, 2),
            )
            result["ok"] = bool(
                count == expected and not bad_timestamps and not corrupt and fps == FPS
            )
        else:
            # This is a metadata check only; full_decode is explicit in the report.
            result["ok"] = bool(stream.frames == expected and fps == FPS and stream.start_time == 0)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--full-decode", action="store_true")
    parser.add_argument(
        "--source", nargs="+", choices=("tactical", "broadcast"), default=["tactical", "broadcast"]
    )
    parser.add_argument("--out", default=str(OUTPUTS_DIR / "data_audit" / "validation.json"))
    args = parser.parse_args(argv)
    report = {}
    for source in args.source:
        paths = games.asset_paths(data_dir=args.data_dir, source=source)
        expected = games.source_frames(source)
        try:
            item = {
                "tracking": audit_tracking(paths["atd_meta"], paths["atd_raw"], expected),
                "homography": audit_homography(paths["homography"], expected),
            }
            print(f"{source}: tracking and homography checked", flush=True)
            if source == "tactical":
                import load_smart_tagging as tagging

                events = tagging.load(data_dir=args.data_dir)
                alignment = tagging.check_time_base(events)
                item["tagging"] = {
                    "instances": len(events),
                    "n_kickoff_tags": alignment["n_kickoff_tags"],
                    "aligned_halves": [
                        {"half": row["half"], "contains_kickoff": row["contains_kickoff"]}
                        for row in alignment["halves"]
                    ],
                    "agrees": alignment["agrees"],
                }
            item["video"] = audit_video(
                paths["video"],
                expected,
                args.full_decode,
                lambda n, seconds, source=source: print(
                    f"{source}: decoded {n:,} frames in {seconds:.0f}s", flush=True
                ),
            )
            item["ok"] = all(item[name]["ok"] for name in ("tracking", "homography", "video"))
            if "tagging" in item:
                item["ok"] &= item["tagging"]["agrees"]
            report[source] = item
            print(
                f"{source}: {'OK' if item['ok'] else 'MISMATCH'}; video "
                f"{item['video'].get('decoded_frames', item['video']['frames_in_header']):,} "
                f"versus {expected:,} reference frames",
                flush=True,
            )
        except (OSError, ValueError, av.FFmpegError) as error:
            report[source] = {"ok": False, "error": str(error)}
            print(f"{source}: ERROR {error}", flush=True)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.out}")
    return 0 if all(item["ok"] for item in report.values()) else 2


if __name__ == "__main__":
    sys.exit(main())
