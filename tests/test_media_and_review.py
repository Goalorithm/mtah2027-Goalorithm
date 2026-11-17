"""Synthetic video checks: exact frame seeking, the full-decode audit and download assembly."""

import os
from fractions import Fraction
from pathlib import Path

import audit_data
import av
import numpy as np
import prepare_data
import pytest

from mtah.paths import games
from mtah.video import FrameReader


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "synthetic.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"preset": "ultrafast"}
        for index in range(12):
            pixels = np.zeros((64, 96, 3), dtype=np.uint8)
            pixels[:, :, 0] = index * 18
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_frame_reader_seeks_by_exact_presentation_timestamp(video):
    with FrameReader(video, expected_frames=12) as reader:
        for index in (0, 1, 11, 5, 6, 0, 10):
            frame = reader.read(index)
            assert frame.pts * frame.time_base == Fraction(index, 25)
        with pytest.raises(ValueError, match="integer"):
            reader.read(-1)
        with pytest.raises(ValueError, match="integer"):
            reader.read(12)
    with pytest.raises(ValueError, match="count or FPS"):
        FrameReader(video, expected_frames=13)


def test_full_video_audit_reports_mismatch_without_substituting_frames(video):
    correct = audit_data.audit_video(video, 12, full_decode=True)
    assert correct["ok"] and correct["decoded_frames"] == 12
    assert correct["bad_timestamps"] == 0 and correct["corrupt_frames"] == 0
    mismatch = audit_data.audit_video(video, 13, full_decode=True)
    assert not mismatch["ok"] and mismatch["decoded_frames"] == 12


def test_download_assembly_hardlinks_without_copying_or_changing_sources(tmp_path):
    source_root, data_dir = tmp_path / "downloads", tmp_path / "data"
    source_root.mkdir()
    for source in ("tactical", "broadcast"):
        for asset, destination in games.asset_paths(data_dir=str(data_dir), source=source).items():
            name = Path(destination).name
            if asset == "video":
                name = name.replace(".mp4", "-002.mp4")
            (source_root / name).write_bytes(f"fixture:{source}:{asset}".encode())
    plan = prepare_data.make_plan(source_root, data_dir)
    assert len(plan) == 9
    prepare_data.apply_plan(plan)
    for item in plan:
        assert os.path.samefile(item["source"], item["destination"])
    original = Path(plan[0]["source"]).read_bytes()
    collision = source_root / "duplicate"
    collision.mkdir()
    (collision / Path(plan[0]["source"]).name).write_bytes(b"different version")
    with pytest.raises(ValueError, match="conflicting downloads"):
        prepare_data.make_plan(source_root, data_dir)
    assert Path(plan[0]["source"]).read_bytes() == original
