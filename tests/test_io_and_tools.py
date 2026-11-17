import labels_to_csv  # tools/, on the path via conftest.py  # noqa: I001
import numpy as np
import pytest

from mtah import A, D, H, X
from mtah.config import load_config
from mtah.io import read_codes, sha256, write_submission
from mtah.paths import EVALUATION_DIR, N_FRAMES, pm
from mtah.scoring import match_stats


def test_submission_is_valid_and_os_independent(tmp_path):
    codes = np.full(N_FRAMES, X, dtype=np.int8)
    codes[11185:12000] = H
    codes[12000:12500] = D
    path = write_submission(tmp_path / "s.csv", codes)
    assert pm.validate_submission(pm.read_label_csv(str(path)), N_FRAMES) is None
    data = path.read_bytes()
    assert b"\r" not in data and data.startswith(b"frame,label\n0,X\n")
    again = write_submission(tmp_path / "t.csv", codes)
    assert sha256(path) == sha256(again)
    assert np.array_equal(read_codes(path), codes)


def test_config_overrides():
    config = load_config(overrides=["possession.window=75", "dead_ball.method=x"])
    assert config["possession"]["window"] == 75
    assert config["dead_ball"]["method"] == "x"
    assert config["team_name"]


def test_match_stats_on_shipped_baseline():
    stats = match_stats(read_codes(EVALUATION_DIR / "results" / "baseline_smooth50.csv"))
    assert stats["n_possessions"] > 100
    assert 0 < stats["dead_share"] < 1
    assert stats["abstain_share"] == 0


def _write(path, text):
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return str(path)


def test_labels_to_csv_builds_dev_and_holdout(tmp_path, monkeypatch):
    windows = _write(
        tmp_path / "w.csv",
        """
jendela,frame_awal,frame_akhir,split,pelabel
V1,30000,30099,dev,Budi
V2,40000,40049,holdout,Sari
""",
    )
    changes = _write(
        tmp_path / "c.csv",
        """
jendela,frame_mulai,label,pelabel,catatan
V1,30000,H,Budi,awal
V1,30040,a,Budi,intersep
V1,30080,D,Budi,keluar
V2,40000,A,Sari,
,,,,
,,,,
""",
    )
    monkeypatch.setattr(labels_to_csv, "LABELS_DIR", tmp_path)
    assert labels_to_csv.main(["--windows", windows, "--changes", changes]) == 0
    dev = read_codes(tmp_path / "validasi_dev.csv")
    assert (dev[30000:30040] == H).all() and (dev[30040:30080] == A).all()
    assert (dev[30080:30100] == D).all() and dev[30100] == X and dev[29999] == X
    holdout = read_codes(tmp_path / "validasi_holdout.csv")
    assert (holdout[40000:40050] == A).all() and int(np.sum(holdout != X)) == 50


@pytest.mark.parametrize(
    "windows_text, changes_text, message",
    [
        (
            "jendela,frame_awal,frame_akhir,split,pelabel\nV1,30000,30099,dev,Budi",
            "jendela,frame_mulai,label,pelabel\nV1,30010,H,Budi",
            "must start at",
        ),
        (
            "jendela,frame_awal,frame_akhir,split,pelabel\nV1,15000,16000,dev,Budi",
            "jendela,frame_mulai,label,pelabel\nV1,15000,H,Budi",
            "public window",
        ),
        (
            "jendela,frame_awal,frame_akhir,split,pelabel\nV1,30000,30099,dev,Budi",
            "jendela,frame_mulai,label,pelabel\nV1,30000,Q,Budi",
            "H, A or D",
        ),
    ],
)
def test_labels_to_csv_rejects_mistakes(tmp_path, capsys, windows_text, changes_text, message):
    windows = _write(tmp_path / "w.csv", windows_text)
    changes = _write(tmp_path / "c.csv", changes_text)
    assert labels_to_csv.main(["--windows", windows, "--changes", changes]) == 1
    assert message in capsys.readouterr().out
