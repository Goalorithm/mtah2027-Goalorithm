"""The competition match as the ATD-realistic model sees it: tracking, video ball, cutaways.

Everything comes from the provided assets by algorithm: the automatic tracking, the ball
re-detected in the tactical video (tools/video_ball*.py, anchored and whole-frame) and the
validity flag of the broadcast homography after synchronising the two recordings.
"""

import _src_path  # noqa: F401
import numpy as np
from atd_features import atd_feature_matrix
from broadcast_ball import half_lags
from broadcast_cut import measured_cutaways

from mtah.load import load_tracking
from mtah.paths import OUTPUTS_DIR, match_runs

VIDEO_TRACK = OUTPUTS_DIR / "video_ball" / "track_full.npz"


def video_ball(n_frames):
    """Re-detected ball in pitch metres for every video frame; NaN where there is none."""
    if not VIDEO_TRACK.is_file():
        raise FileNotFoundError(
            "re-detect the ball first: tools/video_ball.py, video_ball_track.py, "
            "video_ball_full.py and video_ball_merge.py"
        )
    ball = np.full((n_frames, 2), np.nan)
    with np.load(VIDEO_TRACK, allow_pickle=False) as data:
        ball[data["frames"]] = data["ball"]
    return ball


def broadcast_cutaways(tracking, runs, settings):
    """(cutaway indicator on the tactical timeline, synchronisation report)."""
    from load_homography import load as load_homography

    broadcast = load_tracking(source="broadcast")
    lags, report = half_lags(tracking.ball, broadcast.ball, runs, settings)
    valid = load_homography(source="broadcast")[2]
    return measured_cutaways(tracking.n_frames, valid, runs, lags), report


def competition_inputs(cfg):
    tracking, runs = load_tracking(), match_runs()
    if "video" in cfg.get("feature_groups", ["video"]):
        video = video_ball(tracking.n_frames)
    else:
        # A model without video-ball inputs never needs the video re-detection to have run.
        video = np.full((tracking.n_frames, 2), np.nan)
    if "cut" in cfg.get("feature_groups", ["cut"]):
        cut, report = broadcast_cutaways(tracking, runs, cfg["broadcast"])
    else:
        # A model without cutaway inputs never reads the broadcast recording.
        cut, report = np.full(tracking.n_frames, np.nan), None
    return tracking, runs, video, cut, report


def competition_features(cfg, feature_cfg):
    """(features in half precision like the cached profiles, runs, input report)."""
    tracking, runs, video, cut, report = competition_inputs(cfg)
    values = atd_feature_matrix(tracking, feature_cfg, cfg["team_features"], runs, video, cut)
    graded = np.concatenate([np.arange(a, b) for a, b in runs])
    summary = {
        "broadcast_synchronisation": report,
        "automatic_ball_share": float(np.isfinite(tracking.ball[graded]).all(axis=1).mean()),
        "video_ball_share": float(np.isfinite(video[graded]).all(axis=1).mean()),
        "cutaway_share": None if np.isnan(cut).all() else float(np.nanmean(cut[graded])),
    }
    return values.astype(np.float16), runs, summary
