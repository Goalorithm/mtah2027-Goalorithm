"""ATD-like measurement simulation for clean, labelled external games.

The competition tracking is automatic: the ball is missing on 41% of graded frames and about 20.5
objects are tracked on average (organisers' DATA.md). Corrected external tracking has none of
this. IDSSE freezes the ball where play stopped and Metrica blanks it, so a model fitted on either
learns the provider's convention instead of the game. This module re-synthesises the ball and
thins the players of an external game, with every parameter drawn from wide ranges.

There is no video ball here: no component of this workflow is trained, fitted or calibrated on the
provided video. The external reference decides where stoppages are; that is a simulation of the
measurement process on other games, nothing here is an inference input and no BAR-ATH label is read.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from mtah import D, X
from mtah.hmm import sequence_runs
from mtah.load import Tracking
from mtah.paths import FPS, PITCH


def _runs_of(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def sample_parameters(ranges, rng):
    """One draw of every [low, high] range; scalars pass through unchanged."""
    result = {}
    for key in sorted(ranges):
        value = ranges[key]
        if isinstance(value, list):
            if len(value) != 2 or not value[0] <= value[1]:
                raise ValueError(f"simulation range {key} must be [low, high]")
            result[key] = float(rng.uniform(value[0], value[1]))
        else:
            result[key] = value
    return result


def renewal_mask(length, fraction, median, spread, rng):
    """Alternating kept/dropped stretches; dropped lengths are log-normal, never label-aware."""
    mask = np.zeros(length, dtype=bool)
    if fraction <= 0 or length == 0:
        return mask
    if not fraction < 1 or median <= 0 or spread < 0:
        raise ValueError("invalid renewal process parameters")
    mean_drop = median * np.exp(spread * spread / 2)
    mean_keep = mean_drop * (1 - fraction) / fraction
    cursor = int(rng.exponential(mean_keep))
    while cursor < length:
        drop = max(1, int(round(rng.lognormal(np.log(median), spread))))
        mask[cursor : cursor + drop] = True
        cursor += drop + 1 + int(rng.exponential(mean_keep))
    return mask


def stroll(length, speed_mps, cfg, rng):
    """A smooth excursion in metres that ends where it began: a ball being handled."""
    path = np.zeros((length, 2))
    if length < 3 or speed_mps <= 0:
        return path
    velocity = gaussian_filter1d(
        rng.normal(size=(length, 2)), cfg["stroll_smooth_frames"], axis=0, mode="nearest"
    )
    velocity *= speed_mps / FPS / max(np.sqrt((velocity**2).sum(axis=1).mean()), 1e-9)
    path = np.cumsum(velocity, axis=0)
    return path - np.linspace(0, 1, length)[:, None] * path[-1]


def dead_ball_path(previous, velocity, restart, length, rng, cfg, carrier=None, handled=0.0):
    """(positions, first carried frame) for a stoppage: roll on, rest, return, rest.

    previous/restart are the last live and first live positions (None at a period edge);
    velocity is metres per frame at the moment play stopped. carrier, when given, is the track
    of the player who restarts, shifted to meet the restart spot on the first live frame: the
    ball then arrives with that player instead of lying on the spot. handled is the chance that
    the ball is moved about while it waits instead of lying still.
    """
    path = np.full((length, 2), np.nan)
    if length == 0 or (previous is None and restart is None):
        return path, length
    if previous is None:
        path[:] = restart
        return path, length
    roll = int(min(length, rng.integers(cfg["roll_frames"][0], cfg["roll_frames"][1] + 1)))
    decay = rng.uniform(cfg["roll_decay"][0], cfg["roll_decay"][1])
    rolled = previous + np.cumsum(velocity[None, :] * decay ** np.arange(1, roll + 1)[:, None], 0)
    path[:roll] = rolled
    if restart is None or roll == length:
        path[roll:] = rolled[-1]
        return path, length
    remaining = length - roll
    rest = int(
        min(
            rng.integers(cfg["restart_rest_frames"][0], cfg["restart_rest_frames"][1] + 1),
            cfg["restart_rest_share"] * remaining,
        )
    )
    speed = rng.uniform(cfg["return_speed_mps"][0], cfg["return_speed_mps"][1]) / FPS
    distance = float(np.linalg.norm(restart - rolled[-1]))
    travel = int(np.clip(np.ceil(distance / speed), 1, max(remaining - rest, 1)))
    wait = remaining - rest - travel
    if wait < 0:
        rest, wait = max(remaining - travel, 0), 0
    path[roll : roll + wait] = rolled[-1]
    arrival = roll + wait + travel
    moved = rng.random() < handled
    pace = rng.uniform(cfg["stroll_speed_mps"][0], cfg["stroll_speed_mps"][1])
    if moved:
        path[roll : roll + wait] += stroll(wait, pace, cfg, rng)
    carried = carrier is not None and np.isfinite(carrier[arrival - 1 :]).all()
    target = carrier[arrival - 1] if carried else restart
    # A ball kicked or thrown back slows down on the way; exponent 1 is a steady carry.
    ease = rng.uniform(cfg["return_ease"][0], cfg["return_ease"][1])
    alpha = 1 - (1 - np.arange(1, travel + 1)[:, None] / travel) ** ease
    path[roll + wait : arrival] = rolled[-1] + alpha * (target - rolled[-1])
    path[arrival:] = carrier[arrival:] if carried else restart
    return path, arrival if carried else length


def _stoppage_visibility(length, mode, rng, cfg):
    """(visible mask over the stoppage, frames lost before it, frames lost after it)."""
    visible = np.ones(length, dtype=bool)
    before = after = 0
    if mode == "visible":
        return visible, before, after
    if mode == "partial":
        start = int(rng.integers(0, max(length // 2, 1)))
        size = int(rng.integers(max(length // 4, 1), max(length // 2, 2)))
        visible[start : start + size] = False
        return visible, before, after
    keep = int(rng.integers(0, cfg["lost_after_stop_frames"] + 1))
    visible[keep:] = False
    draw = rng.random()
    if draw < cfg["reacquire_just_before"]:
        back = int(rng.integers(0, cfg["just_before_frames"] + 1))
    elif draw < cfg["reacquire_just_before"] + cfg["reacquire_early"]:
        back = int(rng.integers(cfg["just_before_frames"], cfg["early_frames"] + 1))
    else:
        back = 0
        after = int(rng.integers(cfg["late_frames"][0], cfg["late_frames"][1] + 1))
    if back:
        visible[max(length - back, keep) :] = True
    if rng.random() < cfg["lost_before_stop"]:
        before = int(rng.integers(1, cfg["lost_before_frames"] + 1))
    return visible, before, after


def low_pass(ball, cutoff_hz, order, minimum_frames):
    """Zero-phase Butterworth on every observed stretch; short stretches keep their values."""
    if not 0 < cutoff_hz < FPS / 2:
        raise ValueError("low-pass cutoff must lie inside the Nyquist band")
    sos = butter(order, cutoff_hz / (FPS / 2), output="sos")
    result = np.array(ball, dtype=np.float64, copy=True)
    for start, stop in _runs_of(np.isfinite(ball).all(axis=1)):
        if stop - start >= minimum_frames:
            result[start:stop] = sosfiltfilt(sos, ball[start:stop], axis=0)
    return result


def simulate_ball(tracking, reference, runs, params, cfg, rng):
    """(automatic-like ball, ground track, lost mask) for every period.

    The ground track has re-synthesised stoppages. The automatic ball is that track hidden by
    the visibility process and band-limited; a flight stays on its ground track, which is
    where the automatic tracking keeps it.
    """
    length_m, width_m = PITCH
    margin = params["outside_visible_margin_m"]
    ball = np.array(tracking.ball, dtype=np.float64, copy=True)
    hidden = np.zeros(len(ball), dtype=bool)
    leaves = np.zeros(len(ball), dtype=bool)  # stoppages whose ball is lost once off the pitch
    modes = ("missing", "visible", "partial")
    chance = np.array(
        [params["stoppage_missing"], params["stoppage_visible"], 0.0], dtype=np.float64
    )
    chance[2] = max(1.0 - chance[:2].sum(), 0.0)
    chance /= chance.sum()
    for first, last in sequence_runs(len(ball), runs):
        for a, b in _runs_of(reference[first:last] == D):
            start, stop = first + a, first + b
            previous = velocity = restart = None
            if start > first and reference[start - 1] != X and np.isfinite(ball[start - 1]).all():
                previous = ball[start - 1]
                lag = min(cfg["velocity_lag_frames"], start - 1 - first)
                velocity = np.zeros(2)
                if lag > 0 and np.isfinite(ball[start - 1 - lag]).all():
                    velocity = (previous - ball[start - 1 - lag]) / lag
            if stop < last and reference[stop] != X and np.isfinite(ball[stop]).all():
                restart = ball[stop]
            carrier = None
            if restart is not None and rng.random() < params["stoppage_carried"]:
                gap = np.linalg.norm(tracking.players[stop] - restart, axis=1)
                if np.isfinite(gap).any():
                    taker = int(np.nanargmin(gap))
                    carrier = tracking.players[start:stop, taker] + (
                        restart - tracking.players[stop, taker]
                    )
            ball[start:stop], held = dead_ball_path(
                previous,
                velocity,
                restart,
                stop - start,
                rng,
                cfg,
                carrier,
                params["stoppage_handled"],
            )
            mode = modes[int(rng.choice(3, p=chance))]
            visible, before, after = _stoppage_visibility(stop - start, mode, rng, cfg)
            hidden[start:stop] |= ~visible
            leaves[start:stop] = mode != "visible"
            hidden[max(first, start - before) : start] = True
            hidden[stop : min(last, stop + after)] = True
        hidden[first:last] |= renewal_mask(
            last - first,
            params["live_gap_fraction"],
            params["live_gap_median_frames"],
            cfg["live_gap_spread"],
            rng,
        )
    outside = (
        (ball[:, 0] < -margin)
        | (ball[:, 0] > length_m + margin)
        | (ball[:, 1] < -margin)
        | (ball[:, 1] > width_m + margin)
    )
    # Only a stopped ball leaves for the stands. A stoppage tracked throughout keeps its ball
    # in view even behind the line, as on BAR-ATH where a ball six metres behind the goal
    # stays tracked until the goal kick.
    limit = cfg["tracked_margin_m"]  # BAR-ATH never reports a ball further out than this
    beyond = (
        (ball[:, 0] < -limit)
        | (ball[:, 0] > length_m + limit)
        | (ball[:, 1] < -limit)
        | (ball[:, 1] > width_m + limit)
    )
    lost = hidden | (outside & leaves) | (beyond & (reference == D))
    automatic = np.where(lost[:, None], np.nan, ball)
    automatic = low_pass(
        automatic, params["low_pass_hz"], cfg["low_pass_order"], cfg["low_pass_minimum_frames"]
    )
    if params["ball_noise_m"] > 0:
        automatic = automatic + rng.normal(0, params["ball_noise_m"], ball.shape)
    return automatic, ball, lost


def thin_players(tracking, reference, runs, params, cfg, rng):
    """Independent tracklet losses plus shared outages; speeds vanish with positions.

    On BAR-ATH most frames with 16 or fewer tracked players lie in long ball-less stretches,
    so part of the outages is tied to long stoppages; the rest ignores the game state.
    """
    players = np.array(tracking.players, dtype=np.float64, copy=True)
    speeds = np.array(tracking.player_speed, dtype=np.float64, copy=True)
    for first, last in sequence_runs(len(players), runs):
        size = last - first
        for a, b in _runs_of(reference[first:last] == D):
            if b - a < cfg["long_stoppage_frames"] or rng.random() >= params["stoppage_outage"]:
                continue
            span = int((b - a) * rng.uniform(*cfg["stoppage_outage_span"]))
            start = first + a + int(rng.integers(0, b - a - span + 1))
            lost = rng.random(players.shape[1]) < rng.uniform(*cfg["outage_share"])
            players[start : start + span, lost] = np.nan
            speeds[start : start + span, lost] = np.nan
        for slot in range(players.shape[1]):
            lost = renewal_mask(
                size,
                params["player_loss_fraction"],
                params["player_loss_median_frames"],
                cfg["player_loss_spread"],
                rng,
            )
            players[first:last][lost, slot] = np.nan
            speeds[first:last][lost, slot] = np.nan
        outages = rng.poisson(params["outage_per_minute"] * size / FPS / 60)
        for _ in range(outages):
            start = first + int(rng.integers(0, size))
            stop = min(last, start + int(rng.integers(*cfg["outage_frames"])))
            share = rng.uniform(*cfg["outage_share"])
            lost = rng.random(players.shape[1]) < share
            players[start:stop, lost] = np.nan
            speeds[start:stop, lost] = np.nan
    return players, speeds


def simulate_game(tracking, reference, runs, params, cfg, seed):
    """(automatic-like Tracking, an all-missing second ball) of a clean external game; seeded.

    The second ball is kept only so the feature layout stays fixed; the model does not read it.
    """
    reference = np.asarray(reference)
    if len(reference) != tracking.n_frames:
        raise ValueError("reference and tracking lengths differ")
    rng = np.random.default_rng(seed)
    automatic, ball, _ = simulate_ball(tracking, reference, runs, params, cfg, rng)
    players, speeds = thin_players(tracking, reference, runs, params, cfg, rng)
    degraded = Tracking(
        players,
        speeds,
        automatic,
        np.full(tracking.n_frames, np.nan),
        tracking.slot_team,
        tracking.team_names,
    )
    return degraded, np.full((tracking.n_frames, 2), np.nan)


def simulate(tracking, reference, runs, params, cfg, seed):
    """The automatic-like Tracking alone."""
    return simulate_game(tracking, reference, runs, params, cfg, seed)[0]
