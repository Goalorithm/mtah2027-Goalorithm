"""Learned directed team handovers as transition potentials, resetting at each valid run."""

import numpy as np

from mtah import A, D, H, X
from mtah.hmm import _probabilities, sequence_runs
from mtah.paths import pm


def boundary_targets(reference, runs, radius=2):
    if type(radius) is not int or radius < 0:
        raise ValueError("boundary target radius must be a nonnegative integer")
    reference = np.asarray(reference)
    result = np.full(len(reference), X, dtype=np.int8)
    for start, stop in sequence_runs(len(reference), runs):
        for local_a, local_b in pm.valid_runs(reference[start:stop]):
            a, b = start + local_a, start + local_b
            result[a:b] = D  # D means no H/A handover in this separate event classifier.
            nearest = np.full(b - a, radius + 1)
            changes = (
                np.flatnonzero(
                    (reference[a + 1 : b] != reference[a : b - 1])
                    & np.isin(reference[a + 1 : b], [H, A])
                    & np.isin(reference[a : b - 1], [H, A])
                )
                + a
                + 1
            )
            for t in changes:
                lo, hi = max(a, t - radius), min(b, t + radius + 1)
                distance = np.abs(np.arange(lo, hi) - t)
                keep = distance < nearest[lo - a : hi - a]
                result[lo:hi][keep] = reference[t]
                nearest[lo - a : hi - a][keep] = distance[keep]
    return result


def decode_boundaries(probabilities, event_probabilities, model, runs):
    from mtah.supervised import decode_probabilities

    probabilities = np.asarray(probabilities, dtype=float)
    events = np.asarray(event_probabilities, dtype=float)
    _probabilities(probabilities, (len(probabilities), 3), "supervised probabilities")
    _probabilities(events, probabilities.shape, "handover probabilities")
    weight = model["boundary_weight"]
    if isinstance(weight, bool) or not np.isfinite(weight) or weight < 0:
        raise ValueError("invalid boundary weight")
    if weight == 0:
        return decode_probabilities(probabilities, model, runs)
    prior = _probabilities(model["class_prior"], (3,), "supervised class prior")
    event_prior = _probabilities(model["boundary_prior"], (3,), "boundary prior")
    transition = _probabilities(model["transition"], (3, 3), "transition")
    initial = _probabilities(model["initial"], (3,), "initial")
    scale = model["decoder_scale"]
    if (prior <= 0).any() or (event_prior <= 0).any() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid boundary decoder priors or emission scale")
    tiny = np.finfo(float).tiny
    emission = (np.log(np.maximum(probabilities, tiny)) - np.log(prior)) * scale
    evidence = (np.log(np.maximum(events, tiny)) - np.log(event_prior)) * weight
    with np.errstate(divide="ignore"):
        log_transition, log_initial = np.log(transition), np.log(initial)
    result = np.full(len(probabilities), X, dtype=np.int8)
    columns = np.arange(3)
    for start, stop in sequence_runs(len(probabilities), runs):
        back = np.zeros((stop - start, 3), dtype=np.int8)
        score = log_initial + emission[start]
        if not np.isfinite(score.max()):
            raise ValueError("no reachable state at the start of the run")
        score -= score.max()
        for t in range(start + 1, stop):
            candidates = score[:, None] + log_transition
            candidates[H, H] += evidence[t, D]
            candidates[A, A] += evidence[t, D]
            candidates[A, H] += evidence[t, H]
            candidates[H, A] += evidence[t, A]
            parent = candidates.argmax(axis=0)
            back[t - start] = parent
            score = emission[t] + candidates[parent, columns]
            if not np.isfinite(score.max()):
                raise ValueError("no reachable state in boundary decoding")
            score -= score.max()
        state = int(score.argmax())
        result[stop - 1] = state
        for t in range(stop - 1, start, -1):
            state = int(back[t - start, state])
            result[t - 1] = state
    return result
