"""Known-owner silver-label accuracy, separate from complete H/A/D spellF1."""

import numpy as np
from temporal_model import blend, symmetric_infer

from mtah.boundary import decode_boundaries
from mtah.external import anchor_metrics


def partial_validation(net, items, parent, names, mean, scale, device):
    rows = []
    for item in items:
        state, _ = symmetric_infer(net, item["values"], item["runs"], names, mean, scale, device)
        prediction = decode_boundaries(
            blend(item["state"], state, 0.5),
            blend(item["event"], item["event"], 0),
            parent,
            item["runs"],
        )
        metrics = anchor_metrics(item["reference"], prediction)
        rows.append(
            {
                "game": str(item["game"]),
                "profile": str(item["profile"]),
                "known_owner_accuracy": metrics["pass_start_team_accuracy"],
                "known_owner_count": metrics["anchor_count"],
            }
        )
    return float(np.mean([row["known_owner_accuracy"] for row in rows])), rows
