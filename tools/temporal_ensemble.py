"""Frozen multi-context neural probability ensemble; each component keeps its own normalizer."""

import json
from pathlib import Path

import numpy as np
from evaluate_temporal import load_model
from temporal_model import blend, symmetric_infer

from mtah.boundary import decode_boundaries
from mtah.io import sha256
from mtah.paths import ROOT


def load_ensemble(directory, device):
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text())
    receipt = json.loads((directory / "test_freeze.json").read_text())
    for name, expected in receipt["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != expected:
            raise ValueError(f"ensemble artifact changed: {name}")
    for name, expected in metadata["helper_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"ensemble dependency changed: {name}")
    components = []
    for path in metadata["components"]:
        item = (directory / path).resolve()
        if not item.is_relative_to(directory.resolve()):
            raise ValueError("ensemble component path escapes model directory")
        part, _, net, _ = load_model(item, device)
        if part["features"] != metadata["features"]:
            raise ValueError("ensemble components have different feature definitions")
        components.append((part, net))
    if len(components) != 2:
        raise ValueError("this ensemble requires exactly two frozen components")
    parent = json.loads((directory / "parent.json").read_text())
    return metadata, parent, components, receipt


def predict_ensemble(item, components, metadata, parent, device):
    probabilities = []
    for part, net in components:
        state, _ = symmetric_infer(
            net,
            item["values"],
            item["runs"],
            part["features"],
            np.asarray(part["mean"], dtype=np.float32),
            np.asarray(part["scale"], dtype=np.float32),
            device,
        )
        probabilities.append(state)
    state = blend(probabilities[0], probabilities[1], metadata["component_weight"])
    return decode_boundaries(
        blend(item["state"], state, metadata["state_weight"]),
        blend(item["event"], item["event"], 0),
        {**parent, "decoder_scale": metadata["decoder_scale"]},
        item["runs"],
    )
