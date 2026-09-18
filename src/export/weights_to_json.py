"""
Export a trained model to the single JSON file described in B.3 of the master
build document.

One file, two consumers. The browser must never guess the network's shape, the
activation, or the scaler's ranges -- if it guesses and is wrong, the failure
shows up as a plausible-looking wrong angle during a live demo rather than as an
exception. So architecture, weights, scaler, targets, envelope, metrics and
provenance all travel together, and both the Python and the JS forward pass read
them from here.

`tests/test_js_python_parity.py` (P7/P8) asserts the two agree to < 1e-9.

The activation is written as "tanh" rather than "tansig" because that is what
the schema in B.3 specifies and what `Math.tanh` is called in JavaScript. They
are the same function.
"""

import datetime as _dt
import json
import os

from ..ann.network import unpack

_ACT_NAMES = {"tansig": "tanh", "linear": "linear"}


def build_payload(theta, cfg, x_scaler, y_scaler, metrics, provenance):
    layers = unpack(theta, cfg.layer_sizes)

    weights = {}
    for i, (W, b) in enumerate(layers, start=1):
        weights[f"W{i}"] = W.tolist()      # (n_in, n_out); JS does x @ W + b
        weights[f"b{i}"] = b.tolist()

    envelope = {}
    for j, name in enumerate(cfg.inputs):
        envelope[name] = [float(x_scaler.x_min[j]), float(x_scaler.x_max[j])]

    target_envelope = {}
    for k, name in enumerate(cfg.targets):
        target_envelope[name] = [float(y_scaler.x_min[k]), float(y_scaler.x_max[k])]

    return {
        "schema_version": 1,
        "architecture": list(cfg.layer_sizes),
        "activation": _ACT_NAMES[cfg.activation],
        "output_activation": _ACT_NAMES[cfg.output_activation],
        "weights": weights,
        "scaler": {
            "x_min": x_scaler.x_min.tolist(),
            "x_max": x_scaler.x_max.tolist(),
            "y_min": y_scaler.x_min.tolist(),
            "y_max": y_scaler.x_max.tolist(),
            "note": "fitted on training rows only",
        },
        "inputs": list(cfg.inputs),
        "targets": list(cfg.targets),
        "training_envelope": envelope,
        "target_envelope": target_envelope,
        "residual_weights": list(cfg.weights),
        "metrics": metrics,
        "provenance": provenance,
    }


def write_model(path, theta, cfg, x_scaler, y_scaler, metrics, provenance):
    payload = build_payload(theta, cfg, x_scaler, y_scaler, metrics, provenance)
    payload["provenance"].setdefault(
        "exported_at", _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_model(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
