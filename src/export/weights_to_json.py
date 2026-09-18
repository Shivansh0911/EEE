"""
Export a trained model to the single JSON file the browser reads.

One file, two consumers. The browser must never guess the network's shape, the
activation, the input transform, or the scaler's ranges -- if it guesses and is
wrong, the failure shows up as a plausible-looking wrong angle during a live
demo rather than as an exception. So architecture, weights, input transform,
scaler, targets, envelope, metrics and provenance all travel together, and both
the Python and the JavaScript forward pass read them from here.

`tests/test_js_python_parity.py` asserts the two agree to < 1e-9 on 100 random
inputs, running the JS under Node.

The `input_transform` block is the one most easily lost. The scaler's min/max
are in TRANSFORMED space, so a consumer that applied the scaler without first
applying ln to C would produce numbers that look reasonable and are wrong. It is
exported explicitly and the parity test covers it.

The activation is written as "tanh" rather than "tansig" because that is what
`Math.tanh` is called in JavaScript. They are the same function.
"""

import datetime as _dt
import json
import os

from ..ann.network import unpack

_ACT_NAMES = {"tansig": "tanh", "linear": "linear"}


def build_payload(theta, cfg, x_pipeline, y_scaler, metrics, provenance):
    layers = unpack(theta, cfg.layer_sizes)

    weights = {}
    for i, (W, b) in enumerate(layers, start=1):
        weights[f"W{i}"] = W.tolist()      # (n_in, n_out); JS does x @ W + b
        weights[f"b{i}"] = b.tolist()

    target_envelope = {
        name: [float(y_scaler.x_min[k]), float(y_scaler.x_max[k])]
        for k, name in enumerate(cfg.targets)
    }

    return {
        "schema_version": 2,
        "architecture": list(cfg.layer_sizes),
        "activation": _ACT_NAMES[cfg.activation],
        "output_activation": _ACT_NAMES[cfg.output_activation],
        "weights": weights,
        # Applied to the named input BEFORE min-max scaling. See D8.
        "input_transform": dict(cfg.input_transform),
        "scaler": {
            "x_min": x_pipeline.x_min.tolist(),   # in TRANSFORMED space
            "x_max": x_pipeline.x_max.tolist(),
            "y_min": y_scaler.x_min.tolist(),
            "y_max": y_scaler.x_max.tolist(),
            "note": ("fitted on training rows only; x ranges are in transformed "
                     "space, so input_transform must be applied first"),
        },
        "inputs": list(cfg.inputs),
        "targets": list(cfg.targets),
        # Physical units, for the out-of-envelope warning in the UI.
        "training_envelope": x_pipeline.envelope(),
        "target_envelope": target_envelope,
        "residual_weights": list(cfg.weights),
        "saturation_limit": cfg.saturation_limit,
        "metrics": metrics,
        "provenance": provenance,
    }


def write_model(path, theta, cfg, x_pipeline, y_scaler, metrics, provenance):
    payload = build_payload(theta, cfg, x_pipeline, y_scaler, metrics, provenance)
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
