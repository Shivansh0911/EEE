"""
Min-max scaling to [-1, 1].

    x_norm = 2 (x - x_min) / (x_max - x_min) - 1

The rule that matters: x_min and x_max come from the TRAINING rows only, and
are then reused unchanged for validation, test and live inference. Fitting the
scaler on everything leaks the test set's range into training, and with 30 rows
that leak is large. The original Colab notebook already did this correctly.

The fitted scaler is persisted to models/scaler.json because inference in the
browser has to reuse exactly the same numbers -- see B.3 of the master build
document.
"""

import json

import numpy as np


class MinMaxScaler:
    """Fit on train only. Stateless afterwards."""

    def __init__(self, x_min, x_max):
        self.x_min = np.asarray(x_min, dtype=float)
        self.x_max = np.asarray(x_max, dtype=float)
        span = self.x_max - self.x_min
        if np.any(span <= 0):
            bad = np.where(span <= 0)[0].tolist()
            raise ValueError(f"degenerate range in columns {bad}: max <= min")

    @classmethod
    def fit(cls, X):
        X = np.asarray(X, dtype=float)
        return cls(X.min(axis=0), X.max(axis=0))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return 2.0 * (X - self.x_min) / (self.x_max - self.x_min) - 1.0

    def inverse_transform(self, Xn):
        Xn = np.asarray(Xn, dtype=float)
        return (Xn + 1.0) / 2.0 * (self.x_max - self.x_min) + self.x_min

    def to_dict(self):
        return {"min": self.x_min.tolist(), "max": self.x_max.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["min"], d["max"])


def save_scalers(path, x_scaler, y_scaler, inputs, targets):
    """Persist both scalers plus the column names they belong to."""
    payload = {
        "schema_version": 1,
        "note": "fitted on training rows only; reused verbatim for test and inference",
        "inputs": list(inputs),
        "targets": list(targets),
        "x_min": x_scaler.x_min.tolist(),
        "x_max": x_scaler.x_max.tolist(),
        "y_min": y_scaler.x_min.tolist(),
        "y_max": y_scaler.x_max.tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_scalers(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    x = MinMaxScaler(d["x_min"], d["x_max"])
    y = MinMaxScaler(d["y_min"], d["y_max"])
    return x, y, d


def saturation_report(Yn_pred, limit=0.98):
    """
    Rows/outputs whose NORMALISED prediction has run past `limit`.

    A tansig output layer over min-max scaled targets structurally cannot
    predict outside the training range: it asymptotes at +/-1, which maps back
    to y_min and y_max. So a prediction at +/-0.98 is not a confident
    extrapolation, it is the network against its ceiling. The paper's own test
    case 3 (true 50.28 deg, predicted 45.03 deg) is this failure mode, and it is
    the single largest test error in the paper.

    Returns a list of (row_index, output_index, normalised_value).
    """
    Yn_pred = np.atleast_2d(np.asarray(Yn_pred, dtype=float))
    hits = np.argwhere(np.abs(Yn_pred) > limit)
    return [(int(i), int(k), float(Yn_pred[i, k])) for i, k in hits]
