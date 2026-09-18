"""
Per-input transforms, applied BEFORE min-max scaling.

Why this exists, and specifically why C is log-transformed (D8):

C is the ratio of the cathode area to the beam-waist area, and it enters the
physics through gamma = ln(R_c/R_a) -- equation (II) of our own physics model
contains 0.5*ln(C) explicitly. So ln C, not C, is the coordinate the underlying
equations are written in.

It also fixes a resolution problem that has nothing to do with any model. C
spans 5.12 to 306.3, a factor of 60, and the distribution is heavily skewed:
under linear min-max scaling 23 of the 30 rows land in the bottom 15 % of the
input range, compressed into a band the network cannot resolve within. Under
ln C they spread out.

The transform is part of the model, not part of the training script, so it
travels in the exported JSON and the browser applies exactly the same one. If it
did not, Python and JavaScript would disagree and the site would be quietly
wrong.
"""

import numpy as np

FORWARD = {
    "identity": lambda v: v,
    "log": np.log,          # natural log; ln C is the physics coordinate
    "log10": np.log10,
}

INVERSE = {
    "identity": lambda v: v,
    "log": np.exp,
    "log10": lambda v: np.power(10.0, v),
}


def validate(spec, inputs):
    """Reject an unknown transform or a column name that is not an input."""
    for name, kind in (spec or {}).items():
        if name not in inputs:
            raise KeyError(f"input_transform names {name!r}, which is not an input "
                           f"column ({inputs})")
        if kind not in FORWARD:
            raise ValueError(f"unknown transform {kind!r} for {name!r}; "
                             f"known: {sorted(FORWARD)}")


def forward(X, inputs, spec):
    """Apply the per-column transforms. X is (N, n_in) in physical units."""
    validate(spec, inputs)
    X = np.array(X, dtype=float, copy=True)
    X = np.atleast_2d(X)
    for name, kind in (spec or {}).items():
        j = inputs.index(name)
        col = X[:, j]
        if kind in ("log", "log10") and np.any(col <= 0):
            raise ValueError(f"cannot {kind}-transform {name!r}: non-positive values")
        X[:, j] = FORWARD[kind](col)
    return X


def inverse(X, inputs, spec):
    """Back to physical units. Used for reporting the training envelope."""
    validate(spec, inputs)
    X = np.array(X, dtype=float, copy=True)
    X = np.atleast_2d(X)
    for name, kind in (spec or {}).items():
        j = inputs.index(name)
        X[:, j] = INVERSE[kind](X[:, j])
    return X


def describe(spec):
    """One-line human summary, for logs and the exported provenance."""
    if not spec:
        return "none (all inputs used as-is)"
    return ", ".join(f"{k} -> {v}" for k, v in sorted(spec.items()))
