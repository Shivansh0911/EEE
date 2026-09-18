"""
The network itself: 3 -> 3 -> 2 -> n_out, tansig throughout.

Parameter count at n_out = 2 is
    (3*3 + 3) + (3*2 + 2) + (2*2 + 2) = 12 + 8 + 6 = 26,
matching Table 1 of Panahi et al.

This is deliberately from-scratch NumPy. The hand-implemented forward pass,
Jacobian and Levenberg-Marquardt optimizer are the pedagogical point of the
project; a framework would hide exactly the parts that are being demonstrated.

Parameter packing order is the notebook's, unchanged, so the two remain
diffable:  W1, b1, W2, b2, W3, b3, each row-major.  W_l has shape
(n_in_l, n_out_l), i.e. W[m, n] connects incoming unit m to unit n.
"""

import numpy as np


def tansig(z):
    return np.tanh(z)


def tansig_prime_from_output(a):
    """d/dz tanh(z) expressed in terms of a = tanh(z). Cheaper and exact."""
    return 1.0 - a * a


ACTIVATIONS = {
    "tansig": (tansig, tansig_prime_from_output),
    "linear": (lambda z: z, lambda a: np.ones_like(a)),
}


def layer_shapes(layer_sizes):
    """[(n_in, n_out), ...] for each weight matrix."""
    return [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]


def n_params(layer_sizes):
    return sum(a * b + b for a, b in layer_shapes(layer_sizes))


def pack(params_list):
    """params_list is [W1, b1, W2, b2, ...] -> flat vector."""
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in params_list])


def unpack(theta, layer_sizes):
    """Flat vector -> [(W1, b1), (W2, b2), ...]."""
    out = []
    i = 0
    for a, b in layer_shapes(layer_sizes):
        W = theta[i:i + a * b].reshape(a, b)
        i += a * b
        bias = theta[i:i + b]
        i += b
        out.append((W, bias))
    if i != theta.size:
        raise ValueError(f"parameter vector has {theta.size} entries, expected {i}")
    return out


def nguyen_widrow(layer_sizes, rng):
    """
    Nguyen-Widrow initialisation.

    Nguyen D., Widrow B., "Improving the learning speed of 2-layer neural
    networks by choosing initial values of the adaptive weights", IJCNN 1990,
    vol. 3, pp. 21-26.

    Random weights alone leave most neurons responding over the same part of
    the input range, so several of them learn the same feature and the rest of
    the capacity is wasted. Nguyen-Widrow instead gives each of the S neurons in
    a layer a weight vector of the same magnitude beta = 0.7 * S^(1/R) and a
    bias that slides its active region to a different slice of [-1, 1], so the
    layer starts out covering the input space rather than piling up.

    This is what kills the seed lottery: with `--restarts 10` and this init, the
    restarts explore genuinely different basins instead of ten variations on one.
    """
    parts = []
    for R, S in layer_shapes(layer_sizes):
        W = rng.uniform(-1.0, 1.0, size=(R, S))
        norms = np.linalg.norm(W, axis=0)
        norms[norms == 0.0] = 1.0
        beta = 0.7 * (S ** (1.0 / R)) if R > 0 else 0.7
        W = beta * W / norms                      # every neuron gets magnitude beta
        if S > 1:
            b = beta * np.linspace(-1.0, 1.0, S)  # spread the active regions
        else:
            b = np.array([0.0])
        b = b * rng.choice([-1.0, 1.0], size=S)   # arbitrary orientation
        parts.extend([W, b])
    return pack(parts)


def forward(theta, X, layer_sizes, activation="tansig", output_activation="tansig",
            cache=False):
    """
    Forward pass.

    X is (N, n_in) in NORMALISED units. Returns (N, n_out), also normalised.
    With cache=True also returns the per-layer activations, which the analytic
    Jacobian needs.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    layers = unpack(np.asarray(theta, dtype=float), layer_sizes)
    act, _ = ACTIVATIONS[activation]
    out_act, _ = ACTIVATIONS[output_activation]

    a = X
    activations = [a]
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = a @ W + b
        a = out_act(z) if i == last else act(z)
        activations.append(a)

    return (a, activations) if cache else a


def predict(theta, X, cfg):
    """Convenience wrapper that reads the architecture off a TrainConfig."""
    return forward(theta, X, cfg.layer_sizes, cfg.activation, cfg.output_activation)


def residuals(theta, X, Y, cfg):
    """
    Residual vector, length N * n_out, ROW-MAJOR: sample 0's outputs first.

    Per-output weights are applied here, so every consumer -- the loss, the
    Jacobian, the normal equations -- sees the same weighting automatically.
    """
    P = forward(theta, X, cfg.layer_sizes, cfg.activation, cfg.output_activation)
    w = np.asarray(cfg.weights, dtype=float)
    return ((P - np.atleast_2d(Y)) * w).ravel()


def mse(theta, X, Y, cfg):
    e = residuals(theta, X, Y, cfg)
    return float(np.mean(e * e))
