"""
The Jacobian of the residual vector with respect to the 26 parameters.

J has one row per residual and one column per parameter, shape
(N * n_out) x n_params, and J[i][j] answers: if I nudge parameter j, how much
does residual i change? A local sensitivity map, which is what Levenberg-
Marquardt needs to aim its step.

Two implementations:

`analytic_jacobian`  backpropagates EACH RESIDUAL separately. Ordinary backprop
    sums every residual's derivative into one gradient vector, which is all
    gradient descent needs; least-squares needs them kept apart. Exact, one
    backward pass per output, and there is no epsilon to choose.

`fd_jacobian`  the original notebook's finite-difference version, kept as the
    reference the analytic one is tested against (tests/test_jacobian.py,
    max abs difference < 1e-6). Central rather than forward differences: the
    forward form the notebook used carries O(eps) truncation error, which is
    ~1e-5 at the notebook's eps and would swamp the tolerance the test needs.

Why the analytic version is the one that runs: finite differences cost
n_params + 1 = 27 forward passes per Jacobian, every epoch, and the accuracy
depends on choosing eps well -- too large is crude, too small and floating-point
cancellation destroys it.
"""

import numpy as np

from .network import ACTIVATIONS, forward, residuals, unpack


def analytic_jacobian(theta, X, Y, cfg):
    """
    Exact Jacobian by backpropagation.

    Row ordering is row-major over (sample, output), matching
    `network.residuals`: row n * n_out + k is sample n, output k.
    """
    theta = np.asarray(theta, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    layer_sizes = cfg.layer_sizes
    n_out = cfg.n_out
    N = X.shape[0]

    _, d_act = ACTIVATIONS[cfg.activation]
    _, d_out_act = ACTIVATIONS[cfg.output_activation]

    _, acts = forward(theta, X, layer_sizes, cfg.activation,
                      cfg.output_activation, cache=True)
    layers = unpack(theta, layer_sizes)
    last = len(layers) - 1
    w = np.asarray(cfg.weights, dtype=float)

    J = np.zeros((N * n_out, theta.size))

    for k in range(n_out):
        # Seed: d(residual for output k) / d(pre-activation of the output layer).
        # One-hot in k, because this residual sees only its own output unit.
        delta = np.zeros((N, n_out))
        delta[:, k] = d_out_act(acts[-1])[:, k]

        blocks = [None] * (2 * len(layers))
        for i in range(last, -1, -1):
            W, _ = layers[i]
            a_in = acts[i]                                   # (N, n_in_i)
            # dW[n, m, p] = a_in[n, m] * delta[n, p]
            dW = a_in[:, :, None] * delta[:, None, :]
            blocks[2 * i] = dW.reshape(N, -1)
            blocks[2 * i + 1] = delta
            if i > 0:
                delta = (delta @ W.T) * d_act(a_in)          # (N, n_in_i)

        rows = np.arange(N) * n_out + k
        J[rows, :] = w[k] * np.concatenate(blocks, axis=1)

    return J


def fd_jacobian(theta, X, Y, cfg, eps=1e-6, scheme="central"):
    """Finite-difference Jacobian. Reference implementation, used by the tests."""
    theta = np.asarray(theta, dtype=float)
    n_res = residuals(theta, X, Y, cfg).size
    J = np.zeros((n_res, theta.size))

    if scheme == "forward":
        base = residuals(theta, X, Y, cfg)

    for j in range(theta.size):
        if scheme == "central":
            tp = theta.copy(); tp[j] += eps
            tm = theta.copy(); tm[j] -= eps
            J[:, j] = (residuals(tp, X, Y, cfg) - residuals(tm, X, Y, cfg)) / (2 * eps)
        elif scheme == "forward":
            tp = theta.copy(); tp[j] += eps
            J[:, j] = (residuals(tp, X, Y, cfg) - base) / eps
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

    return J
