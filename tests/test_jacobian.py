"""
Proof that the hand-derived analytic Jacobian is right.

The analytic version is a derivation done on paper and then typed in, which is
exactly the kind of code that can be subtly wrong and still train -- a sign
error in one block just makes convergence worse, not obviously broken. So it is
checked against finite differences, which are slow and inelegant but involve no
derivation at all.

Agreement to < 1e-6 in max absolute difference is the assertion. Central
differences at eps = 1e-6 carry truncation error ~eps^2 and roundoff ~1e-10, so
a genuine disagreement has nowhere to hide below the tolerance.
"""

import numpy as np
import pytest

from src.ann.config import TrainConfig
from src.ann.jacobian import analytic_jacobian, fd_jacobian
from src.ann.network import n_params, nguyen_widrow

TOL = 1e-6


def _case(rng, cfg, n_samples=9):
    theta = nguyen_widrow(cfg.layer_sizes, rng)
    X = rng.uniform(-1.0, 1.0, size=(n_samples, cfg.n_in))
    Y = rng.uniform(-1.0, 1.0, size=(n_samples, cfg.n_out))
    return theta, X, Y


def test_param_count_matches_paper_table1():
    cfg = TrainConfig()
    assert cfg.layer_sizes == [3, 3, 2, 2]
    assert n_params(cfg.layer_sizes) == 26


@pytest.mark.parametrize("trial", range(20))
def test_analytic_matches_finite_difference(trial):
    rng = np.random.default_rng(1000 + trial)
    cfg = TrainConfig()
    theta, X, Y = _case(rng, cfg)

    Ja = analytic_jacobian(theta, X, Y, cfg)
    Jf = fd_jacobian(theta, X, Y, cfg)

    assert Ja.shape == (X.shape[0] * cfg.n_out, theta.size)
    assert np.max(np.abs(Ja - Jf)) < TOL


@pytest.mark.parametrize("out_act", ["tansig", "linear"])
@pytest.mark.parametrize("n_out", [1, 2, 3])
def test_analytic_matches_fd_across_shapes(out_act, n_out):
    """The derivation must not quietly depend on n_out == 2 or on tansig output."""
    rng = np.random.default_rng(7 * n_out + len(out_act))
    cfg = TrainConfig(targets=[f"t{i}" for i in range(n_out)], output_activation=out_act)
    theta, X, Y = _case(rng, cfg)

    Ja = analytic_jacobian(theta, X, Y, cfg)
    Jf = fd_jacobian(theta, X, Y, cfg)
    assert np.max(np.abs(Ja - Jf)) < TOL


def test_residual_weights_scale_the_jacobian_rows():
    """
    Per-output weights must reach the Jacobian, not just the loss. If they only
    scaled the residuals, LM would aim its step with the wrong sensitivities.
    """
    rng = np.random.default_rng(42)
    cfg_a = TrainConfig()
    cfg_b = TrainConfig(residual_weights=[3.0, 0.5])
    theta, X, Y = _case(rng, cfg_a)

    Ja = analytic_jacobian(theta, X, Y, cfg_a)
    Jb = analytic_jacobian(theta, X, Y, cfg_b)

    n = X.shape[0]
    even = np.arange(n) * 2       # output 0 rows
    odd = even + 1                # output 1 rows
    assert np.allclose(Jb[even], 3.0 * Ja[even])
    assert np.allclose(Jb[odd], 0.5 * Ja[odd])

    assert np.max(np.abs(Jb - fd_jacobian(theta, X, Y, cfg_b))) < TOL
