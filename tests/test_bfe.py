"""
Tiwary & Basu (1988) focus-electrode and anode synthesis, src/physics/bfe.py.
"""

import math

import numpy as np
import pytest

from src.physics import bfe, lb

# The three guns of the paper's Fig. 2: (rc/ra, rc in mm, theta0 in deg).
GUNS = [(1.985, 13.665, 13.44), (2.389, 10.158, 15.59), (2.393, 17.567, 27.17)]


def test_alpha_series_matches_vaughan_on_the_real_axis():
    y = np.linspace(0.05, 2.5, 30)
    assert np.allclose(bfe.alpha(y).real, lb.a_from_y(y), rtol=1e-13)


def test_complex_inversion_round_trips():
    for a in (0.3 - 0.2j, 1.5 - 1.0j, 2.0 + 0.0j):
        g = bfe.alpha_inv(a, a)
        assert abs(complex(bfe.alpha(g)) - a) < 1e-12


@pytest.mark.parametrize("ratio, rc, th", GUNS)
def test_bfe_leaves_the_rim_at_the_pierce_angle(ratio, rc, th):
    assert bfe.start_angle_deg(rc, th) == pytest.approx(67.5, abs=0.01)


@pytest.mark.parametrize("ratio, rc, th", GUNS)
def test_anode_equipotential_passes_through_the_anode_sphere(ratio, rc, th):
    """The paper's interpolation missed Ra by 0.26-0.41 %; exact tracing hits it."""
    ra = rc / ratio
    e = bfe.electrodes(rc, ra, th)
    z, r = e["anode"][0]
    assert math.hypot(rc - z, r) == pytest.approx(ra, rel=1e-12)


@pytest.mark.parametrize("ratio, rc, th", GUNS)
def test_curves_are_equipotentials(ratio, rc, th):
    ra = rc / ratio
    t0 = math.radians(th)
    for s, level in ((bfe.bfe_s()[1:], 0.0), (bfe.anode_s(math.log(ratio)), 1.0)):
        rho, theta = rc * np.exp(s.real), t0 + s.imag
        v = bfe.potential(rho, theta, rc, ra, t0)
        assert np.max(np.abs(v - level)) < 1e-9


def test_bfe_is_the_same_curve_for_every_gun():
    """The paper's remark: the BFE does not depend on the input parameters, only
    translates in theta. In s = ln(rho/Rc) + j(theta - theta0) it is universal."""
    s = bfe.bfe_s()
    for ratio, rc, th in GUNS:
        P = bfe.to_gun(s, rc, math.radians(th))
        rho = np.hypot(rc - P[:, 0], P[:, 1])
        back = np.log(rho / rc) + 1j * (np.arctan2(P[:, 1], rc - P[:, 0]) - math.radians(th))
        assert np.allclose(back, s, atol=1e-12)


@pytest.mark.parametrize("ratio, rc, th", GUNS)
def test_conformal_potential_is_2d_harmonic_but_not_axisymmetric_laplace(ratio, rc, th):
    """The map makes V harmonic in (ln rho, theta) -- checked to finite-difference
    accuracy -- but the axisymmetric Laplacian keeps V_u + cot(theta) V_theta,
    which the paper drops. It is NOT small; recorded so nobody claims otherwise."""
    res = bfe.laplace_residual(rc, rc / ratio, th)
    assert res["check_2d_harmonic_max"] < 1e-4
    assert res["rms"] > 0.3
