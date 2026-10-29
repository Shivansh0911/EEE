"""
Physics layer: the Langmuir-Blodgett constants, the kf table, the LBClosure
emulator against Vedant's exact solver (Gate G1), and the adapter.
"""

import math

import numpy as np
import pytest

from src.physics import closure, gate_g1, kf, lb, probe_closure
from src.physics import vaughan_adapter as va


# --- constants, derived not asserted ------------------------------------------

def test_cone_constant_is_half_the_sphere_constant():
    assert lb.SPHERE_CONST == pytest.approx(29.33, abs=0.005)
    assert lb.CONE_CONST == pytest.approx(14.665, abs=0.001)
    # Vaughan's printed 14.67 is the same constant rounded.
    assert lb.VAUGHAN_CONE_CONST == pytest.approx(lb.CONE_CONST, rel=5e-4)


def test_old_probe_used_the_full_sphere_constant():
    """The Tier B generator went through probe_closure, which used 29.33 with
    (1 - cos theta): a factor of two too large for a cone. Recorded as D10."""
    assert probe_closure.P_CONST == pytest.approx(2 * lb.CONE_CONST, rel=1e-3)


def test_spread_constant_from_physical_constants():
    assert lb.SPREAD_CONST == pytest.approx(0.17409, abs=5e-6)


def test_y_from_a_matches_vendor_mpmath():
    m = va.module()
    a = np.linspace(0.05, 7.5, 40)
    ym = np.array([float(m.y_from_a(m.mpf(float(x)))) for x in a])
    assert np.max(np.abs(lb.y_from_a(a) - ym)) < 1e-12


def test_vendor_newton_fails_beyond_alpha_7_9():
    """Documented defect in the vendored inversion (20-iteration cap)."""
    a = np.array([8.0])
    assert abs(lb.a_from_y(lb.y_from_a(a))[0] - 8.0) > 1.0
    assert abs(lb.a_from_y(lb.y_from_a(a, n_iter=200))[0] - 8.0) < 1e-9


# --- kf table -----------------------------------------------------------------

def test_kf_table_matches_vendor_function():
    m = va.module()
    rng = np.random.default_rng(3)
    V = np.exp(rng.uniform(math.log(600), math.log(5.5e5), 12))
    y = rng.uniform(0.02, 3.8, 12)
    exact = np.array([m.relativistic_factor(v, math.exp(t)) for v, t in zip(V, y)])
    assert np.max(np.abs(kf.kf(V, y) / exact - 1)) < 1e-6


def test_kf_refuses_to_extrapolate():
    with pytest.raises(ValueError):
        kf.kf(1e6, 1.0)


# --- emulator vs exact --------------------------------------------------------

def test_gate_g1_emulator_matches_exact_solver():
    res = gate_g1.run(n=100, seed=7)
    assert res["n_feasible_both"] >= 80
    assert res["n_feasibility_disagree"] == 0
    assert res["worst_theta_mean_rel"] <= 0.05e-2
    assert res["worst_length_mean_rel"] <= 0.1e-2
    assert res["passed"]


INFEASIBLE_AND_EDGE = [
    (10e3, 0.1, 1.0, 50),     # rc < rw
    (10e3, 5, 0.05, 1),       # no root
    (5e3, 2, 0.3, 0.5),       # no root
    (50e3, 30, 0.2, 20),      # feasible, theta > 56.44 so za invalid
    (4.7e3, 0.5, 0.06, 5),    # feasible, high convergence
]


@pytest.mark.parametrize("case", INFEASIBLE_AND_EDGE)
def test_feasibility_and_za_flags_agree(case):
    ex = va.synthesize(*case)
    em = closure.synthesize(*case)
    assert bool(em["feasible"][0]) == ex["feasible"]
    if ex["feasible"]:
        assert bool(em["za_valid"][0]) == ex["za_valid"]
        assert em["theta"][0] == pytest.approx(ex["theta"], rel=1e-7)


def test_ceeri_klystron_through_both():
    """Kant & Nagaraju 2021 (CEERI 8 MW): 150 kV, 120 A, rw 9.8 mm, 5 A/cm^2.
    Published theta 28.5652 deg. His final solver (with kf = 0.979) gives
    28.797 deg: a 0.81 % shift that his pre-kf 0.51 % claim predates."""
    ex = va.synthesize(150e3, 120, 9.8, 5)
    em = closure.synthesize(150e3, 120, 9.8, 5)
    assert ex["theta"] == pytest.approx(28.797, abs=1e-3)
    assert ex["rel_factor"] == pytest.approx(0.9788, abs=1e-4)
    assert em["theta"][0] == pytest.approx(ex["theta"], rel=1e-6)   # kf table ~1e-7
    assert abs(ex["theta"] - 28.5652) / 28.5652 == pytest.approx(0.0081, abs=5e-4)


def test_theta_raw_and_theta_T_are_distinct():
    r = closure.synthesize(14e3, 0.5, 0.2255, 5)
    th, tT = r["theta"][0], r["thetaT"][0]
    assert tT > th
    assert math.sin(math.radians(th)) == pytest.approx(0.905 * math.sin(math.radians(tT)))
    # geometry is from theta_raw
    assert r["Rc_mm"][0] == pytest.approx(r["rc_mm"][0] / math.sin(math.radians(th)))


def test_from_pcr_roundtrip():
    r = closure.from_pcr(2.0, 1.5, 40.0, V=20e3)
    assert r["muP"][0] == pytest.approx(2.0)
    assert r["C"][0] == pytest.approx(40.0)


def test_adapter_inputs_reproduce_pcr():
    d = va.to_vaughan_inputs(1.2, 0.8, 30.0)
    assert 1e6 * d["I"] / d["V"] ** 1.5 == pytest.approx(1.2)
    rc_cm = math.sqrt(d["I"] / (math.pi * d["Jc"]))
    assert (rc_cm * 10 / 0.8) ** 2 == pytest.approx(30.0)
