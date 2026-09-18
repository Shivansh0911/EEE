"""
The acceptance gate for the physics generator, as a test rather than a claim.

Tier B synthetic rows are only defensible if the model that produced them has
been shown to reproduce a published table it was not free to fit. That evidence
cannot live in prose in a report, because prose does not fail when the code
changes. It lives here.

Three things are asserted, and the second matters as much as the third:

1.  The Langmuir-Blodgett leg reproduces the classical series. This half of the
    engine was always sound.
2.  **The STRICT gate still fails.** Over all 30 published cases the model is at
    MAE 1.66 deg / MRE 5.75 %, against a required 0.5 deg / 1.5 %. That is
    asserted explicitly so that nobody reading a green test suite concludes the
    original gate was passed. It was not. D4 stands for the full envelope.
3.  Over the RESTRICTED envelope C >= 8 the model meets MAE <= 1.0 deg AND
    MRE <= 3.0 %. This, and only this, is what Tier B rests on (D9).

The aperture-lens coefficient c = 0.62 was chosen by sweeping against these same
published cases, so the restricted figure is in-sample. `test_leave_one_out_...`
below refits c without each case and predicts that case, which is the honest
out-of-sample estimate.
"""

import csv
import os

import numpy as np
import pytest

from src.ann import dataset as ds
from src.physics.probe_closure import alpha_and_deriv, solve_case

# The aperture-lens coefficient the sweep in VALIDATION_table2.md lands on. The
# thin-lens value is 1/3; the empirical optimum is almost exactly twice that.
C_LENS = 0.62

# The restricted envelope Tier B is generated over. Not a tuning knob: the
# residual analysis puts the model's failure entirely below this line.
C_MIN = 8.0

# The gate for the restricted envelope. Both must hold.
RESTRICTED_MAE_MAX = 1.0      # degrees
RESTRICTED_MRE_MAX = 3.0      # per cent

# The original strict gate, which is NOT met and is asserted as such.
STRICT_MAE_MAX = 0.5
STRICT_MRE_MAX = 1.5


@pytest.fixture(scope="module")
def table2():
    """The 30 published cases, with our model's prediction for each."""
    path = os.path.join(ds.repo_root(), "data", "processed", "dataset_literature.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30

    out = []
    for r in rows:
        P = float(r["perveance_uperv"])
        rw = float(r["rw_mm"])
        C = float(r["C"])
        out.append({
            "case": int(r["source_case_id"]),
            "P": P, "rw": rw, "C": C,
            "ref": float(r["theta_iterative_deg"]),
            "pred": solve_case(P, rw, C, C_LENS),
        })
    return out


def aggregate(rows):
    err = np.array([r["pred"] - r["ref"] for r in rows])
    ref = np.array([r["ref"] for r in rows])
    return {
        "n": len(rows),
        "mae": float(np.mean(np.abs(err))),
        "mre": float(100 * np.mean(np.abs(err) / np.abs(ref))),
        "mean_err": float(err.mean()),
        "max_abs_err": float(np.max(np.abs(err))),
    }


# --------------------------------------------------------------------------- #
# 1. The Langmuir-Blodgett leg
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gamma", [0.05, 0.10, 0.20])
def test_langmuir_blodgett_matches_the_classical_series(gamma):
    """
    alpha from our ODE integration against the published converging series.

    The series is only usable for small gamma -- it converges poorly by the
    gamma ~ 1.5-2.5 these guns actually need, which is why the engine integrates
    the ODE instead. Agreement at small gamma is what shows the ODE is the right
    ODE.
    """
    a, _ = alpha_and_deriv(gamma)
    series = (gamma + 0.3 * gamma ** 2 + 0.075 * gamma ** 3
              + 0.0143182 * gamma ** 4)
    assert abs(a - series) < 1e-5 * max(1.0, series)


def test_alpha_is_monotone_and_finite_over_the_working_range():
    """The generator inverts alpha by bisection, which needs monotonicity."""
    gs = np.linspace(0.01, 2.8, 200)
    vals = np.array([alpha_and_deriv(g)[0] for g in gs])
    assert np.all(np.isfinite(vals))
    assert np.all(np.diff(vals) > 0)


# --------------------------------------------------------------------------- #
# 2. The strict gate, which still FAILS
# --------------------------------------------------------------------------- #

def test_every_published_case_is_solvable(table2):
    unsolved = [r["case"] for r in table2 if not np.isfinite(r["pred"])]
    assert not unsolved, f"no root found for cases {unsolved}"


def test_strict_gate_over_all_30_cases_still_fails(table2):
    """
    Asserted deliberately, and in the failing direction.

    A green test suite must not leave the impression that the original gate was
    met. It was not: the model is roughly 3x outside it, and D4's conclusion
    holds for the full envelope. If this test ever fails because the model got
    BETTER, that is good news and the whole Tier B restriction should be
    revisited -- which is exactly why it is pinned here.
    """
    agg = aggregate(table2)
    assert agg["n"] == 30
    assert agg["mae"] == pytest.approx(1.66, abs=0.05)
    assert agg["mre"] == pytest.approx(5.75, abs=0.15)
    assert not (agg["mae"] <= STRICT_MAE_MAX or agg["mre"] <= STRICT_MRE_MAX), (
        "the strict gate now passes -- revisit D4, D9 and the C >= 8 restriction"
    )


def test_the_failure_is_confined_to_the_low_convergence_corner(table2):
    """
    The justification for restricting rather than abandoning: the model is
    essentially unbiased above C = 8 and badly biased below it. If that ever
    stops being true, the restriction is no longer principled.
    """
    low = aggregate([r for r in table2 if r["C"] < C_MIN])
    high = aggregate([r for r in table2 if r["C"] >= C_MIN])

    assert low["n"] == 7 and high["n"] == 23
    assert low["mean_err"] < -3.0, "low-C cases should be strongly under-predicted"
    assert abs(high["mean_err"]) < 0.6, "high-C cases should be near-unbiased"
    assert low["mae"] > 3 * high["mae"]


# --------------------------------------------------------------------------- #
# 3. The restricted gate, which Tier B rests on
# --------------------------------------------------------------------------- #

def test_restricted_envelope_meets_the_generator_gate(table2):
    """C >= 8 must satisfy MAE <= 1.0 deg AND MRE <= 3.0 %. Both, not either."""
    agg = aggregate([r for r in table2 if r["C"] >= C_MIN])
    assert agg["n"] == 23
    assert agg["mae"] <= RESTRICTED_MAE_MAX, f"MAE {agg['mae']:.3f} deg"
    assert agg["mre"] <= RESTRICTED_MRE_MAX, f"MRE {agg['mre']:.3f} %"


def test_restricted_envelope_beats_the_papers_own_test_error(table2):
    """
    The arithmetic that makes Tier B worth generating at all.

    D4 blocked synthetic data because a generator carrying 5.75 % error cannot
    train a surrogate to beat the paper's 3.74 % -- the network would learn the
    generator's bias. Over C >= 8 the generator's error is below that target, so
    the objection no longer applies. If this ever fails, Tier B should be
    withdrawn, not patched.
    """
    agg = aggregate([r for r in table2 if r["C"] >= C_MIN])
    assert agg["mre"] < 3.74


def test_tighter_envelope_is_tighter_still(table2):
    """C >= 12 was measured at MAE 0.73 / MRE 1.97. Recorded as a known point."""
    agg = aggregate([r for r in table2 if r["C"] >= 12.0])
    assert agg["n"] == 18
    assert agg["mae"] == pytest.approx(0.73, abs=0.05)
    assert agg["mre"] == pytest.approx(1.97, abs=0.15)


@pytest.mark.slow
def test_leave_one_out_refit_of_c_is_still_within_the_gate():
    """
    The honest out-of-sample number.

    c = 0.62 was chosen by sweeping against these same published cases, so the
    2.80 % above is in-sample and optimistic by an unknown amount. Here c is
    refit from scratch on 22 of the 23 restricted cases and used to predict the
    twenty-third, which removes that optimism.

    Measured at MAE 0.915 deg / MRE 2.894 % -- barely worse than in-sample, and
    still inside the gate. One free parameter against 23 points cannot overfit
    much, and this quantifies rather than assumes it.
    """
    path = os.path.join(ds.repo_root(), "data", "processed", "dataset_literature.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if float(r["C"]) >= C_MIN]

    P = np.array([float(r["perveance_uperv"]) for r in rows])
    rw = np.array([float(r["rw_mm"]) for r in rows])
    C = np.array([float(r["C"]) for r in rows])
    ref = np.array([float(r["theta_iterative_deg"]) for r in rows])

    grid = np.round(np.arange(0.58, 0.661, 0.005), 4)
    pred = np.array([[solve_case(P[i], rw[i], C[i], c) for i in range(len(rows))]
                     for c in grid])

    errs = []
    for i in range(len(rows)):
        others = [j for j in range(len(rows)) if j != i]
        maes = [np.mean(np.abs(pred[g, others] - ref[others]))
                if np.all(np.isfinite(pred[g, others])) else np.inf
                for g in range(len(grid))]
        g = int(np.argmin(maes))
        if np.isfinite(pred[g, i]):
            errs.append(pred[g, i] - ref[i])

    errs = np.array(errs)
    assert len(errs) == len(rows)
    mae = float(np.mean(np.abs(errs)))
    mre = float(100 * np.mean(np.abs(errs) / np.abs(ref[:len(errs)])))
    assert mae <= RESTRICTED_MAE_MAX, f"out-of-sample MAE {mae:.3f} deg"
    assert mre <= RESTRICTED_MRE_MAX, f"out-of-sample MRE {mre:.3f} %"
