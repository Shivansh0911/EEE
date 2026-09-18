"""
Guards on the synthetic data.

Tier B is the part of this project most capable of quietly becoming a lie: a
thousand rows that look exactly like measurements, in the same columns, with the
same dtypes. Everything that keeps them honest is mechanical, and mechanical
things get tested.

The two that matter most:

* no synthetic row may carry the inputs of a held-out test gun, or the
  M1-versus-M3 comparison is measuring a model trained at the test coordinates;
* no synthetic row may be labelled as a test row, or the surrogate ends up being
  scored against its own generator's output.
"""

import math
import os

import numpy as np
import pytest

from src.ann import dataset as ds
from src.data import synth_generator as sg

MASTER = os.path.join(ds.repo_root(), "data", "processed", "dataset_master.csv")
TIER_B = os.path.join(ds.repo_root(), "data", "processed", "dataset_tier_b.csv")

pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def tb():
    if not os.path.exists(TIER_B):
        pytest.skip("Tier B not generated; run python -m src.data.synth_generator")
    return pd.read_csv(TIER_B)


# --------------------------------------------------------------------- honesty

def test_every_row_is_labelled_synthetic(tb):
    assert set(tb["tier"]) == {"B_synthetic"}
    assert set(tb["source_key"]) == {sg.SOURCE_KEY}
    assert set(tb["source_table"]) == {"generated"}
    assert set(tb["theta_source"]) == {"physics_generator"}


def test_every_row_carries_the_generators_accuracy(tb):
    """
    Mandatory by the brief, and the reason is good: anyone reading one row must
    be able to see how accurate the model that produced it was, without opening
    a report.
    """
    assert (tb["generator_mre_pct"] == sg.GENERATOR_MRE_PCT).all()
    assert (tb["generator_envelope"] == sg.GENERATOR_ENVELOPE).all()


def test_the_citation_says_it_is_not_experimental(tb):
    for text in set(tb["source_citation"]):
        assert "NOT experimental data" in text
        assert "2.80%" in text and "C >= 8" in text


def test_notes_flag_every_row_as_model_output(tb):
    assert tb["notes"].str.startswith("SYNTHETIC").all()


# -------------------------------------------------------------------- envelope

def test_rows_sit_inside_the_validated_envelope(tb):
    assert tb["C"].min() >= sg.C_RANGE[0]
    assert tb["C"].max() <= sg.C_RANGE[1]
    assert tb["perveance_uperv"].between(*sg.P_RANGE).all()
    assert tb["rw_mm"].between(*sg.RW_RANGE).all()


def test_no_row_is_below_the_C_floor(tb):
    """
    The floor is the entire justification. Below C = 8 the generator
    under-predicts theta by a mean of -4.29 degrees, which is the error the
    restriction exists to exclude.
    """
    assert (tb["C"] >= 8.0).all()


def test_theta_is_inside_the_accepted_band(tb):
    assert tb["theta_cone_deg"].between(*sg.THETA_RANGE).all()


def test_C_is_sampled_log_uniformly(tb):
    """
    A linear sample over 8..320 would put about three quarters of the rows above
    C = 80. Under a log sample the median sits near the geometric mean.
    """
    geo = math.sqrt(sg.C_RANGE[0] * sg.C_RANGE[1])
    assert abs(float(tb["C"].median()) - geo) / geo < 0.25


def test_samples_are_not_on_a_grid(tb):
    """
    A uniform grid would repeat coordinate values; a low-discrepancy sequence
    gives almost every row a distinct value on every axis. If this fails, the
    network can learn the lattice instead of the physics.
    """
    for col in ("perveance_uperv", "rw_mm", "C"):
        assert tb[col].nunique() > 0.9 * len(tb)


# --------------------------------------------------------------------- leakage

def test_no_synthetic_row_carries_a_held_out_gun_s_inputs(tb):
    """
    All seven of the paper's test guns have C >= 8, so they all qualify as
    anchor candidates. Including them would put the exact test coordinates into
    M3's training set and make the headline M1-vs-M3 comparison meaningless.
    """
    banned = sg.test_triples()
    got = {
        (round(float(p), 6), round(float(r), 6), round(float(c), 6))
        for p, r, c in zip(tb["perveance_uperv"], tb["rw_mm"], tb["C"])
    }
    overlap = banned & got
    assert not overlap, f"{len(overlap)} synthetic row(s) sit on a test gun: {overlap}"


def test_no_synthetic_row_is_ever_a_test_row(tb):
    assert set(tb["split"]) == {"train"}


def test_anchors_come_only_from_the_training_split():
    anchor_cases = {case for _, _, _, case in sg.anchors()}
    master = pd.read_csv(MASTER)
    test_cases = set(master.loc[master["paper_split"] == "test", "source_case_id"])
    assert not (anchor_cases & test_cases)
    assert len(anchor_cases) == 16


# --------------------------------------------------------------------- physics

def test_theta_matches_the_validated_solver(tb):
    """
    Spot-check that the labels really are this solver's output.

    Tolerance is 5e-3 degrees rather than machine precision because the CSV
    stores inputs to four decimals, so re-solving from the written P, rw and C
    is not re-solving from the values the label was computed at. Measured
    discrepancy is ~3e-4 deg typical, 1.1e-3 worst -- entirely that rounding,
    and four orders of magnitude below the generator's own 0.85 deg accuracy.
    """
    rows = tb.sample(12, random_state=0)
    for _, r in rows.iterrows():
        again = sg.theta_for(r["perveance_uperv"], r["rw_mm"], r["C"])
        assert abs(again - r["theta_cone_deg"]) < 5e-3


def test_derived_geometry_is_self_consistent(tb):
    """
    r_c and R_c must be the exact algebra of eqs (2) and (4) on the row's own
    values. Checked relatively: R_c carries a 1/sin(theta) factor that amplifies
    the stored four-decimal rounding into an absolute difference of up to
    2.3e-3 mm on the largest radii, which is 1e-4 relative.
    """
    rc = np.sqrt(tb["C"]) * tb["rw_mm"]
    Rc = rc / np.sin(np.radians(tb["theta_cone_deg"]))
    assert np.max(np.abs(rc - tb["rc_mm"]) / tb["rc_mm"]) < 1e-3
    assert np.max(np.abs(Rc - tb["Rc_mm"]) / tb["Rc_mm"]) < 1e-3


def test_generator_is_deterministic():
    """Same seed, same rows -- otherwise the dataset is not reproducible."""
    a, _ = sg.generate(24, seed=7, verbose=False)
    b, _ = sg.generate(24, seed=7, verbose=False)
    assert [x[:4] for x in a] == [x[:4] for x in b]
