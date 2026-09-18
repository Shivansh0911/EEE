"""
D7 / standing rule 11: nothing with beam_type != "pencil" reaches a training split.

The danger this guards against is specifically that wrong rows look right. A
sheet beam or an annular (magnetron injection gun) beam has a perveance and
something you can call a convergence ratio, so such a row sits in the CSV looking
entirely plausible while obeying different geometry. It would teach the network a
relationship that does not hold, and the error stays invisible until a prediction
is unexplainably off.

The filter is the only thing that catches it, so the filter is tested.
"""

import numpy as np
import pandas as pd
import pytest

from src.ann import dataset as ds
from src.ann.config import TrainConfig


@pytest.fixture(scope="module")
def df():
    return ds.load(cfg=TrainConfig())


def test_every_loaded_row_is_a_pencil_beam(df):
    assert set(df["beam_type"]) == {"pencil"}


def test_a_sheet_beam_row_is_filtered_out(tmp_path):
    """Plant a plausible-looking non-pencil row and check it does not survive."""
    real = pd.read_csv(ds.default_path())
    intruder = real.iloc[0].copy()
    intruder["row_id"] = "SHEET-9001"
    intruder["beam_type"] = "sheet"
    planted = tmp_path / "planted.csv"
    pd.concat([real, intruder.to_frame().T], ignore_index=True).to_csv(planted, index=False)

    loaded = ds.load(path=str(planted), cfg=TrainConfig())
    assert "SHEET-9001" not in set(loaded["row_id"])
    assert len(loaded) == len(real)


def test_paper_split_is_the_paper_s_own(df):
    train, test = ds.paper_split(df)
    assert len(train) == 23 and len(test) == 7
    assert sorted(test["source_case_id"]) == [3, 6, 12, 15, 17, 26, 29]


def test_no_gun_appears_on_both_sides_of_the_split(df):
    train, test = ds.paper_split(df)
    ds.assert_no_test_leakage(train, test)


def test_tier_b_is_confined_to_its_validated_envelope(df):
    """
    Tier B exists as of D9, so "there must be zero synthetic rows" is no longer
    the invariant. These are.

    The strict gate is STILL failed over the full envelope; what changed is that
    the generator was shown accurate over C >= 8 and is restricted to it. If a
    synthetic row ever appears below that floor, the justification for Tier B
    does not cover it.
    """
    tiers = set(df["tier"])
    assert tiers <= {"A_literature", "B_synthetic"}, f"unexpected tier in {tiers}"

    synth = df[df["tier"] == "B_synthetic"]
    if synth.empty:
        pytest.skip("Tier B not generated in this checkout")

    assert (synth["C"] >= 8.0).all(), "synthetic row below the validated C floor"
    assert (synth["theta_source"] == "physics_generator").all()
    assert (synth["generator_mre_pct"] == 2.80).all()
    assert (synth["split"] == "train").all()


def test_no_synthetic_row_can_reach_a_test_split(df):
    """
    The property the whole M1-vs-M3 comparison rests on. Asserted through the
    same call the training code uses, not by inspecting the file.
    """
    for include in (True, False):
        train, test = ds.train_test_frames(df, include_synthetic=include)
        assert (test["tier"] == "A_literature").all()
        assert len(test) == 7


def test_m1_still_sees_exactly_the_real_guns(df):
    """
    Adding 1000 synthetic rows must not have changed M1. Every Tier B row is
    beam_type=pencil, so the D7 filter does NOT separate them -- only the
    explicit tier selector does, which is why this is pinned.
    """
    train, test = ds.paper_split(df)
    assert len(train) == 23 and len(test) == 7
    assert set(train["tier"]) == {"A_literature"}
    assert set(test["tier"]) == {"A_literature"}


def test_second_target_matches_its_closed_form(df):
    """
    Rc_mm must be exactly sqrt(C)*rw/sin(theta) on the row's own values.

    Relative rather than absolute: the CSV stores four decimals, and the
    1/sin(theta) factor amplifies that rounding on the largest radii.
    """
    expected = np.sqrt(df["C"]) * df["rw_mm"] / np.sin(np.radians(df["theta_cone_deg"]))
    assert np.max(np.abs(df["Rc_mm"] - expected) / expected) < 1e-3


def test_derived_columns_are_flagged_as_derived(df):
    for flags in df["derived_fields"]:
        parts = set(str(flags).split(";"))
        assert {"rc_mm", "Rc_mm"} <= parts
