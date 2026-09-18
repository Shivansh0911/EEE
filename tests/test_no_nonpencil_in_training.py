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


def test_tier_b_is_still_empty(df):
    """
    The physics gate has not passed (D4), so there must be zero synthetic rows.
    If this ever fails, either the gate passed and this test needs updating, or
    synthetic data got in without one.
    """
    assert set(df["tier"]) == {"A_literature"}, (
        "non-literature rows present; Tier B is gated on the physics validation "
        "in reports/VALIDATION_table2.md, which currently FAILS"
    )


def test_second_target_matches_its_closed_form(df):
    """Rc_mm must be exactly sqrt(C)*rw/sin(theta), not something near it."""
    expected = np.sqrt(df["C"]) * df["rw_mm"] / np.sin(np.radians(df["theta_cone_deg"]))
    assert np.max(np.abs(df["Rc_mm"] - expected)) < 1e-4


def test_derived_columns_are_flagged_as_derived(df):
    for flags in df["derived_fields"]:
        parts = set(str(flags).split(";"))
        assert {"rc_mm", "Rc_mm"} <= parts
