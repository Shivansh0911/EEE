"""
Gate G2: synthetic rows (tier B2) are never test data, and every one of them is
a design Vedant's solver accepts.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.ann import dataset as ds
from src.data import synth_v2
from src.physics import closure
from src.physics import vaughan_adapter as va

ROOT = ds.repo_root()
V2 = os.path.join(ROOT, synth_v2.OUT)
V1 = os.path.join(ROOT, "data", "processed", "dataset_tier_b.csv")


@pytest.fixture(scope="module")
def syn():
    return pd.read_csv(V2)


def test_v1_is_kept(syn):
    assert os.path.exists(V1)
    assert len(syn) >= 900


def test_no_synthetic_row_in_any_test_split(syn):
    assert set(syn.split) <= {"train", "val"}
    assert (syn.tier == "B2_synthetic").all()
    assert syn.notes.str.startswith("SYNTHETIC").all()
    # The real-gun test frame the evaluation uses is literature only.
    _, test = ds.paper_split(ds.literature_only(ds.load()))
    assert (test.tier == ds.LITERATURE_TIER).all()
    # And the master file's own test split has no synthetic row either.
    master = ds.load()
    assert (master[master.split == "test"].tier == ds.LITERATURE_TIER).all()


def test_no_synthetic_row_sits_on_a_test_gun(syn):
    banned = synth_v2.test_triples()
    keys = set(map(tuple, syn[["perveance_uperv", "rw_mm", "C"]].round(4).to_numpy()))
    assert not keys & banned


def test_envelope_flags_all_true(syn):
    for c in ("env_C_ge_8", "env_theta_raw_le_56", "env_P_rw_in_table2",
              "env_alpha_newton_ok", "za_valid"):
        assert syn[c].all(), c
    assert (syn.C >= 8).all() and (syn.theta_raw_deg <= 56).all()
    assert (syn.theta_cone_deg == syn.thetaT_deg).all()


def test_every_row_passes_feasibility_via_emulator(syn):
    """All rows through the emulator (G1-verified equal to his solver)."""
    r = closure.from_pcr(syn.perveance_uperv, syn.rw_mm, syn.C, V=va.V_REF)
    assert r["feasible"].all()
    assert np.max(np.abs(r["thetaT"] - syn.thetaT_deg) / syn.thetaT_deg) < 1e-6


def test_sampled_rows_pass_his_exact_feasibility_check(syn):
    """A random 24 rows re-run through his exact solver: feasible, same label."""
    s = syn.sample(24, random_state=0)
    ins = [va.to_vaughan_inputs(*t) for t in s[["perveance_uperv", "rw_mm", "C"]].to_numpy()]
    exact = va.synthesize_many([(d["V"], d["I"], d["rw_mm"], d["Jc"]) for d in ins])
    for ex, label in zip(exact, s.thetaT_deg):
        assert ex["feasible"]
        assert ex["thetaT"] == pytest.approx(label, rel=1e-8)
