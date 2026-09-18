"""
Loading the dataset into training matrices.

Two rules are enforced here rather than left to the caller, because both are
the kind of mistake that produces a better-looking number:

1.  PENCIL BEAMS ONLY (standing rule 11, D7). Nothing with
    beam_type != "pencil" reaches a training split. Sheet and annular guns have
    a perveance and something you can call a convergence ratio too, so a wrong
    row looks entirely right sitting in the CSV -- this filter is the only thing
    that catches it.
2.  Tier B and Tier C rows may be TRAINED on but never TESTED on. Testing a
    surrogate against its own generator's output proves nothing.

3.  A model that is supposed to be trained on real guns only must SAY SO, not
    rely on a filter elsewhere happening to exclude synthetic rows. Since D9 the
    master file holds 1030 rows, 1000 of them synthetic, and every one of them
    is `beam_type = pencil` -- so the pencil filter no longer separates them.
    `literature_only()` is the explicit selector, and M1 uses it.
"""

import os

import numpy as np
import pandas as pd

LITERATURE_TIER = "A_literature"


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def default_path():
    return os.path.join(repo_root(), "data", "processed", "dataset_master.csv")


def load(path=None, cfg=None):
    """Read the master CSV and apply the pencil-beam filter."""
    path = path or default_path()
    df = pd.read_csv(path)

    beam_type = cfg.beam_type if cfg is not None else "pencil"
    n_before = len(df)
    df = df[df["beam_type"] == beam_type].reset_index(drop=True)
    dropped = n_before - len(df)
    if dropped:
        print(f"  dropped {dropped} row(s) with beam_type != {beam_type!r} "
              f"(standing rule 11)")
    return df


def literature_only(df):
    """
    Tier A rows and nothing else.

    M1 is defined as "trained on real guns", and that has to be enforced by a
    named call rather than by the absence of synthetic rows in the file. Before
    D9 the two were the same thing; they are not any more.
    """
    return df[df["tier"] == LITERATURE_TIER].reset_index(drop=True)


def synthetic_only(df):
    return df[df["tier"] == "B_synthetic"].reset_index(drop=True)


def train_test_frames(df, include_synthetic):
    """
    The split every model is built from.

    `split` is train/test for Tier A and always train for Tier B, so the test
    frame is real guns by construction. That is asserted here rather than
    assumed, because it is the single property the M1-vs-M3 comparison rests on.
    """
    train = df[df["split"] == "train"]
    if not include_synthetic:
        train = train[train["tier"] == LITERATURE_TIER]
    test = df[df["split"] == "test"]

    assert (test["tier"] == LITERATURE_TIER).all(), (
        "a non-literature row reached the test split; a surrogate must never be "
        "scored against its own generator's output"
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def paper_split(df):
    """
    The paper's own 23/7 split: the seven cases printed with an asterisk in
    Table 2 (3, 6, 12, 15, 17, 26, 29) are test, the rest train.

    This is the split M1 must use. A fresh random 70/15/15 would not be
    comparable to the paper's reported numbers, and with 30 rows the 15 % pieces
    are four rows each, which is noise rather than a validation set.
    """
    lit = literature_only(df)
    train = lit[lit["paper_split"] == "train"].reset_index(drop=True)
    test = lit[lit["paper_split"] == "test"].reset_index(drop=True)
    assert len(train) == 23 and len(test) == 7, (
        f"expected the paper's 23/7 split, got {len(train)}/{len(test)}"
    )
    assert set(test["source_case_id"]) == {3, 6, 12, 15, 17, 26, 29}, (
        "test rows are not the paper's asterisked cases"
    )
    return train, test


def matrices(df, cfg):
    """(X, Y) in physical units, columns ordered as cfg.inputs / cfg.targets."""
    missing = [c for c in cfg.inputs + cfg.targets if c not in df.columns]
    if missing:
        raise KeyError(f"dataset is missing column(s): {missing}. "
                       f"Run `python src/data/build_master.py` first.")
    X = df[cfg.inputs].to_numpy(dtype=float)
    Y = df[cfg.targets].to_numpy(dtype=float)
    return X, Y


def assert_no_test_leakage(train_df, test_df):
    """
    No gun may appear on both sides of the split.

    Duplicate rows across splits are the classic way a test metric comes out
    suspiciously good, and they are invisible unless something looks.
    """
    key = ["perveance_uperv", "rw_mm", "C"]
    a = set(map(tuple, train_df[key].round(6).to_numpy()))
    b = set(map(tuple, test_df[key].round(6).to_numpy()))
    overlap = a & b
    assert not overlap, f"{len(overlap)} gun(s) appear in both train and test: {overlap}"


def kfold_indices(n, k, rng):
    """Shuffled k-fold indices. k = -1 means leave-one-out."""
    if k == -1:
        k = n
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]
