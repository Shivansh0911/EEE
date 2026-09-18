"""
Build data/processed/dataset_master.csv from the verified Tier A file.

This script NEVER re-extracts anything from the PDF and never touches
dataset_literature.csv. It reads that file as-is and adds derived target
columns, so the verified extraction (D1) stays the single source of truth and
the git diff for this phase is exactly "one new column".

Tier B rows are appended here, not in build_tier_a.py, so the verified Tier A
extraction is never rewritten. Tier B exists as of D9 and is restricted to the
C >= 8 envelope where the generator was validated; if data/processed/
dataset_tier_b.csv is absent, the master file is simply Tier A and everything
downstream still works.

Derived column added
--------------------
Rc_mm : spherical cathode radius of curvature, in mm.

    Eq. (2):  r_c = sqrt(C) * r_w        (C is the ratio of circular areas)
    Eq. (4):  R_c = r_c / sin(theta)     (r_c is the chord radius of the cap)
    =>        R_c = sqrt(C) * r_w / sin(theta_cone_deg)

    Panahi R. et al., Results in Physics 73 (2025) 108256, eqs (2) and (4).

NOTE ON NAMING. `rc_mm` (lower case r) already exists and is the cathode disc
radius r_c = sqrt(C)*r_w. `Rc_mm` (upper case R) is the radius of curvature of
the spherical cap, which is larger by a factor 1/sin(theta). They are different
quantities and the case distinction is the standard one in the gun-synthesis
literature. Do not conflate them.

This is a PLACEHOLDER second target under reading C of D3. See reports/DECISIONS.md.
"""

import csv
import math
import os

SRC_NAME = "dataset_literature.csv"
TIER_B_NAME = "dataset_tier_b.csv"
OUT_NAME = "dataset_master.csv"

# Columns Tier B introduces. Tier A rows carry explicit values for them rather
# than blanks, so "no generator was involved" is stated rather than inferred
# from an empty cell.
TIER_B_COLUMNS = {
    "split": "",
    "theta_source": "experiment",
    "generator_mre_pct": "",
    "generator_envelope": "",
}

# Derived targets this script appends, in output order.
DERIVED_COLUMNS = ["Rc_mm"]


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def radius_of_curvature_mm(C, rw_mm, theta_deg):
    """R_c = sqrt(C) * r_w / sin(theta).  Panahi et al. eqs (2) and (4)."""
    return math.sqrt(C) * rw_mm / math.sin(math.radians(theta_deg))


def build(rows):
    out = []
    for r in rows:
        r = dict(r)
        Rc = radius_of_curvature_mm(
            float(r["C"]), float(r["rw_mm"]), float(r["theta_cone_deg"])
        )
        r["Rc_mm"] = f"{Rc:.4f}"

        # derived_fields is a provenance flag, not data: append without
        # duplicating if this script is run twice.
        flags = [f for f in r.get("derived_fields", "").split(";") if f]
        for name in DERIVED_COLUMNS:
            if name not in flags:
                flags.append(name)
        r["derived_fields"] = ";".join(flags)
        out.append(r)
    return out


def load_tier_b(root):
    """Tier B rows, if they have been generated. Absent is not an error."""
    path = os.path.join(root, "data", "processed", TIER_B_NAME)
    if not os.path.exists(path):
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def main():
    root = repo_root()
    src = os.path.join(root, "data", "processed", SRC_NAME)
    dst = os.path.join(root, "data", "processed", OUT_NAME)

    with open(src, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    assert len(rows) == 30, f"expected the 30 verified Tier A rows, got {len(rows)}"
    assert all(r["tier"] == "A_literature" for r in rows), "unexpected tier in Tier A file"
    assert all(r["beam_type"] == "pencil" for r in rows), "non-pencil row in Tier A file"

    out_rows = build(rows)

    # Keep Rc_mm next to rc_mm so the two are read together, not columns apart.
    out_fields = list(fields)
    anchor = out_fields.index("rc_mm") + 1
    for i, name in enumerate(DERIVED_COLUMNS):
        out_fields.insert(anchor + i, name)
    for name in TIER_B_COLUMNS:
        if name not in out_fields:
            out_fields.append(name)

    # Tier A: split mirrors the paper's own division.
    for r in out_rows:
        for name, default in TIER_B_COLUMNS.items():
            r.setdefault(name, default)
        r["split"] = r["paper_split"]

    tb_fields, tb_rows = load_tier_b(root)
    for r in tb_rows:
        assert r["tier"] == "B_synthetic", "non-synthetic row in the Tier B file"
        assert float(r["C"]) >= 8.0, "Tier B row below the validated C floor"
        assert r["split"] == "train", "a synthetic row is marked as a test row"
        # Any column the master has and Tier B does not is left blank rather
        # than invented.
        out_rows.append({k: r.get(k, "") for k in out_fields})

    for name in tb_fields:
        if name not in out_fields:
            raise AssertionError(f"Tier B column {name!r} has nowhere to go in the "
                                 f"master schema; update TIER_B_COLUMNS")

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    n_a = sum(1 for r in out_rows if r["tier"] == "A_literature")
    n_b = sum(1 for r in out_rows if r["tier"] == "B_synthetic")
    n_test = sum(1 for r in out_rows if r["split"] == "test")
    n_train = sum(1 for r in out_rows if r["split"] == "train")
    print(f"Master dataset written: {len(out_rows)} rows "
          f"(Tier A {n_a}, Tier B {n_b}, Tier C 0)")
    print(f"  split: {n_train} train / {n_test} test")
    print(f"  every test row is real: "
          f"{all(r['tier'] == 'A_literature' for r in out_rows if r['split'] == 'test')}")
    print(f"  derived targets added: {', '.join(DERIVED_COLUMNS)}")
    print(f"  {dst}")


if __name__ == "__main__":
    main()
