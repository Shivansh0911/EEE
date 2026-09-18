"""
Build data/processed/dataset_master.csv from the verified Tier A file.

This script NEVER re-extracts anything from the PDF and never touches
dataset_literature.csv. It reads that file as-is and adds derived target
columns, so the verified extraction (D1) stays the single source of truth and
the git diff for this phase is exactly "one new column".

When Tier B is eventually unblocked (D4), its rows are appended here, not in
build_tier_a.py. Today Tier B contributes zero rows.

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
OUT_NAME = "dataset_master.csv"

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

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)

    n_test = sum(1 for r in out_rows if r["paper_split"] == "test")
    print(f"Master dataset written: {len(out_rows)} rows "
          f"(Tier A {len(out_rows)}, Tier B 0, Tier C 0)")
    print(f"  paper split: {len(out_rows) - n_test} train / {n_test} test")
    print(f"  derived targets added: {', '.join(DERIVED_COLUMNS)}")
    print(f"  {dst}")


if __name__ == "__main__":
    main()
