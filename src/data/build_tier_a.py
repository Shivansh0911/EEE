"""
Build Tier A (real literature) dataset from Panahi et al. 2025, Table 2.

Source of the numbers: verbatim text extraction of Table 2 from
data/raw/panahi2025.pdf (page 4 of the PDF, printed page 4). Nothing here is
typed from memory; the TABLE2_RAW block below is a copy of the extracted text.

Provenance, per the paper's own sentence on page 3:
    "cases 1, 2, 9, 12, 25, and 29 are extracted from references [14,18,19],
     whereas the remaining cases are obtained from references [19]."

Reference keys, from the paper's reference list:
    [14] Frost R, Purl O, Johnson H. "Electron guns for forming solid beams of
         high perveance and high convergence." Proc IRE 1962;50(8):1800-7.
    [18] Tiwary U, Basu B. "Noniterative method for the synthesis of convergent
         pierce electron guns." IEEE Trans Electron Devices 1987;34(5):1218-22.
    [19] Yang C, Jia B, Zhu Z. "Improved noniterative method for the synthesis of
         convergent pierce electron guns." IEEE Trans Electron Devices
         2006;53(11):2849-52.

Test cases are those marked with * in Table 2: 3, 6, 12, 15, 17, 26, 29.

All 30 cases are solid, axially symmetric PENCIL beams (beam_type="pencil").
Frost [14], Tiwary [18] and Yang [19] are all solid convergent Pierce guns.
This matters: the spherical Langmuir-Blodgett formulation, the spherical-cap
cathode and C-as-area-ratio all assume circular symmetry, so sheet-beam and
annular-beam guns must never be mixed into this dataset.
Footnote on the iterative column: "True modification has been applied
(sin(theta)/sin(theta_T) = 0.905)."
"""

import csv
import os

# ---------------------------------------------------------------------------
# Verbatim Table 2, as extracted from the PDF. Columns:
#   case  P(uperv)  rw(mm)  C  theta_Exp  theta_ANN  theta_Iter  theta_Nonit  theta_ModNonit
# ---------------------------------------------------------------------------
TABLE2_RAW = """
1   3.00 0.62 207.60 70.00 69.99 69.21 45.03 66.60
2   2.20 0.51 306.30 57.30 57.30 57.26 39.16 57.43
3   2.50 2.49  16.13 50.28 45.03 41.40 30.75 44.68
4   3.67 4.33   5.33 43.98 43.98 39.04 29.37 42.63
5   2.29 2.35  18.11 43.98 43.79 40.27 29.99 43.55
6   2.27 2.72  13.52 42.16 42.45 37.85 28.36 41.12
7   2.93 4.26   5.51 41.81 39.46 34.86 26.45 38.30
8   2.33 3.23   9.59 39.87 39.76 35.63 26.91 38.98
9   1.90 1.35  35.55 38.90 39.43 40.46 30.35 44.09
10  3.27 4.42   5.12 38.12 40.40 36.31 27.34 39.62
11  1.32 1.41  50.30 37.04 36.10 34.60 26.41 38.24
12  1.20 0.76 111.00 35.70 36.09 35.99 27.13 39.30
13  1.52 2.81  12.66 35.08 33.52 30.14 22.85 33.00
14  1.34 1.95  26.30 33.96 35.03 31.92 24.27 35.08
15  1.01 2.00  25.00 30.00 29.74 27.25 20.86 30.09
16  1.32 2.93  11.65 30.00 31.30 27.49 20.95 30.22
17  0.74 1.23  66.10 29.04 27.83 26.21 20.32 29.29
18  1.29 4.05   6.10 26.01 26.01 23.28 17.97 25.88
19  1.07 3.71   7.27 26.39 26.39 22.12 17.09 24.60
20  0.79 2.03  24.27 26.01 25.86 23.86 18.34 26.42
21  0.77 1.83  29.86 24.96 26.21 24.26 18.70 26.94
22  0.53 1.82  30.19 22.99 21.76 20.03 15.52 22.32
23  0.32 1.46  46.91 18.45 18.73 16.36 12.80 18.40
24  0.30 2.01  24.75 17.42 17.55 14.57 11.31 16.24
25  0.55 0.94  11.41 17.40 16.66 17.42 13.43 19.30
26  0.29 1.46  46.91 17.00 18.12 15.56 12.19 17.51
27  0.79 4.13   5.86 16.41 16.41 17.95 13.89 19.97
28  0.33 4.23   5.59 13.02 13.02 11.40  8.84 12.69
29  0.40 0.84  10.61 12.70 12.40 14.59 11.29 16.21
30  0.38 0.77   9.31  9.80 10.26 13.84 10.72 15.40
"""

TEST_CASES = {3, 6, 12, 15, 17, 26, 29}
MULTI_REF_CASES = {1, 2, 9, 12, 25, 29}

PANAHI_CITATION = (
    "Panahi R., Feghhi S.A.H., Sanaye Hajari Sh., Khorsandi M., "
    "'Determination of the Half-Beam cone angle for Pierce electron gun design "
    "using an artificial neural network', Results in Physics 73 (2025) 108256"
)
PANAHI_DOI = "10.1016/j.rinp.2025.108256"

REF_14 = ("Frost R., Purl O., Johnson H., 'Electron guns for forming solid beams "
          "of high perveance and high convergence', Proc. IRE 50(8) (1962) 1800-1807")
REF_18 = ("Tiwary U., Basu B., 'Noniterative method for the synthesis of convergent "
          "Pierce electron guns', IEEE Trans. Electron Devices 34(5) (1987) 1218-1222")
REF_19 = ("Yang C., Jia B., Zhu Z., 'Improved noniterative method for the synthesis of "
          "convergent Pierce electron guns', IEEE Trans. Electron Devices 53(11) (2006) 2849-2852")

FIELDS = [
    "row_id", "tier", "source_key", "source_citation", "source_doi",
    "source_table", "source_case_id", "origin_refs", "origin_citations",
    "paper_split", "perveance_uperv", "rw_mm", "C",
    "theta_cone_deg", "theta_ann_panahi_deg", "theta_iterative_deg",
    "theta_noniterative_deg", "theta_modified_noniter_deg",
    "beam_type", "rc_mm", "derived_fields", "orig_units", "notes",
]


def parse_table():
    rows = []
    for line in TABLE2_RAW.strip().splitlines():
        parts = line.split()
        assert len(parts) == 9, f"bad row: {line}"
        case = int(parts[0])
        P, rw, C, exp, ann, it, nonit, modnonit = (float(x) for x in parts[1:])
        rows.append(dict(case=case, P=P, rw=rw, C=C, exp=exp,
                         ann=ann, it=it, nonit=nonit, modnonit=modnonit))
    assert len(rows) == 30, f"expected 30 rows, got {len(rows)}"
    return rows


def build():
    rows = parse_table()
    out = []
    for r in rows:
        case = r["case"]
        multi = case in MULTI_REF_CASES
        origin_refs = "[14,18,19]" if multi else "[19]"
        origin_cites = " | ".join([REF_14, REF_18, REF_19]) if multi else REF_19
        # Eq. (2): r_c = sqrt(C) * r_w  -- exact algebraic derivation, not a guess
        rc = (r["C"] ** 0.5) * r["rw"]
        out.append({
            "row_id": f"LIT-{case:04d}",
            "tier": "A_literature",
            "source_key": "panahi2025",
            "source_citation": PANAHI_CITATION,
            "source_doi": PANAHI_DOI,
            "source_table": "Table 2",
            "source_case_id": case,
            "origin_refs": origin_refs,
            "origin_citations": origin_cites,
            "paper_split": "test" if case in TEST_CASES else "train",
            "perveance_uperv": f"{r['P']:.2f}",
            "rw_mm": f"{r['rw']:.2f}",
            "C": f"{r['C']:.2f}",
            "theta_cone_deg": f"{r['exp']:.2f}",
            "theta_ann_panahi_deg": f"{r['ann']:.2f}",
            "theta_iterative_deg": f"{r['it']:.2f}",
            "theta_noniterative_deg": f"{r['nonit']:.2f}",
            "theta_modified_noniter_deg": f"{r['modnonit']:.2f}",
            "beam_type": "pencil",
            "rc_mm": f"{rc:.4f}",
            "derived_fields": "rc_mm",
            "orig_units": "P in microperv; rw in mm; C dimensionless; angles in degrees",
            "notes": ("iterative column has true modification applied, "
                      "sin(theta)/sin(theta_T)=0.905"),
        })
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    rows = build()

    raw_path = os.path.join(root, "data", "raw", "literature", "panahi2025_table2.csv")
    proc_path = os.path.join(root, "data", "processed", "dataset_literature.csv")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)

    for path in (raw_path, proc_path):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    n_test = sum(1 for r in rows if r["paper_split"] == "test")
    print(f"Tier A rows written: {len(rows)}  (train {len(rows)-n_test} / test {n_test})")
    print(f"  {raw_path}")
    print(f"  {proc_path}")


if __name__ == "__main__":
    main()
