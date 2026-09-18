# Dataset provenance ledger

Written for a skeptical reader. Every row in `data/processed/dataset_literature.csv`
can be traced from here back to a printed table in a published paper.

**Current totals — Tier A: 30 rows (all `beam_type = pencil`). Tier B: 0 rows
(blocked). Tier C: 0 rows.**

**Scope: pencil beams only** — solid, axially symmetric, circular cross-section.
Sheet-beam, annular/MIG, multi-beam and RF-photocathode guns are out of scope,
because the spherical Langmuir-Blodgett formulation and equations (1)-(7) assume
circular symmetry. Enforced by the `beam_type` column and a test. See D7 in
`reports/DECISIONS.md`.

No row in this dataset was invented, estimated, or produced by a language model
from memory. The one derived column (`rc_mm`) is flagged as derived and comes from
an exact algebraic identity, not an approximation.

---

## Tier A — real guns from the literature

### panahi2025 — 30 rows — PRIMARY SOURCE

> R. Panahi, S.A.H. Feghhi, Sh. Sanaye Hajari, M. Khorsandi,
> "Determination of the Half-Beam cone angle for Pierce electron gun design using
> an artificial neural network", *Results in Physics* **73** (2025) 108256.
> DOI: [10.1016/j.rinp.2025.108256](https://doi.org/10.1016/j.rinp.2025.108256)
> Open access, CC BY-NC.

- **Access route:** full PDF held locally at `data/raw/panahi2025.pdf`.
- **Table used:** Table 2, "Comparison between experimental and calculated results
  for various methods", printed page 4.
- **Rows contributed:** all 30.
- **Extraction method:** programmatic text extraction with `pdfplumber`, not manual
  typing and not from model memory. The extracted block is preserved verbatim
  inside `src/data/build_tier_a.py` (`TABLE2_RAW`) so the transcription can be
  audited against the PDF line by line.
- **Beam type:** all 30 are `pencil`. Frost [14], Tiwary [18] and Yang [19] are
  all solid convergent Pierce guns; none is a sheet-beam or annular design.
- **Columns taken:** case number, perveance (µperv), r_w (mm), C, θ experimental,
  θ predicted by ANN, θ iterative, θ non-iterative, θ modified non-iterative.
- **Split:** the `paper_split` column marks the seven cases printed with an
  asterisk in Table 2 (3, 6, 12, 15, 17, 26, 29) as test and the rest as train.
  This yields 23 train / 7 test, which matches the paper's own statement that
  "the number of data used for training was 23 … whereas those used for testing
  were 7". That agreement is an independent confirmation that the asterisks were
  read correctly.
- **Caveat on the iterative column:** the paper's footnote states the true
  modification `sin(θ)/sin(θ_T) = 0.905` has been applied. Recorded in the `notes`
  column of every row.

### Original upstream sources of the 30 cases

Panahi et al. state on page 3, verbatim:

> "cases 1, 2, 9, 12, 25, and 29 are extracted from references [14,18,19], whereas
> the remaining cases are obtained from references [19]."

This is recorded per row in the `origin_refs` and `origin_citations` columns, so
each row carries both the paper we read it from and the paper it originally came
from.

| Ref | Paper | Cases |
|---|---|---|
| [14] | Frost R., Purl O., Johnson H., "Electron guns for forming solid beams of high perveance and high convergence", *Proc. IRE* 50(8) (1962) 1800–1807 | 1, 2, 9, 12, 25, 29 |
| [18] | Tiwary U., Basu B., "Noniterative method for the synthesis of convergent Pierce electron guns", *IEEE TED* 34(5) (1987) 1218–1222 | 1, 2, 9, 12, 25, 29 |
| [19] | Yang C., Jia B., Zhu Z., "Improved noniterative method for the synthesis of convergent Pierce electron guns", *IEEE TED* 53(11) (2006) 2849–2852 | all 30 |

**These have not been independently retrieved** (see "Unretrieved" below), so the
30 rows are counted once, against Panahi et al., and are *not* double-counted as
separate contributions from Frost / Tiwary / Yang. Going to those three papers
directly is expected to add few if any genuinely new guns, since Panahi et al.
appear to have taken everything usable from them — but it would allow the
attribution to be verified at source, and may add columns (V, I, J_c) that
Panahi et al. did not reproduce.

---

## Unretrieved sources

Logged honestly; none contributed rows.

| Paper | Status | Why it matters |
|---|---|---|
| Vaughan J.R.M., *IEEE TED* 28(1) (1981) 37–41 | Paywalled (IEEE). Direct fetch returned HTTP 418. | Contains the iterative synthesis procedure. **Blocking item for Tier B.** |
| Tiwary & Basu, *IEEE TED* 34(5) (1987) | Paywalled. A secondary copy was located on academia.edu but automated extraction returned garbled equations and was rejected as untrustworthy. | Non-iterative closed form; also blocking. |
| Yang, Jia & Zhu, *IEEE TED* 53(11) (2006) | Paywalled. | Modified non-iterative method; source of 30/30 cases. |
| Frost, Purl & Johnson, *Proc. IRE* 50(8) (1962) | Not retrieved. | Original measurements for 6 cases. |
| Sharma, Sinha & Joshi, *IEEE TED* 48(2) (2001) 395–397 | Not yet attempted. | Anode aperture synthesis; may carry a design table. |
| Gharaati & Mardani, *IEEE TED* 68(1) (2020) 318–323 | Not yet attempted. | Tunable perveance gun; may carry measured cases. |
| Iqbal et al., *NIM-A* 1014 (2021) 165703 | Not yet attempted. | High-power gun beam optics. |
| Alabdullah, *Optik* 268 (2022) 169761 | Not yet attempted. | Thermionic gun, anode shape study. |

---

## Tier B — physics-generated — **NOT GENERATED**

Tier B is gated on the physics engine reproducing Table 2's `θ_Iterative` column
to within MAE ≤ 0.5° or MRE ≤ 1.5 %. **That gate currently fails at MAE 1.66° /
MRE 5.75 %.** Full evidence in [`reports/VALIDATION_table2.md`](../reports/VALIDATION_table2.md).

No synthetic rows have been written. When the gate passes, Tier B rows will be
tagged `tier=B_synthetic` with `source_key=vaughan1981_synthesis` and a citation
that states plainly that they were generated by this project's implementation and
validated against Panahi et al. Table 2.

**What Tier B is:** a verified physics model evaluated at new points in the
operating envelope — the standard basis of surrogate modelling.

**What Tier B is not:** experimental data, measurements, or anything extracted
from a paper. Tier B rows are never to be described as literature-derived, in the
CSV, in the app, in the report, or in conversation.

---

## Corrections found in project inputs

**`ANN testing - Sheet1.csv`, case 11 — transcription error.** θ is recorded as
`37.4`; Table 2 prints `37.04`. A digit transposition. The other 14 rows and all
other columns match Table 2 exactly. The corrected value is used throughout this
dataset; the original sheet is preserved unmodified at
`data/raw/ann_testing_original.csv`.

**Column name `Convergence angle` is a misnomer.** Its values (207.60, 306.30,
16.13 …) are the convergence *ratio* C from Table 2 — cathode area over waist
area, dimensionless. It is not an angle. Renamed to `C` in this dataset. This
matters directly for the "predict the convergence angle" task; see
`reports/DECISIONS.md`.
