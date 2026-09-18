# Decisions log

Running record of judgement calls, with reasoning. This becomes the methodology
section of the write-up.

---

## D1 — Table 2 was extracted programmatically, not typed

**Decision.** Table 2 was pulled from the PDF with `pdfplumber` and the extracted
text block is preserved verbatim inside `src/data/build_tier_a.py`.

**Why.** Transcription of a 30×9 numeric table by hand or from memory is exactly
where silent corruption enters, and it is undetectable later. Keeping the raw
extracted block in the source file means anyone can diff it against the PDF.

**Confirmation it worked.** The asterisk-marked test cases yield a 23/7 split,
which independently matches the paper's own prose ("23 … training, 7 … testing").

---

## D2 — Case 11 in the provided sheet is wrong; corrected

**Decision.** `ANN testing - Sheet1.csv` records θ = 37.4 for case 11. Table 2
prints 37.04. Corrected to 37.04 in the dataset; the original file is kept
unmodified as `data/raw/ann_testing_original.csv`.

**Why.** Verified directly against the PDF. All 14 other rows matched exactly, so
this is an isolated digit transposition rather than a different data source.

---

## D3 — `Convergence angle` renamed to `C`; second ANN output left OPEN

**Decision.** The input column is renamed `C` and documented as the convergence
*ratio*. The definition of the second network output is **not** resolved here.

**Why.** The values in that column (207.60, 306.30, 16.13, 5.33 …) are identical
to Table 2's `C` column, which the paper defines as cathode area over beam waist
area — dimensionless. It is not an angle.

This is not cosmetic, because the task is "extend the ANN to predict the half-beam
convergence angle." Three readings remain live:

| | Reading | Consequence |
|---|---|---|
| A | "convergence half angle" is a synonym for θ — the paper itself uses it that way when bounding the non-iterative method's validity | Nothing to add; the task is already done |
| B | The beam-envelope convergence half-angle at the anode aperture — the trajectory slope where the beam leaves the anode, distinct from the launch angle at the cathode | A genuine second target; derivable, and equation (II) of the physics model computes exactly this quantity |
| C | A second geometric output such as z_ac or R_a | Also genuine, also derivable |

**Note on B.** The beam-optics work done for the physics engine produces the anode
slope directly — `arcsin` of the left-hand side of equation (II). So reading B
becomes available for free the moment the physics gate passes, and is blocked by
the same thing.

**Action required from the project supervisor: confirm which reading is intended.**
The ANN is being written with `n_out` as a config value and the targets as a config
list, so whichever answer comes back changes one line, not the architecture.

### D3 (continued) — `Rc_mm` shipped as a PLACEHOLDER second output

**Decision.** P4 needed a concrete second target to build and test the multi-output
machinery against. The one implemented is

```
Rc_mm = sqrt(C) * r_w / sin(theta_cone_deg)        eqs (2) and (4)
```

the radius of curvature of the spherical cathode cap. It is added as a column by
`src/data/build_master.py` and flagged in `derived_fields`. **This is a
placeholder under reading C, not an answer to D3.** The supervisor still has to
confirm which reading is intended.

**Why this one.** The default is meant to be reading B, the beam-envelope
convergence half-angle at the anode. Reading B is computed by equation (II) of
the physics model, and equation (II) is behind the failed gate (D4) — so
defaulting to B today would mean training against a target carrying 5.75 % of its
own error, which is the same mistake as generating Tier B. `Rc_mm` is instead
*exactly* derivable from quantities already in Table 2, with no physics engine
involved and no error introduced. It is the only candidate second output
available at zero epistemic cost while the gate is down.

**The honest limitation, stated plainly.** `Rc_mm` is a deterministic function of
the three inputs and θ:

```
Rc_mm = f(C, r_w, theta)
```

and θ is output one. So the second output carries **no independent information**.
It exercises the multi-output residual vector, the `(N·n_out) × 26` Jacobian, the
per-output residual weights and the two-column scaler — all of which are real and
tested — but it cannot teach the network anything the first output does not
already contain, and a good score on it is not independent evidence of anything.
It must not be presented as a second physical prediction in the write-up.

Measured cost of carrying it: the same 26 parameters now serve two outputs instead
of one, and θ accuracy is worse for it. See `reports/BENCHMARK.md`.

**Switching readings** is `DEFAULT_TARGETS` in `src/ann/config.py`, plus the
column existing in the dataset. Reading A means `targets=["theta_cone_deg"]` and
n_out drops to 1 (23 parameters). Reading B becomes available the moment the D4
gate passes.

**Note on naming.** `rc_mm` (lower case) already existed and is the cathode *disc*
radius `r_c = sqrt(C)·r_w`. `Rc_mm` (upper case) is the *radius of curvature*,
larger by `1/sin θ`. Different quantities; the case distinction is the standard
one in the gun-synthesis literature, but they are easy to confuse and should not
be read past quickly.

---

## D4 — The physics gate was enforced, and it failed

**Decision.** Tier B synthetic generation has **not** been run. No synthetic rows
exist.

**Why.** The spec requires the physics engine to reproduce Table 2's `θ_Iterative`
column to MAE ≤ 0.5° or MRE ≤ 1.5 %. Best achieved is MAE 1.66° / MRE 5.75 %.
Full evidence in `VALIDATION_table2.md`.

**The decisive argument is arithmetic, not principle.** Tier B exists to train a
surrogate that beats the paper's 3.74 % test MRE. A generator carrying 5.75 % of
its own error cannot produce data that gets a model there — the network would
learn the generator's bias faithfully. A thousand such rows would make the model
worse on real guns while making the dataset look more impressive. That is the
worst possible trade.

**What was salvaged.** The Langmuir–Blodgett half of the engine is verified to
~10⁻⁹ against the classical series and is reusable as-is. The failure is isolated
to the aperture-lens term, and the residuals point at it cleanly (R = +0.79 against
ln C; mean error −4.29° for C < 8 versus −0.35° for C ≥ 8).

---

## D5 — No correction terms were fitted to force a pass

**Decision.** The aperture-lens coefficient was swept, not fitted, and no
additional free parameters were introduced.

**Why.** Tuning parameters against the same 30 points that constitute the
benchmark turns a validation into a curve fit. The argument that makes Tier B
legitimate — "a verified physics model evaluated at new points" — only holds if
the verification was genuinely independent of the fitting. Sweeping one physically
meaningful coefficient and *reporting that it lands near 2/3 rather than the
thin-lens 1/3* is a diagnostic finding. Adding three more coefficients until the
residuals vanish would not be.

---

## D6 — The 30 rows are attributed once, not triple-counted

**Decision.** All 30 rows are counted as Tier A contributions from Panahi et al.,
with the upstream references recorded per row in `origin_refs`, rather than being
split into separate Frost / Tiwary / Yang contributions.

**Why.** We read them from Panahi et al.; we have not independently retrieved the
three upstream papers. Counting them as if they came from four sources would
inflate the apparent breadth of the dataset without adding a single gun.

---

## D7 — Scope restricted to pencil beams, enforced by a column and a test

**Decision.** The dataset is restricted to **pencil beams** — solid, axially
symmetric, circular cross-section. A `beam_type` column was added and all 30 Tier A
rows backfilled as `pencil`. Training, validation and test splits filter on
`beam_type == "pencil"`, and `tests/test_no_nonpencil_in_training.py` enforces it.

**Why.** Every part of the physics assumes circular symmetry: the spherical-cap
cathode with a single R_c, C as a ratio of two circular areas, the spherical
Langmuir-Blodgett solution, and equations (1)-(7).

The danger is specifically that wrong rows look right. A sheet beam or an annular
(magnetron injection gun) beam has a perveance and something you can call a
convergence ratio, so such a row sits in the CSV looking entirely plausible while
obeying different geometry. It would teach the network a relationship that does not
hold, and the error stays invisible until a prediction is unexplainably off.

**Correction this forced.** An earlier draft of the literature-search prompt
included the query "magnetron injection gun / TWT electron gun design table
perveance". MIGs produce annular beams. Removed.

**Accepted cost.** This lowers the reachable literature yield toward the bottom of
the 60-150 estimate, since much modern gun work is sheet-beam or multi-beam for THz
and high-power devices. A smaller physically homogeneous dataset is worth more than
a larger heterogeneous one.

**`unknown` is not `pencil`.** Rows whose beam type cannot be determined are kept
in the CSV but excluded from training. Inability to determine is not permission to
assume.

---

## Open items

1. **D3** — supervisor to confirm the definition of the second output.
2. **D4** — obtain a clean PDF of Vaughan 1981, Tiwary & Basu 1987, or Yang 2006.
   This is the single blocking item for the entire synthetic-data path, and
   therefore for the NSGA-II optimizer that depends on the surrogate.
3. Decide whether to pursue the fallback of restricting Tier B to C ≥ 8, where the
   current model is already unbiased, at the cost of narrowing the design space.
