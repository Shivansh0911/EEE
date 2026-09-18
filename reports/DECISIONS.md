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

> **PARTIALLY SUPERSEDED BY D9.** Tier B now exists, restricted to C ≥ 8 where
> the same model reaches MAE 0.85° / MRE 2.80 %. The strict gate is still failed
> and everything below still describes the full envelope correctly.

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

## D8 — C is log-scaled before normalisation

**Decision.** A per-input transform is applied **before** min-max scaling, and
the default is `input_transform = {"C": "log"}` — natural log. Configured in
`src/ann/config.py`, implemented in `src/ann/transforms.py`, carried in the
exported model JSON so the browser applies the identical transform.

**This reverses an earlier position.** An earlier draft of `BENCHMARK.md`
rejected log-scaling as test-set fitting. That reasoning was wrong on both of
its halves, and the correction is recorded rather than quietly applied.

**Why it is a priori domain knowledge, not selection on test data.**

1. **The paper never states a per-input transform.** It says the data were
   normalized. Linear scaling of C was *our* assumption about what that meant,
   not a claim the paper makes. So "the paper scales C linearly, and we should
   match it" was never true — there was nothing to match. Choosing between
   linear and log is a choice we were always making; the only question is which
   evidence decides it.

2. **ln C is the coordinate the physics is written in.** The synthesis enters
   through γ = ln(R_c/R_a), and equation (II) of our own physics model contains
   `0.5*ln(C)` explicitly. A network given C rather than ln C is being asked to
   learn the logarithm before it can start on the actual relationship, using
   part of a 23-parameter budget.

3. **The evidence predates the ANN entirely.** `VALIDATION_table2.md` reports
   `R(err, lnC) = +0.794` — the correlation of the *physics closure residual*
   with ln C. That was computed from the physics probe, before any network
   existed, and the seven test guns played no part in producing it. It is
   evidence about the coordinate system, not about a model's score.

4. **The linear scaling is measurably degenerate.** C spans 5.12 to 306.3, a
   factor of 59.8. Under linear min-max, **25 of the 30 rows fall in the bottom
   15 %** of the input range and the median row sits at normalised position
   **0.04** — nearly every gun is crushed against −1 and the network has almost
   no resolution to work with. Under ln C the median row sits at **0.295** and
   the rows spread across the interval. This is a property of the data, visible
   without fitting anything.

**The honest caveat, stated plainly.** The improvement from log-scaling was
**observed before the transform was adopted.** It first appeared as diagnostic
E4 in an earlier benchmark run, which measured it against the seven test guns
and reported it. So this is **not a blind pre-registration**: we had seen the
number before making the choice, and a reader is entitled to discount it for
that reason.

What can be said in its defence is that the four arguments above do not depend
on that number — points 1, 2 and 4 could have been made before any model was
trained, and point 3 was in the repository before the ANN existed. What cannot
be said is that we chose it without knowing the answer. Both halves belong in
the record.

**Measured effect** (median test MRE on θ over 30 restarts; full table in
`BENCHMARK.md`): the restart-to-restart spread narrows sharply as well as the
median falling, which is the more interesting half — a model that stops
depending on its initialisation is a better-posed model, not just a
better-scoring one.

---

## D9 — Tier B generated, over a restricted envelope. Partially reverses D4.

**Decision.** 1000 synthetic rows were generated and merged, restricted to
**C ≥ 8**. D4's blanket block on synthetic data is lifted *for that envelope
only*. Everything D4 says about the full envelope still stands.

**What changed.** Nothing about the physics. What changed is a measurement of
where the physics is accurate:

| Subset | n | MAE (deg) | MRE (%) | mean error |
|---|---|---|---|---|
| All 30 cases — the original gate | 30 | 1.66 | 5.75 | −1.27 |
| **C ≥ 8 — the generator's envelope** | **23** | **0.85** | **2.80** | **−0.35** |
| C ≥ 12 | 18 | 0.73 | 1.97 | −0.08 |
| C < 8 — **excluded** | 7 | 4.29 | 15.45 | −4.29 |

This was already visible in `VALIDATION_table2.md` as a residual analysis
(`R(err, lnC) = +0.794`), and was listed there as a fallback. It is now the plan.

**The argument that justifies it is the same arithmetic that blocked it.** D4's
objection was not "the gate says no" — it was that a generator carrying 5.75 %
of its own error cannot train a surrogate to beat the paper's 3.74 %, because
the network would learn the generator's bias faithfully. Over C ≥ 8 the
generator's error is **2.80 %, below 3.74 %**, so that objection does not apply
there. It applies everywhere else, which is why the envelope is restricted
rather than removed.

**The strict gate is still failed and is still asserted as failed.**
`tests/test_physics_reproduces_table2.py::test_strict_gate_over_all_30_cases_still_fails`
exists precisely so a green test suite cannot be mistaken for a passed gate.

**The in-sample objection, measured rather than waved away.** The coefficient
`c = 0.62` was chosen by sweeping against these same 30 published cases, so
2.80 % is in-sample. A leave-one-out refit — `c` re-fitted on 22 of the 23
restricted cases, then used to predict the twenty-third — gives **MAE 0.92° /
MRE 2.89 %**. The optimism is about 0.07°, and the out-of-sample figure is still
inside the gate and still below 3.74 %.

**The excluded corner is named, not hidden.** Seven real guns (cases 4, 7, 10,
18, 19, 27, 28) have C < 8 and are outside the envelope. No synthetic row exists
there. `reports/figures/tier_overlap.png` plots them as EXCLUDED so the coverage
gap is visible rather than implied. The cost is real: the low-convergence,
large-aperture corner of the design space has no synthetic support, and an
optimizer running over Tier B cannot explore it.

**Anchors exclude the test guns — a deliberate deviation.** The plan called for
the real C ≥ 8 triples as anchor points. All 23 qualify, but **all seven
held-out test guns have C ≥ 8**, so anchoring on all of them would put the exact
test coordinates into M3's training set and make the M1-versus-M3 comparison
meaningless. Anchors are the 16 triples that are C ≥ 8 *and* in the paper's
training split, and a test refuses any file containing a test triple.

**Coverage, measured.** 22 of the 23 real in-envelope guns lie inside the
synthetic convex hull. The exception is **case 26 (P = 0.29)**, just below the
envelope's P floor of 0.30 — and it is a held-out test gun, so M3 has no
synthetic support at the coordinates of one of the seven guns it is judged on.

**What Tier B is not.** Not experimental data, not extracted from a paper, and
never to be described as either — in the CSV, the app, the report, or
conversation. Every row carries `tier = B_synthetic`,
`theta_source = physics_generator`, `generator_mre_pct = 2.80`,
`generator_envelope = "C ≥ 8"`, and a citation ending "NOT experimental data".

**M1 is unchanged.** The headline model remains trained on real guns only, and
remains what the site reports. M3 is trained separately on Tier A + Tier B and
tested on held-out **real** guns only. If M3 does not beat M1 there, that is a
result about the generator's accuracy and is reported as one.

**What would reverse this.** A clean PDF of Vaughan 1981, Tiwary & Basu 1987 or
Yang 2006 would fix the aperture-lens term properly and open the full envelope,
making the restriction unnecessary. That remains the single most valuable thing
anyone could hand this project.

---

## Open items

1. **D3** — supervisor to confirm the definition of the second output. Note the
   measured cost of carrying a second output at all (D3 continued, BENCHMARK §7):
   if the answer is reading A, `--model m1` is already it.
2. **D4** — obtain a clean PDF of Vaughan 1981, Tiwary & Basu 1987, or Yang 2006.
   This is the single blocking item for the entire synthetic-data path, and
   therefore for the NSGA-II optimizer that depends on the surrogate.
3. ~~Decide whether to pursue the fallback of restricting Tier B to C ≥ 8~~ —
   **done, D9.** 1000 rows generated over C ≥ 8. The low-convergence corner
   (C < 8) remains without synthetic support, and case 26 sits just outside the
   envelope's P floor.
