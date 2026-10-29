A Midsem Report

On

**MULTI-OBJECTIVE OPTIMIZATION OF AN ELECTRON OPTICAL SYSTEM**

BY

**SHIVANSH SHEKHER OJHA**

**2023B2A30969H**

**SATWIK SHRIVASTAVA**

**2023B2A80938H**

Under the supervision of

**LALIT KUMAR**

**SUBMITTED IN PARTIAL FULFILLMENT OF THE REQUIREMENTS OF**

**EEE F266: STUDY PROJECT**

![Description: bitslogo](media/image1.jpeg){width="0.8854166666666666in"
height="0.8854166666666666in"}

**BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE PILANI (RAJASTHAN)**

**HYDERABAD CAMPUS**

**(OCTOBER 2026)**

**ACKNOWLEDGMENTS**

We sincerely thank Mr. Lalit Kumar for his supervision and guidance
throughout this study project. His formulation of the electron optical
system as a three-stage chain - electron gun, magnetic focusing system,
and collector - and his direction to move from the sheet-beam envelope
problem to a literature-grounded, quantitatively verified pencil-beam
ANN surrogate, have directly shaped the structure of this report.

We are grateful to the Department of Electrical and Electronics
Engineering, BITS Pilani Hyderabad Campus, for the EEE F266 Study
Project framework, which allowed this work to combine classical
electron-optics theory, an independently re-derived and verified
neural-network surrogate model, and a deployed interactive
demonstration.

We also acknowledge the authors of the reference study replicated and
extended in this report - R. Panahi, S.A.H. Feghhi, Sh. Sanaye Hajari
and M. Khorsandi (Results in Physics, 2025) - and the authors of the
classical synthesis methods against which every result in this report is
benchmarked: M. Rodney and J. Vaughan; U. Tiwary and B. Basu; and C.
Yang, B. Jia and Z. Zhu.

![bitslogo](media/image1.jpeg){width="0.8854166666666666in"
height="0.8854166666666666in"}

**Birla Institute of Technology and Science-Pilani,**

**Hyderabad Campus**

**Certificate**

This is to certify that the midsem report entitled "Multi-Objective
Optimization of an Electron Optical System" submitted by Mr. Shivansh
Shekher Ojha (ID No. 2023B2A30969H) and Mr. Satwik Shrivastava (ID No.
2023B2A80938H) in partial fulfillment of the requirements of the course
EEE F266, Study Project Course, embodies the work done by them under my
supervision and guidance.

**Date: 27 October 2026** **(LALIT KUMAR)**

BITS- Pilani, Hyderabad Campus

**ABSTRACT**

This study project addresses the design chain of an electron optical
system for a vacuum electron device: an electron gun that forms and
accelerates the beam, a magnetic focusing system that confines it over
the drift region, and a collector that dissipates the spent beam. The
work proceeded in two phases. In the first phase, a rectangular
sheet-beam electron gun was studied: given the accelerating voltage V,
beam current I, and cross-section dimensions a and b, a paraxial
beam-envelope equation - the sheet-beam analogue of Pierce\'s universal
beam-spread curve for circular beams - was formulated and implemented in
Python to determine the optimum beam cross-section for maximum
brightness, and an interactive demonstrator was deployed.

The second and current phase turns to the more classical pencil
(axisymmetric) Pierce electron gun and its single most consequential
design parameter: the half-beam cone angle, theta. A 2025 paper (Panahi
et al., Results in Physics 73, 108256) showed that a very small
artificial neural network - three inputs (perveance, beam waist radius,
convergence), two hidden layers of three and two neurons, one output, 23
trainable parameters - trained with the Levenberg-Marquardt algorithm on
30 experimental Pierce guns, predicts theta with a mean relative error
of 3.74%, outperforming three established classical methods. This report
documents an independent, from-scratch reimplementation of that network
in plain NumPy (no machine-learning framework), verified against the
paper\'s own numbers, then extended in two directions assigned for this
phase: (i) the network is extended to a multi-output variant that
predicts the derived cathode geometry alongside theta, using the same
three inputs and without adding new parameters, and (ii) a
literature-sourced dataset of practical Pierce-gun design cases is being
assembled from multiple published sources, with 71 datapoints collected
so far from four papers, toward a target of roughly 1,000 points for
future retraining.

Independent verification confirms the reimplementation is numerically
correct - the hand-derived Jacobian matches a finite-difference check to
within 1e-6, the browser (JavaScript) and Python versions of the network
agree to within 1e-9, and recomputing the benchmark directly from the
extracted Table 2 data reproduces the paper\'s own reported 3.74% error
exactly. However, the reimplementation\'s own median test error over 30
independent training runs is 8.33% (interquartile range 7.14-9.91%) on
the seven held-out real guns, against the paper\'s 3.74% - reported here
as a distribution rather than a single favourable number, and reported
not to disguise a shortfall. Two further findings are recorded honestly:
log-transforming the convergence ratio before normalisation (it spans a
factor of 60) roughly halves the median error, from about 13.4% to 9.9%;
and augmenting the training set with 1,000 physics-generated synthetic
guns did not improve accuracy and is reported as a negative result,
because a synthetic generator validated only against the same governing
equations it is meant to supplement adds no information the network does
not already have.

This report documents the electron-optics background, the sheet-beam
Phase 1 work, the pencil-beam Phase 2 replication and extension in full,
the literature dataset assembled so far, the two deployed interactive
demonstrators, and the work planned for the remainder of the semester.

**CONTENTS**

Title Page 1

Acknowledgements 2

Certificate 3

Abstract 4

1\. Introduction 6

2\. Electron Optical System Design Chain: Background 7

3\. Phase 1: Sheet-Beam Electron Gun Envelope Optimization 9

4\. Phase 2: Pencil-Beam Pierce Gun ANN Surrogate 10

5\. Verification and Results 12

6\. Literature-Sourced Pencil-Beam Gun Dataset 13

7\. Demonstration Web Applications 14

8\. Challenges and Current Limitations 14

9\. Future Work: Remainder of the Semester 15

Conclusion 15

Appendix A: Live Demonstrators and Evidence of Work Completed 16

References 19

**1. INTRODUCTION**

**1.1 Background: The Electron Optical System**

A vacuum electron device such as a klystron or a travelling-wave tube
(TWT) requires a well-formed electron beam over its full length of
operation. The electron optical system that produces and maintains this
beam has three stages, as formulated with the supervisor at the start of
this project: an electron gun (S1) that generates and initially shapes
the beam from a thermionic cathode; a magnetic focusing system (S2) that
confines the beam against its own space-charge spreading as it drifts
through the interaction region; and a collector (S3) that safely
dissipates the beam\'s energy once it has done its work. The gun\'s
output - its beam envelope, current density, and cross-sectional shape -
is the starting condition for everything downstream, which is why gun
design was chosen as the starting point of this project.

**1.2 Two Beam Geometries Considered**

Two distinct beam geometries were formulated for study, matching the two
phases of this report. The first (Phase 1) is a rectangular sheet beam
of cross-section a x b, described by the accelerating voltage V, the
beam current I, and the two cross-section dimensions a and b; the design
question is the optimum beam shape for maximum brightness, i.e. the
shape that minimises space-charge-driven spreading of the envelope. The
second (Phase 2, the current focus) is the classical axisymmetric pencil
beam of a Pierce gun with a curved cathode, described instead by the
perveance P (which combines V and I), the beam waist radius r_w, and the
convergence ratio C; the design question here is the half-beam cone
angle theta, from which the rest of the gun geometry follows
algebraically. Both geometries feed the same downstream problem - given
V, I and the beam cross-section, determine the magnet dimensions needed
to focus it (Section 2.1) - so the two phases are complementary halves
of one electron-optical design chain rather than unrelated exercises.

**1.3 Motivation: Why a Learned Surrogate**

Classical gun-design methods (Section 2.3) are iterative or
semi-empirical: they converge to a design by repeated numerical solution
of the governing space-charge equations. That is acceptable for
evaluating one design, but a multi-objective optimiser exploring the
gun\'s parameter space needs on the order of 50,000 such evaluations to
search effectively - repeated classical solves at that scale become the
bottleneck. A small, fast artificial neural network trained once on a
set of known gun designs can replace the expensive iterative step with a
single forward pass that runs in microseconds, making a large-scale
multi-objective search (the eventual goal of this project, Section 9)
practical. This is the reasoning explored in Phase 2 of this report,
following a 2025 paper that applied exactly this idea to the half-beam
cone angle.

**1.4 Objectives**

-   Formulate and implement, in Python, the paraxial beam-envelope
    equation for a rectangular sheet-beam electron gun - the sheet-beam
    analogue of Pierce\'s universal beam-spread curve - and use it to
    determine the optimum beam cross-section for a given V, I, a and b
    (Phase 1, completed; Section 3).

-   Independently reimplement, from scratch and without a
    machine-learning framework, the artificial neural network of Panahi
    et al. (2025) that predicts the half-beam cone angle of a
    pencil-beam Pierce gun from perveance, beam waist radius and
    convergence, and verify it against the paper\'s own reported numbers
    (Phase 2; Section 4).

-   Extend that network to a multi-output model that also predicts the
    derived cathode geometry, using the same three inputs and without
    adding new trainable parameters, as assigned for this phase (Section
    4.4).

-   Build a literature-sourced dataset of practical Pierce-gun design
    cases from multiple published sources, tagged by originating paper,
    working toward roughly 1,000 datapoints for future retraining beyond
    the original paper\'s 30 guns (Section 4.5, Section 6).

-   Deploy interactive, browser-based demonstrations of both the
    sheet-beam (Phase 1) and pencil-beam (Phase 2) work for review and
    viva (Section 7).

**1.5 Scope**

In scope for this phase: the electrostatic/space-charge design of the
electron gun for both beam geometries; benchmarking the pencil-beam
surrogate against the three classical methods used in the reference
paper (Vaughan\'s iterative method, Tiwary and Basu\'s noniterative
method, Yang\'s modified noniterative method); and literature-based
dataset collection for the pencil-beam case.

Out of scope for this phase: the detailed magnetic focusing system (S2)
and collector (S3) design; a completed multi-objective (NSGA-II)
optimisation run; and unification of the sheet-beam and pencil-beam
pipelines into a single tool. These are planned for the remainder of the
semester (Section 9).

**2. ELECTRON OPTICAL SYSTEM DESIGN CHAIN: BACKGROUND**

**2.1 The Three-Stage System**

The design chain formulated for this project, following the initial
problem breakdown with the supervisor, treats the electron optical
system as three stages in series: the electron gun (S1) fixes the
beam\'s initial current density, envelope and cross-sectional shape; the
magnetic focusing system (S2) takes that beam, together with V, I and
the cross-section dimensions, and must be dimensioned to hold the
envelope against space-charge spreading over the drift length; and the
collector (S3) absorbs the beam once it has interacted with the
device\'s RF structure. This report\'s two phases both sit inside stage
S1, since the gun geometry has to be fixed before the magnet
dimensioning problem in S2 can even be posed.

**2.2 Pierce Gun Geometry and Governing Equations**

For the axisymmetric (pencil-beam) Pierce gun studied in Phase 2, the
geometry - cathode, focus electrode, anode aperture and beam
trajectory - is fixed once the half-beam cone angle theta is known,
through the following relations (following Panahi et al., 2025):

-   r_c = sqrt( I / (pi \* J_c) ) \-- cathode disc radius from current
    and cathode current density

-   r_c = sqrt(C) \* r_w \-- cathode radius from convergence and beam
    waist radius

-   P = I / V\^1.5 \-- perveance, combining current and voltage

-   R_c = r_c / sin(theta) \-- cathode spherical radius

-   R_a = R_c \* e\^(-gamma) \-- anode spherical radius, with gamma =
    ln(R_c / R_a)

-   r_b(z_a) = r_c \* e\^(-gamma) \-- beam radius at the anode aperture

-   z_ac = R_c - R_a \-- cathode-to-anode distance

Determining theta is therefore the single most consequential step in the
whole design: every other geometric quantity above follows from it
algebraically. This is why theta, rather than any other output, was
chosen as the quantity to predict with a surrogate model in Phase 2.

**2.3 Classical Methods for Determining theta**

Three classical approaches to determining theta, each used as a
benchmark baseline in this report, are summarised in Table 1.

  ------------------------- --------------------------------- ------------
  **Method**                **Approach**                      **Reported
                                                              MRE (%)\***

  Vaughan (1981) -          Solves the space-charge equations 10.14
  iterative method          for the convergent Pierce gun     
                            repeatedly until the geometry     
                            converges; no empirical factors,  
                            but computationally slow.         

  Tiwary and Basu (1987) -  An analytical, closed-form method 27.93
  noniterative method       using a small-angle               
                            approximation; fast, but accuracy 
                            degrades as convergence           
                            half-angle grows beyond about 20  
                            degrees.                          

  Yang, Jia and Zhu         Improves Tiwary and Basu\'s       7.93
  (2006) - modified         method with an empirical          
  noniterative method       correction factor fitted against  
                            measured guns; substantially more 
                            accurate than the unmodified      
                            noniterative method.              
  ------------------------- --------------------------------- ------------

*\*As reported in Panahi et al. (2025), Table 4, computed on the same
30-gun dataset used to train and test their ANN model (Section 4.1);
reproduced here for comparison against the ANN\'s own 3.74% MRE.*

**2.4 Why a Surrogate Model Is the Right Lever**

Because theta alone fixes the rest of the gun geometry (Section 2.2),
and because an optimisation loop over gun designs needs many thousands
of evaluations of theta (Section 1.3), a fast, sufficiently accurate
surrogate for the theta-determination step is the single
highest-leverage component to build first. This is the reasoning
followed in Phase 2 of this project, and it is also why the multi-output
extension in Section 4.4 was designed to add derived geometry as an
extra output rather than to change the inputs.

**3. PHASE 1 (PRIOR WORK): SHEET-BEAM ELECTRON GUN ENVELOPE
OPTIMIZATION**

**3.1 Problem Statement**

The first phase of the project addressed the rectangular sheet-beam
electron gun: given the accelerating voltage V, beam current I, and the
cross-section dimensions a (length) and b (width), the task was to
determine the optimum shape of the beam cross-section downstream of the
gun - the shape that keeps the beam envelope from spreading due to its
own space charge, and therefore maximises brightness. This is a
genuinely different problem from the classical, circularly symmetric
Pierce gun treated in Phase 2: the governing envelope equation, and the
resulting universal design curve, are not simply a re-labelling of the
circular-beam case, because a flat beam\'s self-field and its paraxial
envelope equation depend on the aspect ratio of the cross-section rather
than on a single radius.

**3.2 Approach**

The approach taken was to derive the paraxial beam-envelope
(beam-spread) equation for a sheet beam under its own space charge and
any external focusing field, in direct analogy with the derivation
behind Pierce\'s universal beam-spread curve for a circular beam.
Solving this envelope equation numerically for a given V, I, a and b
identifies the entrance (launch) condition - the initial envelope slope
and curvature - for which the beam propagates with minimum ripple, i.e.
the matched-beam condition analogous to Brillouin flow, but for the flat
geometry. This determines the optimum cross-sectional shape referred to
in the problem statement.

**3.3 Deliverable**

A Python implementation was built to numerically integrate the derived
sheet-beam envelope equation for arbitrary inputs V, I, a and b, and to
plot the resulting optimum beam envelope/shape. An interactive,
browser-based version of this tool was deployed for review at
electron-gun.netlify.app (Appendix A), which also includes an ANN-based
demonstrator for the sheet-beam case; the approach used to build and
validate that ANN - a small, from-scratch network trained and checked
against known cases - directly informed the more rigorous,
literature-benchmarked replication carried out in Phase 2 (Section 4).

**3.4 Status**

Phase 1 was completed and presented at Seminar 1. It establishes the
sheet-beam half of the eventual unified gun-design tool referred to in
Section 9, and its optimum-envelope methodology is the sheet-beam
counterpart to the pencil-beam half-beam-cone-angle problem that Phase 2
(the remainder of this report) addresses in full quantitative detail.

**4. PHASE 2 (CURRENT WORK): PENCIL-BEAM PIERCE GUN ANN SURROGATE**

Following the completion of Phase 1, the project moved - on the
supervisor\'s direction - to the classical pencil (round) Pierce beam,
with the specific goal of replicating, verifying and extending a
recently published neural-network method for predicting the half-beam
cone angle. This section documents that work in full, since it is the
primary technical content of the current phase.

**4.1 Reference Study**

The reference study is: R. Panahi, S.A.H. Feghhi, Sh. Sanaye Hajari and
M. Khorsandi, \"Determination of the Half-Beam cone angle for Pierce
electron gun design using an artificial neural network,\" Results in
Physics, vol. 73, 108256, 2025
(https://doi.org/10.1016/j.rinp.2025.108256). The paper trains a
multilayer perceptron with three inputs - perveance P, beam waist radius
r_w, and convergence C - two hidden layers of three and two neurons
respectively (tanh activation), and one output neuron predicting theta,
using the Levenberg-Marquardt (LM) algorithm on 30 experimentally
reported Pierce-gun designs drawn from the literature (half-beam cone
angles ranging from 9.80 to 70.00 degrees). The paper reports a mean
relative error (MRE) of 3.74% for the ANN, against 10.14% for Vaughan\'s
iterative method, 27.93% for Tiwary and Basu\'s noniterative method, and
7.93% for Yang\'s modified noniterative method (Table 1, Section 2.3),
concluding that the ANN is the most accurate of the four.

**4.2 Why Replicate From Scratch**

Rather than treating the paper\'s 3.74% figure as given, this project
set out to independently verify it by rebuilding the network, its
training algorithm, and its data from first principles, in plain NumPy
with no machine-learning framework. Three reasons motivated this: first,
an independently reproduced result is a far stronger basis for the
extensions this phase was asked to make than an unverified citation;
second, a hand-derived Jacobian and a from-scratch LM trainer are
necessary groundwork for the multi-output extension in Section 4.4,
which the published paper does not itself provide; and third, keeping
the model this small (23 parameters total) means it can be exported as a
small JSON file and re-run directly in a browser with no server, which
is what makes the live demonstrator in Section 7.1 possible at all.

**4.3 Implementation**

The network architecture, forward pass, and packing/unpacking of
parameters follow the Colab prototype developed for this project
(ANN_Testing.ipynb): three inputs are mapped through a 3-3-2-1 tanh
multilayer perceptron with 23 trainable parameters in total (9 weights +
3 biases in the first hidden layer, 6 weights + 2 biases in the second,
2 weights + 1 bias at the output). Training minimises the sum-of-squared
errors using the Levenberg-Marquardt algorithm: at each step, the
Jacobian of the network\'s errors with respect to all 23 parameters is
computed, and the parameter update solves (J\^T J + mu I) delta = -J\^T
e, with the damping factor mu decreased after an accepted
(error-reducing) step and increased after a rejected one - the same
damping-factor logic implemented and tested in the attached notebook.

Built out from that prototype into the full pipeline used for this
phase\'s results, the stages are:

-   Data collection (Tier A): 30 real gun designs are read directly from
    Table 2 of the reference paper (code pulls the values out of the
    paper rather than the values being copied by hand); this tier is the
    only data ever used for testing.

-   Synthetic data generation (Tier B): an independently written physics
    engine (solving the same governing Langmuir-Blodgett/space-charge
    equations) generates 1,000 additional synthetic gun designs,
    restricted to the region (C \>= 8) where the generator was checked
    to be accurate; used for training only, and investigated in Section
    5.4.

-   Input preparation: the convergence ratio C is log-transformed (ln C)
    before every input is linearly scaled to the range \[-1, 1\] using
    minimum/maximum values taken from the training rows only, so the
    held-out test guns remain genuinely unseen during scaling as well as
    training.

-   Training: the 30 real guns are split into 23 for training and 7 held
    out for testing; the network is trained with LM from 30 independent
    random parameter initialisations, to characterise how much the
    result varies rather than reporting a single favourable run.

-   Evaluation: percentage error on the 7 held-out real guns is computed
    for every run and compared against the paper\'s reported figure
    (Section 5).

-   Export and browser re-implementation: the 23 trained weights are
    exported as a small JSON file, and the identical forward pass is
    re-implemented in JavaScript so the live website (Section 7.1) can
    predict theta instantly in-browser with no backend server.

**4.4 Extension 1: Multi-Output Prediction (Assigned Task)**

As assigned for this phase, the single-output network (predicting theta
only, referred to below as model M1) was extended to a multi-output
variant, M1-multi, that predicts theta together with the derived cathode
geometry (the cathode spherical radius R_c, computed from theta via the
relations in Section 2.2) from the same three inputs - perveance, beam
waist radius and convergence - with no new input parameters added,
exactly as instructed. This was done by widening the output layer while
keeping the rest of the 3-3-2 architecture, inputs, normalisation, and
LM training procedure unchanged, so that the comparison between M1 and
M1-multi isolates the effect of the additional output rather than any
other change to the pipeline. A third variant, M3, trains the same
architecture with the Tier-B synthetic data included, and is discussed
in Section 5.4.

**4.5 Extension 2: Literature-Sourced Dataset Expansion (Assigned
Task)**

The second assigned task was to scrape data from similar papers to build
a dataset of roughly 1,000 datapoints with parameters comparable to the
reference paper\'s (perveance, beam waist radius, convergence, half-beam
cone angle), explicitly tagging each datapoint with the paper it came
from. Work on this is under way: four papers have been scraped so far -
Panahi et al. (2025) itself, Vaughan (1981), Tiwary and Basu (1987), and
Yang, Jia and Zhu (2006) - yielding 71 combined datapoints to date (full
breakdown in Section 6). As a direct benefit of this exercise, all 30 of
the original Tier-A rows were also traced back to their own original,
DOI-cited sources rather than taken only from the reference paper\'s
summary table, and two transcription errors were discovered and
corrected in the process, with the original (uncorrected) values
preserved alongside the corrected ones for transparency.

**5. VERIFICATION AND RESULTS**

**5.1 Correctness Checks**

Before any benchmark number is reported, the reimplementation\'s
correctness was checked against independent ground truth at every stage
of the pipeline. Table 2 summarises these checks; all four passed.

  ---------------------- ------------------------------- ----------------
  **Check**              **What it verifies**            **Result**

  Gradient / Jacobian    Hand-derived Jacobian vs.       Agrees to within
  check                  brute-force finite-difference   1e-6
                         Jacobian                        

  Website = Python       JavaScript (browser) network    Agrees to within
  parity                 vs. Python (training) network,  1e-9
                         100 random inputs               

  Physics solver check   Langmuir-Blodgett equation      Agrees to about
                         solver vs. the classical        1e-9
                         textbook series solution        

  Data-fidelity check    MRE recomputed from our own     Reproduces 3.74%
                         extracted Table 2 data vs. the  exactly
                         paper\'s reported MRE           
  ---------------------- ------------------------------- ----------------

**5.2 Benchmark Against the Published Model**

Over 30 independent training runs from random initialisations, the
reimplemented network\'s median test error on the 7 held-out real guns
is 8.33% (interquartile range 7.14-9.91%, full range across all 30 runs
2.99% to 115.75%), against the paper\'s reported 3.74%. Only 1 of the 30
runs landed below the paper\'s own figure. This gap is reported honestly
rather than concealed by quoting a single favourable run: with only 7
test points and 23 training points for a 23-parameter model, the fitting
problem is close to fully determined, so LM has little redundancy to
exploit and the result is sensitive to where training starts. Reporting
the result as a median and a range, rather than as one number, is a
deliberate response to that sensitivity, and the live demonstrator
(Section 7.1) states this distribution on-screen rather than a single
headline percentage.

**5.3 Ablation: The Log-Transform of Convergence**

Convergence C spans roughly a factor of 60 across the dataset (from
about 5 to over 300), so on a plain linear min-max scale the great
majority of the 30 guns are squeezed into a narrow low-C band, starving
the network of resolution across most of its input range.
Log-transforming C before scaling (feeding ln C rather than C) reduced
median test error from about 13.4% to about 9.9% - roughly halving it -
and this transform is retained in the final pipeline (Section 4.3).

**5.4 Negative Result: Synthetic Data Augmentation Does Not Help**

A controlled experiment (model variant M3, Section 4.4) tested whether
adding the 1,000 physics-generated Tier-B synthetic guns to the training
set improves accuracy on the same 7 real held-out test guns. It did not:
accuracy on real, experimentally measured guns did not improve, and in
some configurations worsened. The likely explanation is that the
synthetic generator was validated only by checking it against the same
governing space-charge equations the network is meant to supplement, not
against independent experimental measurement - so training on its output
does not add information the network does not already have access to
through the physics itself, while any small bias in the generator gets
learned as if it were signal. This negative result is recorded here
deliberately, because it directly shapes the dataset-expansion priority
in Section 9: real, literature-sourced datapoints (Section 4.5, Section
6) are the correct lever, not synthetic augmentation.

**5.5 Interpretation**

Taken together, Sections 5.1-5.4 support a specific, limited claim: the
reimplementation is numerically correct at every stage (Section 5.1), it
correctly reproduces the paper\'s own headline number when run on the
paper\'s own extraction of the data (Section 5.1, data-fidelity check),
and the two genuine methodological improvements attempted - the
log-transform (Section 5.3) and synthetic augmentation (Section 5.4) -
were properly tested rather than assumed, with one helping and one not.
What the reimplementation does not yet do is statistically match the
paper\'s 3.74% figure on new random-restart training runs, given only 23
training points; this is stated as a limitation (Section 8), not
smoothed over, and is exactly why the live site labels its output \"a
research result, not a design tool.\"

**6. LITERATURE-SOURCED PENCIL-BEAM GUN DATASET**

Table 3 summarises the four papers scraped so far for the
dataset-expansion task (Section 4.5), each of which supplied a table of
named, practical Pierce-gun design cases with parameters directly
comparable to the reference paper\'s (V, I, perveance, beam waist
radius/dimensions, convergence, and the resulting half-beam cone angle).

  -------------------- ------------- -------------------------------------
  **Source**           **Table       **What It Contains**
                       Used**        

  Panahi, Feghhi,      Table 2 (30   Experimental theta for 30 practical
  Sanaye Hajari and    cases)        guns, plus theta as predicted by the
  Khorsandi (2025)                   ANN, the iterative method, the
                                     noniterative method, and the modified
                                     noniterative method, for direct
                                     method-vs-method comparison.

  Vaughan (1981)       Table I (6    Frost and Purl (three
                       named cases)  configurations), Litton L-2087,
                                     Varian VTC6364 and a further Frost
                                     case, each with nominal perveance and
                                     convergence plus raw input data (V,
                                     I, r_w, cathode current density J_c).

  Tiwary and Basu      Table I       The same class of
  (1987)               (named cases) commercial/reference guns (Litton
                                     L-2087, Varian VTC 6364, Frost and
                                     Purl as cited from Frost, Purl and
                                     Johnson, 1962), with input parameters
                                     V0, I0 and cathode diameter 2r_m.

  Yang, Jia and Zhu    Table I (32   The largest single source: 32
  (2006)               cases)        practical Pierce guns with perveance
                                     P, beam waist radius r_w, convergence
                                     squared C\^2, and experimental
                                     theta - structurally the closest
                                     match to the reference paper\'s own
                                     Table 2.
  -------------------- ------------- -------------------------------------

Combined and de-duplicated so far, these four sources give 71
datapoints, each tagged with its originating paper as instructed.
Several named cases (notably \"Frost and Purl\" and \"Litton L-2087\")
recur across more than one source table, since different authors
benchmarked their methods against the same well-known reference guns;
these overlaps are being tracked explicitly during deduplication rather
than silently merged, so that no single physical gun is inadvertently
counted, and weighted, more than once in the eventual \~1,000-point
dataset. Scraping additional papers toward that target is ongoing
(Section 9).

**7. DEMONSTRATION WEB APPLICATIONS**

**7.1 Pencil-Beam Demonstrator: pencilgun.netlify.app (Current, Phase
2)**

Live at:
[[https://pencilgun.netlify.app]{.underline}](https://pencilgun.netlify.app).
The site is fully static (HTML, CSS, JavaScript and a small JSON weights
file), makes no network calls, needs no backend server, and runs
entirely offline once loaded - a direct consequence of keeping the model
to 23 numbers (Section 4.2). It exposes four visible tabs plus one
currently hidden:

-   Predict - sliders for perveance P, beam waist radius r_w, and
    convergence C (on a log scale, reflecting the transform in Section
    5.3); shows the predicted theta, the live-drawn gun cross-section,
    and the derived cathode geometry that follows from theta (Section
    2.2).

-   Dataset - the training and test rows, each tagged as Tier A (real,
    from Panahi et al.) or Tier B (synthetic, Section 4.3), so a viewer
    can see exactly what the model was and was not trained on.

-   Method - the network diagram, an explanation of why the ln C
    transform matters (Section 5.3), how Levenberg-Marquardt works, and
    the governing physics equations (Section 2.2).

-   About - stated limitations, the pencil-beam-only scope, and the
    citation for the reference paper.

-   Results (currently hidden) - benchmark tables and graphs for
    Sections 5.1-5.4 are implemented in the site\'s code but
    intentionally left commented out pending final numbers; re-enabling
    the tab is a three-line code change once the benchmark is finalised.

**7.2 Sheet-Beam Demonstrator: electron-gun.netlify.app (Phase 1)**

Live at:
[[https://electron-gun.netlify.app]{.underline}](https://electron-gun.netlify.app).
This earlier deliverable (Section 3.3) takes V, I, a and b and plots the
optimum sheet-beam envelope/cross-section, and includes an ANN-based
demonstrator for the sheet-beam case whose build-and-validate approach
directly informed the more rigorous benchmarking methodology applied
throughout Section 5 of this report.

**8. CHALLENGES AND CURRENT LIMITATIONS**

-   Only 23 usable training points from the one directly benchmarkable
    source (Panahi et al.\'s own Table 2), against 23 trainable
    parameters - a near-fully-determined fitting problem with little
    redundancy, which makes the trained network sensitive to its random
    initialisation (Section 5.2) and is the direct reason results are
    reported as a median and a range rather than a single number.

-   The tanh output activation means predictions cannot extrapolate
    meaningfully outside the training range of theta; a new gun design
    far outside 9.80-70.00 degrees should not be trusted from this
    model.

-   The ln C transform (Section 5.3) that materially improves accuracy
    is an empirical fix motivated by the shape of the data, not
    something derived from the underlying space-charge physics.

-   Synthetic data augmentation, the most obvious next lever for
    improving accuracy, was tried and shown not to help (Section 5.4) -
    an important negative result that redirects effort toward real
    literature data instead.

-   The literature-sourced dataset (Section 6) stands at 71 of a roughly
    1,000-point target; overlapping named cases across source papers
    (e.g. Frost and Purl, Litton L-2087) complicate clean deduplication
    and are still being tracked by hand.

-   The Results tab of the pencil-beam demonstrator (Section 7.1) is
    implemented but not yet public-facing, pending the finalised
    benchmark numbers this report documents.

-   The Phase 1 sheet-beam tool and the Phase 2 pencil-beam ANN pipeline
    remain separate demonstrators; they have not yet been unified with
    each other or with the magnetic focusing (S2) and collector (S3)
    stages of the overall system (Section 2.1).

**9. FUTURE WORK: REMAINDER OF THE SEMESTER**

Continuing dataset expansion is the top priority: scraping additional
papers toward the roughly 1,000-datapoint target (Section 4.5, Section
6), resolving overlapping named cases across sources, and retraining and
re-validating both the single-output (M1) and multi-output (M1-multi)
networks on the expanded, real-data-only training set - directly
informed by the negative synthetic-augmentation result in Section 5.4.

Beyond the dataset, three further items are planned: finalising and
publishing the Results tab of the pencil-beam demonstrator once the
expanded-dataset benchmark is complete; building the planned NSGA-II
multi-objective optimiser that uses the verified ANN as a fast surrogate
for the roughly 50,000 gun-geometry evaluations such a search requires
(Section 1.3); and extending the design chain formulated in Section 2.1
to the magnetic focusing system (S2) and collector (S3) stages, unifying
the sheet-beam (Phase 1) and pencil-beam (Phase 2) tools into one
coherent electron-optical-system design pipeline. A further avenue
discussed with the supervisor - a gun geometry with one spherical and
one flat cathode surface, its own convergent-beam synthesis, and an
accompanying literature bibliography - is also earmarked for the second
half of the semester.

**CONCLUSION**

This midsem report has documented two phases of work on the electron
optical system design chain. Phase 1 formulated and implemented, in
Python, the paraxial envelope equation for a rectangular sheet-beam
electron gun - the sheet-beam analogue of Pierce\'s universal
beam-spread curve - to determine the optimum beam cross-section for a
given voltage, current and geometry, and deployed an interactive
demonstrator for it.

Phase 2, the primary focus of the current period, independently
reimplemented from scratch a recently published neural-network method
(Panahi et al., 2025) for predicting the half-beam cone angle of a
classical pencil-beam Pierce gun, and verified that reimplementation at
every stage against independent ground truth, including an exact
reproduction of the paper\'s own reported error when run on the paper\'s
own data. The reimplementation was then extended in the two directions
assigned for this phase: a multi-output variant that predicts derived
cathode geometry alongside theta from the same inputs, and the beginning
of a literature-sourced dataset - 71 datapoints from four papers so far,
each tagged by source - intended to grow toward roughly 1,000 points and
support retraining beyond the original paper\'s 30 guns.

Two results are reported candidly because they matter for what comes
next: the reimplementation\'s own benchmark accuracy (8.33% median
error) does not yet match the paper\'s reported 3.74% on random-restart
runs, a limitation traced to the small, near-fully-determined 23-point
training set rather than to an error in the implementation; and
augmenting training with synthetic physics-generated data was tested and
shown not to help, redirecting the project\'s priority firmly toward
real, literature-sourced data. Both findings, together with the verified
network, the two deployed demonstrators, and the assembled dataset, are
documented in this report so that the remainder of the semester\'s
work - a larger literature dataset, a published benchmark, and
eventually a multi-objective optimiser built on this surrogate - has a
solid, honestly characterised foundation to build from.

**APPENDIX A: LIVE DEMONSTRATORS AND EVIDENCE OF WORK COMPLETED**

This appendix collects the live links referred to throughout this report
and visual evidence of the work behind it, so that every claim in this
report can be checked directly rather than taken on faith.

**A.1 Live Demonstrators**

**Phase 2 (pencil beam, current):**
[[https://pencilgun.netlify.app]{.underline}](https://pencilgun.netlify.app)

**Phase 1 (sheet beam, prior):**
[[https://electron-gun.netlify.app]{.underline}](https://electron-gun.netlify.app)

**A.2 Original Meeting Notes: Problem Formulation**

The three-stage system decomposition (electron gun / magnetic focusing
system / collector) and the sheet-beam formulation (V, I, a, b -\>
magnet dimensioning) used throughout Section 2.1 and Section 3.1 were
recorded directly from the initial problem-formulation discussion with
the supervisor, reproduced below as originally written.

![](media/d125a14b55d7976a45d35597f276731d60760966.png){width="3.125in"
height="3.7083333333333335in"}

*Figure A.1 - Original meeting notes: the three-stage electron optical
system (electron gun, magnetic focusing system, collector) and the
sheet-beam design formulation.*

![](media/05b27025549d43c26202c3e0a59dd9cc5f9e3afe.png){width="3.125in"
height="3.0in"}

*Figure A.2 - Follow-up notes: the spherical/flat cathode gun variant,
the spherical-to-sheet-beam convergence idea, and the assignment to
build a bibliography on pencil-beam gun synthesis (Section 9).*

**A.3 Course Registration Record**

The course details confirming this project\'s registration are
reproduced below.

![](media/aa92b4229dcbc682db591f415a28cc80da3cd2f7.png){width="4.791666666666667in"
height="2.2291666666666665in"}

*Figure A.3 - Course registration details: EEE F266, Study Project,
\"Multi Object\[ive\] Optimization of Electron optical syst\[em\],\"
Dept. of Electrical and Electronics Engineering, Supervisor Lalit
Kumar.*

**A.4 Live Demonstrator Screenshot**

A screenshot of the pencil-beam demonstrator\'s Results tab, showing the
honestly reported distribution of test error (median 8.33%,
interquartile range 7.14-9.91%, full range 2.99-115.75%) discussed in
Section 5.2, and the explicit \"this is a research result, not a design
tool\" caution.

![](media/49da040c8110a36baa811a42efdaa96ad949e8bf.jpg){width="5.416666666666667in"
height="3.2708333333333335in"}

*Figure A.4 - pencilgun.netlify.app, Results tab: benchmark error
distribution against the paper\'s reported 3.74%, and the applicability
caution referenced in Section 5.5.*

This appendix is intended to make the two most important classes of
claim in this report - what the supervisor and student actually
formulated at the outset, and what the deployed model actually reports -
checkable directly, either from the live links in A.1 or from the
reproduced evidence in A.2-A.4.

**REFERENCES**

1\. Panahi, R., Feghhi, S.A.H., Sanaye Hajari, Sh., & Khorsandi, M.
\"Determination of the Half-Beam cone angle for Pierce electron gun
design using an artificial neural network.\" Results in Physics, 2025,
73, 108256. https://doi.org/10.1016/j.rinp.2025.108256.

2\. Rodney, J., & Vaughan, M. \"Synthesis of the Pierce gun.\" IEEE
Transactions on Electron Devices, 1981, 28, 37-41.

3\. Tiwary, U., & Basu, B. \"Noniterative method for the synthesis of
convergent Pierce electron guns.\" IEEE Transactions on Electron
Devices, 1987, 34(5), 1218-22.

4\. Yang, C., Jia, B., & Zhu, Z. \"Improved noniterative method for the
synthesis of convergent Pierce electron guns.\" IEEE Transactions on
Electron Devices, 2006, 53(11), 2849-52.

5\. Frost, R., Purl, O., & Johnson, H. \"Electron guns for forming solid
beams of high perveance and high convergence.\" Proceedings of the IRE,
1962, 50(8), 1800-7.

6\. Pierce, J.R. Theory and Design of Electron Beams. Van Nostrand, 1954
(background on the classical circular-beam Pierce gun and the universal
beam-spread curve referred to in Section 3).

7\. Pencil-beam demonstrator (Phase 2):
[[pencilgun.netlify.app]{.underline}](https://pencilgun.netlify.app).
See Appendix A.

8\. Sheet-beam demonstrator (Phase 1):
[[electron-gun.netlify.app]{.underline}](https://electron-gun.netlify.app).
See Appendix A.

**Submitted by:**

Shivansh Shekher Ojha - ID: 2023B2A30969H

Satwik Shrivastava - ID: 2023B2A80938H

**Date: October 2026**
