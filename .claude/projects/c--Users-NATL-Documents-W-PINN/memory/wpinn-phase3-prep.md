---
name: wpinn-phase3-prep
description: Phase 3 (inverse interface problems) prep + de-risking PoC — outer-loop strategy validated, joint-Adam fails
metadata:
  type: project
---

Phase 3 of [[wpinn-application-decision]] PREP done + inverse-contrast PoC PASSES (2026-06-16). Code: Elliptic_Interface/phase3_inverse/ (phase3_prep.md = plan; phase3_inverse_contrast_v2.py = working PoC; phase3_inverse_contrast_1d.py = the FAILED joint-Adam version, kept as the cautionary baseline).

Phase 3 = inverse problems (the publication differentiator classical IIM/FEM lack). Targets: (A) inverse contrast a+/- (geometry known); (B) inverse interface location/shape (the headline); (C) joint.

KEY DE-RISKING FINDING:
- Joint Adam over (rho, wavelet coefficients) FAILS: recovers rho~0.9 regardless of true rho, field relL2~0.2. Root cause = the ill-conditioned wavelet basis (same thing that broke Adam in Phase 1 forward). Do NOT do joint gradient descent on coefficients.
- FIX (validated): OUTER-LOOP over the physical parameter + INNER linear-least-squares forward solve (AD-free, ~0.07s). For inverse contrast: golden-section over scalar rho, each eval = one forward solve, minimize interior-data misfit. Parsimonious basis (levels 0..4).
- Results (Ndata=15): noiseless rho recovered to ~1e-9 rel err (rho=4,8,20); 1% noise -> ~2-3% rho error, field relL2 ~1e-3. Graceful degradation. PoC PASSES.
- This outer-loop+fast-forward-solve strategy is THE Phase 3 approach and transfers directly to (B) interface-location recovery (derivative-free optimize over shape params, each eval a cheap forward solve). The forward solver's speed is what makes the inverse tractable.

PHASE 3 EXECUTION DONE + PASS (2026-06-17). Code: phase3_2d.py (all 4 experiments), phase3_figure.py (phase3_summary.png headline). 2D results:
- (A) inverse contrast (circle): rho recovered ~1e-6 noiseless, ~1e-3 at 1% noise (rho=4,8,20).
- (B) inverse location (radius): R ~6e-7 noiseless, ~8e-4 at 1% noise.
- (C) inverse SHAPE (flower r0,eps jointly via Nelder-Mead): (0.5,0.3)->(0.4999,0.3004), ~1e-4 err. THE headline — interface shape from sparse interior data, which classical IIM/FEM can't do natively.
- (D) robustness frontier: graceful; even N=10 pts @ 5% noise -> R to ~0.9%.
Figure phase3_summary.png: recovered flower interface overlaps true exactly from 56 scattered pts (none on the interface) + robustness curves.

STATUS OVERALL: ALL THREE PHASES PASS. Phase 1 [[wpinn-phase1-results]], Phase 2 [[wpinn-phase2-results]], Phase 3 done. Method works end-to-end: 1D+2D, forward+inverse, simple+complex geometry, clean+noisy.

NEXT-ACTION ANALYSIS (2026-06-17): science done; remaining gaps before publication = (1) NOVELTY lock (focused prior-art search: wavelet-operational-matrix interface PINN + inverse shape recovery — biggest binary risk, cheapest, gates everything); (2) credibility audit of inverse results (NOT a classic inverse crime since data is analytic not same-discretization, but need model-MISMATCH test: recover shapes OUTSIDE the parametric family, e.g. 5-lobe/ellipse, + independent-solver cross-check); (3) applied/physical benchmark (manufactured-only reads as method demo, caps venue tier); (4) manuscript. Recommended order: novelty -> credibility -> applied benchmark -> write.