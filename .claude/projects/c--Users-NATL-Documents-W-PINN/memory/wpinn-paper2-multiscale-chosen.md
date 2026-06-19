---
name: wpinn-paper2-multiscale-chosen
description: Chosen paper-#2 application — multi-scale localized-feature W-PINN; why, after exhaustive search; Phase 0+1 pass
metadata:
  type: project
---

CHOSEN paper-#2 application (2026-06-19, after exhaustive web search at user request): **multi-scale localized-feature elliptic/Helmholtz problems** — adaptive multiresolution wavelet W-PINN. Folder Multiscale_Sources/ (plan = multiscale_sources_wpinn_plan.md; phase1_1d/run_phase1_multiscale.py). Supersedes the integro idea ([[wpinn-paper2-integro-verdict]], parked).

**WHY (key strategic finding):** the 2025-26 PINN/PIELM field is SATURATED for single-feature linear problems — point source (2111.01394), corner-singularity+point-source ENRICHMENT (SEPINN, Jin&Zhou 2308.16429), fracture (XPIELM 10^-12, CMAME 2025), biharmonic (RBF/Augmented-PIELM 2310.13947), eigenvalue (Eig-PIELM), inverse source (acoustic 2024, parabolic 2025). W-PINN's single-shot LINEAR solve is a mechanical COUSIN of PIELM, so "linear solve for X" and "enrichment for singularity X" are structurally occupied. The ONE durable edge = genuine MULTIRESOLUTION ADAPTIVITY (multiple features at disparate UNKNOWN scales), which vanilla PINN (spectral bias), RBF/Fourier-PIELM ("degrade on oscillatory/localized"), and enrichment (needs known singular form + count) all structurally lack. Same edge the gear banded-refinement proved ([[wpinn-gear-complex-test]]).

**Defensible novelty differentiator (Phase 0):** (a) MULTIPLE localized features at DISPARATE scales at once + (b) UNKNOWN NUMBER/SCALE recovered via SPARSITY in multiresolution coefficients + (c) AD-free fast forward solve enabling derivative-free inverse over count/scale. Generic multiscale is the source paper's turf (2409.11847) — avoid that framing; lead on (a)+(b). Applied anchor: source localization / load-ID / defect detection with unknown source count.

**Phase 1 PASS (demonstrates the edge):** 1D screened Poisson -u''+u=f, exact u=sin(2 pi x)+exp(-((x-0.5)/0.02)^2) (smooth O(1) + narrow w=0.02 spike). AD-free wavelet op-matrix LSQ. relL2: coarse[0,1,2,3] NF=35 ->1.16e-1 (fails); uniform[0..5] NF=133 ->2.51e-2; [0-3]+band[6,7,8]@spike NF=124 ->4.29e-4 (270x better than coarse, 58x better than uniform-fine, FEWER fns). Op: value psi=Xe^{-X^2/2}, 2nd-deriv psi''=j^2 X(X^2-3)e^{-X^2/2}.

NEXT: Phase 2 (many sources / 2D + vanilla & RBF-PIELM baselines showing their multiscale failure); Phase 3 inverse = recover UNKNOWN number of sources at unknown scales (sparsity + outer-loop+LSQ, reuse interface inverse machinery) — the headline. Then applied benchmark + write-up.
