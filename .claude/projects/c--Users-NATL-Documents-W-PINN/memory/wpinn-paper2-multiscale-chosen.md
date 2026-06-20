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

PHASE 2 DONE (2026-06-19). **1D decisive sweep = the rigorous proof** (phase1_1d/decisive_scaletest.py, decisive.png, table_decisive.png): banded wavelet vs RBF-PIELM (best of 6 widths, MATCHED NF), single spike width w swept — wavelet wins at EVERY scale: w=0.20 3.2e-12 vs 6.7e-5 (5e6x); w=0.05 2.1e-8 vs 1.1e-2; w=0.02 6.5e-5 vs 1.5e-1 (2340x); w=0.005 3.0e-2 vs 1.0e-1 (3.4x). Fixed-width random features saturate at 10-20% as scales separate. Clean controlled headline.

**2D (phase2_2d/run_phase2.py; sol.png, compare.png, baselines.png, table.png, results.json):** 4 bumps widths 0.08-0.20. coarse[0,1,2] 1.99 -> medium[0,1,2,3] 0.23 -> banded+[4,5] 0.10 (ablation works); banded 10% < RBF-PIELM best-width 16% (thesis scales) BUT vanilla PINN (4k Adam, 20x slower) 3.5% wins on this MODERATE field.

HONEST LIMITS (critical, do not oversell 2D): (i) 2D accuracy ceiling ~1e-1 from operational-matrix CONDITIONING — adding levels 6+ DEGRADES (tried: 48%), parsimony is the lever (same as interface Phase 1), NOT more levels/local collocation; (ii) Gaussian bumps are ideal for Gaussian-RBF (unusually strong competitor for that shape); (iii) at moderate 2D separation a trained MLP competes. The DECISIVE multiresolution win needs STRONG separation, proven cleanly in 1D. Also learned: area-weighting starves fine wavelets; unweighted clustering biases coarse — use plain uniform-grid gear recipe.

NEXT (if continuing): Phase 3 inverse (recover unknown #sources at unknown scales via sparsity) is still the intended headline; but RECONSIDER 2D framing — lead the paper with the 1D controlled study; 2D needs either non-Gaussian features (where derivative-wavelets aren't disadvantaged) or a conditioning fix (orthonormal/Daubechies compact wavelets) to be a clean win. Honest status: thesis sound in 1D, 2D needs more work to beat a trained MLP.
