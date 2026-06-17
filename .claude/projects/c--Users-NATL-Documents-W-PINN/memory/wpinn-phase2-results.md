---
name: wpinn-phase2-results
description: Phase 2 (2D circular-interface W-PINN) results — works, conditioning solved by basis parsimony; complex-geometry demo still pending
metadata:
  type: project
---

Phase 2 of [[wpinn-application-decision]] core DONE (2026-06-16). Code: Elliptic_Interface/phase2_2d/ (run_phase2.py = 2D solver + conditioning test + contrast sweep; phase2_analyze.py = richness frontier + figure phase2_summary.png).

Benchmark: 2D circular interface r0=0.5 in (-1,1)^2, manufactured u=(x^2+y^2)/a per region, f=-4 everywhere, homogeneous jumps. Tensor-product Gaussian-derivative wavelets, decomposed (inside/outside disk), normal-deriv on circle is radial; solved AD-free by column-normalized + Tikhonov least squares.

Results:
- 2D formulation WORKS: relL2 ~1e-6, flux-kink [[d_n u]] captured to ~2e-8 across contrast rho=10..10^4 (jump_pred -0.9,-0.99,-0.999,-0.9999 = exact), contrast-ROBUST, ~0.4s, AD-free.
- CONDITIONING OBSTACLE RESOLVED by first-principles basis parsimony: the 1D cond~1e196 came from over-completeness (10 levels for a smooth solution), NOT from interfaces. Using parsimonious 3 levels [0,1,2] -> raw cond 4.5e21 (175 orders better). Column-normalization (Jacobi precond) drops it a further ~4 orders.
- Accuracy<->conditioning frontier (rho=1000, after precond): [0,1] NF=144 relL2 6.4e-5 cond 2.8e16; [0,1,2] NF=625 relL2 4.1e-6 cond 8.6e17; [0,1,2,3] NF=2500 relL2 7.9e-7 cond 1.9e18. ~10x accuracy per added level, modest cond growth. Sweet spot [0,1,2] or [0,1,2,3].

HONEST CAVEATS:
- cond after precond still ~1e16-1e18: fine for SVD least-squares (our justified solver), but would still challenge Adam/gradient training (the PUBLISHED optimizer). For an as-published gradient-trained variant, orthonormal compact-support (Daubechies) wavelets are the next lever.
- 2D accuracy ~1e-6 < 1D's 1e-9: limited by coarse Gaussian wavelets representing the smooth solution + conditioning ceiling.
- 2D vanilla-PINN baseline NOT re-measured; the kink-failure (relL2~0.16) is carried over from Phase 1's clean ablation.

PHASE 2 NOW 100% COMPLETE (2026-06-16):
- (1) Flower/complex-geometry mesh-free demo DONE (phase2_flower.py, phase2_flower.png): level-set u=phi/a, phi=x^2+y^2-r0^2+eps(x^3-3xy^2) harmonic cubic => f=-4, 3-lobe interface, position-varying kink. relL2 ~5e-6 across rho=10..1000, mesh-free (only point classification + normals changed).
- (2) In-2D baseline DONE (phase2_baselines_fast.py, GPU/float32 MLPs) on circle:
  rho=10:   wavelet 4.4e-5 (2.5s) | XPINN 6.3e-3 (75s) | vanilla 5.9e-1 (31s)
  rho=1000: wavelet 1.6e-5 (2.7s) | XPINN 8.6e-2 (78s) | vanilla 2.7e-1 (55s)
  => wavelet beats XPINN ~140x..~5000x, ~30x faster; vanilla fails (can't represent kink). 1D crossover reproduces in 2D, advantage GROWS with contrast.