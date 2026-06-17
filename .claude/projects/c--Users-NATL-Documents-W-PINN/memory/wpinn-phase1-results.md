---
name: wpinn-phase1-results
description: Phase 1 (1D elliptic interface W-PINN) results, verdict PASS, and the conditioning obstacle for Phase 2
metadata:
  type: project
---

Phase 1 of [[wpinn-application-decision]] DONE (2026-06-16). Code: Elliptic_Interface/phase1_1d/ (run_phase1.py = 4-model comparison via Adam; phase1_sweep.py = least-squares solver + contrast sweep; phase1_plot.py = figure). Benchmark: 1D -(a u')'=f, beta=0.4, manufactured homogeneous-interface exact solution u-=sin(pi x), u+=A sin(pi x)+B, A=a-/a+, f=a- pi^2 sin(pi x) both sides, genuine flux-kink.

Results:
- Operators verified vs finite diff: 2e-7 (d/dx), 1e-7 (d2/dx2).
- First-principles claim CONFIRMED: same wavelet basis, decomposed (M1) vs single-expansion (M2) = 360x lower relL2; M2 wavelet & M3 vanilla-MLP both predict ZERO kink (smooth global expansion structurally cannot bend at interface). relL2 ~0.16 for both single-expansion models.
- KEY FIX: M1 under-converged with Adam because the wavelet interface problem is LINEAR/convex -> correct solver is AD-free weighted least-squares (torch.linalg.lstsq gelsd), ~0.03-0.07s.
- Decomposed-wavelet (LSQ) relL2 vs contrast rho: 7.5e-12 (rho=10), 1.4e-11 (100), 4.7e-10 (1000), 4.5e-9 (10^4). Kink captured to ~1e-8. CONTRAST-ROBUST.
- XPINN (decomposed MLP, Adam) DEGRADES with contrast: relL2 1.2e-4 -> 8.9e-3; kink err 3.6e-4 -> 0.123 at rho=10^4. So at rho=10^4 wavelet is ~6 orders more accurate AND ~1000x faster.

VERDICT: PASS. Advantage materializes exactly where predicted (high contrast).

CARRY-FORWARD OBSTACLE for Phase 2 (2D): operational matrix is catastrophically ill-conditioned, cond(A) >> 1e196. Clean result rides on SVD-truncated least-squares absorbing rank-deficiency; also why Adam (the PUBLISHED optimizer) under-converged. Before 2D, fix conditioning: Daubechies/compactly-supported orthonormal wavelets, subdomain-restricted families, or preconditioned/regularized solve. This is the §6 caveat of elliptic_interface_wpinn_plan.md, now quantified.
