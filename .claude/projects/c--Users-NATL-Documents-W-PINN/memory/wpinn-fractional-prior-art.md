---
name: wpinn-fractional-prior-art
description: Prior-art landscape for the "W-PINN on fractional PDEs" novelty claim — what's already taken
metadata:
  type: project
---

Goal: find a novel *application* (not new method) for W-PINN (Pandey/Singh/Behera, arXiv:2409.11847). The repo's W-PINN = solution as fixed multiresolution Gaussian-derivative wavelet expansion `u=Wc+b`; all operators are precomputed matrices applied to `c` (AD-free); a network learns the coefficient vector `c`; collocation-residual + IC/BC loss. Demonstrated strength: boundary/initial layers (SPP example + NTK analysis).

Prior-art check (June 2026) shows "AD-free fractional PINN with precomputed operational matrix" is ALREADY DONE, so it is NOT clean whitespace:
- **Op-matrix fPINN** (arXiv:2401.14081): precomputed Caputo operational matrix, AD replaced by matrix-vector product. BUT global Legendre polynomials, standard coordinate-input MLP, NO singularity treatment. This is the proposal's core mechanism minus wavelets — the main competitor to beat.
- **Laplace-fPINNs** (arXiv:2304.00909): removes fPINN auxiliary points for subdiffusion via Laplace transform.
- **Legendre-wavelet NN for time-fractional Black–Scholes** (MDPI Fractal Fract 6(7) 401): wavelet+NN+fractional op-matrix, but ELM (random weights+least squares), not PINN. Partly occupies the option-pricing applied anchor.
- **Haar-wavelet operational matrix**: classical, no learning.

Therefore the only un-occupied differentiator is: **localized multiresolution wavelets placed densely at the solution's singular layer**, which op-matrix-fPINN/Laplace-fPINN do not do. Headline must be singularity-resolution, NOT "AD-free fractional."

Recommendation (my survey, user chose "survey alternatives"): keep time-fractional subdiffusion as benchmark backbone (Mittag-Leffler exact solution), but lead novelty with the **singularly-perturbed fractional problem** (§2.4 of [[fractional_wpinn_application.md]]) — strongest whitespace + matches the repo's proven SPP strength. Benchmark against op-matrix-fPINN & Laplace-fPINN, not the 2019 fPINN. Optional applied anchor: fractional Fokker–Planck / cable equation (less crowded than option pricing).
