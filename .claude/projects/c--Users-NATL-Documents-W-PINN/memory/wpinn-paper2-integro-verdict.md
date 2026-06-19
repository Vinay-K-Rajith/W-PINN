---
name: wpinn-paper2-integro-verdict
description: Phase 0+1 verdict on the integro-differential/nonlocal W-PINN paper-#2 idea — mechanism works, novelty crowded
metadata:
  type: project
---

Paper-#2 idea = integro-differential/nonlocal W-PINN (folder Integro_Differential/, plan = integro_differential_wpinn_plan.md). Ran Phase 0 (novelty) + Phase 1 (PoC) on 2026-06-19 at user request.

**Phase 1 (capability) = PASS.** phase1_1d/run_phase1_ide.py: 1D Volterra IDE u'+∫_0^x u=2e^x-1, exact u=e^x. Integral operational matrix is ANALYTIC for Gaussian-derivative wavelets: ∫_0^x psi_i = (1/j)(e^{-k^2/2}-e^{-(jx-k)^2/2}). Single AD-free column-norm+Tikhonov LSQ solve. relL2 1e-8 (NF=9) down to 1.4e-10 (NF=68). Machine-accurate, sub-second. The mechanism unquestionably works.

**Phase 0 (novelty) = CAUTION / largely NO-GO as framed.** Focused prior-art shows the lane is crowded (more than interface was):
- PINNIES (arXiv:2409.01899, 2024): PINN for integral operators via precomputed quadrature operator (tensor-vector product), no aux points, Volterra+Fredholm — occupies most of the "op-matrix avoids aux points" claim.
- Rahimkhani 2025 (Wiley jnm.70070): wavelet + operational-matrix-of-integration PINN for fractional delay DEs.
- Peridynamic horizon recovery via PINN (arXiv:2501.03911, 2025): learns kernel/horizon 1D+2D — takes the Phase-3 inverse-horizon headline.
- EPINN 2025, Advanced-PINN arXiv:2501.16370 (integral eqns); Wavelets-based PINN Sci.Rep.2023 (naming).
Remaining differentiators are NARROW/incremental: analytic op-matrix (vs PINNIES quadrature); single-shot CONVEX LINEAR solve (others gradient-train); localized multiresolution (vs global Legendre/Bernstein). Echoes the earlier fractional rejection ([[wpinn-fractional-prior-art]]).

**RECOMMENDATION:** PIVOT paper #2 to **crack/fracture singularity** problems. The r^{1/2} crack-tip singularity is precisely the localized sharp feature the gear's BANDED multiresolution refinement just proved W-PINN crushes ([[wpinn-gear-complex-test]]) where vanilla PINNs fail — cleaner whitespace, strong synergy with the gear result, applied NDT inverse-crack-detection angle. Alternative if user wants to push integro anyway: lead hard on analytic-op-matrix + single-shot-linear-solve + add a new applied PIDE (jump-diffusion option pricing), accepting "incremental over PINNIES" reviewer risk. Integro_Differential/ folder + Phase 1 PoC kept as a working record either way.
