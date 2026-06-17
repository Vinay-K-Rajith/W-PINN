---
name: wpinn-application-decision
description: The chosen novel W-PINN application — elliptic interface (discontinuous-coefficient) problems, and why
metadata:
  type: project
---

Decision (2026-06-16): the novel W-PINN application is **elliptic interface problems** (discontinuous-coefficient elliptic PDEs, $-\nabla\cdot(a\nabla u)=f$ with $a$ jumping across a fixed interface $\Gamma$). Pivoted AWAY from the fractional-PDE proposal.

**Why the pivot:** user clarified the W-PINN selling point is *smoothing / resolving sharp features better than a vanilla PINN* (per its SPP example + NTK analysis), NOT the AD-free property. That kills fractional's justification (AD-impossibility) and the crowded fractional prior art ([[wpinn-fractional-prior-art]]). Interface problems fit the *real* strength: a stationary, localized kink at a known location suits the repo's FIXED multiresolution wavelet basis; the gradient kink is sharp-but-smooth-each-side (no Gibbs, unlike shocks); operator is linear/fixed; vanilla PINNs documented to fail (spectral bias). Not the source paper's existing "multiscale" turf.

**First-principles core:** flux continuity $[a\partial_n u]=0$ with $a^-\neq a^+$ forces a gradient kink ($u$ continuous). A single smooth global expansion CANNOT represent a kink → must DECOMPOSE: two wavelet expansions $u^\pm=W^\pm c^\pm+b^\pm$, one per subdomain, coupled by interface-condition loss terms ($[u]=0$, $[a\partial_n u]=0$) on interface collocation points. Kink emerges from the join. Single-expansion is the motivating ablation (should fail).

**Plan doc:** elliptic_interface_wpinn_plan.md (3 phases: 1D PoC + machinery + ablation; 2D circular/complex geometry + high-contrast sweep + NTK spectral-bias evidence; inverse interface-recovery + competitor benchmarks + write-up). Old fractional doc (fractional_wpinn_application.md) preserved, not deleted.

**Repo assets to reuse:** tensor-product 2D wavelet family (Helmholtz/, Maxwell's Equation/); NTK tooling (SPP/SPP_Ex1_NTK/); operational-matrix pattern (Wfamily.py); coefficient-emitting network (Model.py) — extend to two output heads c-, c+.

**Named competitors to beat:** vanilla PINN, XPINN, IPINN, cusp-capturing PINN; classical immersed-interface method (Li & Ito). Differentiator = AD-free multiresolution decomposition + inverse interface-location recovery.
