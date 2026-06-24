# Multiple disconnected inclusions — results

Decomposed AD-free wavelet W-PINN on **m disjoint circular inclusions** scattered in Ω=(−1,1)²,
each its own subdomain Ω⁻ᵢ (centre cᵢ, radius rᵢ, coefficient a⁻ᵢ) embedded in one shared matrix Ω⁺
(coefficient a⁺). This is the **topology** stress test — m+1 subdomains and m simultaneous interfaces,
not a harder version of one interface — and the realistic composite-material / fibre-suspension case.

## Problem and method

`−∇·(a∇u)=f` with `[[u]]=gᵢ`, `[[a∂ₙu]]=hᵢ` on every Γᵢ and Dirichlet `u=g` on ∂Ω. The exact field
is prescribed **independently and O(1) in every region** — `u⁺=cos(πx/2)cos(πy/2)` in the matrix,
`u⁻ᵢ=cos(κ(x−cxᵢ))cos(κ(y−cyᵢ))` (κ=3) in inclusion i — so the source `f` and the (in general
nonzero) jump data `gᵢ,hᵢ` are read off by autograd (one-time problem data; the solve stays AD-free).

> Why not the gear's `u=Φ/a` level-set construction? A single product level set `Φ=∏ᵢ(|x−cᵢ|²−rᵢ²)`
> collapses the solution amplitude to `O(rᵢ²)` inside small inclusions, making relative error there
> meaningless and imbalancing the least squares. The independent O(1) fields avoid this and give the
> strictly more general **inhomogeneous-jump** interface problem.

**Block-structured basis (the decomposition).** One tensor-product Gaussian-derivative wavelet
expansion *per subdomain*: `u⁺=W₊c₊+b₊` on the matrix, `u⁻ᵢ=Wᵢc⁻ᵢ+b⁻ᵢ` on each inclusion. The
inclusion blocks are **local** and **scaled to each rᵢ** (finest dyadic level `L≈log₂(4/rᵢ)`) — better
conditioned and cleaner to refine than one global "inside" basis spanning the disjoint inclusions. The
matrix block is global-coarse `[0,1,2,3]` + (optionally) a radius-scaled fine band around each Γᵢ.

The resulting system is **arrow-structured**: each inclusion block couples only to the matrix block
through its own interface; inclusions never couple to one another. Solved AD-free by column-normalised
Tikhonov least squares (`tik=1e-10`, interface/BC weights 10), test grid 300×300, K=140, M=240/circle.

## (1) Five well-separated inclusions — contrast sweep ρ = a⁺/a⁻

Config `POOL6[:5]`, NF≈3920 (NF₊=3177, NF₋=[203,151,179,79,131]).

| ρ     | relL2     | L∞        | [[u]] resid | [[a∂ₙu]] resid | per-inclusion relL2                 |
|------:|----------:|----------:|------------:|---------------:|:------------------------------------|
| 10    | 3.33e-04  | 1.13e-03  | 2.57e-04    | 7.89e-03       | 6.3e-5 9.5e-5 1.0e-4 6.1e-5 2.8e-5  |
| 100   | 1.02e-03  | 4.76e-03  | 3.31e-03    | 6.99e-03       | 4.9e-4 3.2e-4 7.9e-4 2.6e-4 4.4e-4  |
| 1000  | 6.88e-04  | 5.98e-03  | 6.51e-03    | 5.19e-03       | 8.0e-4 2.7e-4 1.4e-3 3.3e-4 2.8e-4  |

**Contrast-robust** (relL2 ≈ 1e-3 across two orders of magnitude in ρ) and, critically, the
**per-inclusion** error is uniform across all five inclusions — including the smallest, central one
(r=0.13) — so no inclusion is starved by the shared matrix. Error is localised on the interface rings
(`multi_inclusion.png`, panel c); bulk error is near machine level.

## (2) Heterogeneous contrast — each inclusion its own ρᵢ, one solve

a⁺=1000, ρᵢ = [10, 100, 1000, 50, 500] (composite with five different inclusion materials):

| metric | value | per-inclusion relL2 (by ρᵢ) |
|:-------|------:|:----------------------------|
| relL2  | 1.67e-02 | ρ=10:9.0e-3 · 100:3.2e-3 · 1000:5.7e-3 · 50:4.0e-3 · 500:4.6e-3 |

Every inclusion is solved to <1e-2 in a single shot. Looser than the uniform case because the
widely-varying flux scales (`a⁻` from 1 to 100 against `a⁺`=1000) stress the *single* global interface
weight; a per-interface weight would tighten it, but this already demonstrates true multi-contrast.

## (3) Refinement ablation — is the radius-scaled fine band needed? (ρ=1000)

| matrix basis                     | NF    | relL2     | L∞        |
|:---------------------------------|------:|----------:|----------:|
| global coarse `[0,1,2,3]` only   | 3243  | 6.93e-04  | 5.98e-03  |
| coarse + per-inclusion fine band | 3920  | 6.88e-04  | 5.98e-03  |

**Honest finding:** for *smooth circular* interfaces the band gives **no improvement** (0.7%) at 2.7×
the wall-time. This is the opposite of the gear, where the same band cut relL2 ~13× — because the
band specifically resolves **sharp features** (the gear's tooth-tip cusps), which circles don't have.
The per-inclusion *inside* blocks (always present, radius-scaled) are what carry the local accuracy
here. **The global-coarse matrix basis is the efficient default for smooth inclusions.**

## (4) Topology scaling — m = 2 … 6 inclusions (ρ=100)

| m | NF    | relL2     | worst inclusion relL2 | L∞        |
|--:|------:|----------:|----------------------:|----------:|
| 2 | 3154  | 5.52e-04  | 5.49e-04              | 2.49e-03  |
| 3 | 3497  | 1.12e-03  | 8.62e-04              | 4.74e-03  |
| 4 | 3665  | 1.06e-03  | 8.12e-04              | 4.75e-03  |
| 5 | 3920  | 1.02e-03  | 7.90e-04              | 4.76e-03  |
| 6 | 4148  | 1.01e-03  | 7.95e-04              | 4.79e-03  |

**The decomposition scales past two subdomains.** Both the global relL2 and the worst single-inclusion
error are **flat** from m=2 to m=6 — adding inclusions adds an independent, well-conditioned block and
~250 basis functions each, with no accuracy loss. This is exactly the regime a mesh method pays for
(mesh + remesh around every inclusion) and where the few-parameter mesh-free representation is free.

## Notes

- **Conditioning.** The column-normalised system has cond ~7e26 — the overcomplete Gaussian-derivative
  *frame* is numerically rank-deficient. The accurate solution (relL2 ~3e-4) lives in its
  well-conditioned subspace; this is precisely why the method uses an SVD least squares
  (`lstsq` / `gelsd`) with Tikhonov regularisation rather than the normal equations. Same basis family
  and behaviour as the gear/flower examples.
- **Wall-time.** ~30–60 s per forward solve (dominated by the SVD on the ~20k×4k system, CPU, float64).
- Per-inclusion blocks decouple in the interior; they couple only to the matrix block on their own Γᵢ,
  so the assembly and the system are intrinsically modular in m.

**Reproduce:** `python studies.py` (tables → stdout + `studies_results.json`),
`python make_figure.py` (→ `multi_inclusion.png`), `python multi_inclusion.py` (quick ρ-sweep).

---

## (5) ρ < 1 — contrast soundness (`rho_sweep.py`)

Sweeping ρ = a_out/a_in across **both** regimes on the 5-inclusion forward (worst per-inclusion shown):

| ρ | 0.001 | 0.01 | 0.1 | 0.5 | 1 | 2 | 10 | 100 | 1000 |
|--|--|--|--|--|--|--|--|--|--|
| relL2 | 5.4e-2 | 5.6e-3 | 6.2e-4 | 1.4e-4 | 9.5e-5 | 1.3e-4 | 3.3e-4 | 1.0e-3 | 6.9e-4 |

**ρ<1 is sound, not a failed target.** The method is roughly **contrast-symmetric** about ρ=1, with
relL2 ≤ 6e-4 across ρ ∈ [0.1, 10] (best at ρ=1, no contrast). Only **extreme** ρ ≤ 0.01 (matrix
100–1000× more conductive than the inclusions) degrades (5.6e-3 → 5.4e-2) — the expected near-
insulating-matrix stiffness (the matrix PDE residual scales with the tiny a_out and the flux condition
becomes imbalanced), not a soundness bug. Usable contrast range: **ρ ∈ [0.1, 1000]** at relL2 ≤ ~1e-3.

---

## (6) Multi-parameter inverse — recover (cx, cy, r) for every inclusion (`inverse.py`, `inverse.png`)

The payoff of the block decomposition: recover the **full geometry of all inclusions** (3m unknowns)
from sparse interior measurements. Honest **physical** problem (not manufactured): fixed source `f=1`,
fixed contrast ρ=100, `u=0` on ∂Ω, homogeneous jumps; geometry enters only through where `a` jumps.
Outer `scipy.least_squares` (TRF) over the 12 parameters; inner = a **sub-second** wavelet forward
re-solve (~290 ms) on a geometry-independent precomputed matrix block.

**What works.** 3 of 4 inclusions recover to **~1–2e-2** in position and **~1.4e-2** in radius, and the
recovery is **noise-robust**: 0 %, 0.1 %, 1 % noise all give the same ~5e-2 mean position error — i.e.
the residual error is set by the optimisation/identifiability, *not* by noise sensitivity.

| noise | mean &#124;pos err&#124; | max &#124;pos err&#124; | mean &#124;rad err&#124; | nfev | sec |
|--:|--:|--:|--:|--:|--:|
| 0 % | 4.99e-2 | 9.4e-2 | 1.43e-2 | 152 | 44 |
| 0.1 % | 4.77e-2 | 1.0e-1 | 1.32e-2 | 270 | 75 |
| 1 % | 4.47e-2 | 9.5e-2 | 1.36e-2 | 202 | 55 |

**What is hard (diagnosed, honest).** This is a genuinely **non-convex, ill-posed 12-D** inverse:

1. *Forward map was discontinuous in geometry.* A fixed interior grid reclassified per candidate makes
   J(θ) step as inclusions sweep over grid points → gradient optimisers die in ~2 iterations.
   **Fixed** by geometry-attached collocation (polar interior points in each inclusion's moving frame +
   smooth sigmoid weights on the matrix grid) → J(θ) becomes C¹ and the forward stays sub-second.
2. *Exterior-only data is low-sensitivity / non-identifiable.* With matrix sensors only, Nelder–Mead
   drove the misfit **10× lower** than TRF yet gave **worse** geometry — different geometries fit the
   exterior field almost equally well (classic inverse-conductivity insensitivity).
3. *The frame defeats finite-difference Jacobians.* The cond~1e26 frame makes the `gelsd` solution
   jitter under tiny parameter changes, so TRF terminates prematurely; this is why the repo's inverses
   use derivative-free search. One inclusion consistently lands in a **local minimum** (~9e-2).

**Honest positioning for the paper:** this is a working **local** multi-parameter inverse (recover from
a reasonable prior, robust to 1 % noise), demonstrating the mesh-free forward map drives a 12-parameter
geometry recovery with no remeshing — *not* a globally-convergent solver. A clean global version would
need interior measurements + a derivative-free/global optimiser (CMA-ES) + a better-conditioned basis.

**Reproduce:** `python rho_sweep.py` · `python inverse.py` · `python make_inverse_figure.py`.
