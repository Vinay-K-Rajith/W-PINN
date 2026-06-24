# W-PINN elliptic-interface — combined results (all key numbers in one place)

AD-free, mesh-free **decomposed wavelet least-squares** for `−∇·(a∇u)=f` with coefficient jumps across
an interface. One Gaussian-derivative wavelet expansion per subdomain (`u=Wc+b`), interface conditions
assembled into a **single** column-normalised Tikhonov SVD solve (no training loop, CPU, float64).

**One-line positioning (read first):** the method's win is **against neural-network / mesh-free PINN
methods** (it beats them with one linear solve); it is **not** a competitor to FEM (FEM is faster and
more accurate on fixed geometry — conceded openly). Build the paper on the first claim.

---

## 1. Published-benchmark head-to-head vs the leading mesh-free NN (DCSNN, Hu-Lin-Lai, JCP 2022)
`forward_benchmarks/dcsnn_ex1.py`, `dcsnn_ex2.py` — reproduces their exact problems.

| benchmark | wavelet (this work) | DCSNN reported | factor |
|:----------|--------------------:|---------------:|:------:|
| **Ex1** ellipse, contrast 1000, `u⁻=eˣeʸ/u⁺=sin·sin` | **L∞ 6.0e-7** (NF=5002, 1 solve, 98 s) | L∞ 4.39e-6 (101 params, trained) | **~7× better** |
| **Ex2** non-convex flower, β 10/1 | **relL2 3.2e-6** (NF=5002, 113 s) | relL2 2.63e-4 (501 params, trained) | **~80× better** |

→ Beats DCSNN on **both** its 2D benchmarks with a single AD-free linear solve (no training/GPU).
IIM (DCSNN's cited mesh baseline) needed ~65,792 dof for comparable accuracy on Ex1; wavelet NF=5002.
**This is the headline result.**

## 2. Multi-inclusion (topology) — `multi_inclusion/` (the novelty / composite-materials case)

**Contrast sweep, 5 well-separated inclusions** (per-inclusion error uniform incl. the small central one):

| ρ | 10 | 100 | 1000 |
|--|--|--|--|
| relL2 | 3.3e-4 | 1.0e-3 | 6.9e-4 |

**Topology scaling — relL2 FLAT from m=2→6** (the key result; FEM would remesh per inclusion):

| m | 2 | 3 | 4 | 5 | 6 |
|--|--|--|--|--|--|
| relL2 | 5.5e-4 | 1.1e-3 | 1.1e-3 | 1.0e-3 | 1.0e-3 |

- Heterogeneous per-inclusion contrast (ρᵢ=10..1000 in one solve): relL2 1.7e-2, every inclusion <1e-2.
- Ablation: the radius-scaled fine band (essential for gear cusps, ~13×) is a **no-op for smooth
  circles** (0.7%) — honest finding; global-coarse basis is the efficient default for smooth inclusions.

## 3. Contrast soundness incl. ρ<1 — `multi_inclusion/rho_sweep.py`

| ρ | 0.001 | 0.01 | 0.1 | 0.5 | 1 | 2 | 10 | 100 | 1000 |
|--|--|--|--|--|--|--|--|--|--|
| relL2 | 5.4e-2 | 5.6e-3 | 6.2e-4 | 1.4e-4 | 9.5e-5 | 1.3e-4 | 3.3e-4 | 1.0e-3 | 6.9e-4 |

ρ<1 is **sound** (contrast-symmetric, relL2≤6e-4 for ρ∈[0.1,10]); only extreme ρ≤0.01 degrades in the
*multi-inclusion* case (a conditioning effect — a single interface at contrast 1000 solves to 6e-7, §1).

## 4. Gear (sharp-feature showcase) — `gear/` (existing)

8-tooth gear (~44% tooth depth), banded multiresolution: relL2 ~1e-3, contrast-robust; the band cuts
error ~13× at the cusps (where smooth shallow networks like DCSNN are weakest — state this qualitatively).

## 5. FEM comparison — `fem_reference/` (concede openly; do NOT claim to beat FEM)

| | accuracy | per-solve | total cost, 12 geometries (remesh each) |
|:--|--:|--:|--:|
| conforming FEM P1 | relL2 8e-6 @ 30k dof | 0.013–0.17 s | **0.58 s** |
| wavelet W-PINN | relL2 1.3e-3 @ NF≈2600 | 7.5 s | 68 s |

**FEM dominates** — accuracy, per-solve, and total cost **even while remeshing every geometry (~100×
faster)**. The mesh-free wall-time argument does **not** hold; concede it. FEM is the gold standard for
fixed geometry. (The mesh-free value is vs NN methods and implementation simplicity, not vs FEM speed.)

## 6. Multi-parameter inverse — `multi_inclusion/inverse.py`, `inverse.png` (supporting demo, honest)

Recover `(cx,cy,r)` for 4 inclusions (12 unknowns) from interior data; sub-second forward (290 ms),
fixed-basis re-solve, no remeshing. **Local recovery:** 3/4 inclusions to ~1–2e-2, **noise-robust**
(0/0.1/1% noise give the same ~5e-2 mean — error is optimisation-limited, not noise-limited).
Honestly diagnosed as a hard, non-convex, ill-posed 12-D problem (exterior data is low-sensitivity —
consistent with known Calderón/EIT non-uniqueness; the cond~1e26 frame breaks finite-diff Jacobians).
Frame as a proof-of-concept *local* inverse, not a global solver.

---

## Honest scorecard (for deciding what to claim)

| claim | verdict |
|:------|:--------|
| Beats NN/PINN interface methods (DCSNN) | ✅ strongly (7–80×, single solve) — **lead with this** |
| Scales to many inclusions without remeshing | ✅ relL2 flat m=2→6 |
| Contrast-robust ρ∈[0.1,1000] | ✅ |
| Handles sharp features (gear) | ✅ banded multiresolution |
| Multi-parameter inverse | ⚠️ local only, 3/4 — proof of concept |
| Beats FEM on accuracy / speed / total cost | ❌ no — concede openly |
| Arbitrary high-dynamic-range solutions | ❌ s³/large-box fails — design-point limitation |

**Reproduce everything:** see the per-folder `results.md` (each script is self-contained, CPU, no GPU).
