# FEM gold-standard baseline + standard-benchmark check

Head-to-head of the AD-free wavelet W-PINN against **conforming FEM** (the mesh-based gold standard)
on the *identical* circular elliptic-interface problem, and a check on a standard published benchmark.

## Problem (identical for both solvers)

`−∇·(a∇u)=f` on Ω=(−1,1)², circle Γ={r=0.5}, `u=φ/a` (`φ=x²+y²−0.25`), `f=−Δφ=−4`, homogeneous
jumps `[[u]]=0`, `[[a∂ₙu]]=0`, Dirichlet = exact `u` on the square. Conforming FEM: P1 on a gmsh mesh
whose element edges lie **on** Γ (so both interface conditions hold for free). Wavelet: global-coarse
block basis, one AD-free least-squares solve.

## Accuracy + wall-time vs contrast

**Conforming FEM (P1), `fem_circle.py`** — second-order, contrast-robust:

| ρ | h | nDoF | relL2 | L∞ | solve_s |
|--:|--:|-----:|------:|----:|--------:|
| 10 | 0.100 | 545 | 1.62e-03 | 1.10e-03 | 0.013 |
| 10 | 0.050 | 2005 | 2.45e-04 | 2.80e-04 | 0.005 |
| 10 | 0.025 | 7692 | 4.57e-05 | 7.52e-05 | 0.026 |
| 10 | 0.0125 | 30201 | **8.17e-06** | 1.82e-05 | 0.167 |
| 1000 | 0.0125 | 30201 | 1.08e-05 | 1.81e-05 | 0.160 |

**Wavelet W-PINN (`wavelet_circle.py`)** — same problem:

| ρ | NF | relL2 | L∞ | build+solve_s |
|--:|---:|------:|----:|--------------:|
| 10 | 2623 | 1.36e-03 | 5.61e-04 | 7.6 |
| 1000 | 2623 | 1.20e-03 | 5.47e-04 | 7.3 |

## The honest verdict (read before writing the paper)

**On a single fixed smooth interface, conforming FEM dominates** — it matches the wavelet's accuracy
(1.6e-3) at **545 dof in 0.013 s**, and reaches **8e-6** with refinement; the wavelet sits at ~1.3e-3
for any contrast at NF≈2600 in ~7.5 s. The wavelet method is **not** accuracy- or speed-competitive
with FEM on fixed smooth geometry, and the paper should **not** claim so.

The wavelet method's competitive niche is **mesh-freeness**, where FEM pays a cost the table above
hides (it reports only the linear solve, not meshing):

1. **Moving / many inclusions** — FEM must generate a conforming mesh per geometry; the wavelet just
   relocates blocks (no remeshing). See `../Multi_Inclusion` (m=2..6, flat error, no meshing).
2. **The inverse** — geometry optimisation does *hundreds* of forward solves at *changing* geometry;
   FEM would remesh every iteration. The wavelet forward is a sub-second re-solve on a fixed basis.
3. **Sharp features** — the gear's tooth cusps need local mesh refinement for FEM; the wavelet uses a
   banded multiresolution basis (no mesh).

## Total-cost test for geometry-varying problems (`cost_comparison.py`) — the honest verdict

We tested the strongest pro-mesh-free claim directly: a sweep over **12 geometries** (circle of varying
radius, as an inverse/design loop would visit), FEM **remeshing every time** vs the wavelet method
precomputing its matrix block once and re-solving. Matched ~1.5e-3 accuracy:

| | total (12 geoms) | per-geometry | relL2 |
|:--|--:|--:|--:|
| FEM (conforming remesh each) | **0.58 s** | 0.048 s (0.042 s of it meshing) | 1.5e-3 |
| wavelet (precompute + re-solve) | 68 s | 2.0 s once + 5.5 s/geom | 7.0e-3 |

**FEM is ~100× faster even while remeshing every geometry — and more accurate.** Conforming meshing of
a circle is trivial for gmsh (0.04 s); the wavelet SVD solve (5.5 s) is the bottleneck. **The mesh-free
method does NOT win on wall-time, even for geometry-varying problems.** Do not make that claim.

## So where does the wavelet method actually win? (honest positioning)

**Not against FEM.** FEM beats it on accuracy, on per-solve speed, and on total cost even with remeshing,
for these smooth-interface problems. The paper should NOT position the method as an FEM competitor.

**Against other mesh-free / neural methods — decisively.** On the DCSNN benchmarks above it is ~7×–80×
more accurate than the leading mesh-free NN, with a **single linear solve (no iterative training, no
GPU, no hyper-parameter tuning)**. That is the genuine contribution: *a training-free, mesh-free
least-squares solver that outperforms PINN/neural interface methods*. Frame the whole paper there.

## Published-benchmark head-to-head — DCSNN Example 1 (`dcsnn_ex1.py`)

Reproduces **Hu, Lin & Lai, *JCP* 2022, Example 1** exactly: ellipse `(x/0.2)²+(y/0.5)²=1`, contrast
**β⁻/β⁺ = 1000** (β⁻=1, β⁺=1e-3), inhomogeneous jumps, exact `u⁻=eˣeʸ`, `u⁺=sin x sin y`.

| method | L∞ error | relL2 | "dof" | how |
|:-------|---------:|------:|------:|:----|
| **wavelet W-PINN (this work)** | **6.04e-07** | 7.44e-07 | NF=5002 | one AD-free linear lstsq, 98 s CPU |
| wavelet W-PINN (finer) | 3.64e-07 | 3.00e-07 | NF=12303 | one lstsq, 1216 s |
| DCSNN (Hu-Lin-Lai 2022) | 4.39e-06 | — | 101 params | gradient-descent training |
| IIM (their baseline) | ~comparable | — | ~65,792 | mesh-based |

**DCSNN Example 2** (`dcsnn_ex2.py`) — **non-convex flower** interface `r=½+sin(5θ)/7`, β⁻/β⁺=10/1,
`u⁻=e^(x²+y²)`, `u⁺=0.1(x²+y²)²−0.01·log(2r)`:

| method | relL2 | L∞ | "dof" |
|:-------|------:|----:|------:|
| **wavelet W-PINN (this work)** | **3.20e-06** | 9.25e-06 | NF=5002 (113 s) |
| DCSNN (Hu-Lin-Lai 2022) | 2.63e-04 | — | 501 params (trained) |

→ **~80× more accurate than DCSNN** on a non-convex interface, single linear solve.

**The wavelet method beats the leading mesh-free-NN method (DCSNN) on BOTH its 2D benchmarks (ellipse
~7× in L∞, flower ~80× in relL2), using a single linear solve (no training, no GPU)**, at far fewer
unknowns than the IIM mesh.
Honest caveat: DCSNN uses only 101 parameters (far more parameter-efficient); the wavelet trade is
a larger but still modest basis in exchange for being non-iterative/AD-free. This O(1)-amplitude,
contrast-1000 single-interface case is solved to 6e-7 — so the ρ≤0.01 degradation seen in the
5-inclusion sweep is a multi-inclusion *conditioning* effect, not an intrinsic high-contrast limit.

## Standard-benchmark check

- **Quadratic-solution circle** (`u=φ/a`, the Li–Ito-style manufactured interface): relL2 **1.3e-3**,
  i.e. the wavelet method reproduces a standard interface benchmark at the level of a coarse FEM mesh.
- **IFE `s³` benchmark** (`benchmark_ife.py`, circle (0.5,0.5) r0=0.4, `u=(r²−r₀²)³/β`): **fails on our
  (−1,1)² box** (relL2 ~ O(1)). Honest cause: the IFE papers pose it on the *unit* square where the
  circle fills the domain and `u` is O(1); on the larger box `u=s³~r⁶` reaches ~80 at the corners — a
  dynamic range the *decaying* wavelet basis cannot represent. The method is built for O(1)-amplitude
  interface solutions; high-degree polynomial growth on a large box is outside its design point.
  A faithful match needs the solver re-posed on (0,1)² (TODO).

Reproduce: `python fem_circle.py` · `python wavelet_circle.py` · `python benchmark_ife.py`
(requires `gmsh`, `meshio`, `scikit-fem`).
