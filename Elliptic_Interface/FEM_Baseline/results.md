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

So the answer to "is it competitive?" is: **yes, in the mesh-free / geometry-flexible regime;
no, as a raw accuracy/speed replacement for FEM on a fixed smooth interface.**

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
