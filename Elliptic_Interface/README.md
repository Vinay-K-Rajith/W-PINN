# Elliptic interface — AD-free wavelet least-squares

Training-free, mesh-free wavelet least-squares solver for `−∇·(a∇u)=f` with coefficient jumps
across an interface. One Gaussian-derivative wavelet expansion per subdomain (`u=Wc+b`); interface
and boundary conditions assembled into a **single** column-normalised Tikhonov SVD solve (`gelsd`) —
no training loop, CPU, float64. Every script is self-contained and runs without a GPU.

## Layout

| folder | paper section | contents |
|:-------|:--------------|:---------|
| `docs/` | — | `COMBINED_RESULTS.md` (all numbers), `HOW_TO_WRITE_THE_PAPER.md`, `VIVA_QA.md`, `paper_intro_methodology.md`, elsarticle template |
| `forward_benchmarks/` | §3.1 (headline) | head-to-head vs DCSNN (Hu–Lin–Lai, JCP 2022): `dcsnn_ex1.py`, `dcsnn_ex2.py` — beats both with one linear solve |
| `gear/` | §3.2 | sharp-feature showcase: `run_gear.py` (+ `baselines_gear.py`, `convergence_study.py`, `helmholtz_robustness.py`, table makers) |
| `multi_inclusion/` | §3.3, §5 | topology scaling + contrast sweep (`multi_inclusion.py`, `rho_sweep.py`, `studies.py`) and the inverse demo (`inverse.py`) |
| `fem_reference/` | §4 | honest FEM comparison: `fem_circle.py`, `wavelet_circle.py`, `cost_comparison.py`, `benchmark_ife.py` |
| `_archive/` | — | superseded scaffolding (1D→2D→inverse dev phases; original torch/notebook versions) — kept for provenance, not used by the paper |

## Positioning (read `docs/COMBINED_RESULTS.md` first)

The method's win is **against neural-network / mesh-free PINN methods** (it beats them with one linear
solve). It is **not** a competitor to FEM — FEM is faster and more accurate on fixed geometry, conceded
openly in §4. Build the paper on the first claim.

## Reproduce

Each script is standalone (CPU, float64). Cross-folder scripts in `forward_benchmarks/` and
`fem_reference/` import the shared assembly from `multi_inclusion/multi_inclusion.py`.
