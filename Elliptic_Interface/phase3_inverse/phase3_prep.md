# Phase 3 prep — Inverse elliptic-interface problems with W-PINN

> The publication differentiator. Forward interface solvers are crowded; **recovering the coefficient
> jump or the interface location/shape from sparse interior data** is something classical IIM/FEM do
> not do natively and most PINN-interface papers do not demonstrate. This is the result that moves the
> paper from CAMWA-tier toward CMAME/JCP-tier (see probability tables in the project notes).

## 1. Problem statements (rising novelty)

**(A) Inverse contrast** — geometry (Γ) known, `a^-,a^+` unknown.
Given sparse, possibly noisy measurements `{u(x_i^data)}`, recover `a^±` (equivalently ρ=a^+/a^-).
- Operational matrices stay fixed ⇒ precompute-once survives.
- Caveat: the residual `a^±·(D2 c^±)` is *bilinear* in (a^±, c^±) ⇒ no longer linear least squares ⇒ joint
  iterative solve (Adam over coefficients + the scalar a^±), or alternating LSQ/parameter steps.
- Identifiability: the data misfit anchors the otherwise-ambiguous scale; needs ≥ a few interior points.

**(B) Inverse interface location/shape** — `a^±` known, Γ unknown (THE headline).
Parametrize Γ by few parameters θ_Γ (circle: center+radius; flower: r0, ε, rotation; general: low-order
Fourier shape coefficients). Recover θ_Γ from sparse interior data.
- Γ controls point classification (which residual uses a^- vs a^+) and interface-point locations ⇒ the
  matrices partially depend on Γ; precompute-once partially breaks.
- **Outer-loop strategy (leverages the fast forward solver):** for each candidate θ_Γ, run the AD-free
  least-squares forward solve (~0.4 s) and evaluate data misfit J(θ_Γ)=‖u_pred−u_data‖². Minimize J over
  θ_Γ with a derivative-free optimizer (Nelder–Mead / CMA-ES / coarse grid → local refine). A few hundred
  forward solves = minutes. The cheap forward solve is what makes this tractable.

**(C) Joint** — both a^± and Γ unknown (stretch; nest A inside B's outer loop).

## 2. Experimental design

- **Synthetic data:** sample the manufactured exact solution at N scattered interior points; add Gaussian
  noise σ (relative). Sweep N ∈ {5,10,20,50} and σ ∈ {0, 1e-3, 1e-2}.
- **Metrics:** parameter recovery error |ρ̂−ρ|/ρ and |θ̂_Γ−θ_Γ|; reconstructed-field rel-L²; robustness
  curves vs N and σ; cost (forward solves / wall-clock).
- **Pre-committed success:** recover ρ to <1% from N≤20 noiseless points (A); recover interface radius/shape
  to <2% from N≤50 points with 1% noise (B). Demonstrate graceful degradation with noise.

## 3. Plan
1. **(A) PoC** in 1D — joint Adam over (a^+, coefficients) + data misfit. Validate the inverse machinery cheaply. → `phase3_inverse_contrast_1d.py`
2. **(A) 2D** — inverse contrast on the circle benchmark.
3. **(B) 2D** — interface-radius recovery via outer-loop forward solves; then flower shape (r0, ε).
4. **(C)** joint, and noise/robustness sweeps. Figure: recovery vs N vs σ.
5. Write-up: position as the capability classical interface solvers lack.

## 4. Risks
- **Bilinearity (A):** joint Adam may be slow/local-min-prone; mitigate with alternating LSQ-in-c / step-in-a,
  or a good initial guess from a coarse ρ-grid.
- **Discontinuous reclassification (B):** point in/out assignment is non-smooth in θ_Γ ⇒ derivative-free outer
  loop (not gradient) is the robust choice; J(θ_Γ) is smooth enough for Nelder–Mead despite this.
- **Identifiability/noise:** sparse + noisy data may be ill-posed; report the N–σ frontier honestly, add mild
  regularization/priors on θ_Γ if needed.
- **Conditioning:** same caveat as Phase 2; keep the parsimonious basis.
