# W-PINN for Integro-Differential & Nonlocal Problems — project plan (paper #2)

Status: PLAN (2026-06-19). Separate paper from the elliptic-interface work. Outer folder created;
no code yet. See memory `wpinn-gear-complex-test.md` / `wpinn-application-decision.md` for context.

## 1. Thesis (the one clean sentence)
A W-PINN solution is a fixed wavelet expansion `u = W c` whose operators are PRECOMPUTED matrices
applied to the coefficients `c` (AD-free). **For integral operators this AD-free property is a
genuine, not cosmetic, advantage: you cannot autodifferentiate an integral.** A precomputed wavelet
*integral* operational matrix encodes `∫K(x,t)u(t)dt` exactly as cheaply as it encodes `d²u/dx²` —
so integro-differential and nonlocal PDEs become the same convex linear solve we already use, while
standard (AD-based) PINNs must bolt on auxiliary quadrature points (the clumsy fPINN route).

This is the differentiator the interface paper deliberately did NOT use (there the selling point was
kink-resolution by decomposition). Here the selling point is **the operational matrix encodes the
integral** — distinct, defensible, and under-explored for PINNs.

## 2. Why this is whitespace (prior-art posture — confirm in Phase 0)
- **fPINN** (Pang/Lu/Karniadakis, SIAM JSC, arXiv:18M1229845): handles fractional/nonlocal via
  AUXILIARY quadrature points + AD. The pain point we remove. Main accuracy/cost competitor.
- **Op-matrix fPINN** (arXiv:2401.14081): precomputed Caputo operational matrix — but GLOBAL Legendre
  polynomials, standard MLP, no localized/multiresolution basis, no nonlocal-PDE / kernel-learning.
- **Peridynamic / nonlocal PINN** (Haghighat PDDO; inverse-peridynamics arXiv:2312.11316): uses the
  peridynamic differential operator as a conv filter or plain MLP — NOT a wavelet operational matrix,
  NOT AD-free coefficient solve.
- **Classical wavelet operational matrices for integral equations** (Haar/Legendre): no learning,
  no physics-informed framing, not for nonlocal PDEs.
→ Open lane = **localized multiresolution wavelet operational-matrix, AD-free, physics-informed,
  for integro-differential / nonlocal PDEs (+ inverse kernel/horizon recovery)**.

### PHASE 0 VERDICT (2026-06-19) — DONE, and it is a CAUTION, not a clean go
Focused prior-art (3 searches) shows the lane is MORE CROWDED than the interface lane was:
- **PINNIES** (arXiv:2409.01899, 2024): PINN for integral-operator problems via a PRECOMPUTED
  quadrature operator (tensor-vector product), no auxiliary points, Volterra+Fredholm+optimal
  control. This is most of the "operational matrix avoids aux points" claim already.
- **Rahimkhani 2025** (Wiley IJNM, jnm.70070): PINN for delay Hilfer FRACTIONAL DEs using a wavelet
  + operational matrix of integration. = wavelet operational matrix + PINN + integro/fractional.
- **EPINN (2025)** & **Advanced-PINN-with-residuals** (arXiv:2501.16370): continuous/complex integral
  equations with PINNs.
- **Peridynamic horizon recovery via PINN** (arXiv:2501.03911, 2025): learns the horizon/kernel
  (V-shaped/tent/distributed), 1D+2D — directly occupies our Phase-3 inverse-horizon headline.
- **Wavelets-based PINN** (Sci.Rep. 2023, PMC9938906): wavelet-activation PINN (occupies the naming).
Net: the headline "AD-free wavelet operational matrix encodes the integral" + "inverse horizon" is
LARGELY OCCUPIED. Remaining differentiators are NARROW: (a) the integral op-matrix here is ANALYTIC
for Gaussian-derivative wavelets (not quadrature, unlike PINNIES); (b) single-shot CONVEX LINEAR
solve (PINNIES/Rahimkhani still gradient-train); (c) localized multiresolution vs global polynomials.
These are incremental, not a clean new-application story. Mirrors the earlier FRACTIONAL rejection.

### PHASE 1 RESULT (2026-06-19) — DONE, mechanism fully validated
phase1_1d/run_phase1_ide.py: 1D Volterra IDE u'+\int_0^x u = 2e^x-1, exact u=e^x. The Volterra
integral operational matrix is ANALYTIC: \int_0^x psi_i = (1/j)(e^{-k^2/2}-e^{-(jx-k)^2/2}).
Single AD-free column-norm+Tikhonov LSQ solve. relL2: (0,1)=3.8e-8, (0,1,2)=5.3e-9,
(0,1,2,3)=1.8e-9, (0,1,2,3,4)=1.4e-10 (NF=68). Machine-accurate, sub-second. The TECHNICAL pieces
all work; novelty (not capability) is the issue.

### RECOMMENDATION (after Phase 0+1)
The mechanism is excellent but the lane is crowded. Two honest paths:
  (A) Proceed only if we lead HARD on the narrow differentiators (analytic op-matrix + single-shot
      convex linear solve + localized multiresolution) AND add a genuinely new applied PIDE; accept
      reviewer risk of "incremental over PINNIES/Rahimkhani".
  (B) PIVOT paper #2 to CRACK / FRACTURE singularity problems instead: r^{1/2} crack-tip singularity
      is exactly the localized sharp feature the gear's BANDED multiresolution refinement just
      proved W-PINN crushes (and vanilla PINNs fail). Cleaner whitespace, strong synergy with the
      gear result, applied NDT inverse angle. Recommended.

## 3. New machinery to build (the only genuinely new piece)
The **integral operational matrix** `G` with `G[m,i] = ∫ K(x_m, t) ψ_i(t) dt` for each collocation
point `x_m` and wavelet `ψ_i`. Then `(∫K u)(x_m) = (G c)_m` — a matrix-vector product, AD-free.
- Gaussian-derivative wavelets have semi-analytic integrals against common kernels (Gaussian,
  exponential, polynomial, weakly-singular `|x-t|^{-s}`) — precompute once per problem.
- Volterra (variable upper limit `∫₀ˣ`) vs Fredholm (fixed domain) → two builders.
- Everything else (value, derivative, Laplacian matrices; column-norm + Tikhonov LSQ solve; the
  coefficient network in Model.py) is REUSED from the interface repo.

## 4. Phased plan (same de-risking discipline that made paper #1 work)

**Phase 0 — Novelty lock (cheapest, gates everything).** One focused prior-art pass on
"wavelet operational matrix + PINN + integro-differential / nonlocal PDE + inverse kernel". Confirm
the specific combination is open before writing code. Binary go/no-go.

**Phase 1 — 1D integro-differential PoC (the analog of the 1D interface PoC).**
- Build & validate the Volterra integral operational matrix `G` (semi-analytic vs numerical quad).
- Benchmark IDE with exact solution, e.g. `u'(x) + ∫₀ˣ u(t)dt = f(x)`, `u=sin x` or `e^x`.
- Show AD-free linear solve → machine accuracy; report the full metric suite (relL2, MSE, RMSE,
  MAE, L∞, rel L∞) — same table format as the gear.
- Baseline: a quadrature-PINN (fPINN-style auxiliary points) — compare accuracy AND wall-clock /
  #operations. Expected headline: same/better accuracy, far cheaper, no auxiliary points.
- Add a Fredholm case + a weakly-singular kernel to show the matrix handles the hard kernels.

**Phase 2 — Nonlocal PDE (1D→2D) + W-PINN's proven sharp-feature strength.**
- Nonlocal diffusion / peridynamic-type: `∫ K_δ(x,y)(u(y)-u(x)) dy = f` (nonlocal Laplacian,
  horizon δ). Manufactured solutions for exact error.
- Show the operational matrix handles the nonlocal operator where AD-PINN needs dense auxiliary
  quadrature; sweep the horizon δ (the nonlocality strength) like we swept contrast ρ.
- If a boundary/interior layer appears, place wavelets densely there → reuse the documented SPP/NTK
  strength. 2D via tensor-product family (reuse Helmholtz/Maxwell/interface family).

**Phase 3 — Applied benchmark + inverse (venue-tier lift + the differentiator).**
- Applied PIDE: option pricing under JUMP-DIFFUSION (Merton / Kou) — a partial integro-differential
  equation, genuinely applied, less crowded than vanilla Black–Scholes. Alt: anomalous subdiffusion
  / fractional cable equation (neuroscience).
- INVERSE: recover the kernel parameter or peridynamic horizon δ (or kernel shape) from sparse
  interior data — reuse the outer-loop + fast-forward-solve strategy that worked for interface
  shape recovery. This is what classical quadrature solvers can't do natively and the headline
  novelty beyond "forward solver".
- Competitor table: quadrature-fPINN, op-matrix-fPINN, vanilla PINN.

## 5. Repo assets to reuse
Wavelet family + operational-matrix pattern (`Wfamily.py`); AD-free column-norm + Tikhonov LSQ solve
(interface notebooks); coefficient-emitting network (`Model.py`); NTK tooling (`SPP/SPP_Ex1_NTK/`);
canonical folder layout (config / Wfamily / Model / `WPINN_*.ipynb` / sol.png + results.md + table.png).

## 6. Named competitors to beat
fPINN (quadrature aux points), op-matrix fPINN (global Legendre), peridynamic-PINN (PDDO/MLP),
vanilla PINN. Differentiator = AD-free *localized multiresolution* wavelet integral operational
matrix + mesh-free + fast inverse kernel/horizon recovery.

## 7. Biggest risks
- Novelty (Phase 0) — cheapest to check, do first.
- Weakly-singular kernels: the integral operational matrix entries need careful (semi-analytic /
  singularity-subtracted) quadrature — the main technical risk in Phase 1/2.
- Conditioning at strong nonlocality / fine kernels — expect the same column-norm + Tikhonov fix;
  watch it as we did for high ρ.
- Nonlinear IDEs would break the single-shot linear solve → outer Newton/Picard with the same
  matrices (note as extension, not v1).
