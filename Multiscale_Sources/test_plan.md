# Test & benchmark plan — Multiscale-Sources W-PINN (CMAME)

Due-process roadmap of the experiments needed to support the CMAME claims. NOTHING here is run yet; each
entry states **goal / setup / metric / pass criterion / which claim it backs / priority / depends-on /
status**. Claims (A)(B)(C) and the novelty are defined in `cmame_novelty.md`; results so far in
`results.md`; plan in `multiscale_sources_wpinn_plan.md`.

Legend — Priority: **P0** venue-deciding · **P1** core paper · **P2** strengthens · **P3** nice-to-have.
Status: TODO / PARTIAL / DONE.

---

## A. A posteriori error control (backbone of claim A)

### T-A1 — Measure the stability constant C_stab = ‖error‖ / ‖residual‖   [P0, TODO]
- **Goal:** turn the a posteriori bound `‖u−u_h‖ ≤ C_stab·‖residual‖` from "diagnosed" to "demonstrated".
- **Setup:** 1D and 2D screened-Poisson; for a sweep of (Tikhonov tik, RRQR filtering on/off, basis size)
  record solution error AND PDE+BC residual; compute the ratio C_stab and the smallest singular value of
  the (filtered) operator matrix.
- **Metric:** C_stab vs tik; C_stab vs RRQR threshold; correlation of C_stab with 1/σ_min (inf-sup proxy).
- **Pass:** C_stab is finite and *controlled* — shrinks as RRQR removes the near-null space and as tik is
  tuned; the residual indicator is shown RELIABLE (error ≤ C·residual) and EFFICIENT (residual ≤ c·error).
- **Backs:** claim A. **Depends:** rrqr_2d.py, diagnose_solve.py (exist).

### T-A2 — Residual indicator drives refinement to a target tolerance (reliability in practice)   [P1, TODO]
- **Goal:** show the greedy stopping rule (residual < tol) yields the intended solution accuracy.
- **Setup:** run the greedy to several residual tolerances; plot achieved relL2 vs residual indicator.
- **Metric:** monotone, near-linear error–residual relationship across tolerances/problems.
- **Pass:** residual indicator predicts error within a stable constant across cases (1D done implicitly in
  greedy_residual.py; needs explicit plot + 2D).
- **Backs:** claim A. **Depends:** T-A1, greedy_residual.py.

### T-A3 — Solution-selection regularizer vs plain Tikhonov   [P2, TODO]
- **Goal:** close the residual→error gap toward the 2e-2 best-fit floor (results.md §2a).
- **Setup:** compare plain Tikhonov, column-norm Tikhonov, and a fit-/norm-anchored penalty (e.g. small
  data-fit anchor or weighted-ℓ2/ℓ1 on coefficients) on the 2D 4-bump case.
- **Metric:** relL2 floor reached; C_stab improvement.
- **Pass:** a regularizer reaches ≤ 2–3e-2 in 2D (matching best-fit), beating plain Tikhonov's 4.5e-2.
- **Backs:** claim A (+ honest 2D story). **Depends:** T-A1.

---

## B. Approximation rate / the moat vs random features (claim B)

### T-B1 — Best-N-term convergence of the greedy   [P1, TODO]
- **Goal:** empirical best-N-term rate — error vs N (# selected atoms) on a log-log plot.
- **Setup:** 1D and 2D; greedy refinement; record error vs N; fit the algebraic rate; compare to the
  theoretical wavelet N-term rate for the manufactured solution's regularity (Besov class).
- **Metric:** observed exponent in error ~ C·N^(−s); agreement with predicted s.
- **Pass:** greedy achieves (near-)optimal N-term rate; uniform refinement is provably worse (shown).
- **Backs:** claim B. **Depends:** greedy in 2D (T-D1).

### T-B2 — Greedy vs single-scale random features at matched N, scale-separation sweep   [P0, PARTIAL]
- **Goal:** the rigorous "beats RBF/Fourier-PIELM" result — adaptive multiresolution wins at every scale.
- **Setup:** sweep feature width w of a single localized feature on a smooth background; compare greedy
  (auto) vs best-of-many-widths RBF-PIELM AND Fourier-feature PIELM at matched N_F.
- **Metric:** relL2 vs w for each method.
- **Pass:** greedy beats both random-feature methods across the whole w-range; the random methods saturate
  as scales separate (1D RBF case already shown — decisive_scaletest.py; ADD Fourier-feature baseline + 2D).
- **Backs:** claim B. **Depends:** none (extend existing 1D sweep).

---

## C. Forward solver — harder regimes (robustness; claim B support)

### T-C1 — Many features, unknown count & disparate scales   [P1, TODO]
- **Goal:** scale the headline beyond 1–4 features.
- **Setup:** 1D and 2D with m = 1,2,4,8,16 localized features at random locations and widths spanning
  ≥ one decade; greedy auto-refine (no locations given).
- **Metric:** relL2 and N_F vs m; does greedy localize ALL features?
- **Pass:** accuracy flat / mildly growing with m; greedy finds every feature (recall ≈ 1).
- **Backs:** claims A,B; the "unknown count" selling point. **Depends:** T-D1 for 2D.

### T-C2 — Helmholtz / higher wavenumber κ   [P2, TODO]
- **Goal:** push past the mild κ=1 screening into oscillatory regime (stronger vs spectral-bias methods).
- **Setup:** sweep κ ∈ {1, 5, 10, 20, 40}; multiscale RHS; check resolution per wavelength.
- **Metric:** relL2 vs κ; points-per-wavelength needed; comparison to vanilla PINN (spectral bias) at each κ.
- **Pass:** method holds accuracy where vanilla PINN degrades with κ; document any κ ceiling honestly.
- **Backs:** competitive positioning. **Depends:** none.

### T-C3 — Sensitivity / ablation of the greedy hyperparameters   [P2, TODO]
- **Goal:** show the method is not fragile.
- **Setup:** vary batch size per greedy step, candidate level cap Lmax, start-level set, collocation density,
  tik; on a fixed multiscale case.
- **Metric:** relL2 spread; cost (N_F, wall-time).
- **Pass:** results stable across reasonable ranges; recommend default settings.
- **Backs:** reproducibility / reviewer trust. **Depends:** T-D1.

---

## D. 2D consolidation (lift 2D from "competitive" to "clean")

### T-D1 — Greedy refinement in 2D with proper (1e-6) regularization + RRQR backend   [P0, TODO]
- **Goal:** carry the 1D headline into 2D now that the ceiling is diagnosed as regularization, not basis.
- **Setup:** 2D 4-bump case; residual-greedy selection (operator dictionary) with tik≈1e-6, RRQR-filtered
  solve as the backend; spike/bump locations NOT given.
- **Metric:** relL2 vs N_F; localization of fine atoms at the bumps; vs vanilla PINN (3.5e-2) and oracle band.
- **Pass:** greedy ≤ ~3–4e-2 in 2D, auto-localizing all bumps, at fewer functions than the oracle.
- **Backs:** claims A,B; closes the honest 2D gap. **Depends:** greedy_residual.py (1D), rrqr_2d.py.

### T-D2 — Strong-scale-separation 2D (the decisive multiresolution win)   [P1, TODO]
- **Goal:** the clean 2D analogue of the 1D decisive result.
- **Setup:** 2D with widths spanning a full decade (e.g., 0.4 and 0.02 simultaneously); greedy vs RBF-PIELM
  vs vanilla PINN at matched budget.
- **Metric:** relL2 for each; cost.
- **Pass:** greedy decisively wins as separation grows (mirrors 1D decisive_scaletest.py).
- **Backs:** claim B. **Depends:** T-D1.

---

## E. Inverse problem — the venue-deciding headline (claims A + unification + C)

### T-E1 — Recover known-count sources from sparse/boundary data   [P0, TODO]
- **Goal:** baseline inverse: locations/strengths of a KNOWN number of localized sources from sparse
  interior or boundary measurements, via the fast forward solve + sparse recovery in the coefficients.
- **Setup:** generate data from a forward solve; recover coefficients by sparsity-promoting LSQ; map to
  source locations/strengths.
- **Metric:** location error, strength error, vs # measurements.
- **Pass:** accurate recovery with modest data; noise-free.
- **Backs:** unification + claim C. **Depends:** T-D1; reuse interface inverse outer-loop machinery.

### T-E2 — UNKNOWN number/scale of sources (the differentiator)   [P0, TODO]
- **Goal:** the claim no competitor can match — recover an a-priori-unknown COUNT and SCALE via the SAME
  residual-greedy selector used in the forward solve.
- **Setup:** data from k sources (k hidden); greedy/sparse model selection over count & scale; report how
  many found vs true.
- **Metric:** detection (precision/recall on count), location/scale/strength error; model-selection curve.
- **Pass:** correct count recovered across k; graceful behavior at over/under-estimation.
- **Backs:** the headline. **Depends:** T-E1.

### T-E3 — Noise robustness & identifiability of the inverse   [P1, TODO]
- **Goal:** honest robustness characterization (the real technical risk per the plan §7).
- **Setup:** add 0.1/1/5/10% noise to measurements; vary sensor count/placement; regularization path.
- **Metric:** recovery error vs noise; failure modes; an identifiability/ill-posedness discussion.
- **Pass:** stable to ~1% noise; degradation characterized, not hidden.
- **Backs:** claim C credibility. **Depends:** T-E2.

### T-E4 — Sources embedded in a heterogeneous (interface) medium   [P2, TODO]
- **Goal:** the CMAME engineering anchor — load/defect/source identification behind piecewise-coefficient
  inclusions (fuses paper #1 interface machinery).
- **Setup:** interface medium + hidden localized sources; recover sources through the heterogeneity.
- **Metric:** recovery accuracy with the interface present vs absent.
- **Pass:** works through heterogeneity; demonstrates the less-crowded niche.
- **Backs:** claim C. **Depends:** T-E2, interface machinery.

---

## F. Competitor & reference benchmarks (the comparison table)

### T-F1 — Baseline matrix at matched budget   [P1, PARTIAL]
- **Methods:** greedy W-PINN (ours) · vanilla PINN (Adam) · RBF-PIELM (best width) · Fourier-feature PIELM ·
  uniform-refinement wavelet (ablation) · (where applicable) singularity-enrichment.
- **Metric:** relL2, L∞, N_F / #params, wall-time, GPU? — one table per problem (1D, 2D moderate, 2D
  strong-sep, inverse).
- **Pass:** ours leads on multiscale/unknown-scale cases at no-training/CPU cost; concede where it doesn't.
- **Backs:** positioning. **Depends:** T-D1, T-D2, T-E2. (vanilla PINN + RBF-PIELM already in run_phase2.py.)

### T-F2 — FEM reference for forward accuracy/cost (honest concession)   [P2, TODO]
- **Goal:** state the honest FEM position (per [[wpinn-method-limits]]: FEM wins on fixed smooth geometry).
- **Setup:** adaptive-mesh FEM on the multiscale forward problem; compare accuracy/cost.
- **Metric:** relL2 vs dof and wall-time.
- **Pass:** report truthfully; emphasize mesh-free + inverse where the method's value is.
- **Backs:** honesty / reviewer trust. **Depends:** scikit-fem/gmsh (present).

---

## G. Reproducibility & paper artifacts   [P1, TODO]
- Deterministic seeds; results.json per experiment; one figure per claim (greedy.png exists for the 1D
  headline); a single `run_all` script; environment/version note.
- Convergence/error data archived; tables auto-generated (make_table.py pattern).

---

## Suggested execution order (critical path to CMAME)
1. **T-A1** (C_stab measurement) — backbone of claim A; cheap, uses existing scripts.   [P0]
2. **T-D1** (2D greedy + RRQR + proper reg) — closes the 2D gap; unblocks most 2D/inverse tests.   [P0]
3. **T-E1 → T-E2** (inverse, known then unknown count) — the venue-deciding headline.   [P0]
4. **T-B2 + T-D2** (rate/competitor sweeps) — the moat vs random features.   [P0/P1]
5. **T-A2/T-A3, T-B1, T-C1** — solidify claims A,B.   [P1]
6. **T-E3, T-E4, T-F1/T-F2, T-C2/T-C3, G** — robustness, application, comparison, reproducibility.   [P1/P2]
