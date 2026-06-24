# W-PINN for Multi-Scale Localized-Feature Problems — project plan (paper #2, chosen)

Status: PLAN + Phase 0 (novelty) + Phase 1 (PoC) DONE (2026-06-19). Replaces the integro-differential
idea (Integro_Differential/, parked — mechanism works but lane crowded; see
wpinn-paper2-integro-verdict memory). Separate paper from the elliptic-interface work (paper #1).

## 1. The honest landscape (why THIS, after extensive search)
The 2025-26 PINN/PIELM field is SATURATED for single-feature linear problems: point sources
(arXiv:2111.01394), corner singularities + point sources via ENRICHMENT (SEPINN, Jin & Zhou,
arXiv:2308.16429), fracture (XPIELM, 10^-12, CMAME 2025), biharmonic (RBF/Augmented-PIELM,
arXiv:2310.13947), eigenvalues (Eig-PIELM 2025), inverse source (acoustic 2024, parabolic 2025).
W-PINN's single-shot LINEAR solve is mechanically a COUSIN of PIELM, so "linear solve for X" and
"enrichment for singularity X" are structurally occupied.

**The one durable, hard-to-contest edge = genuine MULTIRESOLUTION ADAPTIVITY:** resolving features
at MULTIPLE, a-priori-UNKNOWN scales/locations simultaneously. This defeats:
  * vanilla PINN  -> spectral bias (cannot fit fine localized features);
  * RBF / Fourier PIELM -> single-scale random/global features, documented to "degrade on
    oscillatory/localized solutions";
  * enrichment (SEPINN/XPIELM) -> needs the ANALYTIC singular form AND known count/location.
So the application must make multi-scale, multi-feature, unknown-count resolution the ESSENTIAL
difficulty. That is exactly what wavelets are for, and what the gear banded-refinement result proved.

## 2. Thesis (one sentence)
An AD-free multiresolution wavelet W-PINN resolves elliptic/Helmholtz fields containing localized
features at DISPARATE, UNKNOWN scales via ADAPTIVE BANDED refinement (finest wavelets placed only
where/at the scale needed), and -- because the forward solve is a single fast linear system --
RECOVERS an UNKNOWN NUMBER of localized sources/loads at unknown scales from boundary/sparse data
(a sparse multiresolution inverse problem that enrichment- and single-scale methods cannot do).

## 3. Phase 0 VERDICT (2026-06-19) — defensible, with a sharp differentiator
Forward multi-scale solving is partly covered (multiscale is the source paper's, arXiv:2409.11847,
broad turf). The OPEN, defensible niche is the combination:
  (a) MULTIPLE localized features across DISPARATE scales at once (not one source/singularity);
  (b) UNKNOWN NUMBER & SCALE of features -> recovered by SPARSITY in the multiresolution coefficients
      (enrichment needs known count/form -> cannot; single-scale PIELM cannot);
  (c) AD-free fast forward solve -> tractable derivative-free inverse over feature count/scale.
Applied anchor: source localization / load identification / defect detection where the NUMBER of
sources is unknown. Differentiator vs all named competitors is (b)+(a): adaptivity + sparsity, not
"solve one PDE".

## 4. Phase 1 RESULT (2026-06-19) — PASS, and it demonstrates the edge
phase1_1d/run_phase1_multiscale.py: 1D screened Poisson -u''+u=f, exact u=sin(2 pi x)+exp(-((x-0.5)
/0.02)^2) (smooth O(1) background + narrow w=0.02 spike = two disparate scales). AD-free wavelet
operational-matrix LSQ solve. relL2:
  coarse [0,1,2,3]      (NF=35)  -> 1.16e-1  (cannot resolve the spike)
  uniform fine [0..5]   (NF=133) -> 2.51e-2  (expensive, still poor)
  [0-3]+band[6,7,8]@spike(NF=124)-> 4.29e-4  (270x better than coarse, 58x better than uniform-fine,
                                              with FEWER functions). 
=> adaptive banded multiresolution refinement is the cheap, decisive fix -- the durable edge, in 1D.

## 4b. PHASE 2 RESULTS (2026-06-19) — thesis PROVEN in 1D; honest limits in 2D
**1D decisive scale-separation sweep (the rigorous competitor proof)** — phase1_1d/decisive_scaletest.py,
decisive.png, table_decisive.png. Banded wavelet W-PINN vs RBF-PIELM (best of 6 widths, MATCHED N_F),
single sharp spike of width w on a smooth background, sweep w:
  w=0.20: 3.2e-12 vs 6.7e-5 (5e6x) ; w=0.10: 7.4e-11 vs 4.2e-4 ; w=0.05: 2.1e-8 vs 1.1e-2 ;
  w=0.02: 6.5e-5 vs 1.5e-1 (2340x) ; w=0.01: 2.8e-3 vs 1.4e-1 (48x) ; w=0.005: 3.0e-2 vs 1.0e-1 (3.4x).
=> adaptive multiresolution beats fixed-width random features at EVERY scale; RBF-PIELM saturates at
10-20% as scales separate (a fixed width + random centres cannot resolve spike AND background at a
fixed budget). THIS is the clean, controlled headline result.

**2D screened-Poisson, 4 bumps (widths 0.08..0.20), phase2_2d/run_phase2.py** (sol.png, compare.png,
baselines.png, table.png, results.json): coarse [0,1,2] 1.99 -> medium [0,1,2,3] 0.23 -> banded
[0,1,2,3]+[4,5]@bumps 0.10 (banded ablation works); banded (10%) < RBF-PIELM best-width (16%) -- thesis
scales -- but vanilla PINN (4k Adam, 20x slower) reaches 3.5% on this MODERATE field.
HONEST 2D LIMITS (important): (i) 2D accuracy ceiling ~1e-1 from operational-matrix CONDITIONING --
adding levels 6+ DEGRADES the solution (48%), parsimony is the lever (same as interface Phase 1), not
more levels; (ii) Gaussian bumps are the ideal case for Gaussian-RBF, an unusually strong competitor
for that shape; (iii) at MODERATE 2D separation a trained MLP competes. The decisive multiresolution
win needs STRONG scale separation (proven cleanly in 1D).

## 4c. PHASE 2 — AUTOMATIC residual-greedy refinement (2026-06-24) — the sharpened headline
After a fresh lit pass (June 2026), "wavelet multiresolution for multiscale" is now CROWDED (source
paper arXiv:2409.11847 + PIMWNN CPC 2025 S0010465525004874 + wavelet-quantum-PINN arXiv:2512.08256),
and "sparse multiresolution inverse source" overlaps mature weighted-sparsity source-ID work
(arXiv:2206.06069/2212.04187/2012.11280) and CORSING wavelet compressed-sensing PDE solves. The
manual banded refinement (place finest wavelets where you ALREADY know the spike is) is the weak point
a reviewer attacks and the inverse problem forbids. THE FIX = make refinement AUTOMATIC and residual-
driven: matching pursuit / weak-greedy on the PDE-OPERATOR dictionary -- score each candidate wavelet
by |a_i . r|/||a_i|| (a_i = operator column, r = PDE residual), add top-batch, re-solve. AD-free makes
this cheap (residual + all correlations are matrix products of pre-stored operator columns). THEORY
MOAT vs RFM/PIELM: structured multiresolution dictionary => best-N-term / weak-greedy a-posteriori
optimality, which single-scale random features provably lack.

RESULT (phase1_1d/greedy_residual.py, greedy.png, greedy_results.json) -- SAME 1D problem, spike
location x0 NOT given to the selector:
  coarse [0,1,2,3]            NF= 35  relL2 1.16e-1
  uniform fine [0..5]         NF=133  relL2 2.51e-2
  oracle band[6,7,8]@spike    NF=124  relL2 4.29e-4   (MANUAL, knows x0)
  GREEDY (auto, no x0)        NF=113  relL2 9.36e-6   <-- BEATS the oracle ~46x at FEWER functions
The greedy DISCOVERS the spike: finest level (l=8) atoms are 67% within 0.1 of x0, l=7 58%, l=6 24%
(a refinement pyramid centred on x0 -- greedy.png panel b). This is the paper's Figure 1: automatic
residual-driven multiresolution refinement, no a-priori location, beating the manual oracle.
NEXT: (i) carry into 2D, where the conditioning ceiling (~1e-1) blocks deep levels -> add RRQR local-
feature filtering + preconditioning (arXiv:2506.17626, cond -11 orders) so the greedy can use fine
levels in 2D; (ii) the SAME residual-greedy selector becomes the inverse source-count/location detector.

2D CEILING DIAGNOSED (2026-06-24, phase2_2d/diagnose_*.py + rrqr_2d.py, results.md §2):
the old ~0.10 ceiling is a residual-to-error STABILITY/REGULARIZATION effect, NOT the basis (best-fit
L2-projection reaches 2e-2, even 7e-3 with Gaussian atoms), NOT raw conditioning (RRQR cuts cond -9
orders 3.5e17->4e8 with NO accuracy gain, but HALVES the basis = efficiency win), and NOT collocation
aliasing (denser grid monotonically WORSENS: 4.5e-2 @K120 -> 9.4e-2 @K220). The lever is Tikhonov
strength: tik=1e-6 + modest K gives relL2 4.48e-2 -- below the old 0.10 and COMPETITIVE with the trained
vanilla PINN (3.5e-2) via one linear solve. Open refinement: a solution-selection regularizer (norm-/fit-
anchored) to reach the 2e-2 best-fit floor. So 2D is no longer a clear loss; it's competitive-with-PINN.

## 5. Phased plan
- **Phase 2 — forward, harder:** (i) MANY sources at several scales in 1D/2D; (ii) 2D screened
  Poisson / Helmholtz with multi-scale localized loads; (iii) ablation vs uniform refinement and a
  vanilla-PINN / RBF-PIELM baseline (show their multiscale failure); reuse tensor-product family.
- **Phase 3 — inverse (the headline):** recover an UNKNOWN NUMBER of sources at unknown
  locations/scales from boundary/sparse interior data, via sparsity-promoting recovery in the
  multiresolution coefficients + outer search over scale (reuse interface inverse machinery:
  outer-loop + fast LSQ). Robustness to noise; model-selection (how many sources?).
- **Phase 4 — applied benchmark + write-up:** a real source-localization / load-ID case; competitor
  table (vanilla PINN, RBF/Fourier-PIELM, enrichment where applicable).

## 6. Repo assets to reuse
Wavelet family + operational matrices (Wfamily.py); column-norm + Tikhonov LSQ; banded-refinement
builder (from gear, Elliptic_Interface/Gear_Interface); inverse outer-loop+LSQ (Phase 3 interface);
canonical folder layout (config / Wfamily / Model / WPINN_*.ipynb / sol.png + results.md + table.png).

## 7. Biggest risks
- Novelty sharpness: must lead on (a)+(b) [multi-scale + unknown count via sparsity], not generic
  multiscale (source paper) or single source (taken). Keep the framing tight.
- Inverse identifiability / sparsity regularization at high noise (the real technical risk).
- 2D cost of fine bands (manage as the gear did: band only, not global finest level).
