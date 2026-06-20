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
