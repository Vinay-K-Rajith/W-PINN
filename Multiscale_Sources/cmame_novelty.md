# Mathematical novelty & CMAME positioning — Multiscale-Sources W-PINN

Target venue: **CMAME** (Computer Methods in Applied Mechanics and Engineering). CMAME is an applied
*methods* journal — rigorous but not pure-math; it rejects "another PINN variant with nice plots" unless
the mathematical contribution is a precise, defensible claim, positioned sharply against the literature.
This document defines that claim and the theory/experiments needed to support it. Companion to
`multiscale_sources_wpinn_plan.md`; results to date in `results.md`.

---

## 0. Working title (chosen 2026-06-24)
> **Resolving Hidden Scales: Adaptive Wavelet Least-Squares for Multiscale and Inverse Source Problems**

12 words, ~98 chars (in the 10–15 word / 80–110 char triage-optimal window). "Hidden Scales" = the moat
(features at a-priori-UNKNOWN scales/locations); the rest encodes the two novelty pillars: adaptive
multiresolution refinement + the forward↔inverse unification.

Title rules for this submission:
- **No "AD-free" / "training-free" / "no-GPU" in the title** — implementation properties; put them in the
  abstract + contributions list as selling points.
- **No "PINN" / "physics-informed" in the title** — the method has no network and no training; "PINN"
  misleads CMAME reviewers and invites "this isn't a PINN" objections. Call it a wavelet least-squares /
  collocation method; position *against* PINNs in the intro.
- If signalling the engineering anchor, "Inverse Source Problems" → "Inverse Source Identification".
- Theory-forward alternative (only if claim (B) is proven/verified): "Best-N-Term Adaptive Wavelet
  Least-Squares for Multiscale and Inverse Source Problems".

---

## 1. What is NOT the novelty (do not claim these — reviewers will reject)
- **AD-free operational matrices** → an *implementation* advantage (no training, no GPU, float64 on CPU),
  not mathematics. Keep as a selling point, not a theorem.
- **Wavelets for multiscale PDEs** → occupied (arXiv:2409.11847 source paper; PIMWNN, CPC 2025
  S0010465525004874; wavelet-quantum-PINN arXiv:2512.08256).
- **Matching pursuit / OMP / weak-greedy / adaptive wavelets** → classical (Cohen–Dahmen–DeVore best-N-term
  optimality; reduced-basis weak-greedy a posteriori). The *ingredients* are known.

## 2. The mathematical novelty (stated precisely)
The new object is ONE algorithm + its error-control theory:

> A **residual-driven greedy selection on the PDE-OPERATOR dictionary** — atoms scored by
> `|a_i · r| / ‖a_i‖`, where `a_i = (−Δ + κ²)ψ_i` is the operator image of a multiresolution wavelet and
> `r` is the PDE residual — together with an **a posteriori error estimate** controlling the solution error
> by this residual, yielding near-best-N-term convergence for solutions in the multiscale (Besov)
> regularity class, where the SAME selector solves the inverse source-recovery problem.

Two genuinely new pieces:
1. **Greedy on the OPERATOR dictionary, not the function dictionary.** Classical adaptive wavelets greedily
   approximate a *known* function; here atoms are selected by their *PDE-operator* correlation with the
   residual of an *unknown* solution. The selection rule and its analysis are specific to the operator
   least-squares setting (atoms are `(−Δ+κ²)ψ_i`, not `ψ_i`).
2. **Forward↔inverse unification.** The identical residual-greedy operator drives forward refinement
   (where/at-what-scale to add a wavelet) AND inverse detection (unknown source count/location/scale).
   Both are *sparse selection in the same multiresolution operator dictionary* — the conceptual
   contribution no competitor (enrichment SEPINN/XPIELM, single-scale RFM/PIELM, weighted-sparsity source
   ID) has.

## 3. Theoretical results required to "properly define" the novelty (ranked)

### (A) MUST HAVE — a posteriori error estimate with an explicit stability constant
Reliability/efficiency bound:
```
        ‖u − u_h‖  ≤  C_stab · ( ‖PDE residual‖ + ‖BC residual‖ )
```
This is the heart of the paper. Our own 2D experiment (`results.md` §2) EXPOSED why it matters: residual
small but error not ⇒ `C_stab` is large/degrading. So it cannot be hand-waved. The paper-defining move:
- identify `C_stab` as the **discrete inf-sup / frame-stability constant** of the operator least-squares;
- show how column-normalised **Tikhonov regularisation** controls it (data exists: tik=1e-6 → relL2
  4.48e-2 vs the suboptimal tik=1e-8 → 0.10);
- demonstrate the residual→error gap closing under regularisation / RRQR filtering.
This converts the honest 2D "negative" into a *characterised mechanism* — the CMAME-grade contribution.

### (B) STRONG TO HAVE — best-N-term / weak-greedy convergence rate
For the target class (smooth background + localised features ⇒ Besov `B^s_{p,q}`, `p<2`, the sparse class
wavelets are optimal for), the greedy attains the dictionary's near-optimal N-term rate — which
single-scale random features provably CANNOT (no adaptive N-term rate). This is the rigorous form of
"beats RBF-PIELM" and the moat vs PIELM/RFM. Full proof for an overcomplete frame is hard (not a Riesz
basis; cond ~1e26); defensible route = prove under a frame/restricted-isometry-type assumption, then
verify the assumption numerically.

### (C) ENGINEERING ANCHOR (CMAME-specific)
CMAME wants applied-mechanics relevance. Anchor the inverse as **load / defect / source identification in a
heterogeneous (interface) medium** — fuses with the interface machinery (paper #1), and is where
(A)+(B)+the unification all pay off. Less crowded than homogeneous acoustic inverse source.

## 4. The honest hard part (where reviewers will probe)
All theory rests on `C_stab` for an **overcomplete, numerically rank-deficient frame**. Two honest options:
- **Tame the frame so the theory is clean:** RRQR column filtering (measured −9 orders, 3.5e17→4e8;
  `rrqr_2d.py`) makes the kept set well-conditioned ⇒ state stability for the *filtered* basis, giving the
  a posteriori bound a real constant. RRQR is then not just efficiency — it ENABLES the theorem. (Preferred.)
- Or state the bound with the **empirically measured** `C_stab` and be explicit it is verified, not proven,
  for the full frame.

## 5. Paper spine for CMAME
1. Lead with (A): a posteriori-controlled residual-greedy refinement, `C_stab` made explicit
   (RRQR-stabilised) — forward, with the 1D headline (auto-greedy beats the manual oracle 46×, `greedy.png`).
2. Support with (B): N-term rate argument vs single-scale random features (the RBF-PIELM comparison).
3. Unify forward↔inverse (§2.2); deliver the inverse as the differentiator.
4. Anchor in (C): an identification application in a heterogeneous medium.
Forward solver alone = ICOSAHOM/JSC-strength; (A)+(B)+(C) = CMAME.

## 6. Status vs this spine (2026-06-24)
- (A) a posteriori bound: residual indicator implemented & drives the 1D greedy; `C_stab` mechanism
  diagnosed in 2D (regularisation lever) but **not yet measured/plotted as error/residual ratio vs RRQR &
  tik** — the next experiment, and the empirical backbone of (A).
- (B) rate: 1D scale-separation sweep already shows the multiresolution advantage vs RBF-PIELM
  (`phase1_1d/decisive_scaletest.py`); the N-term *theorem* is unwritten.
- (C) application: not started; reuse interface machinery from paper #1.
- Inverse headline: not started — the venue-deciding piece for CMAME.
