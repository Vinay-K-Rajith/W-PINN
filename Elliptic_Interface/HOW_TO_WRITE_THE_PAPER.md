# How to write the paper — optimal layout & guidance

Companion to `COMBINED_RESULTS.md` (all numbers) and the per-folder `results.md` files.

## 0. The thesis (every sentence serves this)

> *A training-free, mesh-free wavelet least-squares solver for elliptic interface problems that matches
> or beats neural-network (PINN) methods at the cost of a single linear solve, scales to multi-inclusion
> topologies, and handles sharp and irregular interfaces.*

**The single most important framing decision:** position the method **against neural-network / mesh-free
PINN methods (which it beats)** — NOT against FEM (which beats it on speed and accuracy). We measured
both. This reframe is honest *and* makes the paper stronger, because your win is real and clean there.

## 1. Target venue & tone

CAMWA / Engineering with Computers (Springer/Elsevier, applied-math/engineering) — composite-materials
application angle. Numerical-analysis reviewers will check: (a) a fair baseline, (b) conditioning, (c) a
convergence/limitations discussion. Pre-empt all three; an honest Limitations section *raises* the score.

## 2. Section-by-section layout

**Abstract.** Put two numbers in it: the DCSNN head-to-head (relL2 3.2e-6 vs 2.6e-4 on the flower, **no
training**) and topology scaling (relL2 flat m=2→6). Those two earn the read.

**1. Introduction.**
- Motivate via composite materials (fibers/particles/pores in a matrix) → `−∇·(a∇u)=f` with jumps.
- Survey the mesh-free/NN-interface line and state the gap: those methods **train** (iterative, GPU,
  tuning); mesh methods **remesh** per geometry. Your method is **both mesh-free and training-free** — a
  convex linear least-squares solved once. That is the contribution sentence.
- Cite: DCSNN (arXiv 2106.05587), cusp-capturing PINN (2210.08424), piecewise-DNN (2005.04847),
  MTNN domain-decomposition (2502.19893), interfaced-operator-net (2308.14537).

**2. Method.**
- Per-subdomain Gaussian-derivative wavelet expansions; interface/BC conditions → one column-normalised
  Tikhonov SVD (`gelsd`). Emphasise: **linear in the coefficients ⇒ convex ⇒ one solve, no training.**
- **Be upfront about the frame.** State cond is large (~1e26) because the basis is an overcomplete frame,
  and that the SVD+Tikhonov solve is precisely why it is robust. Owning this disarms the obvious reviewer.
- Banded multiresolution refinement (for sharp features) and per-inclusion block structure (for topology).

**3. Forward results — present in THIS order for impact:**
- **3.1 Head-to-head vs NN methods (LEAD).** The two DCSNN tables (`dcsnn_ex1/ex2.py`). This is your
  strongest, cleanest, most defensible result. Make the "single solve vs training" point explicit.
- **3.2 Sharp features.** Gear + banded refinement; note this is where smooth shallow NNs (DCSNN) are
  weakest (cusps) — you may state this qualitatively without re-implementing DCSNN on the gear.
- **3.3 Multi-inclusion topology (your novelty).** Flat relL2 m=2→6 table + `multi_inclusion.png`;
  heterogeneous contrast; the honest band ablation. This is the composite-materials core.
- **3.4 Contrast robustness.** ρ∈[0.1,1000]; note ρ<1 is sound and the ρ≤0.01 multi-inclusion dip is a
  conditioning effect (single interface at contrast 1000 solves to 6e-7).

**4. Comparison to FEM (handle with complete honesty — this is a strength, not a weakness).**
- Show the FEM tables openly: FEM wins on accuracy, per-solve, AND total cost even while remeshing
  (`cost_comparison.py`, ~100×). **Do not spin this.**
- Then state the honest scope: *FEM remains the gold standard for fixed geometry; we do not replace it.
  Our contribution is within the mesh-free/training-free class, where we outperform neural methods, and
  our representation is trivial to implement (no meshing pipeline, no mesh-robustness failures on
  complex/sharp/touching geometries).* This candor is what a good reviewer rewards.

**5. Inverse problem (supporting demo — keep modest).**
- Local multi-parameter recovery from interior data; sub-second forward; no remeshing; robust to 1% noise.
- Frame the partial recovery as **consistent with known ill-posedness** (cite Calderón/EIT inclusion-
  detection: shape-opt JCP 2019 S0021999119308289, Calderón 1902.04462). This turns a soft result into a
  theory-aware one. Do NOT make it a headline.

**6. Limitations (write a real one).**
- Larger basis than DCSNN (per-parameter less efficient); slower than FEM on fixed geometry; inverse is
  local-only; high-dynamic-range solutions on large domains fail (the s³ benchmark, `benchmark_ife.py`).
- Stating these explicitly is a credibility multiplier.

**7. Conclusion.** Training-free + mesh-free, beats NN methods at one linear solve, scales in topology,
enables geometry inversion. Future work: better-conditioned basis, global optimiser for the inverse,
re-pose on the unit square for high-dynamic-range benchmarks.

## 3. Figures/tables checklist (you already have these)
- DCSNN head-to-head table (§3.1) — **the money table.**
- `multi_inclusion.png` (exact/pred/error, 5 inclusions) + topology-scaling table (§3.3).
- Gear `sol.png` (existing).
- FEM-vs-wavelet tables incl. total-cost (§4).
- `inverse.png` (true vs recovered circles) (§5).

## 4. What to DOWNPLAY or cut
- Don't claim to beat FEM (you don't) — concede it cleanly in §4.
- Don't headline the inverse (it's a demo).
- Footnote the failed s³ benchmark in Limitations only.
- Don't over-report the ρ=0.001 dip as a flaw (it's multi-inclusion conditioning).

## 5. Honest verdict on readiness
With §3.1 (beat DCSNN ×2), §3.3 (topology), and an honest §4/§6, this is a **solid ~7/10** — a
comfortable mid-tier Springer accept and a credible journal submission **provided** you position against
NN methods, not FEM. The DCSNN wins are the spine; everything else supports them.
