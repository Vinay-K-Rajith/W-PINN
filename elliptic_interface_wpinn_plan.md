# Elliptic Interface Problems as a Novel Application of Wavelet-PINN (W-PINN)

> **Goal of this document.** Define a *novel application* — not a new architecture — for the wavelet-PINN of Pandey, Singh & Behera (arXiv:2409.11847). W-PINN is used as published: the solution is expanded in a fixed family of scaled/translated wavelets, the coefficients are learned, and every differential operator is applied through a **precomputed operational matrix** (no automatic differentiation). The only thing that changes is the *problem class* the method is pointed at. We point it at **elliptic PDEs with a discontinuous coefficient across a fixed interface**, because the method's one demonstrated, repeatable advantage — resolving sharp, localized features that a vanilla PINN smears (its SPP boundary-layer example) — is *exactly* what an interface kink demands.

---

## 0. Why this target, from first principles

The W-PINN value proposition, read off the existing code (the SPP example and its NTK analysis), is **not** "AD is impossible here" — it is **"multiresolution wavelets resolve a sharp, stationary, localized feature that a smooth MLP under-resolves because of spectral bias / ill-conditioning."** A good application must therefore satisfy:

1. **The solution has a sharp but *stationary, localized* feature** — a kink, layer, or steep gradient at a *known* location, so a *fixed* multiresolution wavelet family can place fine scales exactly there. (The repo's wavelet family is static — `Wfamily.py` builds it once — so a *moving* front, e.g. Allen–Cahn, fights the method; a *fixed* interface fits it.)
2. **The feature is sharp but smooth at fine scale**, not a true jump. The Gaussian-derivative wavelets in this repo are smooth and would Gibbs-ring on a discontinuity in $u$ itself. A *gradient* kink (with $u$ continuous) is the ideal case: smooth on each side, non-smooth only at the join.
3. **The governing operator is linear with a fixed coefficient field**, so the operational matrix is genuinely precomputed once and reused.
4. **Vanilla PINNs are documented to fail on it**, so there is a real baseline to beat.

The discontinuous-coefficient elliptic problem hits all four. It is also *not* on the method's existing turf: the source paper is titled "…for **multiscale problems**," so multiscale/high-frequency is already claimed; an interface **kink** is a distinct failure mode (non-smoothness, not multiple smooth scales). And it sidesteps the crowded fractional-PINN literature (operational-matrix fPINN, arXiv:2401.14081, already did "AD-free fractional").

---

## 1. The equations, defined properly

### 1.1 Model problem

On a bounded domain $\Omega\subset\mathbb{R}^d$ split by a fixed interface $\Gamma$ into $\Omega=\Omega^-\cup\Gamma\cup\Omega^+$,

$$
-\nabla\cdot\big(a(x)\nabla u(x)\big)=f(x)\ \text{in }\Omega^\pm,
\qquad
a(x)=\begin{cases}a^-(x), & x\in\Omega^-\\[2pt] a^+(x), & x\in\Omega^+\end{cases}
$$

with the **interface (jump) conditions**

$$
[\![u]\!]_\Gamma \;:=\; u^+ - u^- \;=\; g_D \quad\text{on }\Gamma,
\qquad
[\![a\,\partial_n u]\!]_\Gamma \;:=\; a^+\partial_n u^+ - a^-\partial_n u^- \;=\; g_N \quad\text{on }\Gamma,
$$

(usually $g_D=g_N=0$: continuity of solution and of normal flux), plus an outer boundary condition $u=h$ on $\partial\Omega$. Here $\partial_n$ is the normal derivative on $\Gamma$ (outward from $\Omega^-$).

**The defining feature.** With $a^-\neq a^+$, flux continuity forces $[\![\partial_n u]\!]\neq0$: $u$ is continuous but **kinked** at $\Gamma$. This is the localized non-smoothness the method must resolve.

### 1.2 Why a single global expansion is wrong (the first-principles core)

If we wrote one smooth expansion $u=\sum_n c_n\Psi_n + b$ over all of $\Omega$, then $\nabla u$ would be continuous across $\Gamma$ by construction, so $[\![a\,\partial_n u]\!]=(a^+-a^-)\partial_n u\neq0$ unless $\partial_n u=0$. **A single $C^1$ expansion structurally cannot satisfy the flux condition** — it can only approximate the kink by superposing ever-finer wavelets. That is precisely the vanilla-PINN limitation re-dressed.

The fix is **domain decomposition**: one smooth wavelet expansion per subdomain,

$$
u^-(x)=\sum_n c^-_n\,\Psi_n(x)+b^-\ \text{ on }\Omega^-,
\qquad
u^+(x)=\sum_n c^+_n\,\Psi_n(x)+b^+\ \text{ on }\Omega^+,
$$

each smooth *inside* its subdomain, with the kink produced **at the join** by enforcing $[\![u]\!]$ and $[\![a\partial_n u]\!]$ as constraints on interface collocation points. Each piece is exactly the regime where smooth wavelets excel; the multiresolution levels are concentrated near $\Gamma$ to resolve the steep behaviour approaching the kink. This is the W-PINN analogue of immersed-interface / domain-decomposition PINNs — but AD-free and multiresolution.

### 1.3 Benchmarks (with exact references)

- **B1 — 1D two-domain.** $\Omega=(0,1)$, interface $\beta\in(0,1)$, $a^\pm$ constant. Manufactured solution: prescribe smooth $u^-,u^+$, back-compute $f$, $g_D$, $g_N$, and boundary data. Cheapest possible validation of the whole operational-matrix + interface-coupling machinery; closed-form error.
- **B2 — 2D circular interface (standard).** $\Omega=(-1,1)^2$, $\Gamma=\{|x|=r_0\}$, $a^-$ inside, $a^+$ outside; a classical immersed-interface benchmark with known exact solutions. Sweeps in contrast $\rho=a^+/a^-$ and radius $r_0$.
- **B3 — Complex geometry.** A star-/flower-shaped interface to demonstrate the **mesh-free** advantage (no remeshing as geometry changes).
- **B4 — High-contrast stress test.** $\rho\in\{10,10^2,10^3,10^4\}$ — the interface analogue of $\epsilon\to0$ in the repo's SPP example; the regime where vanilla PINNs and naive methods degrade most.

---

## 2. How W-PINN applies — *unchanged* — to these equations

The published machinery (`Wfamily.py`: precompute $W$, $D_1W$, $D_2W$ by sampling the analytic wavelets and their analytic derivatives at collocation points; `Model.py`: a network emits the coefficient vector plus a trainable bias; loss = residual MSE assembled from matrix–vector products) carries over with **two coefficient vectors instead of one and an extra interface-residual term**. Nothing else changes; there is no AD in any operator.

Let $\{x_i^-\}\subset\Omega^-$, $\{x_i^+\}\subset\Omega^+$ be interior collocation points, $\{x_i^\Gamma\}\subset\Gamma$ interface points, $\{x_i^{\partial}\}\subset\partial\Omega$ boundary points. Precompute, **once**:

| Matrix | Definition | Use |
|---|---|---|
| $W^\pm,\,D_2W^\pm$ | wavelets / Laplacian-of-wavelets at $\{x_i^\pm\}$ | interior residual |
| $W^\pm_\Gamma,\,\partial_nW^\pm_\Gamma$ | wavelets / normal-derivative at $\{x_i^\Gamma\}$ (from each side) | interface conditions |
| $W^\pm_\partial$ | wavelets at outer boundary points in the owning subdomain | outer BC |

The normal-derivative columns use $\partial_n\Psi_n = n_x\,\partial_x\Psi_n + n_y\,\partial_y\Psi_n$, all analytic. The residuals are then pure matrix–vector products:

$$
\underbrace{-a^\pm\,(D_2W^\pm c^\pm) - f}_{\text{interior, }\Omega^\pm},
\qquad
\underbrace{(W^+_\Gamma c^+ + b^+) - (W^-_\Gamma c^- + b^-) - g_D}_{[\![u]\!]},
\qquad
\underbrace{a^+(\partial_nW^+_\Gamma c^+) - a^-(\partial_nW^-_\Gamma c^-) - g_N}_{[\![a\partial_n u]\!]} .
$$

(For non-constant $a^\pm(x)$ use the divergence form $-\nabla a^\pm\!\cdot\!\nabla u - a^\pm\Delta u$, with $\nabla a^\pm$ a fixed, precomputed pointwise weight — still no AD, still precomputed once.) The loss is the published residual + interface + boundary mean-squared form:

$$
\mathcal{L}=\lambda_r\big(\|\,\text{res}^-\|^2+\|\,\text{res}^+\|^2\big)
+\lambda_\Gamma\big(\|[\![u]\!]\|^2+\|[\![a\partial_n u]\!]\|^2\big)
+\lambda_b\,\|u-h\|^2_{\partial\Omega}.
$$

We keep the repo's coefficient-emitting network design (`Model.py`), now with **two output heads** ($c^-,c^+$) and two trainable biases; the 2D tensor-product wavelet family already exists in the repo (`Helmholtz/`, `Maxwell's Equation/`), and the NTK tooling for the spectral-bias argument already exists (`SPP/SPP_Ex1_NTK/`).

---

## 3. Novelty — honest positioning

| Existing approach | Mechanism at the interface | Limitation this targets |
|---|---|---|
| **Vanilla / XPINN / cusp-capturing / IPINN** | smooth MLP(s), kink captured by special activations, augmented inputs, or subnet-per-domain with AD | spectral bias smears the kink; flux jump hard; conditioning degrades at high contrast |
| **Immersed Interface / FEM / IIM** | grid + interface-fitted corrections | not a learning framework; remeshing for new geometry; no native inverse-problem path |
| **Source W-PINN** | single multiresolution expansion for *smooth multiscale* problems | not yet applied to a *non-smooth* interface kink |
| **This proposal** | **two multiresolution wavelet expansions, coupled by precomputed interface operational matrices; fully AD-free** | — |

**Honest reading.** Two things are *not* claimed novel: (i) using NNs / domain decomposition for interface problems (XPINN/IPINN own that), and (ii) wavelets as a numerical basis for interface PDEs (classical analysis owns that). What appears open is the *combination*: the W-PINN solution-as-wavelet-coefficients / precomputed-operational-matrix framework, **decomposed across the interface with multiresolution concentrated at $\Gamma$**, shown to resolve the flux-kink that smooth-MLP PINNs under-resolve, AD-free, and mesh-free across geometry. The paper lives or dies on **one regime where the win is unambiguous**: the high-contrast kink.

---

## 4. Three-phase plan

### Phase 1 — Formulation & 1D proof of concept *(validate the machinery and the central hypothesis cheaply)*

**Do:**
1. Derive and freeze the two-domain formulation (§2). Implement the 1D operational matrices $W^\pm, D_1W^\pm, D_2W^\pm$ and the interface matrices $W^\pm_\Gamma,\,D_1W^\pm_\Gamma$ from the repo's analytic Gaussian-derivative wavelets.
2. Build benchmark **B1** with a manufactured exact solution (prescribe $u^\pm$, back-compute $f,g_D,g_N,h$).
3. **Unit-test the operators first** (researcher discipline): verify $D_2W^\pm c\to u''^\pm$ and $D_1W^\pm_\Gamma c\to u'^\pm(\beta)$ at expected order on a *known* coefficient vector — before any training.
4. Train AD-free; report relative $L^2$, max error, and specifically the **gradient-jump error** $|[\![\partial_n u]\!]_{\text{pred}}-[\![\partial_n u]\!]_{\text{exact}}|$ at $\beta$.
5. **Ablation that is itself a result:** the naive *single-expansion* W-PINN vs the decomposed one vs a vanilla MLP-PINN. Predicted ordering: decomposed ≫ single-expansion ≈ vanilla near the kink.

**Go/no-go gate:** decomposed W-PINN resolves the kink to near machine-limited quadrature accuracy on B1, and beats both baselines at/near $\beta$. If the single-expansion already matches it, the central premise is wrong — stop and reconsider.

### Phase 2 — 2D, geometry, and the smoothing claim *(establish the advantage where it is supposed to live)*

**Do:**
1. Extend to 2D using the existing tensor-product wavelet family; add the normal-derivative interface matrices ($\partial_n = n_x\partial_x+n_y\partial_y$).
2. Benchmark **B2** (circular interface, exact solution); then **B3** (star/flower interface) for the mesh-free claim.
3. **Graded multiresolution at $\Gamma$:** concentrate fine wavelet levels in a band around the interface; quantify accuracy vs basis count.
4. **High-contrast sweep B4** ($\rho$ up to $10^3$–$10^4$): the headline stress test.
5. **Spectral-bias evidence:** reuse the repo's NTK tooling (`SPP_Ex1_NTK`) to show the vanilla PINN's NTK under-weights the high-frequency content at the kink while the wavelet basis does not — the mechanistic explanation of the win, not just a number.

**Go/no-go gate:** on B2/B4, lower error (especially in flux/gradient near $\Gamma$) than vanilla PINN at equal or lower cost, sustained as $\rho$ grows; mesh-free success on B3 without re-tuning the basis topology.

### Phase 3 — Differentiators, hardening & write-up *(turn an advantage into a defensible paper)*

**Do:**
1. **Inverse problem** (the genuine framework-level novelty): from sparse interior measurements, recover (a) the contrast $a^\pm$, and/or (b) the **interface location/shape** $\Gamma$ itself — a capability FEM/IIM lack natively and where the soft-constraint learning formulation shines. This is the strongest claim available and should anchor the paper's contribution.
2. **Head-to-head** vs named competitors: vanilla PINN, XPINN/IPINN, cusp-capturing PINN — accuracy *and* wall-clock, on identical B2/B4 problems.
3. **Conditioning / basis study:** Gaussian-derivative vs compactly-supported (Daubechies) wavelets for $W^\pm$ conditioning and sparsity at high resolution (the source paper's own caveat).
4. **Robustness:** loss-weight sensitivity ($\lambda_\Gamma$ vs $\lambda_r$), interface-point density, noise on inverse-problem data.
5. Write the focused application paper anchored on the high-contrast kink + the inverse-interface result.

**Exit criterion:** see §5.

---

## 5. Pre-committed success criterion (fix before running)

A result counts only if, on a discontinuous-coefficient elliptic problem whose solution has a flux-kink that a vanilla PINN demonstrably under-resolves, the decomposed multiresolution W-PINN achieves **either** lower relative $L^2$/flux error at equal cost **or** equal accuracy at lower cost — *and* sustains it into the high-contrast regime ($\rho\gtrsim10^3$) — **AD-free and mesh-free**. The inverse recovery of the interface location from sparse data, if achieved, is the decisive differentiator over classical interface solvers. Merely reproducing a vanilla PINN with no accuracy or cost edge is a sanity check, not a result.

---

## 6. Risks & first-principles caveats

- **Interface-matrix accuracy is where the work is.** Normal derivatives on a curved $\Gamma$ in 2D must be sampled and weighted carefully; correct normals $n(x)$ and adequate interface-point density are prerequisites. Build and unit-test these once.
- **Loss balancing.** Interface residuals and interior residuals live on different scales; $\lambda_\Gamma$ may need tuning (or a hard-constraint variant). Report sensitivity rather than hiding it.
- **Curse of dimensionality persists.** Basis count grows like $\sum_j O(2^{jd})$ (the source paper's own limitation). Keep claims to 1D/2D.
- **Wavelet smoothness vs true jumps.** This method is for *gradient* kinks with $u$ continuous. A discontinuity in $u$ itself (e.g. $g_D\neq0$ with a jump) would Gibbs-ring; handle by absorbing the prescribed jump into the two-expansion split, not by asking one basis to jump.
- **Single-expansion temptation.** Resist it; §1.2 is the reason. Keep it only as the motivating ablation.

---

## 7. Bottom line

Discontinuous-coefficient elliptic problems let W-PINN be applied **exactly as published** — same wavelet basis, same precomputed-operational-matrix residual, no AD — to a problem whose solution carries precisely the sharp, *stationary, localized* feature (a flux-induced kink) that the method's own SPP example proves it resolves better than a vanilla PINN. The principled formulation is two multiresolution expansions coupled at the interface (never one global expansion). The defensible product is a focused application paper anchored on the **high-contrast kink** and an **inverse interface-recovery** result — a clean, in-scope, novel *application* of wavelet-PINN, not a new method.

*Prior-art to verify before submission: IPINN / cusp-capturing PINN / XPINN for interface problems; immersed-interface method (Li & Ito); and that the specific multiresolution-wavelet-operational-matrix decomposition has not been published. Source method — Pandey, Singh & Behera, arXiv:2409.11847.*
