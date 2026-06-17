# Study Guide — Understanding the W-PINN Elliptic-Interface Project

> Premise: thinking can be outsourced, understanding cannot. This guide lists every topic you must
> genuinely own to defend, extend, and judge this project. Each item has: **why it matters here**,
> **what to master**, and a **self-test** ("you understand it when you can…"). Many self-tests are
> drawn from things that already decided Phase 1 — if you can answer those, you own the result.
>
> Priority key:  ★★★ = load-bearing for this project · ★★ = important · ★ = context.
> Suggested order is top to bottom; the tiers build on each other.

---

## Tier 0 — PDE & elliptic foundations

### 0.1 ★★★ Second-order elliptic PDEs and the divergence form
- **Why here:** our equation is $-\nabla\cdot(a(x)\nabla u)=f$. Everything rests on what this operator means.
- **Master:** divergence (conservation) form vs non-divergence form; what $a(x)$ (diffusivity/conductivity) physically represents; why ellipticity ($a>0$) guarantees a unique smooth solution *within* a region; Dirichlet vs Neumann vs Robin boundary conditions.
- **Self-test:** Explain why, with $a$ piecewise-constant, the equation reduces to $-a^\pm\Delta u=f$ *inside* each subdomain but **not** across the interface. Why can't you just write $-a(x)u''=f$ everywhere in 1D?

### 0.2 ★★ Weak / variational formulation
- **Why here:** the "natural" interface conditions ($[\![u]\!]=0$, $[\![a\partial_n u]\!]=0$) fall out of the weak form; it's also why FEM handles interfaces and why flux continuity is the physical statement.
- **Master:** multiply by a test function, integrate by parts, where the flux term $a\partial_n u$ appears on the boundary; the notion that the weak solution lives in $H^1$ even when the classical second derivative doesn't exist at Γ.
- **Self-test:** Derive the flux-continuity condition $[\![a\partial_n u]\!]=0$ from the weak form (no external source on Γ). Why is it the *flux*, not the *gradient*, that is continuous?

### 0.3 ★★ Solution regularity and singularities
- **Why here:** the whole project is "wavelets resolve a sharp feature." You must know exactly *what* the feature is.
- **Master:** $u\in C^0$ but $u\notin C^1$ at Γ (a kink); the gradient jump $[\![\partial_n u]\!]=(a^-/a^+-1)\partial_n u^-$; in 2D with a curved interface or corners, stronger singularities can appear.
- **Self-test:** For our 1D benchmark, derive $[\![u']\!] = (A-1)\pi\cos(\pi\beta)$ with $A=a^-/a^+$. Predict how the kink sharpens as contrast $\rho=a^+/a^-\to\infty$.

---

## Tier 1 — The interface problem

### 1.1 ★★★ Interface (jump) conditions
- **Why here:** these two conditions are the entire coupling between subdomains; they are the loss terms in M1/M4.
- **Master:** $[\![u]\!]=g_D$ (continuity / prescribed solution jump) and $[\![a\partial_n u]\!]=g_N$ (flux continuity / prescribed flux jump); the homogeneous case $g_D=g_N=0$; the outward normal $n$ and its sign convention.
- **Self-test:** Why does a single globally-smooth ($C^1$) approximant *force* $[\![\partial_n u]\!]=0$, and therefore *cannot* satisfy flux continuity when $a^-\neq a^+$ unless $\partial_n u=0$? (This is the first-principles core of the whole project — Phase 1 confirmed it: M2/M3 predicted a zero kink.)

### 1.2 ★★ Manufactured solutions for interface problems
- **Why here:** B1 (and all our error numbers) come from a manufactured exact solution.
- **Master:** prescribe smooth $u^\pm$, back-compute $f$, $g_D$, $g_N$, BCs; how to *design* $u^\pm$ so the homogeneous conditions hold (we used $A=a^-/a^+$, $B=\sin(\pi\beta)(1-A)$).
- **Self-test:** Build a *different* manufactured interface solution (e.g. polynomial pieces) with $g_D=g_N=0$ and verify the conditions by hand.

### 1.3 ★ Classical interface solvers (the comparison baseline)
- **Why here:** the novelty claim is "vs classical methods, ours is mesh-free + learns + inverse-capable."
- **Master:** the Immersed Interface Method (Li & Ito) and interface-fitted FEM at a conceptual level — what they do well, why remeshing is their pain point.
- **Self-test:** State one thing classical IIM/FEM do better than us, and one thing we can do that they can't natively (hint: Phase 3).

---

## Tier 2 — Wavelets & multiresolution

### 2.1 ★★★ Wavelets, scales, translations, multiresolution analysis (MRA)
- **Why here:** the solution basis. $\Psi_{j,k}(x)=\sqrt{2^j}\psi(2^j x-k)$ — level $j$ = scale, $k$ = location.
- **Master:** mother wavelet; how increasing $j$ localizes/refines; how a sum over $(j,k)$ builds a function; why placing fine levels near a feature = "multiresolution refinement"; the repo's specific wavelet $\psi(z)=-z e^{-z^2/2}$ (Gaussian-derivative / Mexican-hat family).
- **Self-test:** Given the repo's `gaussian`, derive its 1st and 2nd derivatives analytically and confirm they match `D1tgaussian`/`D2tgaussian`. Why does a *smooth* wavelet superposition struggle to represent a true discontinuity (Gibbs) but handle a *kink-by-decomposition* fine?

### 2.2 ★★ Tensor-product wavelets in 2D
- **Why here:** Phase 2 is 2D; the repo's Helmholtz/Maxwell families are tensor products.
- **Master:** $\Psi(x,y)=\psi_x(x)\psi_y(y)$; how the family count grows like $\sum_j O(2^{jd})$ (the curse of dimensionality the source paper warns about).
- **Self-test:** Estimate the family size for a 2D problem with $J$ levels and explain why this caps the method at low dimension.

### 2.3 ★★ Wavelet choice & conditioning (Gaussian-derivative vs Daubechies)
- **Why here:** Phase 1 found cond(A) ≫ 10¹⁹⁶ with Gaussian wavelets — the #1 obstacle for Phase 2.
- **Master:** non-orthogonal redundant frames (Gaussian-derivative) vs orthonormal compactly-supported (Daubechies); how overlap/redundancy creates near-linear-dependence → huge condition numbers; sparsity of compactly-supported bases.
- **Self-test:** Explain mechanistically *why* a redundant Gaussian-derivative family produces a near-singular collocation matrix, and why orthonormal compact support would help.

---

## Tier 3 — PINNs and the W-PINN method

### 3.1 ★★★ Standard PINNs and their failure modes
- **Why here:** the baseline we beat; understanding *why* it fails is the scientific case.
- **Master:** collocation residual loss; automatic differentiation to form $u_{xx}$; soft constraints for BC/IC; the documented pathologies — spectral bias (low frequencies learned first), ill-conditioned loss landscapes, trouble with sharp/stiff features.
- **Self-test:** Why did the vanilla MLP (M3) plateau at rel-L² ≈ 0.16 on our kink? Tie it to spectral bias, not just "needs more training."

### 3.2 ★★★ The W-PINN reformulation (Pandey–Singh–Behera)
- **Why here:** this is *the method*. Read arXiv:2409.11847 closely.
- **Master:** solution as $u=\sum c_n\Psi_n+b$ (fixed basis, learned coefficients); derivatives via **precomputed operational matrices** $W,D_1W,D_2W$ (sampled analytic wavelet derivatives); no AD in the operator; loss = residual + BC/IC MSE; the coefficient-emitting network design in `Model.py`.
- **Self-test:** Walk through `WPINN_main.ipynb` line by line and explain what `torch.mv(DWt, c)` computes and why it needs no autograd. Why is the network input the (constant) collocation set, and what is actually being trained?

### 3.3 ★★ AD-free operational matrices
- **Why here:** the mechanical heart; in our problem it's `D2_m @ cM` etc.
- **Master:** how an operator linear in $u$ with a fixed basis becomes a matrix that's built once and reused; how nonlinear *pointwise* terms (e.g. $u^3$) are still fine; what must stay linear/fixed (the differential operator) for precompute-once to hold.
- **Self-test:** Explain why `D2W` can be precomputed once for our elliptic operator but could *not* be precomputed if $a$ depended on $u$.

---

## Tier 4 — The project's own ideas

### 4.1 ★★★ The decomposition principle
- **Why here:** the load-bearing insight of this whole application; Phase 1's central confirmed result.
- **Master:** the solution is smooth *inside* each subdomain and non-smooth *only at the join*; therefore use one smooth wavelet expansion per subdomain ($u^\pm=W^\pm c^\pm+b^\pm$) coupled by interface losses; the kink emerges from the coupling, not from any single basis; single-expansion is the motivating ablation that *should* fail.
- **Self-test:** Without looking, reproduce the argument for why M1 (decomposed) beat M2 (single) by ~360× even though they use the *identical* wavelet family. State the prediction you'd make before running it.

### 4.2 ★★★ Why this is a *good fit* for W-PINN specifically
- **Why here:** you'll be asked "why interface problems?" — own the answer.
- **Master:** the four-filter argument (stationary feature at known location ⇒ fits the static basis; sharp-but-smooth kink ⇒ no Gibbs; linear fixed-coefficient operator ⇒ genuine precomputed matrix; vanilla PINN documented to fail ⇒ real baseline). See `study`'s companion plan doc.
- **Self-test:** Explain why a *moving* front (Allen–Cahn) is a worse fit for this exact method than a *fixed* interface, in terms of the static wavelet family.

---

## Tier 5 — Linear algebra & optimization (where Phase 1 turned)

### 5.1 ★★★ Least squares and the linear/convex structure
- **Why here:** the optimization fix that revealed the method's true accuracy (7e-12 instead of 4e-4).
- **Master:** for a PDE *linear* in $u$ and a *fixed* basis, the W-PINN objective is a convex quadratic in the coefficients → the global optimum is a (weighted) linear least-squares solution; no iterative training needed; normal equations vs QR vs SVD (`gelsd`) solvers.
- **Self-test:** Explain why Adam *under-converged* M1 but a direct `lstsq` nailed it. When is iterative training actually necessary (hint: nonlinear $f(u)$, or the MLP baselines)?

### 5.2 ★★★ Conditioning, rank-deficiency, and SVD-truncated solves
- **Why here:** the central Phase 1 finding (cond ≫ 10¹⁹⁶) and the #1 Phase 2 risk.
- **Master:** condition number and how it amplifies error; why a redundant basis → numerically rank-deficient $A$; how SVD-based least squares (`gelsd`) discards tiny singular values (effective regularization) and still returns an accurate min-norm solution; why this same ill-conditioning *breaks gradient descent*.
- **Self-test:** We got rel-L² = 4.5e-9 from a matrix with cond ≈ 10¹⁹⁶. Explain how both facts can be true simultaneously, and why this success is "fragile" / not something to rely on in 2D. Propose two fixes.

### 5.3 ★★ Spectral bias and the NTK lens
- **Why here:** the *mechanistic* explanation for why MLP PINNs miss sharp features; the repo has NTK tooling (`SPP_Ex1_NTK`).
- **Master:** Neural Tangent Kernel intuition; eigenvalue spectrum ⇒ which frequencies train fast; why a wavelet basis sidesteps the bias by representing localized high-frequency content directly.
- **Self-test:** Sketch how you'd use the repo's NTK code to *show* (not just assert) that a vanilla PINN under-weights the kink's high-frequency content.

### 5.4 ★★ Loss weighting / multi-objective balance
- **Why here:** `W_IFACE`, `W_BC` in our code; single-point constraints vs many interior points.
- **Master:** why interface/BC terms (few points) must be up-weighted against the interior residual (many points); sensitivity to these weights; the idea of hard vs soft constraints.
- **Self-test:** Predict what happens to the kink accuracy if `W_IFACE` is set to 1 vs 50, and why.

---

## Tier 6 — Baselines, competitors, methodology

### 6.1 ★★ XPINN / IPINN / cusp-capturing PINN
- **Why here:** XPINN (decomposed MLP, our M4) is the *strongest* competitor; the others are the named prior art.
- **Master:** XPINN = subdomain MLPs + interface losses (can represent kinks — that's why M4 worked); IPINN/cusp-capturing = special architectures/inputs for the kink; how our method differs (AD-free, multiresolution, least-squares-solvable).
- **Self-test:** At ρ=10 XPINN *beat* our wavelet method on accuracy; at ρ=10⁴ it lost by ~6 orders. Explain the crossover in terms of what each basis can resolve and how Adam scales with stiffness.

### 6.2 ★★ Experimental methodology
- **Why here:** credibility of every claim.
- **Master:** relative $L^2$ vs max error vs *flux/gradient-jump* error (the interface-specific metric); test on a dense independent grid; unit-testing operators *before* training; controlled sweeps (contrast as the difficulty knob); separating "basis capability" (lstsq) from "optimizer" (Adam).
- **Self-test:** Why is the gradient-jump error at Γ a more honest metric for this problem than global rel-L²? Construct a case where rel-L² looks fine but the interface physics is wrong.

---

## Tier 7 — What's ahead (own these before Phase 2/3)

### 7.1 ★★★ Conditioning remedies
- Daubechies / compactly-supported orthonormal wavelets; subdomain-restricted (truncated-support) families; preconditioning; Tikhonov/SVD regularization. **Self-test:** which of these preserves the "precompute-once, AD-free" property, and which changes the method's character?

### 7.2 ★★ 2D geometry: normals, curved interfaces, interface quadrature
- Computing $\partial_n = n_x\partial_x+n_y\partial_y$ on a curved Γ; placing interface collocation points; the standard circular-interface benchmark. **Self-test:** how does the flux condition change for a curved interface vs the flat 1D case?

### 7.3 ★★ Inverse problems (the Phase-3 differentiator)
- Recovering the contrast $a^\pm$ or the interface *location/shape* Γ from sparse interior data; why the soft-constraint learning formulation enables this and classical solvers don't natively. **Self-test:** pose the interface-recovery problem as an optimization — what are the unknowns, the data, the loss?

---

## Core reading (in priority order)
1. **Pandey, Singh & Behera**, wavelet-PINN — arXiv:2409.11847 (the method). ★★★
2. A standard PINN reference (Raissi, Perdikaris, Karniadakis 2019) — for the baseline & spectral bias. ★★
3. **Li & Ito**, *The Immersed Interface Method* — interface conditions & the classical baseline. ★★
4. An XPINN paper (Jagtap & Karniadakis) — the strongest competitor. ★★
5. A wavelet/MRA primer (Daubechies, *Ten Lectures on Wavelets*, ch.1–3) — basis & conditioning. ★★
6. Any NTK-for-PINNs note (Wang, Yu, Perdikaris) — the spectral-bias mechanism. ★

## The five questions you should be able to answer cold
1. Why can't a single smooth expansion represent the interface kink? (Tier 1.1 / 4.1)
2. Why is the W-PINN operator AD-free, and what must stay linear for that to hold? (Tier 3.3)
3. Why did least-squares beat Adam for our wavelet model? (Tier 5.1)
4. How can cond(A) ≈ 10¹⁹⁶ and rel-L² ≈ 1e-9 both be true, and why is that fragile? (Tier 5.2)
5. Why does the wavelet advantage over XPINN *grow* with contrast? (Tier 6.1)
