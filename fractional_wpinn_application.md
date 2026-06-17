# Fractional PDEs as a Novel Application of Wavelet-PINN (W-PINN)

> **Goal of this document.** Identify and define a *novel application* for the existing wavelet-PINN method of Pandey, Singh & Behera. The objective is **not** to build a new architecture, loss, or operator-learning scheme. W-PINN is used exactly as published — solution expanded in a fixed wavelet basis, coefficients learned, all operators applied through precomputed matrices, no automatic differentiation. The only thing that changes relative to the source paper is the *equation class* the method is pointed at.

---

## 1. Why fractional PDEs are the right target

W-PINN's machinery rewards a specific kind of problem. From the method's structure, a good application must satisfy three conditions:

1. **The governing operator is an integro-differential / nonlocal operator**, so that the AD-free operational matrix is the genuine advantage (for a plain Laplacian, ordinary AD-PINNs are already fine and the wavelet trick buys little).
2. **The operator is linear in $u$ with a fixed kernel**, so the operational matrix can be precomputed *once* and reused every iteration (pointwise nonlinear reaction terms are still fine — the source paper already handles $u^3$, $u^5$ — what must stay linear/fixed is the *integral/derivative operator* itself).
3. **Solutions carry sharp gradients, boundary layers, or weak singularities**, the regime where the source paper already demonstrated that wavelet multiresolution beats standard PINNs.

Fractional PDEs satisfy all three cleanly, and the source paper itself names them as the natural next target: its conclusion explicitly points to "fractional differential equations [which] often present non-locality and memory effects, where the localized resolution capabilities of wavelets can offer a significant advantage."

There is also a clean lineage worth noting: the fractional Laplacian is the $\delta\to\infty$ limit of a peridynamic nonlocal operator. Fractional operators are, in effect, the **well-posed, fixed-kernel members of the same nonlocal-integral family** that motivated the peridynamics idea — but without the solution-dependent bond-breaking that would destroy the precompute-once property. We keep the nonlocal-integral character that makes the application interesting and discard the part that fights the method.

---

## 2. The equations, defined properly

### 2.1 Fractional operators

**Caputo time-fractional derivative.** For order $\alpha\in(0,1)$ and a function $u(\cdot,t)$ on $t\in(0,T]$, the (left) Caputo derivative is the singular convolution

$$
{}^{C}_{0}D_t^{\alpha}\,u(x,t)\;=\;\frac{1}{\Gamma(1-\alpha)}\int_0^t \frac{\partial_\tau u(x,\tau)}{(t-\tau)^{\alpha}}\,d\tau,
\qquad 0<\alpha<1 .
$$

For $\alpha\in(1,2)$ the kernel exponent and the order of the inner derivative shift accordingly:

$$
{}^{C}_{0}D_t^{\alpha}\,u(x,t)\;=\;\frac{1}{\Gamma(2-\alpha)}\int_0^t \frac{\partial_\tau^2 u(x,\tau)}{(t-\tau)^{\alpha-1}}\,d\tau .
$$

The defining features that break standard PINNs: this operator is **nonlocal in time** (it depends on the entire history $[0,t]$), it does **not obey the chain rule**, and consequently **automatic differentiation cannot form it** — the original fPINN had to resort to a hybrid of AD (for integer-order terms) plus a separate numerical discretization (for the fractional term), with auxiliary collocation points.

**Riemann–Liouville derivative** (used for boundary handling and as the alternative convention), $\alpha\in(0,1)$:

$$
{}^{RL}_{\;0}D_t^{\alpha}\,u(x,t)\;=\;\frac{1}{\Gamma(1-\alpha)}\frac{\partial}{\partial t}\int_0^t \frac{u(x,\tau)}{(t-\tau)^{\alpha}}\,d\tau .
$$

**Fractional Laplacian / Riesz space-fractional operator.** For order $s\in(0,1)$ in one spatial dimension, the fractional Laplacian is the principal-value singular integral

$$
(-\Delta)^{s} u(x)\;=\;c_{1,s}\,\mathrm{P.V.}\!\int_{\mathbb{R}}\frac{u(x)-u(y)}{|x-y|^{1+2s}}\,dy,
\qquad
c_{1,s}=\frac{4^{s}\,\Gamma\!\big(\tfrac{1}{2}+s\big)}{\sqrt{\pi}\,\lvert\Gamma(-s)\rvert},
$$

equivalently characterized by its Fourier symbol,

$$
\widehat{(-\Delta)^{s} u}(\xi)=\lvert\xi\rvert^{2s}\,\hat{u}(\xi).
$$

The Riesz derivative $\partial^{\beta}/\partial|x|^{\beta}$ of order $\beta=2s\in(0,2)$ is the symmetric combination of left/right Riemann–Liouville derivatives and carries Fourier symbol $-|\xi|^{\beta}$; on $\mathbb{R}$ it coincides (up to sign) with $-(-\Delta)^{s}$. Either form is an explicit, **fixed**, nonlocal integral operator — exactly condition (2) of §1.

### 2.2 Primary target — time-fractional subdiffusion

$$
\boxed{\;
{}^{C}_{0}D_t^{\alpha}\,u(x,t)\;=\;\kappa\,\partial_{xx}u(x,t)\;+\;f(x,t),
\qquad x\in(0,1),\; t\in(0,T],\; \alpha\in(0,1),
\;}
$$

$$
u(x,0)=u_0(x),\qquad u(0,t)=u(1,t)=0 .
$$

This is the canonical subdiffusion / anomalous-transport model. Its decisive property for our purposes is a **weak singularity at $t=0$**: even for smooth data, the solution behaves like $u(x,t)=u_0(x)+O(t^{\alpha})$, so $\partial_t u \sim t^{\alpha-1}\to\infty$ as $t\to 0^{+}$. There is a sharp initial layer in time. Uniform-collocation methods (including standard fPINN) systematically under-resolve it; **wavelet multiresolution placed densely near $t=0$ resolves it** — this is the showcase, the direct analogue of the boundary layers the source paper already conquered, moved into the time-fractional setting.

A clean closed-form benchmark exists. With $f\equiv 0$ and $u_0(x)=\sin(\pi x)$, the exact solution is the Mittag-Leffler function

$$
u(x,t)=E_{\alpha}\!\big(-\kappa\pi^{2} t^{\alpha}\big)\,\sin(\pi x),
\qquad
E_{\alpha}(z)=\sum_{m=0}^{\infty}\frac{z^{m}}{\Gamma(\alpha m+1)},
$$

using the identity ${}^{C}_{0}D_t^{\alpha}\big[E_{\alpha}(-\lambda t^{\alpha})\big]=-\lambda\,E_{\alpha}(-\lambda t^{\alpha})$. This gives an exact reference with the characteristic singular $t$-behaviour for error reporting, and the manufactured-solution route ($u$ prescribed, $f$ back-computed) supplies controllable test cases at will.

### 2.3 Benchmark target — space-fractional advection–diffusion

To enable a head-to-head comparison against the original fPINN benchmark:

$$
\partial_t u(x,t)\;+\;v\,\partial_x u(x,t)\;=\;-\,K\,(-\Delta)^{s}\,u(x,t)\;+\;f(x,t),
\qquad s\in(\tfrac12,1),
$$

on a bounded domain with the fractional operator restricted appropriately (volume constraints in $\mathbb{R}\setminus(0,1)$, the nonlocal analogue of boundary conditions). Sweeping $s$ traces the local ($s\to1$) to strongly nonlocal regimes and lets us report accuracy and cost against fPINN / MC-fPINN on identical problems.

### 2.4 Stress target — singularly perturbed fractional problem

To lean directly on W-PINN's demonstrated edge, a fractional problem with a small parameter producing a steep layer, e.g.

$$
\epsilon\,(-\Delta)^{s} u(x) + b(x)\,\partial_x u(x) + c(x)\,u(x) = f(x),
\qquad 0<\epsilon\ll 1 ,
$$

whose solution develops a thin transition layer as $\epsilon\to0$ — the fractional counterpart of the source paper's Example 1.

---

## 3. How W-PINN applies — *unchanged* — to these equations

This is the crux of the "application, not architecture" claim. The source paper expands the solution in a fixed family of scaled/translated mother wavelets,

$$
u(x,t)=\sum_{n} c_n\,\Psi_n(x,t)+B,
\qquad \Psi_{j,k}(z)=\sqrt{2^{j}}\,\psi\!\left(2^{j}z-k\right),
$$

with the $c_n$ learned and $B$ a trainable bias, and it evaluates derivatives through **precomputed matrices** $W$, $D_1W$, $D_2W$ whose columns are the wavelets and their derivatives sampled at the collocation points. The fractional application adds **one more matrix of exactly the same kind** — nothing else changes.

Because every operator here is linear in $u$ and the basis is fixed, applying a fractional operator $\mathcal{D}^{\alpha}$ to the expansion commutes with the sum:

$$
\mathcal{D}^{\alpha} u \;=\; \sum_n c_n\,\mathcal{D}^{\alpha}\Psi_n \;=\; \big(\mathbf{W}^{\alpha} c\big),
\qquad
\big(\mathbf{W}^{\alpha}\big)_{in}=\big(\mathcal{D}^{\alpha}\Psi_n\big)(z_i).
$$

For the **Caputo time operator**, each column is

$$
\big(\mathbf{W}^{\alpha}_{t}\big)_{in}
=\frac{1}{\Gamma(1-\alpha)}\int_0^{t_i}\frac{\Psi_n'(\tau)}{(t_i-\tau)^{\alpha}}\,d\tau,
$$

where $\Psi_n'$ is known **analytically** (the mother wavelets are smooth, closed-form), so even the inner derivative needs no AD; the singular integral is evaluated once by a stable graded/product-integration quadrature and stored. For the **fractional Laplacian**, the cleanest route is the Fourier symbol,

$$
(-\Delta)^{s}\Psi_n(x)=\mathcal{F}^{-1}\!\big[\,|\xi|^{2s}\,\widehat{\Psi_n}(\xi)\,\big](x),
$$

again computed once per basis function (the wavelets have known/computable Fourier transforms).

The subdiffusion residual then assembles purely from matrix–vector products:

$$
\underbrace{\mathbf{W}^{\alpha}_{t}\,c}_{{}^{C}D_t^{\alpha}u}
\;-\;\kappa\,\underbrace{D_2W_x\,c}_{\partial_{xx}u}
\;-\;f \;=\; 0,
$$

and the loss is the same residual + initial/boundary mean-squared form the source paper uses. **There is no automatic differentiation anywhere, no auxiliary collocation points, and no per-iteration re-discretization of the fractional operator** — the matrix $\mathbf{W}^{\alpha}_{t}$ is built once before training. This is precisely the property the method was designed around, now carrying the fractional operator.

---

## 4. Novelty — honest positioning

What exists, and where this sits relative to it:

| Existing approach | Mechanism for the fractional operator | Limitation this application targets |
|---|---|---|
| **fPINN** (Pang, Lu & Karniadakis, 2019) | Black-box MLP for $u$; **hybrid** — AD for integer-order terms, separate numerical discretization for the fractional term | Awkward hybrid; auxiliary points; grid discretization brings a curse of dimensionality |
| **MC-fPINN** | Monte-Carlo evaluation of the singular fractional integral | High variance from the singular integrand; many hyperparameters; convergence sensitivity |
| **Classical wavelet–Galerkin / collocation for fractional PDEs** | Wavelets as a numerical basis (no learning) | Not a learning framework; no NN coefficient model, no PINN-style soft constraints / inverse-problem path |
| **This proposal (W-PINN application)** | Solution in a fixed wavelet basis; fractional operator as a **single precomputed operational matrix**; coefficients learned; **fully AD-free** | — |

The honest reading: **the application is the contribution, not the method.** Two things are *not* claimed as novel — (i) using neural networks for fractional PDEs (fPINN/MC-fPINN own that), and (ii) using wavelets as a numerical basis for fractional operators (classical numerical analysis owns that). What appears genuinely open is the *combination*: running the W-PINN solution-as-wavelet-coefficients / precomputed-operational-matrix framework on fractional PDEs, and showing that it removes fPINN's hybrid-AD/auxiliary-point overhead **and** resolves the $t=0$ (or boundary-layer) singularity that uniform-collocation PINNs miss. Because it is an "obvious" extension of a mature field, the paper lives or dies on a **specific equation where the win is unambiguous** — not on breadth.

For W-PINN-application fit, this target scores well on the criteria of §1 (fixed singular kernel, linear operator, AD genuinely blocked, singularities in the solution), which is why it was selected over peridynamic fracture (where bond-breaking makes the operator solution-dependent and defeats the precompute-once advantage).

---

## 5. Minimal experiment plan and pre-committed success criterion

1. **1D time-fractional subdiffusion (§2.2)** with the Mittag-Leffler exact solution. Build $\mathbf{W}^{\alpha}_t$ on a wavelet basis with extra resolution levels near $t=0$. Verify $\mathbf{W}^{\alpha}_t c \to {}^{C}D_t^{\alpha}u$ at the expected quadrature order; train AD-free; report relative $L^2$ error and wall-clock against fPINN and MC-fPINN; and show the multiresolution resolves the initial layer that uniform fPINN under-resolves. Sweep $\alpha\in\{0.3,0.5,0.7,0.9\}$.
2. **Space-fractional advection–diffusion (§2.3)** as the direct fPINN benchmark; sweep $s$.
3. **Singularly perturbed fractional problem (§2.4)** to exercise the boundary-layer regime, sweeping $\epsilon$.

**Success criterion (fix before running):** AD-free wall-clock *and* relative $L^2$ accuracy that beat fPINN / MC-fPINN on a problem whose solution has a weak singularity that uniform-collocation PINNs demonstrably miss. Matching them at equal accuracy with lower cost — or higher accuracy at the singularity at equal cost — counts; anything that merely reproduces fPINN with no efficiency or accuracy edge is a sanity check, not a result.

---

## 6. Caveats to plan around

- **Building $\mathbf{W}^{\alpha}$ accurately is where the real work is.** The Caputo and fractional-Laplacian kernels are singular; the operational matrix must be assembled with a stable graded/product-integration quadrature (time) or the Fourier-symbol route (space). Do this carefully once; it is the method's whole leverage.
- **Wavelet choice matters.** The source paper's smooth Gaussian/Mexican-hat/Morlet wavelets are adequate (they superpose to approximate the singular $t$-profile), but watch conditioning at high resolution; compactly supported orthonormal wavelets (Daubechies) may give a better-conditioned, sparser $\mathbf{W}^{\alpha}$ and are worth comparing.
- **Curse of dimensionality persists.** The basis count grows like $\sum_j O(2^{jd})$ — the source paper's own stated limitation. Keep claims to 1D/2D; do not market this against high-dimensional fractional solvers.

---

## 7. Bottom line

Fractional PDEs let W-PINN be applied **exactly as published** — one extra precomputed operational matrix, no architectural change — to a nonlocal, fixed-kernel, AD-resistant equation class whose solutions carry precisely the singularities the method already resolves, and which the source paper itself flagged as the ideal next step. The defensible product is a focused application paper anchored on time-fractional subdiffusion, benchmarked AD-free against fPINN / MC-fPINN, that wins clearly on the $t=0$ singularity. That is a clean, in-scope, novel *application* of wavelet-PINN — not a new method.

*Prior-art attributions should be verified before submission: fPINN — Pang, Lu & Karniadakis, SIAM J. Sci. Comput. 41(4), 2019 (arXiv:1811.08967); MC-fPINN and tempered/high-dimensional variants — e.g. arXiv:2406.11708; source method — Pandey, Singh & Behera, arXiv:2409.11847v3.*
