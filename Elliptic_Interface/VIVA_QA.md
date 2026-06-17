# Viva Q&A — Wavelet-PINN for Elliptic Interface Problems

> Anticipated examiner questions with crisp, defensible answers. Organised by theme.
> The five you MUST answer cold are starred (★).

---

## A. Domain, geometry, boundary & initial conditions (the questions your seniors asked)

**Q. What is your computational domain, and why?**
The outer domain is the square $\Omega=(-1,1)^2$. It is just a bounding box that holds the *interface*; the physics of interest is the material interface inside it, not the outer boundary. A square gives a trivial place to impose the outer boundary condition and is the standard domain in the interface-method literature, so results are comparable.

**★Q. Why a rectangular domain — why not a complex domain?**
Two separate things are being conflated: the **outer boundary** and the **interface**. The hard, interesting geometry in this problem is the *interface* (where the coefficient jumps), and that is where I *did* use complex geometry — the 3-lobed flower. The outer boundary is deliberately kept simple (a square) because its shape is irrelevant to the contribution and only adds bookkeeping. The method is mesh-free (collocation-based), so a complex *outer* domain is also handled — it just wouldn't demonstrate anything new. So: simple outer domain on purpose, complex *interface* by design.

**Q. Why did you do the circle before the flower?**
Standard validate-then-generalise practice: (1) the circle is the canonical benchmark everyone reports, enabling direct comparison; (2) it has a clean closed-form exact solution; (3) its normal is trivially radial, which isolates the *solver* from geometry-handling — if it failed on a circle there is no point trying harder shapes; (4) the flower then proves generality (varying normals, Newton-found interface points). You earn the hard geometry by nailing the easy one.

**★Q. What are your boundary conditions?**
Dirichlet on the outer square boundary: $u=h$ given (taken from the manufactured exact solution). Plus two **interface conditions** on $\Gamma$: continuity of the solution $[\![u]\!]=0$ and continuity of the flux $[\![a\,\partial_n u]\!]=0$. The interface conditions are *not* boundary conditions — they are internal coupling between the two subdomains.

**Q. What is your initial condition?**
There isn't one. This is an **elliptic (steady-state)** problem — it has no time variable, so no initial condition is needed or meaningful. Only boundary conditions and the interface conditions close the problem. (If the examiner pushes: a time-dependent version, e.g. parabolic interface diffusion, *would* need an initial condition; that's future work.)

**Q. Could you use Neumann or Robin boundary conditions instead of Dirichlet?**
Yes — they enter as different soft-constraint loss terms (Neumann uses the normal-derivative matrix on the boundary instead of the value matrix). Dirichlet was chosen because the manufactured benchmark supplies clean boundary values and it's the standard test.

**Q. Why is the interface circle/flower placed inside, not touching the boundary?**
To keep the two subdomains well-separated and the outer boundary entirely in one region ($\Omega^+$), which makes the boundary condition unambiguous. Interfaces touching the boundary add corner singularities that are a separate complication.

---

## B. The PDE and the physics

**Q. State your governing equation.**
$-\nabla\cdot(a(x)\nabla u)=f$ on $\Omega$, with $a$ piecewise constant: $a=a^-$ inside $\Gamma$, $a^+$ outside.

**Q. What does $a$ represent physically?**
A material coefficient — thermal/electrical conductivity, diffusivity, or dielectric permittivity — that jumps across the interface between two materials. The contrast $\rho=a^+/a^-$ is the conductivity ratio.

**★Q. The solution is continuous but you keep saying "kink." Explain.**
Flux continuity $[\![a\,\partial_n u]\!]=0$ with $a^-\neq a^+$ forces $[\![\partial_n u]\!]\neq0$: the normal derivative jumps even though $u$ itself is continuous. So $u$ has a gradient **kink** (a $C^0$-but-not-$C^1$ point) at the interface. That kink is the feature the method must resolve.

**Q. Why is the *flux* continuous and not the gradient?**
Physical conservation: no source/sink sits on the interface, so the normal flux $a\,\partial_n u$ (e.g. heat flux) must be continuous. Since $a$ jumps, the gradient must jump to compensate. It's a consequence of integrating the PDE across a thin pillbox straddling $\Gamma$ (the divergence theorem / weak form).

**Q. Where does $f=-4$ come from in your benchmark?**
Manufactured solution. I pick a level-set $\varphi$ (the interface is $\varphi=0$) and set $u=\varphi/a$ per region. Then $f=-\nabla\cdot(a\nabla(\varphi/a))=-\Delta\varphi$. For $\varphi=x^2+y^2-r_0^2$ (+ a harmonic cubic for the flower), $\Delta\varphi=4$, so $f\equiv-4$. The construction makes both jump conditions hold automatically and gives an exact solution to measure error against.

---

## C. The method (Wavelet-PINN)

**★Q. What is a Wavelet-PINN and how does it differ from a standard PINN?**
A standard PINN represents $u$ by an MLP and uses automatic differentiation to form the PDE operator. A Wavelet-PINN represents $u$ as a fixed sum of scaled/translated wavelets, $u=\sum_n c_n\Psi_n+b$, learns the coefficients $c_n$, and applies every differential operator through a **precomputed operational matrix** (the wavelets' analytic derivatives sampled at the collocation points). No automatic differentiation is used to build the operator.

**Q. What is the operational matrix and why is it an advantage?**
A matrix whose columns are the wavelets (or their analytic Laplacian/derivatives) evaluated at the collocation points. Because the basis is fixed and the operator is linear, this matrix is built **once** and reused; applying the operator is then a matrix-vector product. It removes per-iteration autodiff and gives a linear system.

**★Q. Why two wavelet expansions instead of one? (the core idea)**
A single smooth ($C^1$) global expansion *cannot represent the kink* — it would force $[\![\partial_n u]\!]=0$, violating flux continuity. The solution is smooth *inside* each subdomain and non-smooth only at the join. So I use one expansion per subdomain, $u^\pm=W^\pm c^\pm+b^\pm$, coupled by the interface conditions; the kink emerges from the *coupling*, not from any single basis. This decomposition is the load-bearing idea — my ablation shows it gives ~360× lower error than a single expansion with the *same* basis.

**Q. Is this just XPINN with wavelets?**
Structurally related (both decompose at the interface), but the mechanism differs: XPINN uses MLP subnetworks trained by gradient descent with autodiff; mine uses fixed wavelet expansions with precomputed operational matrices, solved by least squares — AD-free and, for linear problems, non-iterative. Empirically mine beats XPINN by ~140×–5000× and is ~30× faster at high contrast.

---

## D. Wavelets

**Q. Which wavelet do you use and why?**
The Gaussian-derivative ("Mexican-hat"-type) wavelet, $\psi(z)=-z\,e^{-z^2/2}$ in 1D, tensor-producted in 2D — the same family as the source paper. It's smooth with closed-form derivatives (so the operational matrices are analytic) and is good at sharp-but-smooth features.

**Q. What do the levels and translations mean?**
$\Psi_{j,k}(z)=$ wavelet at scale $j=2^{\text{level}}$ and location $k$. Higher level = finer scale = more localised. Summing over levels and translations is the multiresolution representation; placing finer levels near a feature refines it there.

**Q. Why only 3 levels [0,1,2]? Isn't more better?**
The subdomain solutions are smooth, so coarse levels suffice — the kink is handled by decomposition, not by fine wavelets. Crucially, *over-completeness* (too many fine levels) is what wrecked the conditioning (I measured cond ≈ $10^{196}$ with 10 levels in 1D vs $10^{21}$ with 3 levels in 2D). So I match basis richness to solution smoothness: parsimony is both accurate and well-conditioned. There's an accuracy↔conditioning frontier (~10× accuracy per added level, modest conditioning cost) and [0,1,2]–[0,1,2,3] is the sweet spot.

**Q. Would Daubechies (orthonormal, compact support) wavelets be better?**
Likely better-conditioned and sparser, which matters if you train by gradient descent. I use least squares, where the current conditioning is acceptable, so it wasn't necessary — but it's the natural next lever and a listed future direction.

---

## E. Solver / optimisation

**★Q. Why do you solve by least squares instead of training like a normal PINN?**
For a PDE *linear* in $u$ with a *fixed* basis, the W-PINN loss is a convex quadratic in the coefficients — its global minimum is a (weighted) linear least-squares solution. So gradient descent (Adam) is the wrong tool: it's slow and, on the ill-conditioned wavelet basis, fails to converge (I observed this directly). The least-squares solve is exact, AD-free, and sub-second. Adam is only needed for genuinely nonlinear terms.

**Q. Your condition number was astronomically large yet the answer is accurate. How?**
The least-squares solver uses an SVD (`gelsd`), which truncates the negligible singular values — effectively a regularisation. The true solution lives in the well-conditioned subspace (the coarse, low-frequency wavelets), so the SVD recovers it accurately despite the formal ill-conditioning. I also column-normalise (Jacobi preconditioning) and add mild Tikhonov to make this robust. It's a known, principled way to handle redundant bases.

**Q. What do the loss weights $w_{\text{iface}}, w_{\text{bc}}$ do?**
They balance the few interface/boundary equations against the many interior residual equations so the constraints aren't drowned out. I report sensitivity to them rather than hiding it.

---

## F. Results & validation

**Q. How do you know your answer is correct?**
Three layers: (1) I unit-test the operational matrices against finite differences (~$10^{-7}$) *before* training; (2) I compare against a manufactured exact solution and report relative $L^2$, max error, and the interface-specific *flux-kink* error; (3) I sweep contrast and compare against baselines (vanilla MLP, XPINN).

**Q. Why report the kink/flux error separately from $L^2$?**
Because global $L^2$ can look fine while the interface physics (the flux jump) is wrong. The kink error is the honest, problem-specific metric.

**Q. What happens as the contrast $\rho$ grows?**
My method stays accurate ($\sim10^{-5}$–$10^{-6}$) and the flux-kink is captured exactly, while the XPINN baseline degrades (kink error grows from $10^{-3}$ to $\sim0.12$ at $\rho=10^4$). The advantage *grows* with contrast — the regime where vanilla methods struggle most.

**Q. Why does vanilla PINN fail here?**
Spectral bias: a smooth MLP cannot place a kink and smears it; it plateaus at ~16% relative error and predicts essentially zero gradient jump. This is the documented failure mode the method is designed to beat.

---

## G. The inverse problem

**Q. What is the inverse problem and why does it matter?**
Forward: given the coefficient/geometry, find $u$. Inverse: given sparse, noisy interior measurements of $u$, recover the unknown contrast *or the interface shape*. Recovering the interface location/shape from data is something classical FEM/IIM solvers don't do natively — it's the contribution's differentiator.

**★Q. How do you solve the inverse problem?**
Outer-loop over the few physical parameters (contrast, or shape coefficients) $\times$ inner AD-free least-squares forward solve. For each candidate, I solve the forward problem (~0.4 s) and measure the interior data misfit; a derivative-free optimiser (golden-section / Nelder–Mead) minimises it. The *speed* of the forward solve is what makes the inverse tractable.

**Q. Isn't near-exact recovery just an "inverse crime"?**
No, and I tested it. (1) The data is the *analytic* solution, independent of the wavelet discretisation, so the data-generating model differs from the inversion model. (2) Recovery error tracks the forward discretisation floor (~$10^{-6}$), not machine precision — a true inverse crime gives ~$10^{-12}$. (3) I ran a model-mismatch test: when I invert an ellipse truth with a *wrong* (circle) model family, the data residual *cannot* fall below ~$6\times10^{-3}$ — so a mis-specified model is *diagnosable* from the residual, the opposite of an artifact. The correct family recovers it exactly.

**Q. How robust is recovery to noise and sparse data?**
Graceful: even 10 data points at 5% noise recover the radius to ~0.9%; 1% noise gives ~$10^{-3}$. I report the full noise×data-density frontier.

**Q. Is the inverse problem identifiable / unique?**
With known forcing and boundary data, varying the geometry changes the interior solution measurably, so sparse interior data disambiguates the parameters. For very sparse or very noisy data it becomes ill-posed; that's why I report the frontier and would add priors/regularisation if needed.

---

## H. Novelty, positioning, limitations

**★Q. What exactly is novel here?**
Not "a neural method for interface problems" — that space is crowded (cusp-capturing PINN, IG-PINN, AE-PINN, INN, I-PINN, DR-PINN, XPINN), all MLP-based. The novelty is the *mechanism × application*: the **first application of the AD-free wavelet operational-matrix W-PINN to elliptic interface problems** — domain-decomposed, least-squares-solvable, contrast-robust, with interface-shape recovery. The differentiator is the wavelet/operational-matrix route plus the inverse capability.

**Q. So is the contribution a new method or an application?**
An application (by design — the goal was to find a novel *application* of the wavelet-PINN, not invent an architecture). The method is used as published; the equation class and the inverse extension are new for it.

**Q. What are the limitations?**
(1) Conditioning of the operational matrix needs care (parsimony / preconditioning); (2) basis count grows like $O(2^{jd})$, so it's a low-dimensional (1D/2D) method — curse of dimensionality; (3) the clean least-squares route relies on the problem being linear; (4) accuracy in 2D (~$10^{-6}$) is below the 1D level, limited by coarse Gaussian wavelets + conditioning; (5) interface touching the boundary / strong corner singularities are not yet handled.

**Q. Where would you publish, and honestly at what level?**
A focused application paper at the mid-tier computational-mathematics level (CAMWA / AMC class), positioned against the MLP interface-PINN methods. The inverse-shape-recovery result is what could push it toward a higher tier with an added physical benchmark.

---

## I. Quick-fire definitions (likely rapid-fire)

- **Elliptic PDE:** steady-state, no time; e.g. $-\nabla\cdot(a\nabla u)=f$. (vs parabolic = diffusion-in-time, hyperbolic = waves.)
- **Interface:** the curve/surface where the coefficient $a$ is discontinuous.
- **Jump $[\![\cdot]\!]$:** the difference of a quantity across the interface (outside minus inside).
- **Collocation point:** a point where the PDE residual is enforced.
- **Manufactured solution:** a chosen exact $u$ from which $f$ and BCs are back-computed, for error measurement.
- **Multiresolution:** representing a function at multiple scales simultaneously via wavelet levels.
- **Spectral bias:** the tendency of MLPs to learn low frequencies first and under-resolve sharp features.
- **Mesh-free:** no grid/mesh; only scattered collocation points — so changing geometry needs no remeshing.
