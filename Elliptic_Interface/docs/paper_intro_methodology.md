# Wavelet-PINN for Elliptic Interface Problems — Introduction & Methodology (draft)

## Checklist of important points

**Introduction:**
1. What is an elliptic interface problem: `-∇·(a∇u)=f` with `a` jumping across a curve.
2. Why it matters: heat conduction between two materials, electrostatics/dielectrics, fluid flow, biology.
3. The key difficulty: `u` continuous but its gradient has a *kink* (flux continuity).
4. Existing approaches and gaps: FEM/IIM need meshing; standard PINNs smear the kink (spectral bias).
5. Our idea (one line): apply AD-free Wavelet-PINN (operational-matrix) with domain decomposition.
6. Contributions list.

**Methodology:**
1. Problem statement + governing equation + interface (jump) conditions.
2. Geometry, domain, collocation points.
3. Wavelet representation `u = Wc + b` and the wavelet family.
4. Two-expansion (domain decomposition) core idea.
5. Operational matrices (value, Laplacian, normal-derivative).
6. Loss / linear system + least-squares (SVD) solver.
7. Manufactured solution for validation.

---

## 1. Introduction

Many problems in science and engineering are described by partial differential equations
(PDEs) in which the material property is not the same everywhere. A very common case is when
two different materials are joined together — for example, two metals with different thermal
conductivity, or two regions with different dielectric constant. The curve (in 2D) where the
two materials meet is called the **interface**, and the material coefficient `a(x)` suddenly
*jumps* from one value to another as we cross this interface. Such problems are known as
**elliptic interface problems**, and the governing equation is

$$-\nabla\cdot\big(a(x)\,\nabla u\big) = f, \qquad x\in\Omega,$$

where `a(x)` is **piecewise constant**: it equals `a⁻` inside the interface and `a⁺` outside.
The ratio `ρ = a⁺/a⁻` is called the **contrast**.

These problems appear everywhere in practice — heat conduction across the boundary of two
solids, electric fields in materials with different permittivity, groundwater flow through
different soil layers, and modelling of biological cells.

The main difficulty in interface problems is the behaviour of the solution at the interface.
By physics, no heat (or flux) is created or lost at the interface, so the **normal flux
`a ∂u/∂n` must be continuous** across it. At the same time the solution `u` itself is
continuous. But since `a` jumps and the product `a ∂u/∂n` must stay continuous, the gradient
`∂u/∂n` is *forced to jump*. This means the solution has a **kink** at the interface — it is
continuous (`C⁰`) but not smooth (`C¹`). Capturing this kink correctly is the real challenge.

Classical numerical methods such as the Finite Element Method (FEM) and the Immersed Interface
Method (IIM) can solve these problems, but they usually require generating a mesh that fits the
interface, which is hard for complicated shapes and must be redone when the shape changes.

Recently, **Physics-Informed Neural Networks (PINNs)** have become popular because they are
mesh-free. However, a standard PINN represents `u` using a single smooth MLP. Because of
**spectral bias**, an MLP prefers smooth functions and tends to *smear out* the sharp kink at
the interface, giving large errors, especially when the contrast `ρ` is high.

In this work, we apply a **Wavelet-PINN (W-PINN)** based on the **operational-matrix** approach
to the elliptic interface problem. Instead of an MLP, we write the solution as a fixed sum of
wavelets and only learn their coefficients. All derivatives are applied through pre-computed
matrices, so no automatic differentiation is needed, and for these linear problems the whole
thing reduces to a fast least-squares solve. To capture the kink, we use **two separate wavelet
expansions** — one for each side of the interface — coupled by the interface conditions.

**Main contributions:**
- First application of the AD-free wavelet operational-matrix W-PINN to elliptic interface problems.
- A domain-decomposition idea (two wavelet expansions) so the gradient kink emerges from the coupling.
- A single least-squares (SVD) solve — AD-free and sub-second.
- Accuracy maintained at very high contrast `ρ`, where standard PINNs fail.
- Extension to the inverse problem — recovering contrast or interface shape from sparse, noisy data.

---

## 2. Methodology

### 2.1 Problem statement
On the square domain `Ω = (-1,1)²`, a closed curve `Γ` divides it into the inside region `Ω⁻`
and outside region `Ω⁺`. The governing equation is

$$-\nabla\cdot(a\,\nabla u) = f \quad\text{in } \Omega, \qquad
a=\begin{cases}a^- & \text{in }\Omega^-\\ a^+ & \text{in }\Omega^+\end{cases}$$

Dirichlet condition `u = h` on the outer boundary, and two interface conditions on `Γ`:

$$[\![u]\!] = 0, \qquad [\![a\,\partial_n u]\!] = 0.$$

Here `[[·]]` is the jump (outside minus inside) and `∂ₙ` is the outward-normal derivative.
This is a steady-state (elliptic) problem — no time variable, no initial condition.

### 2.2 Geometry and collocation points
The method is mesh-free; we use scattered points:
- **Interior points** (90×90 grid), split into `Ω⁻` and `Ω⁺` — PDE residual enforced here.
- **Interface points** (240 on `Γ`) with outward normals — interface conditions enforced here.
- **Boundary points** on the four square edges — Dirichlet condition enforced here.
- **Test points** (fine grid) — used only for error measurement.

Two interface shapes are tested: a **circle** (`r₀ = 0.5`, radial normal) as the benchmark,
and a 3-lobed **flower** shape to demonstrate complex geometry with varying normals.

### 2.3 Wavelet representation
The solution is written as a fixed sum of wavelets with unknown coefficients:

$$u(x,y) = \sum_{n} c_n\,\Psi_n(x,y) + b = W\,c + b,$$

We use the **Gaussian-derivative ("Mexican-hat") wavelet family**, tensor-producted in 2D. It
is smooth and has **closed-form derivatives**, needed to build the operational matrices. Each
wavelet `Ψⱼ,ₖ` has scale `j = 2^level` and location `k`. We use only **three levels [0,1,2]** —
a small (parsimonious) basis, since the solution is smooth inside each subdomain and the kink is
handled by decomposition, not by fine wavelets. Too many levels harms conditioning.

### 2.4 Core idea — two expansions (domain decomposition)
A single global expansion cannot represent the kink (it forces `[[∂ₙu]] = 0`). So we use one
expansion per subdomain:

$$u^- = W^-c^- + b^- \text{ in } \Omega^-, \qquad u^+ = W^+c^+ + b^+ \text{ in } \Omega^+,$$

coupled only through the interface conditions. The kink emerges from the coupling.

### 2.5 Operational matrices
Built once from the wavelets' analytic derivatives at the collocation points, then reused:
- **Value matrix `W`** — for boundary and continuity conditions.
- **Laplacian matrix `L`** — for the PDE residual (`a` constant in each region).
- **Normal-derivative matrix `Dₙ = nₓ Dₓ + n_y D_y`** — for the flux condition.

Applying any operator is just a matrix-vector product (no autodiff).

### 2.6 Linear system and solver
Stacking all conditions gives linear equations in `(c⁻, c⁺, b⁻, b⁺)`:

| Condition | Equation |
|---|---|
| PDE residual in `Ω⁻` | `a⁻ L⁻ c⁻ = -f` |
| PDE residual in `Ω⁺` | `a⁺ L⁺ c⁺ = -f` |
| Continuity on `Γ` | `W⁻c⁻ + b⁻ = W⁺c⁺ + b⁺` |
| Flux on `Γ` | `a⁻ Dₙ⁻c⁻ = a⁺ Dₙ⁺c⁺` |
| Dirichlet boundary | `W_bc c⁺ + b⁺ = h` |

Since the PDE is linear and the basis fixed, the loss is a convex quadratic in the coefficients.
We solve the stacked system directly as a **weighted linear least-squares problem** via SVD
(`gelsd`). Weights `w_iface`, `w_bc` balance the few interface/boundary equations against the
many interior ones. SVD truncates negligible singular values (mild regularisation); we also
column-normalise (Jacobi preconditioning) and add small Tikhonov for robustness. Solution is
machine-accurate and sub-second.

### 2.7 Validation — manufactured solution
Using the method of manufactured solutions, pick a level-set `φ` (interface `φ = 0`) and set
`u = φ/a` per region. With `φ = x² + y² - r₀²`, both jump conditions hold automatically and
`f = -Δφ = -4`. Knowing exact `u`, we report relative L² error, max error, and the
interface **flux-kink error**, which directly checks the interface physics.
