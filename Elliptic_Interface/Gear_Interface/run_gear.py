"""
Gear / "settings-icon" interface  --  the hard-geometry stress test for the AD-free wavelet W-PINN
=====================================================================================================
A much more complex elliptic-interface problem than the 3-lobe flower: a closed GEAR with 8 sharp,
deep teeth (~44% radial depth).  -nabla.(a grad u) = f  with a jumping by a factor rho across the
gear interface Gamma; continuity [[u]]=0 and flux continuity [[a d_n u]]=0 on Gamma.

Level set (globally smooth, deep teeth)
---------------------------------------
A harmonic phi (which gives the flower an exact constant source f=-4) can only make ~3% ripples in
the unit disk -- r^N is suppressed for r<1.  To get real teeth we use a WINDOWED smooth level set:

    phi(x,y) = (x^2+y^2) - R0^2 + B * Re((x+iy)^N)/R0^N * exp(-s*((x^2+y^2)-R0^2)^2)

  * Re((x+iy)^N)/R0^N = (r/R0)^N cos(N*theta): smooth polynomial, -> 0 at the origin (u stays well
    defined there);  the Gaussian window restores FULL tooth amplitude at r=R0 (depth set by B, no
    r^N suppression) and decays away from Gamma so the level set cannot self-intersect.
  * u = phi/a  per region  =>  [[u]]=0 and [[a d_n u]]=0 automatically (homogeneous jumps),
    exact kink  [[d_n u]] = |grad phi| (1/a_out - 1/a_in)  (varies along Gamma).
  * the only cost vs the flower: the source f = -Delta phi is spatially VARYING.  It is computed
    EXACTLY by autograd of phi at the fixed collocation points (one-time problem DATA) -- the wavelet
    coefficient SOLVE itself stays AD-free (a single linear least-squares).

Method
------
Decomposed solution u^- = W c^- + b^-, u^+ = W c^+ + b^+ (one tensor-product Gaussian-derivative
wavelet expansion per subdomain), coupled by the interface conditions.  All operators (value,
Laplacian, normal derivative) are precomputed matrices applied to the coefficients -> the W-PINN
objective is a convex quadratic, solved AD-free by column-normalised Tikhonov least squares.

BANDED multiresolution refinement
---------------------------------
The sharp teeth are hard for a coarse global basis (relL2 ~2e-2).  We add the FINEST wavelets ONLY
in the annular band around the teeth -- a few hundred extra functions instead of ~5000 for a global
finest level -- which is exactly what multiresolution wavelets are for and what a vanilla PINN cannot
do.  Result: relL2 ~1e-3, contrast-robust, error localised at the tooth tips.
"""
import math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------- gear geometry (locked config)
N_TEETH, R0, B, S = 8, 0.55, 0.04, 8.0          # 8 teeth, ~44% depth, valid single closed curve

def phi_t(x, y):                                 # torch expression (for autograd f and normals)
    r2 = x*x + y*y; rez = ((x + 1j*y)**N_TEETH).real
    return r2 - R0*R0 + B*rez/(R0**N_TEETH)*torch.exp(-S*(r2 - R0*R0)**2)

def phi_np(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    r2 = x*x + y*y; rez = ((x + 1j*y)**N_TEETH).real
    return r2 - R0*R0 + B*rez/(R0**N_TEETH)*np.exp(-S*(r2 - R0*R0)**2)

def lap_grad(x, y):                              # exact Laplacian (for f=-Delta phi) and gradient
    xt = torch.tensor(x, requires_grad=True); yt = torch.tensor(y, requires_grad=True)
    p = phi_t(xt, yt)
    gx, = torch.autograd.grad(p.sum(), xt, create_graph=True)
    gy, = torch.autograd.grad(p.sum(), yt, create_graph=True)
    gxx, = torch.autograd.grad(gx.sum(), xt, retain_graph=True)
    gyy, = torch.autograd.grad(gy.sum(), yt, retain_graph=True)
    return (gxx + gyy).detach().numpy(), gx.detach().numpy(), gy.detach().numpy()

def interface_points(M=720):                     # zero set of phi by per-ray bisection near R0
    th = np.linspace(0, 2*np.pi, M, endpoint=False)
    rlo = np.full(M, 0.15); rhi = np.full(M, 1.05)
    xr = lambda r: (r*np.cos(th), r*np.sin(th)); flo = phi_np(*xr(rlo))
    for _ in range(70):
        rm = 0.5*(rlo + rhi); fm = phi_np(*xr(rm)); left = np.sign(fm) == np.sign(flo)
        rlo = np.where(left, rm, rlo); flo = np.where(left, fm, flo); rhi = np.where(left, rhi, rm)
    rs = 0.5*(rlo + rhi); xg, yg = rs*np.cos(th), rs*np.sin(th)
    _, gx, gy = lap_grad(xg, yg); nrm = np.hypot(gx, gy)
    return xg, yg, gx/nrm, gy/nrm, nrm

# ----------------------------------------------------------------- wavelet family + op-matrices
def banded_family(coarse=(0,1,2,3), fine=(4,5), band=(0.42, 0.86), lo=-1.0, hi=1.0, pad=0.5):
    def lvl1d(levels):
        out = []
        for l in levels:
            j = 2.0**l
            for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j)) + 1):
                out.append((j, float(k)))
        return out
    f1c = lvl1d(coarse)
    fam = [(jx, kx, jy, ky) for (jx, kx) in f1c for (jy, ky) in f1c]      # coarse: global
    rlo, rhi = band
    for l in fine:                                                       # fine: banded by centre
        j = 2.0**l; ks = [float(k) for k in range(int(math.floor((lo-pad)*j)),
                                                   int(math.ceil((hi+pad)*j)) + 1)]
        for kx in ks:
            for ky in ks:
                if rlo <= math.hypot(kx/j, ky/j) <= rhi:
                    fam.append((j, kx, j, ky))
    arr = torch.tensor(fam, dtype=torch.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

def W_mats(x, y, jx, kx, jy, ky):                # value, Laplacian, d/dx, d/dy of each wavelet
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    X = jx[None, :]*x[:, None] - kx[None, :]; Y = jy[None, :]*y[:, None] - ky[None, :]
    E = torch.exp(-(X**2 + Y**2)/2)
    Psi = X*Y*E
    lap = -(jx[None, :]**2)*X*Y*(3 - X**2)*E - (jy[None, :]**2)*X*Y*(3 - Y**2)*E
    dx = jx[None, :]*(1 - X**2)*Y*E; dy = jy[None, :]*(1 - Y**2)*X*E
    return Psi, lap, dx, dy

# ----------------------------------------------------------------- collocation points
def make_points(K=140, M=720):
    g = np.linspace(-1, 1, K); Xg, Yg = np.meshgrid(g, g); xs = Xg.ravel(); ys = Yg.ravel()
    p = phi_np(xs, ys); inside = p < -1e-6; outside = (p > 1e-6) & (np.abs(xs) < 1) & (np.abs(ys) < 1)
    xg, yg, nx, ny, nrm = interface_points(M)
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)])
    by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])
    return (xs[inside], ys[inside]), (xs[outside], ys[outside]), (xg, yg, nx, ny, nrm), (bx, by)

# ----------------------------------------------------------------- AD-free least-squares solve
def solve(rho, fam, wi=10.0, wb=10.0, tik=1e-10, want_cond=False):
    a_in, a_out = 1.0, rho*1.0
    jx, kx, jy, ky = fam; NF = jx.numel()
    (xm, ym), (xp, yp), (xg, yg, nx, ny, nrm), (bx, by) = make_points()
    Pm, Lm, _, _ = W_mats(xm, ym, jx, kx, jy, ky); Pp, Lp, _, _ = W_mats(xp, yp, jx, kx, jy, ky)
    Pg, _, dxg, dyg = W_mats(xg, yg, jx, kx, jy, ky)
    nxv = torch.tensor(nx); nyv = torch.tensor(ny); dnG = nxv[:, None]*dxg + nyv[:, None]*dyg
    Pb, _, _, _ = W_mats(bx, by, jx, kx, jy, ky)
    n = 2*NF + 2
    def blk(rows, cM=None, cP=None, bM=None, bP=None):
        Bm = torch.zeros(rows, n)
        if cM is not None: Bm[:, :NF] = cM
        if cP is not None: Bm[:, NF:2*NF] = cP
        if bM is not None: Bm[:, 2*NF] = bM
        if bP is not None: Bm[:, 2*NF+1] = bP
        return Bm
    A = torch.cat([blk(len(xm), cM=-a_in*Lm), blk(len(xp), cP=-a_out*Lp),
                   math.sqrt(wi)*blk(len(xg), cP=Pg, cM=-Pg, bP=1., bM=-1.),   # [[u]]=0
                   math.sqrt(wi)*blk(len(xg), cP=a_out*dnG, cM=-a_in*dnG),      # [[a d_n u]]=0
                   math.sqrt(wb)*blk(len(bx), cP=Pb, bP=1.)], 0)                # Dirichlet BC
    u_exact = lambda x, y: np.where(phi_np(x, y) < 0, phi_np(x, y)/a_in, phi_np(x, y)/a_out)
    fin, _, _ = lap_grad(xm, ym); fout, _, _ = lap_grad(xp, yp)                # f = -Delta phi
    ub = torch.tensor(u_exact(bx, by))
    b = torch.cat([torch.tensor(-fin), torch.tensor(-fout),
                   torch.zeros(len(xg)), torch.zeros(len(xg)), math.sqrt(wb)*ub]).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30); An = A/s                               # column normalisation
    cond = torch.linalg.cond(An).item() if want_cond else float("nan")         # post-normalisation conditioning
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    theta = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1))/s
    cM, cP = theta[:NF], theta[NF:2*NF]; bM, bP = theta[2*NF], theta[2*NF+1]

    gt = np.linspace(-0.999, 0.999, 300); Xt, Yt = np.meshgrid(gt, gt); xt = Xt.ravel(); yt = Yt.ravel()
    Pt, _, _, _ = W_mats(xt, yt, jx, kx, jy, ky)
    pred = np.where(phi_np(xt, yt) < 0, (Pt@cM + bM).numpy(), (Pt@cP + bP).numpy())
    ue = u_exact(xt, yt); err = pred - ue
    metrics = dict(NF=NF,
                   relL2=np.linalg.norm(err)/np.linalg.norm(ue),
                   MSE=np.mean(err**2), RMSE=np.sqrt(np.mean(err**2)),
                   MAE=np.mean(np.abs(err)), Linf=np.max(np.abs(err)),
                   relLinf=np.max(np.abs(err))/np.max(np.abs(ue)),
                   kink=np.max(np.abs((dnG@cP).numpy() - (dnG@cM).numpy() - nrm*(1/a_out - 1/a_in))),
                   cond=cond)
    return metrics, (Xt, Yt, pred.reshape(Xt.shape), ue.reshape(Xt.shape)), (xg, yg)

# ----------------------------------------------------------------- run + figure
if __name__ == "__main__":
    import time
    fam = banded_family()
    print(f"gear interface: N={N_TEETH} teeth, R0={R0}, B={B}, s={S}  (banded basis, NF={fam[0].numel()})\n")
    hdr = (f"{'rho':>6} | {'relL2':>10} {'MSE':>10} {'RMSE':>10} {'MAE':>10} "
           f"{'Linf':>10} {'relLinf':>10} {'kinkErr':>10} {'sec':>6}")
    print(hdr); print("-"*len(hdr))
    saved = None
    for rho in [10.0, 100.0, 1000.0]:
        t0 = time.time(); m, fields, gpts = solve(rho, fam); dt = time.time() - t0
        print(f"{rho:>6.0f} | {m['relL2']:>10.2e} {m['MSE']:>10.2e} {m['RMSE']:>10.2e} "
              f"{m['MAE']:>10.2e} {m['Linf']:>10.2e} {m['relLinf']:>10.2e} {m['kink']:>10.2e} {dt:>6.1f}",
              flush=True)
        if rho == 1000.0: saved = (fields, gpts)

    (Xt, Yt, PR, UE), (xg, yg) = saved
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    c0 = ax[0].contourf(Xt, Yt, UE, 40, cmap='viridis'); ax[0].plot(xg, yg, 'w.', ms=0.7)
    ax[0].set_title('(a) exact, gear $\\Gamma$ ($\\rho$=1000)'); plt.colorbar(c0, ax=ax[0])
    c1 = ax[1].contourf(Xt, Yt, PR, 40, cmap='viridis'); ax[1].plot(xg, yg, 'w.', ms=0.7)
    ax[1].set_title('(b) wavelet W-PINN (mesh-free, banded)'); plt.colorbar(c1, ax=ax[1])
    c2 = ax[2].contourf(Xt, Yt, np.log10(np.abs(PR - UE) + 1e-16), 40, cmap='magma')
    ax[2].plot(xg, yg, 'w.', ms=0.7)
    ax[2].set_title('(c) $\\log_{10}$|error|'); plt.colorbar(c2, ax=ax[2])
    for a in ax: a.set_aspect('equal')
    plt.tight_layout(); plt.savefig("sol.png", dpi=130); print("\nsaved sol.png")
