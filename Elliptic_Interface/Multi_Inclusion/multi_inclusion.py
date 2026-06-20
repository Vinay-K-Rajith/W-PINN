"""
Multiple disconnected inclusions -- topology stress test for the AD-free wavelet W-PINN
=======================================================================================
Several closed circular interfaces inside Omega=(-1,1)^2, each its own Omega^- region, all
embedded in one shared Omega^+ matrix.  -nabla.(a grad u)=f, a jumping by rho across every
interface; [[u]]=0 and [[a d_n u]]=0 on each circle.

Manufactured solution (same trick as the gear, generalised to many interfaces)
------------------------------------------------------------------------------
Smooth level set  phi(x,y) = prod_i ( |x-c_i|^2 - R_i^2 ).
  * disjoint disks => phi<0 inside exactly one disk, phi>0 outside all of them; zero set = union
    of the circles.  Near circle i, phi ~ (local quadratic)*(const) so grad phi is radial there.
  * u = phi/a per region  =>  [[u]]=0 and [[a d_n u]]=0 automatically; exact kink
    [[d_n u]] = |grad phi| (1/a_out - 1/a_in)  (varies along each circle).
  * source f = -Delta phi (spatially varying) computed EXACTLY by autograd at the fixed
    collocation points -- one-time DATA; the coefficient solve stays AD-free (linear lstsq).

Method: two-expansion decomposition u^- = W c^- + b^-, u^+ = W c^+ + b^+, coupled by the
interface conditions on every circle.  Banded multiresolution refinement: the finest wavelets are
placed ONLY in an annulus around EACH inclusion (scaled per radius), the coarse levels are global.
"""
import math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

# ----------------------------------------------------------------- geometry: 5 mixed-size inclusions
#                       cx,    cy,    R
INCL = np.array([(-0.45, -0.45, 0.27),
                 ( 0.48,  0.50, 0.20),
                 (-0.50,  0.52, 0.15),
                 ( 0.50, -0.48, 0.22),
                 ( 0.02,  0.00, 0.13)], dtype=float)

def phi_t(x, y):
    p = torch.ones_like(x)
    for cx, cy, R in INCL:
        p = p * ((x - cx)**2 + (y - cy)**2 - R*R)
    return p

def phi_np(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); p = np.ones_like(x)
    for cx, cy, R in INCL:
        p = p * ((x - cx)**2 + (y - cy)**2 - R*R)
    return p

def lap_grad(x, y):                              # exact Laplacian (f=-Delta phi) and gradient
    xt = torch.tensor(x, requires_grad=True); yt = torch.tensor(y, requires_grad=True)
    p = phi_t(xt, yt)
    gx, = torch.autograd.grad(p.sum(), xt, create_graph=True)
    gy, = torch.autograd.grad(p.sum(), yt, create_graph=True)
    gxx, = torch.autograd.grad(gx.sum(), xt, retain_graph=True)
    gyy, = torch.autograd.grad(gy.sum(), yt, retain_graph=True)
    return (gxx + gyy).detach().numpy(), gx.detach().numpy(), gy.detach().numpy()

def interface_points(M_each=170):                # exact circles; normals/|grad phi| from autograd
    xs, ys = [], []
    for cx, cy, R in INCL:
        th = np.linspace(0, 2*np.pi, M_each, endpoint=False)
        xs.append(cx + R*np.cos(th)); ys.append(cy + R*np.sin(th))
    xg = np.concatenate(xs); yg = np.concatenate(ys)
    _, gx, gy = lap_grad(xg, yg); nrm = np.hypot(gx, gy)
    return xg, yg, gx/nrm, gy/nrm, nrm           # outward normal = grad phi / |grad phi|

# ----------------------------------------------------------------- wavelet family + op-matrices
def banded_family(coarse=(0, 1, 2, 3), fine=(4, 5), halfband=0.13, lo=-1.0, hi=1.0, pad=0.5):
    def lvl1d(levels):
        out = []
        for l in levels:
            j = 2.0**l
            for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j)) + 1):
                out.append((j, float(k)))
        return out
    f1c = lvl1d(coarse)
    fam = [(jx, kx, jy, ky) for (jx, kx) in f1c for (jy, ky) in f1c]      # coarse: global
    for l in fine:                                                       # fine: banded per inclusion
        j = 2.0**l
        ks = [float(k) for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j)) + 1)]
        for kx in ks:
            for ky in ks:
                px, py = kx/j, ky/j
                for cx, cy, R in INCL:                                   # keep if near ANY circle
                    if abs(math.hypot(px-cx, py-cy) - R) <= halfband:
                        fam.append((j, kx, j, ky)); break
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
def make_points(K=140, M_each=170):
    g = np.linspace(-1, 1, K); Xg, Yg = np.meshgrid(g, g); xs = Xg.ravel(); ys = Yg.ravel()
    p = phi_np(xs, ys)
    inside = p < -1e-9
    outside = (p > 1e-9) & (np.abs(xs) < 1) & (np.abs(ys) < 1)
    xg, yg, nx, ny, nrm = interface_points(M_each)
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)])
    by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])
    return (xs[inside], ys[inside]), (xs[outside], ys[outside]), (xg, yg, nx, ny, nrm), (bx, by)

# ----------------------------------------------------------------- AD-free least-squares solve
def solve(rho, fam, wi=10.0, wb=10.0, tik=1e-10, want_cond=False, K=140, M_each=170, NT=300):
    a_in, a_out = 1.0, rho*1.0
    jx, kx, jy, ky = fam; NF = jx.numel()
    (xm, ym), (xp, yp), (xg, yg, nx, ny, nrm), (bx, by) = make_points(K, M_each)
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
    cond = torch.linalg.cond(An).item() if want_cond else float("nan")
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    theta = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1))/s
    cM, cP = theta[:NF], theta[NF:2*NF]; bM, bP = theta[2*NF], theta[2*NF+1]

    gt = np.linspace(-0.999, 0.999, NT); Xt, Yt = np.meshgrid(gt, gt); xt = Xt.ravel(); yt = Yt.ravel()
    uM = np.empty(xt.size); uP = np.empty(xt.size)                             # chunked to bound RAM
    for i in range(0, xt.size, 4000):
        Pt, _, _, _ = W_mats(xt[i:i+4000], yt[i:i+4000], jx, kx, jy, ky)
        uM[i:i+4000] = (Pt@cM + bM).numpy(); uP[i:i+4000] = (Pt@cP + bP).numpy()
    pred = np.where(phi_np(xt, yt) < 0, uM, uP)
    ue = u_exact(xt, yt); err = pred - ue
    metrics = dict(NF=NF, n_iface=len(xg),
                   relL2=np.linalg.norm(err)/np.linalg.norm(ue),
                   Linf=np.max(np.abs(err)), relLinf=np.max(np.abs(err))/np.max(np.abs(ue)),
                   kink=np.max(np.abs((dnG@cP).numpy() - (dnG@cM).numpy() - nrm*(1/a_out - 1/a_in))),
                   cond=cond)
    return metrics, (Xt, Yt, pred.reshape(Xt.shape), ue.reshape(Xt.shape)), (xg, yg)

# ----------------------------------------------------------------- run + figure
if __name__ == "__main__":
    import sys, time
    smoke = "--smoke" in sys.argv
    fam = banded_family()
    print(f"multi-inclusion: {len(INCL)} circles R in [{INCL[:,2].min():.2f},{INCL[:,2].max():.2f}]"
          f"  (banded basis, NF={fam[0].numel()})\n")
    rhos = [100.0] if smoke else [10.0, 100.0, 1000.0]
    hdr = f"{'rho':>6} | {'NF':>6} {'nIface':>7} {'relL2':>10} {'Linf':>10} {'relLinf':>10} {'kinkErr':>10} {'sec':>6}"
    print(hdr); print("-"*len(hdr)); saved = None
    for rho in rhos:
        t0 = time.time(); m, fields, gpts = solve(rho, fam); dt = time.time() - t0
        print(f"{rho:>6.0f} | {m['NF']:>6} {m['n_iface']:>7} {m['relL2']:>10.2e} {m['Linf']:>10.2e} "
              f"{m['relLinf']:>10.2e} {m['kink']:>10.2e} {dt:>6.1f}", flush=True)
        if rho == rhos[-1]: saved = (fields, gpts)

    (Xt, Yt, PR, UE), (xg, yg) = saved
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
    c0 = ax[0].contourf(Xt, Yt, UE, 40, cmap='viridis'); ax[0].plot(xg, yg, 'w.', ms=0.5)
    ax[0].set_title(f'(a) exact, {len(INCL)} inclusions'); plt.colorbar(c0, ax=ax[0])
    c1 = ax[1].contourf(Xt, Yt, PR, 40, cmap='viridis'); ax[1].plot(xg, yg, 'w.', ms=0.5)
    ax[1].set_title('(b) wavelet W-PINN (mesh-free)'); plt.colorbar(c1, ax=ax[1])
    c2 = ax[2].contourf(Xt, Yt, np.log10(np.abs(PR - UE) + 1e-16), 40, cmap='magma')
    ax[2].plot(xg, yg, 'w.', ms=0.5)
    ax[2].set_title('(c) $\\log_{10}$|error|'); plt.colorbar(c2, ax=ax[2])
    for a in ax: a.set_aspect('equal')
    plt.tight_layout(); plt.savefig("sol_multi.png", dpi=130); print("\nsaved sol_multi.png")
