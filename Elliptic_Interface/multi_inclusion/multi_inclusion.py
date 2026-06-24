"""
Multiple disconnected inclusions  --  the TOPOLOGY stress test for the AD-free wavelet W-PINN
=====================================================================================================
A step up from the single gear/flower interface: several disjoint closed inclusions scattered in the
unit square, each its own subdomain Omega^-_i with its own contrast, all embedded in one shared
matrix Omega^+.  This changes the *topology* (m+1 subdomains, m simultaneous interfaces), not just
the curvature -- the genuine multi-subdomain stress test, and the realistic composite-material case
(fibre-reinforced media, suspensions of cells/particles) where a mesh method must mesh and remesh
around every inclusion while this few-parameter mesh-free representation handles "m circles at these
centres/radii" trivially (and makes the inverse "recover the inclusions from data" a clean demo).

    -div(a grad u) = f  in Omega=(-1,1)^2,   a = a_in,i inside inclusion i,  a = a_out in the matrix.
    [[u]] = g_i  and  [[a d_n u]] = h_i  on every interface Gamma_i;   u = g on dOmega (Dirichlet).

Manufactured solution (O(1) in every region, general inhomogeneous jumps)
-------------------------------------------------------------------------
We prescribe an INDEPENDENT smooth O(1) exact field per subdomain and read off the data:

    u^+(x,y)   = cos(pi x/2) cos(pi y/2)                       (matrix; = 0 on dOmega)
    u^-_i(x,y) = cos(kappa (x-cx_i)) cos(kappa (y-cy_i))       (inclusion i; distinct per inclusion)

  * f^+ = -a_out Delta u^+,   f^-_i = -a_in,i Delta u^-_i      (sources, by autograd -- problem DATA)
  * [[u]]_i      = (u^+ - u^-_i)|_Gamma_i                       (continuity data on Gamma_i)
  * [[a d_n u]]_i = (a_out d_n u^+ - a_in,i d_n u^-_i)|_Gamma_i (flux data on Gamma_i)
  * g = u^+|_dOmega                                             (Dirichlet data)

This avoids the dynamic-range pathology of a single product level set u=Phi/a (whose amplitude
collapses like O(r_i^2) inside small inclusions, making relative error there meaningless); every
region carries an O(1) solution, and the method handles the resulting NONZERO jump data on the RHS
-- a strictly more general interface problem than the homogeneous-jump gear/flower.

Block-structured basis (the multi-subdomain decomposition)
----------------------------------------------------------
One wavelet expansion PER subdomain:  u^+ = W_+ c^+ + b^+  on the matrix, and  u^-_i = W_i c^-_i + b^-_i
on each inclusion.  A single global "inside" basis spanning disjoint inclusions wastes basis on the
gaps and conditions worse; per-inclusion LOCAL blocks (scaled to each r_i) are better-conditioned and
cleaner to refine.  The matrix block is global-coarse + a fine band around every Gamma_i, the band
width scaled to that inclusion's radius (small inclusions get proportionally finer wavelets).

The resulting least-squares system is ARROW-structured: each inclusion block couples only to the
matrix block through its own interface; inclusions never couple to each other -> well-conditioned and
scales past two subdomains.  Solved AD-free by column-normalised Tikhonov least squares (one lstsq).
"""
import math, time, numpy as np, torch
torch.set_default_dtype(torch.float64)

KAPPA = 3.0                                       # wavenumber of the per-inclusion exact field


# ----------------------------------------------------------------- exact fields + autograd data
def u_out_t(x, y):
    return torch.cos(math.pi*x/2)*torch.cos(math.pi*y/2)

def u_out_np(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return np.cos(np.pi*x/2)*np.cos(np.pi*y/2)

def u_in_t(x, y, c):
    return torch.cos(KAPPA*(x - c[0]))*torch.cos(KAPPA*(y - c[1]))

def u_in_np(x, y, c):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return np.cos(KAPPA*(x - c[0]))*np.cos(KAPPA*(y - c[1]))

def _lap_grad(fn, x, y):                          # exact Laplacian and gradient of fn(x,y) via autograd
    xt = torch.tensor(np.asarray(x, float), requires_grad=True)
    yt = torch.tensor(np.asarray(y, float), requires_grad=True)
    p = fn(xt, yt)
    gx, = torch.autograd.grad(p.sum(), xt, create_graph=True)
    gy, = torch.autograd.grad(p.sum(), yt, create_graph=True)
    gxx, = torch.autograd.grad(gx.sum(), xt, retain_graph=True)
    gyy, = torch.autograd.grad(gy.sum(), yt, retain_graph=True)
    return (gxx + gyy).detach().numpy(), gx.detach().numpy(), gy.detach().numpy()

def which_inclusion(x, y, incl, eps=1e-6):        # index of containing inclusion, or -1 for matrix
    x = np.asarray(x, float); y = np.asarray(y, float)
    idx = np.full(x.shape, -1, dtype=int)
    for i, (cx, cy, r) in enumerate(incl):
        idx[(x - cx)**2 + (y - cy)**2 < (r - eps)**2] = i
    return idx


# ----------------------------------------------------------------- wavelet atoms + op-matrices
def W_mats(x, y, jx, kx, jy, ky):                 # value, Laplacian, d/dx, d/dy of each atom
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
    X = jx[None, :]*x[:, None] - kx[None, :]; Y = jy[None, :]*y[:, None] - ky[None, :]
    E = torch.exp(-(X**2 + Y**2)/2)
    Psi = X*Y*E
    lap = -(jx[None, :]**2)*X*Y*(3 - X**2)*E - (jy[None, :]**2)*X*Y*(3 - Y**2)*E
    dx = jx[None, :]*(1 - X**2)*Y*E; dy = jy[None, :]*(1 - Y**2)*X*E
    return Psi, lap, dx, dy

def _grid_atoms(cx, cy, lo, hi, levels, keep):
    """tensor-product Gaussian-derivative atoms on dyadic levels, centred near (cx,cy) over the box
    [cx+lo, cx+hi] x [cy+lo, cy+hi], kept only where keep(distance-to-centre, j) is True."""
    fam = []
    for l in levels:
        if l < 0: continue
        j = 2.0**l
        ksx = range(int(math.floor((cx + lo)*j)), int(math.ceil((cx + hi)*j)) + 1)
        ksy = range(int(math.floor((cy + lo)*j)), int(math.ceil((cy + hi)*j)) + 1)
        for kxv in ksx:
            for kyv in ksy:
                if keep(math.hypot(kxv/j - cx, kyv/j - cy), j):
                    fam.append((j, float(kxv), j, float(kyv)))
    return fam

def fine_level(r, alpha=4.0):                     # finest dyadic level so atom width ~ r/alpha
    return int(round(math.log2(alpha / r)))


def build_basis(incl, coarse=(0, 1, 2, 3), pad=0.5, lo=-1.0, hi=1.0,
                n_in_levels=3, in_pad=1.0, band_cells=1.6, alpha=4.0):
    """Returns (atoms_out, [atoms_in_i]).  atoms_out: global coarse over the square + a radius-scaled
    fine band around every Gamma_i.  atoms_in_i: a local n-level block scaled to inclusion i."""
    def lvl1d(levels):
        out = []
        for l in levels:
            j = 2.0**l
            for k in range(int(math.floor((lo - pad)*j)), int(math.ceil((hi + pad)*j)) + 1):
                out.append((j, float(k)))
        return out
    f1c = lvl1d(coarse)
    fam_out = [(jx, kx, jy, ky) for (jx, kx) in f1c for (jy, ky) in f1c]      # global coarse
    for (cx, cy, r) in (incl if band_cells > 0 else []):                      # per-inclusion fine band
        Lf = fine_level(r, alpha)
        for l in (Lf - 1, Lf):
            j = 2.0**l; half = band_cells / j
            fam_out += _grid_atoms(cx, cy, -(r + half + 1.0/j), (r + half + 1.0/j), [l],
                                   keep=lambda d, jj, r=r, half=half: (r - half) <= d <= (r + half))
    atoms_in = []                                                            # local inclusion blocks
    for (cx, cy, r) in incl:
        Lf = fine_level(r, alpha)
        levels = [Lf - (n_in_levels - 1) + t for t in range(n_in_levels)]
        ai = _grid_atoms(cx, cy, -(r + in_pad/2.0**Lf), (r + in_pad/2.0**Lf), levels,
                         keep=lambda d, j, r=r: d <= r + in_pad/j)
        atoms_in.append(ai)
    to_t = lambda fam: tuple(torch.tensor(fam, dtype=torch.float64).T) if fam else None
    return to_t(fam_out), [to_t(a) for a in atoms_in]


# ----------------------------------------------------------------- collocation points
def make_points(incl, K=140, M=240):
    g = np.linspace(-1, 1, K); Xg, Yg = np.meshgrid(g, g); xs = Xg.ravel(); ys = Yg.ravel()
    idx = which_inclusion(xs, ys, incl)
    inside_sets = [(xs[idx == i], ys[idx == i]) for i in range(len(incl))]
    out_mask = (idx == -1) & (np.abs(xs) < 1 - 1e-9) & (np.abs(ys) < 1 - 1e-9)
    outside = (xs[out_mask], ys[out_mask])
    ifaces = []                                   # per-inclusion interface points (analytic normals)
    for (cx, cy, r) in incl:
        th = np.linspace(0, 2*np.pi, M, endpoint=False)
        ifaces.append((cx + r*np.cos(th), cy + r*np.sin(th), np.cos(th), np.sin(th)))
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)])
    by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])
    return inside_sets, outside, ifaces, (bx, by)


# ----------------------------------------------------------------- AD-free least-squares solve
def solve(incl, a_in, a_out, basis=None, K=140, M=240, wi=10.0, wb=10.0, tik=1e-10,
          want_cond=False, want_fields=False):
    if basis is None:
        basis = build_basis(incl)
    fam_out, fams_in = basis
    m = len(incl)
    cen = [(c[0], c[1]) for c in incl]
    NFo = fam_out[0].numel(); NFi = [f[0].numel() for f in fams_in]
    off_in = [NFo + int(np.sum(NFi[:i])) for i in range(m)]                  # column layout:
    nc = NFo + int(np.sum(NFi)); bplus = nc; bminus = [nc + 1 + i for i in range(m)]
    n = nc + 1 + m                                  # [c^+ | c^-_0..c^-_{m-1} | b^+ | b^-_0..b^-_{m-1}]

    inside_sets, outside, ifaces, (bx, by) = make_points(incl, K, M)
    rows, rhs = [], []
    newblk = lambda r: torch.zeros(r, n)

    # matrix interior PDE residual: -a_out * Lout c^+ = f^+
    xo, yo = outside
    _, Lo, _, _ = W_mats(xo, yo, *fam_out)
    Bo = newblk(len(xo)); Bo[:, :NFo] = -a_out*Lo
    lo_, _, _ = _lap_grad(u_out_t, xo, yo)
    rows.append(Bo); rhs.append(torch.tensor(-a_out*lo_))                    # f^+ = -a_out * Delta u^+

    # inclusion interior PDE residual: -a_in,i * Lin_i c^-_i = f^-_i
    for i, (xi, yi) in enumerate(inside_sets):
        if len(xi) == 0: continue
        _, Li, _, _ = W_mats(xi, yi, *fams_in[i])
        Bi = newblk(len(xi)); Bi[:, off_in[i]:off_in[i]+NFi[i]] = -a_in[i]*Li
        li, _, _ = _lap_grad(lambda x, y, c=cen[i]: u_in_t(x, y, c), xi, yi)
        rows.append(Bi); rhs.append(torch.tensor(-a_in[i]*li))

    # interface conditions on each Gamma_i (inhomogeneous: jump data on the RHS)
    iface_data = []
    for i, (xg, yg, nx, ny) in enumerate(ifaces):
        Pog, _, dxo, dyo = W_mats(xg, yg, *fam_out)
        Pig, _, dxi, dyi = W_mats(xg, yg, *fams_in[i])
        nxv = torch.tensor(nx); nyv = torch.tensor(ny)
        dnO = nxv[:, None]*dxo + nyv[:, None]*dyo
        dnI = nxv[:, None]*dxi + nyv[:, None]*dyi
        uo = u_out_np(xg, yg); ui = u_in_np(xg, yg, cen[i])
        _, gox, goy = _lap_grad(u_out_t, xg, yg)
        _, gix, giy = _lap_grad(lambda x, y, c=cen[i]: u_in_t(x, y, c), xg, yg)
        gjump = uo - ui                                                     # [[u]]_i
        hjump = a_out*(gox*nx + goy*ny) - a_in[i]*(gix*nx + giy*ny)         # [[a d_n u]]_i
        # [[u]] = g :  (Po c^+ + b^+) - (Pi c^-_i + b^-_i) = g
        Bc = newblk(len(xg)); Bc[:, :NFo] = Pog; Bc[:, bplus] = 1.0
        Bc[:, off_in[i]:off_in[i]+NFi[i]] = -Pig; Bc[:, bminus[i]] = -1.0
        rows.append(math.sqrt(wi)*Bc); rhs.append(math.sqrt(wi)*torch.tensor(gjump))
        # [[a d_n u]] = h :  a_out (dnO c^+) - a_in,i (dnI c^-_i) = h
        Bf = newblk(len(xg)); Bf[:, :NFo] = a_out*dnO
        Bf[:, off_in[i]:off_in[i]+NFi[i]] = -a_in[i]*dnI
        rows.append(math.sqrt(wi)*Bf); rhs.append(math.sqrt(wi)*torch.tensor(hjump))
        iface_data.append((i, Pog, dnO, Pig, dnI, gjump, hjump))

    # Dirichlet boundary: Pbc c^+ + b^+ = g
    Pb, _, _, _ = W_mats(bx, by, *fam_out)
    Bb = newblk(len(bx)); Bb[:, :NFo] = Pb; Bb[:, bplus] = 1.0
    rows.append(math.sqrt(wb)*Bb); rhs.append(math.sqrt(wb)*torch.tensor(u_out_np(bx, by)))

    A = torch.cat(rows, 0); b = torch.cat(rhs, 0).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30); An = A/s
    cond = torch.linalg.cond(An).item() if want_cond else float("nan")
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    theta = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1)) / s

    cP = theta[:NFo]; bP = theta[bplus]
    cM = [theta[off_in[i]:off_in[i]+NFi[i]] for i in range(m)]
    bM = [theta[bminus[i]] for i in range(m)]

    # ---- evaluate on a dense test grid ----
    gt = np.linspace(-0.999, 0.999, 300); Xt, Yt = np.meshgrid(gt, gt)
    xt = Xt.ravel(); yt = Yt.ravel(); idx = which_inclusion(xt, yt, incl)
    ue = u_out_np(xt, yt).copy()
    for i in range(m):
        ue[idx == i] = u_in_np(xt[idx == i], yt[idx == i], cen[i])
    Pt, _, _, _ = W_mats(xt, yt, *fam_out)
    pred = (Pt @ cP + bP).numpy()
    for i in range(m):
        mi = idx == i
        if mi.any():
            Pti, _, _, _ = W_mats(xt[mi], yt[mi], *fams_in[i])
            pred[mi] = (Pti @ cM[i] + bM[i]).numpy()
    err = pred - ue

    per_incl = []
    for i in range(m):
        mi = idx == i
        per_incl.append(np.linalg.norm(err[mi]) / max(np.linalg.norm(ue[mi]), 1e-30))

    # interface residuals: max |predicted jump - imposed jump| over all Gamma_i
    ujump_err = fjump_err = 0.0
    for (i, Pog, dnO, Pig, dnI, gjump, hjump) in iface_data:
        upred = (Pog @ cP + bP).numpy() - (Pig @ cM[i] + bM[i]).numpy()
        fpred = a_out*(dnO @ cP).numpy() - a_in[i]*(dnI @ cM[i]).numpy()
        ujump_err = max(ujump_err, np.max(np.abs(upred - gjump)))
        fjump_err = max(fjump_err, np.max(np.abs(fpred - hjump)))

    metrics = dict(NFo=NFo, NFi=NFi, NF=n,
                   relL2=np.linalg.norm(err)/np.linalg.norm(ue),
                   RMSE=np.sqrt(np.mean(err**2)), MAE=np.mean(np.abs(err)),
                   Linf=np.max(np.abs(err)), relLinf=np.max(np.abs(err))/np.max(np.abs(ue)),
                   ujump=ujump_err, fjump=fjump_err, per_incl=per_incl, cond=cond)
    fields = (Xt, Yt, pred.reshape(Xt.shape), ue.reshape(Xt.shape), idx.reshape(Xt.shape)) \
        if want_fields else None
    return metrics, fields


# ----------------------------------------------------------------- physical forward solve (for inverse)
def solve_physical(incl, a_in, a_out, f_func, g_func, basis=None, K=120, M=200,
                   wi=10.0, wb=10.0, tik=1e-10, precomp=None):
    """PHYSICAL interface problem (homogeneous jumps [[u]]=0, [[a d_n u]]=0) with a GIVEN source
    f_func(x,y) and Dirichlet data g_func(x,y).  No manufactured field -> the geometry enters only
    through where the coefficient a jumps.  Returns a predictor u(xq,yq).  Used as the inner forward
    map of the inverse: the matrix (global-coarse) block is geometry-independent and may be precomputed
    once via `precomp` (P_all,L_all at the fixed grid + P_bc); only the moving inclusion blocks and
    interface matrices are rebuilt per candidate geometry -> fast (sub-second) forward evaluations."""
    if basis is None:
        basis = build_basis(incl, band_cells=0.0)            # smooth circles: no fine band needed
    fam_out, fams_in = basis
    m = len(incl); cen = [(c[0], c[1]) for c in incl]
    NFo = fam_out[0].numel(); NFi = [f[0].numel() for f in fams_in]
    off_in = [NFo + int(np.sum(NFi[:i])) for i in range(m)]
    nc = NFo + int(np.sum(NFi)); bplus = nc; bminus = [nc + 1 + i for i in range(m)]
    n = nc + 1 + m

    g = np.linspace(-1, 1, K); Xg, Yg = np.meshgrid(g, g); xs = Xg.ravel(); ys = Yg.ravel()
    idx = which_inclusion(xs, ys, incl)
    out_mask = (idx == -1) & (np.abs(xs) < 1 - 1e-9) & (np.abs(ys) < 1 - 1e-9)
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)]); by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])

    if precomp is None:                                       # matrix-block matrices at the fixed grid
        P_all, L_all, _, _ = W_mats(xs, ys, *fam_out)
        P_bc, _, _, _ = W_mats(bx, by, *fam_out)
    else:
        P_all, L_all, P_bc = precomp
    rows, rhs = [], []; newblk = lambda r: torch.zeros(r, n)

    xo, yo = xs[out_mask], ys[out_mask]                       # matrix interior: -a_out L c+ = f
    Bo = newblk(len(xo)); Bo[:, :NFo] = -a_out*L_all[out_mask]
    rows.append(Bo); rhs.append(torch.tensor(f_func(xo, yo)))
    for i in range(m):                                        # inclusion interiors: -a_in,i L_i c-_i = f
        mi = idx == i; xi, yi = xs[mi], ys[mi]
        if len(xi) == 0: continue
        _, Li, _, _ = W_mats(xi, yi, *fams_in[i])
        Bi = newblk(len(xi)); Bi[:, off_in[i]:off_in[i]+NFi[i]] = -a_in[i]*Li
        rows.append(Bi); rhs.append(torch.tensor(f_func(xi, yi)))
    for i, (cx, cy, r) in enumerate(incl):                    # interfaces: homogeneous jumps
        th = np.linspace(0, 2*np.pi, M, endpoint=False)
        xg = cx + r*np.cos(th); yg = cy + r*np.sin(th); nx = np.cos(th); ny = np.sin(th)
        Pog, _, dxo, dyo = W_mats(xg, yg, *fam_out)
        Pig, _, dxi, dyi = W_mats(xg, yg, *fams_in[i])
        nxv = torch.tensor(nx); nyv = torch.tensor(ny)
        dnO = nxv[:, None]*dxo + nyv[:, None]*dyo; dnI = nxv[:, None]*dxi + nyv[:, None]*dyi
        Bc = newblk(M); Bc[:, :NFo] = Pog; Bc[:, bplus] = 1.0
        Bc[:, off_in[i]:off_in[i]+NFi[i]] = -Pig; Bc[:, bminus[i]] = -1.0
        rows.append(math.sqrt(wi)*Bc); rhs.append(torch.zeros(M))
        Bf = newblk(M); Bf[:, :NFo] = a_out*dnO; Bf[:, off_in[i]:off_in[i]+NFi[i]] = -a_in[i]*dnI
        rows.append(math.sqrt(wi)*Bf); rhs.append(torch.zeros(M))
    Bb = newblk(len(bx)); Bb[:, :NFo] = P_bc; Bb[:, bplus] = 1.0       # Dirichlet boundary
    rows.append(math.sqrt(wb)*Bb); rhs.append(math.sqrt(wb)*torch.tensor(g_func(bx, by)))

    A = torch.cat(rows, 0); b = torch.cat(rhs, 0).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30); An = A/s
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    theta = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1)) / s
    cP = theta[:NFo]; bP = theta[bplus]
    cM = [theta[off_in[i]:off_in[i]+NFi[i]] for i in range(m)]; bM = [theta[bminus[i]] for i in range(m)]

    def predict(xq, yq):
        xq = np.asarray(xq, float); yq = np.asarray(yq, float); iq = which_inclusion(xq, yq, incl)
        Pq, _, _, _ = W_mats(xq, yq, *fam_out); out = (Pq @ cP + bP).numpy()
        for i in range(m):
            mi = iq == i
            if mi.any():
                Pi, _, _, _ = W_mats(xq[mi], yq[mi], *fams_in[i]); out[mi] = (Pi @ cM[i] + bM[i]).numpy()
        return out
    return predict


def matrix_precomp(fam_out, K):                              # geometry-independent matrix-block matrices
    g = np.linspace(-1, 1, K); Xg, Yg = np.meshgrid(g, g); xs = Xg.ravel(); ys = Yg.ravel()
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)]); by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])
    P_all, L_all, _, _ = W_mats(xs, ys, *fam_out)
    P_bc, _, _, _ = W_mats(bx, by, *fam_out)
    return P_all, L_all, P_bc


# ----------------------------------------------------------------- default config + quick run
# 6 mutually well-separated inclusions of varying size; prefixes give the m=2..6 topology sweep.
POOL6 = [(-0.50, -0.50, 0.17), (0.50, -0.50, 0.15), (-0.50, 0.50, 0.16),
         (0.50, 0.50, 0.19), (0.00, 0.00, 0.13), (0.00, -0.58, 0.12)]
INCL5 = POOL6[:5]

if __name__ == "__main__":
    incl = INCL5
    basis = build_basis(incl)
    print(f"{len(incl)} inclusions   NF_out={basis[0][0].numel()}   "
          f"NF_in={[f[0].numel() for f in basis[1]]}\n")
    hdr = (f"{'rho':>6} | {'relL2':>10} {'RMSE':>10} {'MAE':>10} {'Linf':>10} "
           f"{'relLinf':>10} {'ujumpErr':>10} {'fjumpErr':>10} | {'per-incl relL2':>30} {'sec':>6}")
    print(hdr); print("-"*len(hdr))
    for rho in [10.0, 100.0, 1000.0]:
        a_in = [1.0]*len(incl); a_out = rho
        t0 = time.time(); mtr, _ = solve(incl, a_in, a_out, basis=basis); dt = time.time() - t0
        pis = " ".join(f"{p:.1e}" for p in mtr['per_incl'])
        print(f"{rho:>6.0f} | {mtr['relL2']:>10.2e} {mtr['RMSE']:>10.2e} {mtr['MAE']:>10.2e} "
              f"{mtr['Linf']:>10.2e} {mtr['relLinf']:>10.2e} {mtr['ujump']:>10.2e} "
              f"{mtr['fjump']:>10.2e} | {pis:>30} {dt:>6.1f}", flush=True)
