"""
Reproduce DCSNN Example 1 (Hu, Lin, Lai, JCP 2022) -- a PUBLISHED mesh-free benchmark, head-to-head.
=====================================================================================================
Domain (-1,1)^2.  Ellipse interface:  phi = (x/0.2)^2 + (y/0.5)^2 - 1 = 0.
Coefficients:  beta^- = 1 (inside ellipse),  beta^+ = 1e-3 (outside)  -> contrast 1000.
Exact:  u^- = exp(x)exp(y),   u^+ = sin(x) sin(y)   (INHOMOGENEOUS jumps).
  f^- = -beta^- * Delta u^- = -2 exp(x)exp(y)
  f^+ = -beta^+ * Delta u^+ =  2e-3 sin(x) sin(y)
  [[u]]   = (u^+ - u^-)|_Gamma ,   [[beta d_n u]] = (beta^+ d_n u^+ - beta^- d_n u^-)|_Gamma
  Dirichlet g = u^+ on the square.
Reported by DCSNN: L_inf = 4.39e-6 with 101 parameters; they note IIM needs ~65,792 dof for similar.
We solve with the AD-free wavelet least-squares (single linear solve, no training) and report L_inf/relL2.
"""
import sys, os, math, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "multi_inclusion"))
from multi_inclusion import W_mats
torch.set_default_dtype(torch.float64)

AX, BY = 0.2, 0.5                                 # ellipse semi-axes
B_IN, B_OUT = 1.0, 1e-3
phi = lambda x, y: (x/AX)**2 + (y/BY)**2 - 1.0
u_in = lambda x, y: np.exp(x)*np.exp(y)
u_out = lambda x, y: np.sin(x)*np.sin(y)
f_in = lambda x, y: -2.0*np.exp(x)*np.exp(y)      # -beta^- Delta u^-
f_out = lambda x, y: 2e-3*np.sin(x)*np.sin(y)     # -beta^+ Delta u^+  (Delta sin sin = -2 sin sin)
# gradients for the flux jump
gux_in = lambda x, y: np.exp(x)*np.exp(y); guy_in = lambda x, y: np.exp(x)*np.exp(y)
gux_out = lambda x, y: np.cos(x)*np.sin(y); guy_out = lambda x, y: np.sin(x)*np.cos(y)


def family(coarse, lo=-1.0, hi=1.0, pad=0.5):
    fam = []
    for l in coarse:
        j = 2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            fam.append((j, float(k)))
    out = [(jx, kx, jy, ky) for (jx, kx) in fam for (jy, ky) in fam]
    arr = torch.tensor(out, dtype=torch.float64).T
    return arr[0], arr[1], arr[2], arr[3]


def solve(coarse_in, coarse_out, K=170, M=700, wi=10.0, wb=10.0, tik=1e-10):
    fin = family(coarse_in); fout = family(coarse_out)
    NFi = fin[0].numel(); NFo = fout[0].numel(); n = NFi + NFo + 2
    bI, bO = NFi + NFo, NFi + NFo + 1
    g = np.linspace(-1, 1, K); X, Y = np.meshgrid(g, g); xs = X.ravel(); ys = Y.ravel()
    p = phi(xs, ys); ins = p < -1e-9; out = (p > 1e-9) & (np.abs(xs) < 1) & (np.abs(ys) < 1)
    th = np.linspace(0, 2*np.pi, M, endpoint=False)
    xg = AX*np.cos(th); yg = BY*np.sin(th)
    nx = xg/AX**2; ny = yg/BY**2; nn = np.hypot(nx, ny); nx, ny = nx/nn, ny/nn   # outward normal ~ grad phi
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)]); by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])

    rows, rhs = [], []; newb = lambda r: torch.zeros(r, n)
    # interior PDE
    _, Li, _, _ = W_mats(xs[ins], ys[ins], *fin)
    Bi = newb(int(ins.sum())); Bi[:, :NFi] = -B_IN*Li
    rows.append(Bi); rhs.append(torch.tensor(f_in(xs[ins], ys[ins])))
    _, Lo, _, _ = W_mats(xs[out], ys[out], *fout)
    Bo = newb(int(out.sum())); Bo[:, NFi:NFi+NFo] = -B_OUT*Lo
    rows.append(Bo); rhs.append(torch.tensor(f_out(xs[out], ys[out])))
    # interface [[u]] and [[beta d_n u]]
    Pgi, _, dxi, dyi = W_mats(xg, yg, *fin); Pgo, _, dxo, dyo = W_mats(xg, yg, *fout)
    nxv = torch.tensor(nx); nyv = torch.tensor(ny)
    dnI = nxv[:, None]*dxi + nyv[:, None]*dyi; dnO = nxv[:, None]*dxo + nyv[:, None]*dyo
    gj = u_out(xg, yg) - u_in(xg, yg)
    hj = B_OUT*(gux_out(xg, yg)*nx + guy_out(xg, yg)*ny) - B_IN*(gux_in(xg, yg)*nx + guy_in(xg, yg)*ny)
    Bc = newb(M); Bc[:, NFi:NFi+NFo] = Pgo; Bc[:, bO] = 1.0; Bc[:, :NFi] = -Pgi; Bc[:, bI] = -1.0
    rows.append(math.sqrt(wi)*Bc); rhs.append(math.sqrt(wi)*torch.tensor(gj))
    Bf = newb(M); Bf[:, NFi:NFi+NFo] = B_OUT*dnO; Bf[:, :NFi] = -B_IN*dnI
    rows.append(math.sqrt(wi)*Bf); rhs.append(math.sqrt(wi)*torch.tensor(hj))
    # boundary (square in the outside region)
    Pb, _, _, _ = W_mats(bx, by, *fout); Bb = newb(len(bx)); Bb[:, NFi:NFi+NFo] = Pb; Bb[:, bO] = 1.0
    rows.append(math.sqrt(wb)*Bb); rhs.append(math.sqrt(wb)*torch.tensor(u_out(bx, by)))

    A = torch.cat(rows, 0); b = torch.cat(rhs, 0).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30); An = A/s
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    th_ = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1))/s
    cI, cO, biI, biO = th_[:NFi], th_[NFi:NFi+NFo], th_[bI], th_[bO]

    gt = np.linspace(-0.999, 0.999, 300); Xt, Yt = np.meshgrid(gt, gt); xt = Xt.ravel(); yt = Yt.ravel()
    pt = phi(xt, yt); ue = np.where(pt < 0, u_in(xt, yt), u_out(xt, yt))
    PtI, _, _, _ = W_mats(xt, yt, *fin); PtO, _, _, _ = W_mats(xt, yt, *fout)
    pr = np.where(pt < 0, (PtI@cI+biI).numpy(), (PtO@cO+biO).numpy())
    err = pr - ue
    return dict(NF=n, relL2=np.linalg.norm(err)/np.linalg.norm(ue), Linf=np.max(np.abs(err)))


if __name__ == "__main__":
    print("DCSNN Example 1 (ellipse, contrast 1000, u-=exp*exp / u+=sin*sin) -- wavelet W-PINN\n")
    print(f"{'basis(in/out)':>18} {'NF':>6} {'relL2':>11} {'Linf':>11} {'sec':>6}")
    for ci, co in [((0,1,2,3), (0,1,2,3)), ((0,1,2,3,4), (0,1,2,3))]:
        t0 = time.time(); m = solve(ci, co); dt = time.time()-t0
        print(f"{str(ci)+'/'+str(co):>18} {m['NF']:>6} {m['relL2']:>11.3e} {m['Linf']:>11.3e} {dt:>6.1f}",
              flush=True)
    print("\nDCSNN reported: Linf=4.39e-6 (101 params, trained); IIM ~65,792 dof for comparable accuracy.")
