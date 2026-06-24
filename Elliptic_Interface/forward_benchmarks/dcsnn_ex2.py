"""
Reproduce DCSNN Example 2 (Hu, Lin, Lai, JCP 2022) -- NON-CONVEX flower interface, head-to-head.
=====================================================================================================
Domain (-1,1)^2.  Flower interface (polar):  r(theta) = 1/2 + sin(5 theta)/7  (5 petals, non-convex).
Coefficients:  beta^- = 10 (inside flower),  beta^+ = 1 (outside).
Exact:  u^- = exp(x^2+y^2),   u^+ = 0.1 (x^2+y^2)^2 - 0.01 log(2 sqrt(x^2+y^2))   (inhomogeneous jumps).
  s = x^2+y^2.   Delta u^- = (4+4s) e^s  (=> f^- = -beta^- (4+4s) e^s)
                 Delta u^+ = 1.6 s        (log r is harmonic => f^+ = -beta^+ 1.6 s)
  grad u^- = 2(x,y) e^s ;  grad u^+ = 0.4 s (x,y) - 0.01 (x,y)/s
DCSNN reported: relative L2 = 2.634e-4 with 501 parameters (trained).
"""
import sys, os, math, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "multi_inclusion"))
from multi_inclusion import W_mats
torch.set_default_dtype(torch.float64)

B_IN, B_OUT = 10.0, 1.0
rflower = lambda th: 0.5 + np.sin(5*th)/7.0
s_ = lambda x, y: x*x + y*y
u_in = lambda x, y: np.exp(s_(x, y))
u_out = lambda x, y: 0.1*s_(x, y)**2 - 0.01*np.log(2*np.sqrt(s_(x, y)))
f_in = lambda x, y: -B_IN*(4 + 4*s_(x, y))*np.exp(s_(x, y))
f_out = lambda x, y: -B_OUT*1.6*s_(x, y)
gx_in = lambda x, y: 2*x*np.exp(s_(x, y)); gy_in = lambda x, y: 2*y*np.exp(s_(x, y))
gx_out = lambda x, y: 0.4*s_(x, y)*x - 0.01*x/s_(x, y)
gy_out = lambda x, y: 0.4*s_(x, y)*y - 0.01*y/s_(x, y)

def inside(x, y):
    th = np.arctan2(y, x); return np.hypot(x, y) < rflower(th)

def family(coarse, lo=-1.0, hi=1.0, pad=0.5):
    fam = []
    for l in coarse:
        j = 2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            fam.append((j, float(k)))
    out = [(jx, kx, jy, ky) for (jx, kx) in fam for (jy, ky) in fam]
    a = torch.tensor(out, dtype=torch.float64).T
    return a[0], a[1], a[2], a[3]

def solve(coarse_in, coarse_out, K=180, M=900, wi=10.0, wb=10.0, tik=1e-10):
    fin = family(coarse_in); fout = family(coarse_out)
    NFi = fin[0].numel(); NFo = fout[0].numel(); n = NFi + NFo + 2
    bI, bO = NFi + NFo, NFi + NFo + 1
    g = np.linspace(-1, 1, K); X, Y = np.meshgrid(g, g); xs = X.ravel(); ys = Y.ravel()
    ins = inside(xs, ys); ins &= s_(xs, ys) > 1e-8                      # avoid the origin singularity
    out = (~ins) & (np.abs(xs) < 1) & (np.abs(ys) < 1)
    th = np.linspace(0, 2*np.pi, M, endpoint=False); rr = rflower(th); rp = 5*np.cos(5*th)/7.0
    xg = rr*np.cos(th); yg = rr*np.sin(th)
    dx = rp*np.cos(th) - rr*np.sin(th); dy = rp*np.sin(th) + rr*np.cos(th)   # tangent
    nx, ny = dy, -dx; sgn = np.sign(nx*xg + ny*yg); nx *= sgn; ny *= sgn      # outward normal
    nn = np.hypot(nx, ny); nx, ny = nx/nn, ny/nn
    nb = np.linspace(-1, 1, K)
    bx = np.concatenate([nb, nb, -np.ones(K), np.ones(K)]); by = np.concatenate([-np.ones(K), np.ones(K), nb, nb])

    rows, rhs = [], []; newb = lambda r: torch.zeros(r, n)
    _, Li, _, _ = W_mats(xs[ins], ys[ins], *fin)
    Bi = newb(int(ins.sum())); Bi[:, :NFi] = -B_IN*Li
    rows.append(Bi); rhs.append(torch.tensor(f_in(xs[ins], ys[ins])))
    _, Lo, _, _ = W_mats(xs[out], ys[out], *fout)
    Bo = newb(int(out.sum())); Bo[:, NFi:NFi+NFo] = -B_OUT*Lo
    rows.append(Bo); rhs.append(torch.tensor(f_out(xs[out], ys[out])))
    Pgi, _, dxi, dyi = W_mats(xg, yg, *fin); Pgo, _, dxo, dyo = W_mats(xg, yg, *fout)
    nxv = torch.tensor(nx); nyv = torch.tensor(ny)
    dnI = nxv[:, None]*dxi + nyv[:, None]*dyi; dnO = nxv[:, None]*dxo + nyv[:, None]*dyo
    gj = u_out(xg, yg) - u_in(xg, yg)
    hj = B_OUT*(gx_out(xg, yg)*nx + gy_out(xg, yg)*ny) - B_IN*(gx_in(xg, yg)*nx + gy_in(xg, yg)*ny)
    Bc = newb(M); Bc[:, NFi:NFi+NFo] = Pgo; Bc[:, bO] = 1.0; Bc[:, :NFi] = -Pgi; Bc[:, bI] = -1.0
    rows.append(math.sqrt(wi)*Bc); rhs.append(math.sqrt(wi)*torch.tensor(gj))
    Bf = newb(M); Bf[:, NFi:NFi+NFo] = B_OUT*dnO; Bf[:, :NFi] = -B_IN*dnI
    rows.append(math.sqrt(wi)*Bf); rhs.append(math.sqrt(wi)*torch.tensor(hj))
    Pb, _, _, _ = W_mats(bx, by, *fout); Bb = newb(len(bx)); Bb[:, NFi:NFi+NFo] = Pb; Bb[:, bO] = 1.0
    rows.append(math.sqrt(wb)*Bb); rhs.append(math.sqrt(wb)*torch.tensor(u_out(bx, by)))

    A = torch.cat(rows, 0); b = torch.cat(rhs, 0).unsqueeze(1)
    sc = A.norm(dim=0).clamp_min(1e-30); An = A/sc
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    thv = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1))/sc
    cI, cO, biI, biO = thv[:NFi], thv[NFi:NFi+NFo], thv[bI], thv[bO]

    gt = np.linspace(-0.999, 0.999, 300); Xt, Yt = np.meshgrid(gt, gt); xt = Xt.ravel(); yt = Yt.ravel()
    inst = inside(xt, yt); ue = np.where(inst, u_in(xt, yt), u_out(xt, yt))
    PtI, _, _, _ = W_mats(xt, yt, *fin); PtO, _, _, _ = W_mats(xt, yt, *fout)
    pr = np.where(inst, (PtI@cI+biI).numpy(), (PtO@cO+biO).numpy())
    err = pr - ue
    return dict(NF=n, relL2=np.linalg.norm(err)/np.linalg.norm(ue), Linf=np.max(np.abs(err)))

if __name__ == "__main__":
    print("DCSNN Example 2 (flower r=1/2+sin(5t)/7, beta-/+=10/1, u-=exp(s)/u+=0.1s^2-0.01log) -- wavelet\n")
    print(f"{'basis(in/out)':>20} {'NF':>6} {'relL2':>11} {'Linf':>11} {'sec':>6}")
    for ci, co in [((0,1,2,3), (0,1,2,3)), ((0,1,2,3,4), (0,1,2,3))]:
        t0 = time.time(); m = solve(ci, co); dt = time.time()-t0
        print(f"{str(ci)+'/'+str(co):>20} {m['NF']:>6} {m['relL2']:>11.3e} {m['Linf']:>11.3e} {dt:>6.1f}",
              flush=True)
    print("\nDCSNN reported: relative L2 = 2.63e-4 (501 params, trained).")
