"""
Gap 1 -- HEAD-TO-HEAD baselines on the IDENTICAL gear interface problem (paper-standard).
=========================================================================================
A CAMWA referee wants the method measured against competitors on the SAME benchmark, not
just an internal ablation.  This runs four solvers on the exact gear problem of run_gear.py
(same geometry, same source f=-Delta phi, same contrast rho, same collocation points):

  1. wavelet W-PINN (banded multiresolution, AD-free linear LSQ)   <- this paper
  2. RBF-PIELM     (single-scale Gaussian random features, best of a width sweep, matched N_F,
                    SAME decomposed least-squares assembly)        <- single-scale competitor
  3. XPINN-MLP     (two tanh MLPs + interface losses, Adam)        <- strongest NN competitor
  4. vanilla MLP   (one tanh MLP, Adam)                            <- standard PINN baseline

The RBF-PIELM goes through the identical decomposed assembly as the wavelet method, so the
ONLY difference is multiresolution-wavelet vs single-scale-random basis -- the clean test of
the paper's thesis.  Literature numbers (DCSNN, cusp/I-PINN) are reported separately in
results.md with citations, on their own (smoother) benchmarks.

Run:  python baselines_gear.py     (writes baselines_results.json)
"""
import json, math, time, numpy as np, torch
from run_gear import phi_np, lap_grad, make_points, banded_family, W_mats
torch.set_default_dtype(torch.float64)

RHO_LIST = [10.0, 1000.0]
SEED = 0

def u_exact(x, y, a_in, a_out):
    p = phi_np(x, y); return np.where(p < 0, p/a_in, p/a_out)

# ---- shared geometry / data (computed once, identical for every solver) ----
(XM, YM), (XP, YP), (XG, YG, NXG, NYG, NRM), (BX, BY) = make_points()
GT = np.linspace(-0.999, 0.999, 300); _Xt, _Yt = np.meshgrid(GT, GT)
XT, YT = _Xt.ravel(), _Yt.ravel()
FIN, _, _ = lap_grad(XM, YM); FOUT, _, _ = lap_grad(XP, YP)        # f = -Delta phi (one-time data)

# ================= generic decomposed least-squares (basis-agnostic) =================
def decomposed_lsq(rho, eval_basis, NF, wi=10.0, wb=10.0, tik=1e-10):
    """eval_basis(x,y) -> (value, laplacian, d/dx, d/dy) matrices [npts, NF]. Mirrors run_gear.solve."""
    a_in, a_out = 1.0, rho
    Pm, Lm, _, _ = eval_basis(XM, YM); Pp, Lp, _, _ = eval_basis(XP, YP)
    Pg, _, dxg, dyg = eval_basis(XG, YG)
    nxv = torch.tensor(NXG); nyv = torch.tensor(NYG); dnG = nxv[:, None]*dxg + nyv[:, None]*dyg
    Pb, _, _, _ = eval_basis(BX, BY)
    n = 2*NF + 2
    def blk(rows, cM=None, cP=None, bM=None, bP=None):
        Bm = torch.zeros(rows, n)
        if cM is not None: Bm[:, :NF] = cM
        if cP is not None: Bm[:, NF:2*NF] = cP
        if bM is not None: Bm[:, 2*NF] = bM
        if bP is not None: Bm[:, 2*NF+1] = bP
        return Bm
    A = torch.cat([blk(len(XM), cM=-a_in*Lm), blk(len(XP), cP=-a_out*Lp),
                   math.sqrt(wi)*blk(len(XG), cP=Pg, cM=-Pg, bP=1., bM=-1.),
                   math.sqrt(wi)*blk(len(XG), cP=a_out*dnG, cM=-a_in*dnG),
                   math.sqrt(wb)*blk(len(BX), cP=Pb, bP=1.)], 0)
    ub = torch.tensor(u_exact(BX, BY, a_in, a_out))
    b = torch.cat([torch.tensor(-FIN), torch.tensor(-FOUT),
                   torch.zeros(len(XG)), torch.zeros(len(XG)), math.sqrt(wb)*ub]).unsqueeze(1)
    s = A.norm(dim=0).clamp_min(1e-30); An = A/s
    Aa = torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb = torch.cat([b, torch.zeros(n, 1)])
    theta = (torch.linalg.lstsq(Aa, bb, driver="gelsd").solution.squeeze(1))/s
    cM, cP, bM, bP = theta[:NF], theta[NF:2*NF], theta[2*NF], theta[2*NF+1]
    Pt, _, _, _ = eval_basis(XT, YT)
    pred = np.where(phi_np(XT, YT) < 0, (Pt@cM + bM).numpy(), (Pt@cP + bP).numpy())
    ue = u_exact(XT, YT, a_in, a_out); err = pred - ue
    return dict(relL2=np.linalg.norm(err)/np.linalg.norm(ue), Linf=np.max(np.abs(err)), NF=int(NF))

# ---- basis 1: banded wavelet (this paper) ----
def wavelet_eval_factory():
    jx, kx, jy, ky = banded_family()
    return (lambda x, y: W_mats(x, y, jx, kx, jy, ky)), int(jx.numel())

# ---- basis 2: single-scale Gaussian RBF (PIELM), matched N_F, width sweep ----
def rbf_eval_factory(NF, eps, seed=SEED):
    rng = np.random.default_rng(seed)
    cx = torch.tensor(rng.uniform(-1, 1, NF)); cy = torch.tensor(rng.uniform(-1, 1, NF))
    e2 = eps*eps
    def ev(x, y):
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1); y = torch.as_tensor(y, dtype=torch.float64).reshape(-1)
        DX = x[:, None] - cx[None, :]; DY = y[:, None] - cy[None, :]; R2 = DX*DX + DY*DY
        E = torch.exp(-e2*R2)
        val = E; lap = E*(4*e2*e2*R2 - 4*e2); dx = -2*e2*DX*E; dy = -2*e2*DY*E
        return val, lap, dx, dy
    return ev

# ================= neural-network competitors (autograd, Adam) =================
class MLP(torch.nn.Module):
    def __init__(s, w=64, d=4):
        super().__init__(); L = [torch.nn.Linear(2, w), torch.nn.Tanh()]
        for _ in range(d-1): L += [torch.nn.Linear(w, w), torch.nn.Tanh()]
        L += [torch.nn.Linear(w, 1)]; s.net = torch.nn.Sequential(*L)
        for m in s.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight); torch.nn.init.zeros_(m.bias)
    def forward(s, p): return s.net(p)

def _lap(u, p):
    g = torch.autograd.grad(u, p, torch.ones_like(u), create_graph=True)[0]
    ux, uy = g[:, :1], g[:, 1:2]
    uxx = torch.autograd.grad(ux, p, torch.ones_like(ux), create_graph=True)[0][:, :1]
    uyy = torch.autograd.grad(uy, p, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    return g, uxx + uyy

def vanilla_mlp(rho, epochs=8000, lr=1e-3):
    torch.manual_seed(SEED)
    a_in, a_out = 1.0, rho
    xc = np.concatenate([XM, XP]); yc = np.concatenate([YM, YP])
    P = torch.tensor(np.stack([xc, yc], 1), requires_grad=True)
    av = torch.tensor(np.where(phi_np(xc, yc) < 0, a_in, a_out)).reshape(-1, 1)
    fv = torch.tensor(np.concatenate([FIN, FOUT])).reshape(-1, 1)
    Pb = torch.tensor(np.stack([BX, BY], 1)); ub = torch.tensor(u_exact(BX, BY, a_in, a_out)).reshape(-1, 1)
    net = MLP(); opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); u = net(P); _, L = _lap(u, P)
        loss = (-av*L - fv).pow(2).mean() + 10*(net(Pb) - ub).pow(2).mean()
        loss.backward(); opt.step()
    pr = net(torch.tensor(np.stack([XT, YT], 1))).detach().numpy().ravel()
    ue = u_exact(XT, YT, a_in, a_out); err = pr - ue
    return dict(relL2=np.linalg.norm(err)/np.linalg.norm(ue), Linf=np.max(np.abs(err)))

def xpinn_mlp(rho, epochs=8000, lr=1e-3):
    torch.manual_seed(SEED)
    a_in, a_out = 1.0, rho
    Pm = torch.tensor(np.stack([XM, YM], 1), requires_grad=True)
    Pp = torch.tensor(np.stack([XP, YP], 1), requires_grad=True)
    fm = torch.tensor(FIN).reshape(-1, 1); fp = torch.tensor(FOUT).reshape(-1, 1)
    G = torch.tensor(np.stack([XG, YG], 1), requires_grad=True); nrm = torch.tensor(np.stack([NXG, NYG], 1))
    Pb = torch.tensor(np.stack([BX, BY], 1)); ub = torch.tensor(u_exact(BX, BY, a_in, a_out)).reshape(-1, 1)
    nm, npp = MLP(), MLP(); opt = torch.optim.Adam(list(nm.parameters()) + list(npp.parameters()), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        uM = nm(Pm); _, LM = _lap(uM, Pm); uP = npp(Pp); _, LP = _lap(uP, Pp)
        rM = (-a_in*LM - fm).pow(2).mean(); rP = (-a_out*LP - fp).pow(2).mean()
        uMg = nm(G); gMg = torch.autograd.grad(uMg, G, torch.ones_like(uMg), create_graph=True)[0]
        uPg = npp(G); gPg = torch.autograd.grad(uPg, G, torch.ones_like(uPg), create_graph=True)[0]
        cont = (uPg - uMg).pow(2).mean()
        flux = (a_out*(gPg*nrm).sum(1) - a_in*(gMg*nrm).sum(1)).pow(2).mean()
        bc = (npp(Pb) - ub).pow(2).mean()
        (rM + rP + 10*(cont + flux) + 10*bc).backward(); opt.step()
    pm = nm(torch.tensor(np.stack([XT, YT], 1))).detach().numpy().ravel()
    pp = npp(torch.tensor(np.stack([XT, YT], 1))).detach().numpy().ravel()
    pred = np.where(phi_np(XT, YT) < 0, pm, pp); ue = u_exact(XT, YT, a_in, a_out); err = pred - ue
    return dict(relL2=np.linalg.norm(err)/np.linalg.norm(ue), Linf=np.max(np.abs(err)))

if __name__ == "__main__":
    wav_eval, NF_W = wavelet_eval_factory()
    RBF_EPS = [3.0, 5.0, 8.0, 12.0, 18.0]      # width sweep; report best-width RBF-PIELM
    out = {}
    print(f"gear head-to-head baselines (N_F[wavelet]={NF_W}, N_F[RBF]={NF_W} matched)\n")
    for rho in RHO_LIST:
        print(f"===== rho = {rho:.0f} =====")
        t = time.time(); w = decomposed_lsq(rho, wav_eval, NF_W); tw = time.time()-t
        print(f"  wavelet W-PINN (banded) : relL2={w['relL2']:.3e}  Linf={w['Linf']:.3e}  ({tw:.0f}s)", flush=True)
        rbf_best = None
        for eps in RBF_EPS:
            t = time.time(); r = decomposed_lsq(rho, rbf_eval_factory(NF_W, eps), NF_W); tr = time.time()-t
            print(f"  RBF-PIELM eps={eps:<5.1f}     : relL2={r['relL2']:.3e}  Linf={r['Linf']:.3e}  ({tr:.0f}s)", flush=True)
            if rbf_best is None or r['relL2'] < rbf_best['relL2']: rbf_best = dict(r, eps=eps)
        t = time.time(); xp = xpinn_mlp(rho); tx = time.time()-t
        print(f"  XPINN-MLP (Adam 8k)     : relL2={xp['relL2']:.3e}  Linf={xp['Linf']:.3e}  ({tx:.0f}s)", flush=True)
        t = time.time(); va = vanilla_mlp(rho); tv = time.time()-t
        print(f"  vanilla MLP (Adam 8k)   : relL2={va['relL2']:.3e}  Linf={va['Linf']:.3e}  ({tv:.0f}s)\n", flush=True)
        out[str(int(rho))] = dict(wavelet=w, rbf_pielm_best=rbf_best, xpinn=xp, vanilla=va, NF=NF_W)
    with open("baselines_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("saved baselines_results.json")
