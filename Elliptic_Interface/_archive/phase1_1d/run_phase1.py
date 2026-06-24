"""
Phase 1 - 1D elliptic interface problem, proof of concept for Wavelet-PINN.

Problem (homogeneous interface conditions, manufactured exact solution):
    -(a(x) u'(x))' = f(x),  x in (0,1),  interface at x = beta
    a = a_minus (x<beta),  a_plus (x>beta)
    [[u]] = 0,  [[a u']] = 0  at beta;  Dirichlet BCs at 0,1.

Exact:  u_- = sin(pi x)              on (0,beta)
        u_+ = A sin(pi x) + B        on (beta,1),  A = a_minus/a_plus,  B = sin(pi beta)(1-A)
        => f = a_minus pi^2 sin(pi x)  on BOTH sides (continuous), genuine gradient kink at beta.

Models compared:
    M1  decomposed wavelet W-PINN   (the proposal: two expansions + interface coupling)
    M2  single-expansion wavelet    (ablation: one global smooth expansion, no interface dof)
    M3  vanilla single MLP PINN     (standard baseline, AD)
    M4  decomposed MLP / XPINN      (isolates "basis" from "decomposition")

Wavelet = Gaussian-derivative (Mexican-hat family) reused verbatim from the repo's Wfamily.py.
"""

import time, math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)          # 1D is tiny -> use double for clean error numbers
device = torch.device("cpu")
torch.manual_seed(0); np.random.seed(0)

PI = math.pi

# ----------------------------------------------------------------------------- problem
beta    = 0.4
a_minus = 1.0
RHO     = 10.0                                    # contrast a_plus/a_minus  (swept later)

def make_problem(rho):
    a_plus = rho * a_minus
    A = a_minus / a_plus
    B = math.sin(PI*beta) * (1.0 - A)
    def u_exact(x):
        x = np.asarray(x)
        return np.where(x < beta, np.sin(PI*x), A*np.sin(PI*x) + B)
    def du_exact(x):
        x = np.asarray(x)
        return np.where(x < beta, PI*np.cos(PI*x), A*PI*np.cos(PI*x))
    def f(x):                                     # same on both sides
        return a_minus * PI**2 * np.sin(PI*np.asarray(x))
    u0 = 0.0
    u1 = A*math.sin(PI*1.0) + B                   # = B
    jump_dudx = (A - 1.0)*PI*math.cos(PI*beta)    # exact [[u']] at beta
    return dict(a_plus=a_plus, A=A, B=B, u_exact=u_exact, du_exact=du_exact,
                f=f, u0=u0, u1=u1, jump_dudx=jump_dudx)

# ----------------------------------------------------------------------------- wavelet family
def build_family(J=7, pad=0.5, lo=0.0, hi=1.0):
    rows = []
    for lvl in range(J):
        j = 2.0**lvl
        kmin = int(math.floor((lo-pad)*j)); kmax = int(math.ceil((hi+pad)*j))
        for k in range(kmin, kmax+1):
            rows.append((j, float(k)))
    fam = torch.tensor(rows, dtype=torch.float64, device=device)
    return fam[:,0], fam[:,1]                      # j-vector, k-vector

def W_mats(x, jv, kv):
    """Return (Psi, dPsi, d2Psi) matrices of shape [len(x), n_family]."""
    x = torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1)
    T = jv[None,:]*x[:,None] - kv[None,:]
    e = torch.exp(-(T**2)/2)
    Psi   = -T*e
    dPsi  = jv[None,:]*(T**2 - 1.0)*e
    d2Psi = (jv[None,:]**2)*T*(3.0 - T**2)*e
    return Psi, dPsi, d2Psi

# ----------------------------------------------------------------------------- collocation
N_INT = 256
x_minus = np.linspace(0.0, beta, N_INT+2)[1:-1]          # interior of Omega^-
x_plus  = np.linspace(beta, 1.0, N_INT+2)[1:-1]          # interior of Omega^+
x_all   = np.sort(np.concatenate([x_minus, x_plus]))
x_test  = np.linspace(0.0, 1.0, 4001)
xb = np.array([beta])

jv, kv = build_family()
NF = jv.numel()

# ----------------------------------------------------------------------------- operator unit test
def unit_test_operators():
    xg = np.linspace(0.02, 0.98, 23)
    Psi, dPsi, d2Psi = W_mats(xg, jv, kv)
    h = 1e-5
    Pp,_,_ = W_mats(xg+h, jv, kv); Pm,_,_ = W_mats(xg-h, jv, kv)
    d1_fd = (Pp-Pm)/(2*h); d2_fd = (Pp-2*Psi+Pm)/(h*h)
    e1 = (d1_fd-dPsi).abs().max().item()/dPsi.abs().max().item()
    e2 = (d2_fd-d2Psi).abs().max().item()/d2Psi.abs().max().item()
    return e1, e2

# ----------------------------------------------------------------------------- helpers
def metrics(pred, prob):
    ue = prob["u_exact"](x_test)
    relL2 = np.linalg.norm(pred-ue)/np.linalg.norm(ue)
    emax  = np.max(np.abs(pred-ue))
    return relL2, emax

def eval_wavelet(c, b, x, jv, kv):
    Psi,_,_ = W_mats(x, jv, kv)
    return (Psi @ c + b).detach().cpu().numpy()

# precompute matrices on fixed point sets (used by wavelet models) -------------
P_m, _, D2_m   = W_mats(x_minus, jv, kv)
P_p, _, D2_p   = W_mats(x_plus , jv, kv)
P_all,_,D2_all = W_mats(x_all  , jv, kv)
Pg_m, Dg_m, _  = W_mats(xb, jv, kv)
Pg_p, Dg_p, _  = W_mats(xb, jv, kv)
P0,_,_         = W_mats([0.0], jv, kv)
P1,_,_         = W_mats([1.0], jv, kv)

W_IFACE, W_BC = 50.0, 50.0          # loss weights for single-point constraints

# =============================================================================== M1
def train_M1(prob, epochs=8000, lr=1e-2):
    f_m = torch.tensor(prob["f"](x_minus)); f_p = torch.tensor(prob["f"](x_plus))
    ap = prob["a_plus"]
    cM = torch.zeros(NF, requires_grad=True); cP = torch.zeros(NF, requires_grad=True)
    bM = torch.zeros(1, requires_grad=True);  bP = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([cM,cP,bM,bP], lr=lr)
    t0=time.time()
    for ep in range(epochs):
        opt.zero_grad()
        rM = -a_minus*(D2_m@cM) - f_m
        rP = -ap     *(D2_p@cP) - f_p
        cont = (Pg_p@cP + bP) - (Pg_m@cM + bM)
        flux = ap*(Dg_p@cP) - a_minus*(Dg_m@cM)
        bc0 = (P0@cM + bM) - prob["u0"]
        bc1 = (P1@cP + bP) - prob["u1"]
        loss = rM.pow(2).mean()+rP.pow(2).mean() \
             + W_IFACE*(cont.pow(2).mean()+flux.pow(2).mean()) \
             + W_BC*(bc0.pow(2).mean()+bc1.pow(2).mean())
        loss.backward(); opt.step()
    dt=time.time()-t0
    pred = eval_wavelet(cM,bM,x_test,jv,kv); pp = eval_wavelet(cP,bP,x_test,jv,kv)
    pred = np.where(x_test<beta, pred, pp)
    # predicted gradient jump at beta
    duM = float((Dg_m@cM).item()); duP = float((Dg_p@cP).item())
    return pred, dt, loss.item(), duP-duM

# =============================================================================== M2
def train_M2(prob, epochs=8000, lr=1e-2):
    f_all = torch.tensor(prob["f"](x_all))
    avec = torch.tensor(np.where(x_all<beta, a_minus, prob["a_plus"]))
    c = torch.zeros(NF, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([c,b], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        r = -avec*(D2_all@c) - f_all
        bc0 = (P0@c + b) - prob["u0"]; bc1 = (P1@c + b) - prob["u1"]
        loss = r.pow(2).mean() + W_BC*(bc0.pow(2).mean()+bc1.pow(2).mean())
        loss.backward(); opt.step()
    pred = eval_wavelet(c,b,x_test,jv,kv)
    Pg,Dg,_ = W_mats(xb,jv,kv)
    dujump = 0.0   # single C^1 expansion -> structural gradient jump is ~0
    return pred, loss.item(), dujump

# =============================================================================== MLP
class MLP(torch.nn.Module):
    def __init__(self, w=50, d=4):
        super().__init__()
        layers=[torch.nn.Linear(1,w), torch.nn.Tanh()]
        for _ in range(d-1): layers += [torch.nn.Linear(w,w), torch.nn.Tanh()]
        layers += [torch.nn.Linear(w,1)]
        self.net=torch.nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight); torch.nn.init.zeros_(m.bias)
    def forward(self,x): return self.net(x)

def d2(u, x):
    g = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    gg= torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True)[0]
    return g, gg

# =============================================================================== M3 vanilla single MLP
def train_M3(prob, epochs=8000, lr=1e-3):
    net=MLP()
    xc=torch.tensor(x_all).reshape(-1,1).requires_grad_(True)
    favec=torch.tensor(prob["f"](x_all)).reshape(-1,1)
    avec=torch.tensor(np.where(x_all<beta,a_minus,prob["a_plus"])).reshape(-1,1)
    x0=torch.tensor([[0.0]]); x1=torch.tensor([[1.0]])
    opt=torch.optim.Adam(net.parameters(),lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        u=net(xc); _,uxx=d2(u,xc)
        r=-avec*uxx-favec
        bc=(net(x0)-prob["u0"]).pow(2)+(net(x1)-prob["u1"]).pow(2)
        loss=r.pow(2).mean()+W_BC*bc.mean()
        loss.backward(); opt.step()
    pred=net(torch.tensor(x_test).reshape(-1,1)).detach().cpu().numpy().ravel()
    return pred, loss.item()

# =============================================================================== M4 decomposed MLP (XPINN)
def train_M4(prob, epochs=8000, lr=1e-3):
    netM,netP=MLP(),MLP()
    xm=torch.tensor(x_minus).reshape(-1,1).requires_grad_(True)
    xp=torch.tensor(x_plus ).reshape(-1,1).requires_grad_(True)
    fm=torch.tensor(prob["f"](x_minus)).reshape(-1,1)
    fp=torch.tensor(prob["f"](x_plus )).reshape(-1,1)
    ap=prob["a_plus"]
    xbt=torch.tensor([[beta]]).requires_grad_(True)
    x0=torch.tensor([[0.0]]); x1=torch.tensor([[1.0]])
    opt=torch.optim.Adam(list(netM.parameters())+list(netP.parameters()),lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        uM=netM(xm); _,uxxM=d2(uM,xm)
        uP=netP(xp); _,uxxP=d2(uP,xp)
        rM=-a_minus*uxxM-fm; rP=-ap*uxxP-fp
        uMb=netM(xbt); gMb=torch.autograd.grad(uMb,xbt,torch.ones_like(uMb),create_graph=True)[0]
        uPb=netP(xbt); gPb=torch.autograd.grad(uPb,xbt,torch.ones_like(uPb),create_graph=True)[0]
        cont=(uPb-uMb); flux=ap*gPb-a_minus*gMb
        bc=(netM(x0)-prob["u0"]).pow(2)+(netP(x1)-prob["u1"]).pow(2)
        loss=rM.pow(2).mean()+rP.pow(2).mean()+W_IFACE*(cont.pow(2).mean()+flux.pow(2).mean())+W_BC*bc.mean()
        loss.backward(); opt.step()
    pm=netM(torch.tensor(x_test).reshape(-1,1)).detach().numpy().ravel()
    pp=netP(torch.tensor(x_test).reshape(-1,1)).detach().numpy().ravel()
    pred=np.where(x_test<beta,pm,pp)
    duP=float(gPb.item()); duM=float(gMb.item())
    return pred, loss.item(), duP-duM

# =============================================================================== run
if __name__=="__main__":
    print(f"family size NF = {NF},  interior pts/side = {N_INT},  beta={beta}")
    e1,e2 = unit_test_operators()
    print(f"[unit-test] max rel err  d/dx: {e1:.2e}   d2/dx2: {e2:.2e}")

    prob = make_problem(RHO)
    print(f"\n=== contrast rho = {RHO} ===  exact [[u']]@beta = {prob['jump_dudx']:+.4f}")

    t=time.time(); p1,dt1,l1,j1 = train_M1(prob);
    r,m = metrics(p1,prob); print(f"M1 decomposed-wavelet : relL2={r:.3e}  max={m:.3e}  [[u']]_pred={j1:+.4f}  loss={l1:.2e}  {time.time()-t:.1f}s")

    t=time.time(); p2,l2,j2 = train_M2(prob)
    r,m = metrics(p2,prob); print(f"M2 single-wavelet     : relL2={r:.3e}  max={m:.3e}  [[u']]_pred~{j2:+.4f}  loss={l2:.2e}  {time.time()-t:.1f}s")

    t=time.time(); p3,l3 = train_M3(prob)
    r,m = metrics(p3,prob); print(f"M3 vanilla-MLP        : relL2={r:.3e}  max={m:.3e}  loss={l3:.2e}  {time.time()-t:.1f}s")

    t=time.time(); p4,l4,j4 = train_M4(prob)
    r,m = metrics(p4,prob); print(f"M4 decomposed-MLP/XPINN: relL2={r:.3e}  max={m:.3e}  [[u']]_pred={j4:+.4f}  loss={l4:.2e}  {time.time()-t:.1f}s")

    np.savez("phase1_results.npz", x_test=x_test, ue=prob["u_exact"](x_test),
             p1=p1,p2=p2,p3=p3,p4=p4)
    print("\nsaved phase1_results.npz")
