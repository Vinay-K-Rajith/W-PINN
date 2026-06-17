"""
Phase 1 - optimization + analysis.
(1) Replace Adam with the correct optimizer for the LINEAR wavelet problem: a weighted
    least-squares solve of the convex W-PINN objective (AD-free, ~instant) = exact optimum.
(2) Sweep contrast rho to find where the decomposed-wavelet method holds vs degrades,
    and compare against the strongest baseline (decomposed MLP / XPINN, M4 from run_phase1).
"""
import time, math, numpy as np, torch
from run_phase1 import (build_family, W_mats, make_problem, beta, a_minus,
                        MLP, d2, train_M4)

torch.set_default_dtype(torch.float64)
PI = math.pi

# richer collocation so the linear system is well over-determined
N_INT = 600
x_minus = np.linspace(0.0, beta, N_INT+2)[1:-1]
x_plus  = np.linspace(beta, 1.0, N_INT+2)[1:-1]
x_test  = np.linspace(0.0, 1.0, 4001)
jv, kv  = build_family(J=7)
NF = jv.numel()

P_m,_,D2_m = W_mats(x_minus, jv, kv)
P_p,_,D2_p = W_mats(x_plus , jv, kv)
Pg_m,Dg_m,_= W_mats([beta], jv, kv)
Pg_p,Dg_p,_= W_mats([beta], jv, kv)
P0,_,_     = W_mats([0.0], jv, kv)
P1,_,_     = W_mats([1.0], jv, kv)

def lstsq_decomposed(prob, w_iface=1.0, w_bc=1.0):
    """Solve the convex W-PINN least-squares system for theta=[cM,cP,bM,bP]."""
    ap = prob["a_plus"]; n = 2*NF + 2
    Z = torch.zeros
    def col(cM=None,cP=None,bM=None,bP=None,m=1):
        blk = torch.zeros(m, n)
        if cM is not None: blk[:, :NF] = cM
        if cP is not None: blk[:, NF:2*NF] = cP
        if bM is not None: blk[:, 2*NF] = bM
        if bP is not None: blk[:, 2*NF+1] = bP
        return blk
    rows=[]; rhs=[]
    # interior residuals  -a u'' = f
    rows.append(col(cM=-a_minus*D2_m, m=len(x_minus))); rhs.append(torch.tensor(prob["f"](x_minus)))
    rows.append(col(cP=-ap*D2_p,      m=len(x_plus ))); rhs.append(torch.tensor(prob["f"](x_plus )))
    # interface continuity  u+ - u- = 0
    rows.append(math.sqrt(w_iface)*col(cP=Pg_p,cM=-Pg_m,bP=1.0,bM=-1.0,m=1)); rhs.append(torch.zeros(1))
    # interface flux  a+ u+' - a- u-' = 0
    rows.append(math.sqrt(w_iface)*col(cP=ap*Dg_p,cM=-a_minus*Dg_m,m=1));     rhs.append(torch.zeros(1))
    # boundary
    rows.append(math.sqrt(w_bc)*col(cM=P0,bM=1.0,m=1)); rhs.append(torch.tensor([prob["u0"]]))
    rows.append(math.sqrt(w_bc)*col(cP=P1,bP=1.0,m=1)); rhs.append(torch.tensor([prob["u1"]]))
    A = torch.cat(rows,0); bb = torch.cat(rhs,0)
    t0=time.time()
    theta = torch.linalg.lstsq(A, bb.unsqueeze(1), driver="gelsd").solution.squeeze(1)
    dt=time.time()-t0
    cM,cP = theta[:NF], theta[NF:2*NF]; bM,bP = theta[2*NF], theta[2*NF+1]
    pm = (W_mats(x_test,jv,kv)[0]@cM + bM).numpy()
    pp = (W_mats(x_test,jv,kv)[0]@cP + bP).numpy()
    pred = np.where(x_test<beta, pm, pp)
    duM=float((Dg_m@cM).item()); duP=float((Dg_p@cP).item())
    return pred, dt, duP-duM, float(torch.linalg.cond(A).item())

def metrics(pred, prob):
    ue=prob["u_exact"](x_test)
    return np.linalg.norm(pred-ue)/np.linalg.norm(ue), np.max(np.abs(pred-ue))

if __name__=="__main__":
    print(f"NF={NF}, interior/side={N_INT}\n")
    print(f"{'rho':>6} | {'wavelet-LSQ relL2':>17} {'max':>9} {'kink_err':>9} {'cond(A)':>9} {'t(s)':>6} | {'XPINN relL2':>11} {'kink_err':>9}")
    print("-"*100)
    for rho in [10.0, 100.0, 1000.0, 10000.0]:
        prob = make_problem(rho)
        pred,dt,jpred,cond = lstsq_decomposed(prob, w_iface=1.0, w_bc=1.0)
        r,m = metrics(pred,prob); jerr=abs(jpred-prob["jump_dudx"])
        # XPINN comparison (more epochs at high contrast)
        ep = 8000 if rho<=100 else 16000
        p4,l4,j4 = train_M4(prob, epochs=ep)
        r4,m4 = metrics(p4,prob); jerr4=abs(j4-prob["jump_dudx"])
        print(f"{rho:>6.0f} | {r:>17.3e} {m:>9.2e} {jerr:>9.2e} {cond:>9.1e} {dt:>6.3f} | {r4:>11.3e} {jerr4:>9.2e}")
