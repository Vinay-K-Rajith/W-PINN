"""
Phase 3 PoC (A): inverse CONTRAST recovery in 1D (geometry known, a^+ unknown).
Given sparse noisy measurements of u, jointly recover the unknown contrast rho=a^+/a^- and the field.
Reuses the Phase-1 wavelet machinery. Joint Adam over (log_rho, coefficients, biases) + data misfit.
"""
import sys, os, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_1d"))
from run_phase1 import (build_family, W_mats, make_problem, beta, a_minus, jv, kv, NF,
                        D2_m, D2_p, P_m, P_p, Pg_m, Dg_m, Pg_p, Dg_p, P0, P1, x_test, x_minus, x_plus)
torch.set_default_dtype(torch.float64); torch.manual_seed(0); np.random.seed(0)

def run(rho_true=8.0, Ndata=15, noise=1e-3, rho_init=2.0, epochs=15000, lr=5e-3):
    prob = make_problem(rho_true)
    # ---- synthetic sparse measurements from the exact solution ----
    xd = np.sort(np.random.rand(Ndata))                     # scattered interior points
    ud = prob["u_exact"](xd) * (1 + noise*np.random.randn(Ndata))
    Pd_m,_,_ = W_mats(xd[xd<beta], jv, kv); Pd_p,_,_ = W_mats(xd[xd>=beta], jv, kv)
    ud_m = torch.tensor(ud[xd<beta]); ud_p = torch.tensor(ud[xd>=beta])
    fm = torch.tensor(prob["f"](x_minus)); fp = torch.tensor(prob["f"](x_plus))

    # ---- unknowns: log_rho (>0 via exp), coefficients, biases ----
    log_rho = torch.zeros(1, requires_grad=True);
    with torch.no_grad(): log_rho += math.log(rho_init)
    cM=torch.zeros(NF,requires_grad=True); cP=torch.zeros(NF,requires_grad=True)
    bM=torch.zeros(1,requires_grad=True);  bP=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([log_rho,cM,cP,bM,bP], lr=lr)
    Wd, Wb = 50.0, 50.0
    for ep in range(epochs):
        opt.zero_grad(); ap = torch.exp(log_rho)             # a^+ = rho (a^-=1)
        rM = -a_minus*(D2_m@cM) - fm
        rP = -ap*(D2_p@cP) - fp
        cont = (Pg_p@cP+bP)-(Pg_m@cM+bM)
        flux = ap*(Dg_p@cP) - a_minus*(Dg_m@cM)
        bc = (P0@cM+bM-prob["u0"]).pow(2)+(P1@cP+bP-prob["u1"]).pow(2)
        data = (Pd_m@cM+bM-ud_m).pow(2).mean()+(Pd_p@cP+bP-ud_p).pow(2).mean()
        loss = rM.pow(2).mean()+rP.pow(2).mean()+50*(cont.pow(2).mean()+flux.pow(2).mean())+Wb*bc.mean()+Wd*data
        loss.backward(); opt.step()
    rho_hat=float(torch.exp(log_rho).item())
    Pt,_,_=W_mats(x_test,jv,kv)
    pm=(Pt@cM+bM).detach().numpy(); pp=(Pt@cP+bP).detach().numpy()
    field=np.where(x_test<beta,pm,pp); ue=prob["u_exact"](x_test)
    relL2=np.linalg.norm(field-ue)/np.linalg.norm(ue)
    return rho_hat, abs(rho_hat-rho_true)/rho_true, relL2

if __name__=="__main__":
    print("Inverse CONTRAST recovery (1D), a^-=1, recover rho=a^+\n")
    print(f"{'rho_true':>9} {'Ndata':>6} {'noise':>7} | {'rho_hat':>9} {'rel_err':>9} {'field_relL2':>11}")
    print("-"*60)
    for rho_true in [4.0, 8.0, 20.0]:
        for noise in [0.0, 1e-2]:
            rh, re, rl = run(rho_true=rho_true, Ndata=15, noise=noise)
            print(f"{rho_true:>9.1f} {15:>6} {noise:>7.0e} | {rh:>9.4f} {re:>9.2e} {rl:>11.3e}")
