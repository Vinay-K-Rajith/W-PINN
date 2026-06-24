r"""
Diagnose the operator-solve residual-to-error gap: best-fit reaches ~2e-2 but the PDE solve plateaus
at ~0.1. Hypotheses: (i) collocation UNDER-samples the oscillatory Laplacian of fine wavelets
(aliasing); (ii) Tikhonov too strong/weak. Sweep collocation density K and tik for the banded basis.
"""
import math, numpy as np, torch
import run_phase2 as P
torch.set_default_dtype(torch.float64)

def colloc(K):
    g=np.linspace(-1,1,K); Xg,Yg=np.meshgrid(g,g); xi=Xg.ravel(); yi=Yg.ravel()
    keep=(np.abs(xi)<1)&(np.abs(yi)<1); return xi[keep],yi[keep]

def solve(co,fi,K,tik,wb=10.0):
    xi,yi=colloc(K)
    jx,kx,jy,ky=P.family(co,fi); NF=jx.numel()
    Pi,Li=P.W_mats(xi,yi,jx,kx,jy,ky); Pb,_=P.W_mats(P.BX,P.BY,jx,kx,jy,ky)
    A=torch.cat([-Li+P.KAPPA**2*Pi, math.sqrt(wb)*Pb],0)
    b=torch.cat([torch.tensor(P.f_rhs(xi,yi)), math.sqrt(wb)*torch.tensor(P.u_exact(P.BX,P.BY))]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30)
    Aa=torch.cat([A/s, math.sqrt(tik)*torch.eye(NF)]); bb=torch.cat([b, torch.zeros(NF,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    Pt,_=P.W_mats(P.XTr,P.YTr,jx,kx,jy,ky); pred=(Pt@c).numpy()
    return NF, float(np.linalg.norm(pred-P.UE_T)/np.linalg.norm(P.UE_T))

if __name__=="__main__":
    co,fi=(0,1,2,3),(4,5)
    print("banded [0,1,2,3]+[4,5] -- operator solve relL2 vs collocation density K and Tikhonov tik\n")
    print(f"{'K':>5} {'intpts':>7} | " + " ".join(f"tik={t:.0e}".rjust(11) for t in (1e-6,1e-8,1e-10,1e-12)))
    for K in (120,160,220,300):
        xi,_=colloc(K); row=[]
        for tik in (1e-6,1e-8,1e-10,1e-12):
            NF,rl2=solve(co,fi,K,tik); row.append(rl2)
        print(f"{K:>5d} {len(xi):>7d} | " + " ".join(f"{r:>11.2e}" for r in row), flush=True)
