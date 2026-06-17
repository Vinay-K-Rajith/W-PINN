"""
Phase 3 PoC (A) - CORRECTED: inverse contrast via OUTER-LOOP over rho + inner linear forward solve.
Avoids the bilinear/ill-conditioned joint-Adam failure. Known forcing f and boundary data; unknown rho.
For each candidate rho: AD-free least-squares forward solve -> predicted u -> interior data misfit.
Minimize misfit over the scalar rho (golden-section). Parsimonious well-conditioned basis.
"""
import sys, os, math, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1_1d"))
from run_phase1 import build_family, W_mats, make_problem, beta, a_minus
torch.set_default_dtype(torch.float64); np.random.seed(0)

# parsimonious, well-conditioned basis (Phase-2 lesson) + well-determined collocation
jv, kv = build_family(J=5)                       # levels 0..4
NF = jv.numel()
xm = np.linspace(0, beta, 402)[1:-1]; xp = np.linspace(beta, 1, 402)[1:-1]
_,_,D2m_ = (lambda P,_d,D2: (P,_d,D2))(*W_mats(xm,jv,kv)); D2m = W_mats(xm,jv,kv)[2]
D2p = W_mats(xp,jv,kv)[2]
Pgm,Dgm,_ = W_mats([beta],jv,kv); Pgp,Dgp,_ = W_mats([beta],jv,kv)
P0 = W_mats([0.0],jv,kv)[0]; P1 = W_mats([1.0],jv,kv)[0]

def forward(rho, f_m, f_p, u0, u1, wi=1.0, wb=1.0):
    """AD-free least-squares forward solve for given rho. Returns (cM,cP,bM,bP)."""
    n=2*NF+2
    def blk(rows,cM=None,cP=None,bM=None,bP=None):
        B=torch.zeros(rows,n)
        if cM is not None:B[:,:NF]=cM
        if cP is not None:B[:,NF:2*NF]=cP
        if bM is not None:B[:,2*NF]=bM
        if bP is not None:B[:,2*NF+1]=bP
        return B
    A=torch.cat([blk(len(xm),cM=-a_minus*D2m), blk(len(xp),cP=-rho*D2p),
                 math.sqrt(wi)*blk(1,cP=Pgp,cM=-Pgm,bP=1.,bM=-1.),
                 math.sqrt(wi)*blk(1,cP=rho*Dgp,cM=-a_minus*Dgm),
                 math.sqrt(wb)*blk(1,cM=P0,bM=1.), math.sqrt(wb)*blk(1,cP=P1,bP=1.)],0)
    b=torch.cat([f_m, f_p, torch.zeros(2),
                 math.sqrt(wb)*torch.tensor([u0]), math.sqrt(wb)*torch.tensor([u1])]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30)
    th=(torch.linalg.lstsq(A/s, b, driver="gelsd").solution.squeeze(1))/s
    return th[:NF], th[NF:2*NF], th[2*NF], th[2*NF+1]

def predict(c, b, x): return (W_mats(x,jv,kv)[0]@c + b)

def invert(rho_true=8.0, Ndata=15, noise=1e-3):
    prob = make_problem(rho_true)
    f_m = torch.tensor(prob["f"](xm)); f_p = torch.tensor(prob["f"](xp))
    u0, u1 = prob["u0"], prob["u1"]                       # known boundary data
    xd = np.sort(np.random.rand(Ndata))
    ud = prob["u_exact"](xd)*(1+noise*np.random.randn(Ndata)); ud=torch.tensor(ud)
    xdm, xdp = xd[xd<beta], xd[xd>=beta]
    def misfit(rho):
        cM,cP,bM,bP = forward(rho, f_m, f_p, u0, u1)
        e=0.0
        if len(xdm): e=e+(predict(cM,bM,xdm)-ud[:len(xdm)]).pow(2).sum()
        if len(xdp): e=e+(predict(cP,bP,xdp)-ud[len(xdm):]).pow(2).sum()
        return float(e.item())
    # golden-section search on rho in [1.2, 40]
    a,b=1.2,40.0; gr=(math.sqrt(5)-1)/2
    c,d=b-gr*(b-a),a+gr*(b-a); fc,fd=misfit(c),misfit(d)
    for _ in range(40):
        if fc<fd: b,d,fd=d,c,fc; c=b-gr*(b-a); fc=misfit(c)
        else: a,c,fc=c,d,fd; d=a+gr*(b-a); fd=misfit(d)
    rho_hat=(a+b)/2
    cM,cP,bM,bP=forward(rho_hat,f_m,f_p,u0,u1)
    xt=np.linspace(0,1,2000)
    fm_=predict(cM,bM,xt).numpy(); fp_=predict(cP,bP,xt).numpy(); field=np.where(xt<beta,fm_,fp_)
    ue=prob["u_exact"](xt); relL2=np.linalg.norm(field-ue)/np.linalg.norm(ue)
    return rho_hat, abs(rho_hat-rho_true)/rho_true, relL2

if __name__=="__main__":
    print("Inverse CONTRAST via outer-loop (golden-section) + inner LSQ forward\n")
    print(f"{'rho_true':>9} {'Ndata':>6} {'noise':>7} | {'rho_hat':>9} {'rel_err':>9} {'field_relL2':>11}")
    print("-"*60)
    for rho_true in [4.0, 8.0, 20.0]:
        for noise in [0.0, 1e-2]:
            rh,re,rl = invert(rho_true=rho_true, Ndata=15, noise=noise)
            print(f"{rho_true:>9.1f} {15:>6} {noise:>7.0e} | {rh:>9.4f} {re:>9.2e} {rl:>11.3e}")
