r"""
Phase-2 2D test: does RRQR LOCAL-FEATURE FILTERING break the 2D conditioning ceiling?

Documented limit (results.json + memory): the AD-free 2D wavelet solve plateaus at relL2 ~1e-1 because
adding fine levels (>=6) DEGRADES the solution -- the overcomplete operational-matrix frame becomes
numerically rank-deficient (cond ~1e16+), so extra fine atoms inject noise instead of resolution.

Fix (arXiv:2506.17626, "Local Feature Filtering for ... Random Feature Methods"): rank-revealing QR
with column pivoting on the column-normalised operator matrix A. Pivoted QR orders columns by how much
NEW, well-conditioned information each adds; keep the leading columns whose pivot |R_ii| stays above
tau*|R_11| and DROP the redundant tail. This is principled, automatic parsimony -- it removes exactly
the ill-conditioned redundant atoms that cause the degradation, and (cond(A)=cond(R)) we can read the
conditioning straight off R.

Test: build a RICH basis (coarse [0,1,2,3] + banded fine [4,5,6] @bumps) that NORMALLY degrades, then
compare a direct Tikhonov solve vs an RRQR-filtered solve. Report cond and relL2 for each.
"""
import math, json, numpy as np, torch
import scipy.linalg as sla
import run_phase2 as P                     # reuse family / W_mats / collocation / metrics
torch.set_default_dtype(torch.float64)

KAPPA=P.KAPPA; XI,YI=P.XI,P.YI; BX,BY=P.BX,P.BY; XTr,YTr=P.XTr,P.YTr

def assemble(coarse, fine, wb=10.0):
    jx,kx,jy,ky=P.family(coarse,fine); NF=jx.numel()
    Pi,Li=P.W_mats(XI,YI,jx,kx,jy,ky); Pb,_=P.W_mats(BX,BY,jx,kx,jy,ky)
    A=torch.cat([-Li+KAPPA**2*Pi, math.sqrt(wb)*Pb],0)
    b=torch.cat([torch.tensor(P.f_rhs(XI,YI)), math.sqrt(wb)*torch.tensor(P.u_exact(BX,BY))]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30)                          # column normalisation
    return A, b, s, (jx,kx,jy,ky)

def cond_of(An):
    # cond(A) = cond(R); compute R via economic pivoted QR (cheap n x n SVD)
    R = sla.qr(An.numpy(), mode='r', pivoting=False)[0]
    sv = np.linalg.svd(R, compute_uv=False)
    return float(sv[0]/max(sv[-1],1e-300))

def solve_direct(coarse, fine, tik=1e-8):
    A,b,s,fam=assemble(coarse,fine); An=A/s; NF=An.shape[1]
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(NF)]); bb=torch.cat([b, torch.zeros(NF,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    Pt,_=P.W_mats(XTr,YTr,*fam); pred=(Pt@c).numpy()
    return NF, P.metrics(pred), cond_of(An)

def solve_rrqr(coarse, fine, tau=1e-7, tik=1e-10):
    """RRQR column filtering: keep columns whose pivot |R_ii| > tau*|R_11|, then LSQ on the kept set."""
    A,b,s,fam=assemble(coarse,fine); An=(A/s).numpy(); NF=An.shape[1]
    R,Pcol=sla.qr(An, mode='r', pivoting=True)                # rank-revealing pivoted QR
    diag=np.abs(np.diag(R)); keep_mask=diag > tau*diag[0]
    keep=np.sort(Pcol[keep_mask]); NFk=keep.size
    Ak=torch.tensor(An[:,keep]); sk=s[keep]
    Aa=torch.cat([Ak, math.sqrt(tik)*torch.eye(NFk)]); bb=torch.cat([b, torch.zeros(NFk,1)])
    ck=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/sk
    jx,kx,jy,ky=fam; Pt,_=P.W_mats(XTr,YTr,jx[keep],kx[keep],jy[keep],ky[keep]); pred=(Pt@ck).numpy()
    return NF, NFk, P.metrics(pred), cond_of(Ak)

if __name__=="__main__":
    print(f"2D RRQR conditioning test  (interior pts={len(XI)})\n")
    out={}
    configs=[("medium [0,1,2,3]",        (0,1,2,3), ()),
             ("banded [0,1,2,3]+[4,5]",   (0,1,2,3), (4,5)),
             ("rich [0,1,2,3]+[4,5,6]",   (0,1,2,3), (4,5,6))]   # the one that normally DEGRADES
    print(f"{'config':28} | {'NF':>5} {'relL2(direct)':>13} {'cond(direct)':>12} || "
          f"{'NFkept':>6} {'relL2(RRQR)':>11} {'cond(RRQR)':>11}")
    print("-"*108)
    for name,co,fi in configs:
        NF,md,cd = solve_direct(co,fi)
        NF2,NFk,mr,cr = solve_rrqr(co,fi)
        out[name]=dict(NF=NF, relL2_direct=md['relL2'], cond_direct=cd,
                       NF_kept=NFk, relL2_rrqr=mr['relL2'], cond_rrqr=cr,
                       Linf_direct=md['Linf'], Linf_rrqr=mr['Linf'])
        print(f"{name:28} | {NF:>5d} {md['relL2']:>13.2e} {cd:>12.2e} || "
              f"{NFk:>6d} {mr['relL2']:>11.2e} {cr:>11.2e}", flush=True)
    json.dump(out, open("rrqr_results.json","w"), indent=2)
    print("\nsaved rrqr_results.json")
