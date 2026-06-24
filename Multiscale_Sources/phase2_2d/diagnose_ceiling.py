r"""
Diagnose the 2D accuracy ceiling: is it the SOLVE (operator/conditioning) or the BASIS (best-fit)?

Test: directly L2-fit u_exact with the SAME wavelet basis (value matrix only, no PDE operator).
 - If best-fit relL2 ~ PDE-solve relL2 (~0.1)  => the BASIS cannot represent the field (basis limit).
 - If best-fit relL2 << PDE-solve relL2        => the OPERATOR/conditioning is the bottleneck.
Also try an EVEN (non-zero-mean) tensor basis [psi_even = (1-X^2)e^{-X^2/2}] to see if the odd
Gaussian-derivative wavelet (zero at its centre) is the culprit for isotropic positive bumps.
"""
import math, numpy as np, torch
import run_phase2 as P
torch.set_default_dtype(torch.float64)
XI,YI,XTr,YTr,UE_T=P.XI,P.YI,P.XTr,P.YTr,P.UE_T

def fit(jx,kx,jy,ky, basis="odd", tik=1e-10):
    def Psi(x,y):
        x=torch.as_tensor(x).reshape(-1); y=torch.as_tensor(y).reshape(-1)
        X=jx[None,:]*x[:,None]-kx[None,:]; Y=jy[None,:]*y[:,None]-ky[None,:]; E=torch.exp(-(X**2+Y**2)/2)
        if basis=="odd":  return (X*Y)*E                      # Gaussian-derivative (current), odd*odd
        if basis=="even": return (1-X**2)*(1-Y**2)*E          # even Mexican-hat-like, nonzero at centre
        if basis=="gauss":return E                            # plain Gaussian (matches bump shape)
    A=Psi(XI,YI); b=torch.tensor(P.u_exact(XI,YI)).unsqueeze(1); NF=A.shape[1]
    s=A.norm(dim=0).clamp_min(1e-30); An=A/s
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(NF)]); bb=torch.cat([b, torch.zeros(NF,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    pred=(Psi(XTr,YTr)@c).numpy(); return float(np.linalg.norm(pred-UE_T)/np.linalg.norm(UE_T))

if __name__=="__main__":
    print("Best-fit (L2-projection) ceiling test -- no PDE operator, just represent u_exact\n")
    for co,fi in [((0,1,2,3),()),((0,1,2,3),(4,5)),((0,1,2,3,4,5),())]:
        jx,kx,jy,ky=P.family(co,fi); NF=jx.numel()
        ro=fit(jx,kx,jy,ky,"odd"); re=fit(jx,kx,jy,ky,"even"); rg=fit(jx,kx,jy,ky,"gauss")
        print(f"coarse={co} fine={fi}  NF={NF:5d} | best-fit relL2: "
              f"odd-deriv={ro:.2e}  even={re:.2e}  gauss={rg:.2e}", flush=True)
