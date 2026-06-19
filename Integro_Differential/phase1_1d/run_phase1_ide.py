r"""
Phase 1 PoC -- 1D Volterra integro-differential equation with the AD-free wavelet W-PINN.

Problem:   u'(x) + \int_0^x u(t) dt = f(x),  x in [0,1],  u(0)=1.
Exact:     u(x) = e^x  =>  u' = e^x,  \int_0^x e^t dt = e^x - 1  =>  f(x) = 2 e^x - 1.

W-PINN: u(x) = sum_i c_i psi_i(x) + b, with the repo's 1D Gaussian-derivative wavelet
    psi(x) = X e^{-X^2/2},  X = j x - k.
The three operators are ALL precomputed matrices applied to c (AD-free):
  value        psi_i(x)            = X e^{-X^2/2}
  derivative   psi_i'(x)           = j (1 - X^2) e^{-X^2/2}
  INTEGRAL     \int_0^x psi_i dt   = (1/j)( e^{-k^2/2} - e^{-(jx-k)^2/2} )      <-- ANALYTIC, exact
The Volterra integral operational matrix G[m,i] = \int_0^{x_m} psi_i is therefore exact (no
quadrature, no auxiliary points -- the thing AD-based PINNs cannot do for an integral).
Residual at x_m:  sum_i c_i (psi_i'(x_m) + G[m,i]) + b * x_m = f(x_m)   (bias integral = b*x).
Linear in (c, b) -> single column-normalised Tikhonov least-squares solve.
"""
import math, numpy as np, torch
torch.set_default_dtype(torch.float64)

# ---- 1D Gaussian-derivative wavelet family -------------------------------------------------
def family(levels=(0,1,2,3), lo=0.0, hi=1.0, pad=0.5):
    fam=[]
    for l in levels:
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            fam.append((j,float(k)))
    arr=torch.tensor(fam, dtype=torch.float64)
    return arr[:,0], arr[:,1]

def val(x, j, k):          # psi(x)
    X=j[None,:]*x[:,None]-k[None,:]; return X*torch.exp(-X**2/2)
def der(x, j, k):          # psi'(x)
    X=j[None,:]*x[:,None]-k[None,:]; return j[None,:]*(1-X**2)*torch.exp(-X**2/2)
def integ(x, j, k):        # \int_0^x psi dt  (analytic)
    X=j[None,:]*x[:,None]-k[None,:]
    return (torch.exp(-(k[None,:]**2)/2) - torch.exp(-X**2/2))/j[None,:]

# ---- problem -------------------------------------------------------------------------------
u_exact = lambda x: np.exp(x)
f_rhs   = lambda x: 2*np.exp(x) - 1.0

def solve(levels=(0,1,2,3), Nc=400, wb=10.0, tik=1e-12):
    j,k = family(levels); NF=j.numel()
    xc = torch.linspace(0,1,Nc)                     # collocation points
    D  = der(xc,j,k); G = integ(xc,j,k)             # residual operator = D + G  (+ bias*x)
    n  = NF+1
    A  = torch.zeros(Nc+1, n)
    A[:Nc,:NF] = D + G; A[:Nc,NF] = xc              # bias integral term = b*x ; bias' = 0
    x0 = torch.zeros(1)
    A[Nc,:NF] = val(x0,j,k); A[Nc,NF] = 1.0         # BC u(0)=1  (sqrt(wb) weight below)
    A[Nc,:] *= math.sqrt(wb)
    b = torch.zeros(Nc+1,1)
    b[:Nc,0] = torch.tensor(f_rhs(xc.numpy())); b[Nc,0] = math.sqrt(wb)*1.0
    s = A.norm(dim=0).clamp_min(1e-30); An=A/s
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb=torch.cat([b, torch.zeros(n,1)])
    theta=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    c, bias = theta[:NF], theta[NF]
    # evaluate on a dense test grid
    xt = torch.linspace(0,1,2000)
    pred=(val(xt,j,k)@c + bias).numpy(); ue=u_exact(xt.numpy()); err=pred-ue
    return dict(NF=NF, relL2=np.linalg.norm(err)/np.linalg.norm(ue),
                MSE=np.mean(err**2), RMSE=math.sqrt(np.mean(err**2)), MAE=np.mean(np.abs(err)),
                Linf=np.max(np.abs(err)), relLinf=np.max(np.abs(err))/np.max(np.abs(ue)))

if __name__=="__main__":
    print("1D Volterra integro-differential eqn, AD-free wavelet operational-matrix W-PINN")
    print("u'(x) + \\int_0^x u dt = 2e^x-1,  u(0)=1,  exact u=e^x\n")
    hdr=f"{'levels':>12} {'NF':>4} | {'relL2':>10} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'Linf':>10} {'relLinf':>10}"
    print(hdr); print("-"*len(hdr))
    for lv in [(0,1),(0,1,2),(0,1,2,3),(0,1,2,3,4)]:
        m=solve(lv)
        print(f"{str(lv):>12} {m['NF']:>4d} | {m['relL2']:>10.2e} {m['MSE']:>10.2e} {m['RMSE']:>10.2e} "
              f"{m['MAE']:>10.2e} {m['Linf']:>10.2e} {m['relLinf']:>10.2e}", flush=True)
