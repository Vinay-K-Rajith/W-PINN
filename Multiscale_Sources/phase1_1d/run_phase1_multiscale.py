r"""
Phase 1 PoC -- 1D MULTISCALE screened-Poisson problem with the AD-free wavelet W-PINN.

    -u''(x) + kappa^2 u(x) = f(x),  x in [0,1],  Dirichlet BCs from the exact solution.

Exact solution has TWO disparate scales -- a smooth O(1) background + a narrow localized spike:
    u(x) = sin(2 pi x)  +  A * exp(-((x - x0)/w)^2),   w << 1   (here w = 0.02).
This is the regime where every competing method fails for a different reason:
  * vanilla PINN  -> spectral bias cannot fit the narrow spike;
  * RBF / Fourier PIELM -> single-scale random/global features "degrade on oscillatory/localized"
    solutions (documented);
  * singularity-enriched methods -> need the analytic singular form, which a generic localized
    feature at an unknown scale does not provide.
W-PINN's MULTIRESOLUTION wavelet basis resolves both scales at once, and adding the finest level
ONLY in a band around the spike (adaptive banded refinement, as in the gear) is the cheap fix.

All operators are precomputed matrices applied to the coefficients (AD-free):
  value        psi(x)   = X e^{-X^2/2},                 X = j x - k
  second der   psi''(x) = j^2 X (X^2 - 3) e^{-X^2/2}
Operator row: (-psi'' + kappa^2 psi).  Linear in c -> single column-normalised Tikhonov LSQ solve.
"""
import math, numpy as np, torch
torch.set_default_dtype(torch.float64)

KAPPA = 1.0
A, X0, W = 1.0, 0.5, 0.02          # spike amplitude / centre / width (the fine scale)

def u_exact(x):
    return np.sin(2*np.pi*x) + A*np.exp(-((x - X0)/W)**2)

def f_rhs(x):
    # f = -u'' + kappa^2 u, computed analytically
    upp_smooth = -(2*np.pi)**2*np.sin(2*np.pi*x)
    g = np.exp(-((x - X0)/W)**2)
    upp_spike = A*g*(4*(x - X0)**2/W**4 - 2.0/W**2)
    u = u_exact(x)
    return -(upp_smooth + upp_spike) + KAPPA**2*u

# ---- 1D Gaussian-derivative wavelet family, with optional banded finest level ---------------
def family(coarse=(0,1,2,3), fine=(), band=(0.40,0.60), lo=0.0, hi=1.0, pad=0.5):
    fam=[]
    for l in coarse:
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            fam.append((j,float(k)))
    for l in fine:                                  # finest levels only inside the spike band
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            if band[0] <= k/j <= band[1]:
                fam.append((j,float(k)))
    arr=torch.tensor(fam, dtype=torch.float64)
    return arr[:,0], arr[:,1]

def val(x, j, k):
    X=j[None,:]*x[:,None]-k[None,:]; return X*torch.exp(-X**2/2)
def d2(x, j, k):
    X=j[None,:]*x[:,None]-k[None,:]; return (j[None,:]**2)*X*(X**2-3)*torch.exp(-X**2/2)

def solve(coarse, fine=(), band=(0.40,0.60), Nc=1200, wb=10.0, tik=1e-12):
    j,k = family(coarse, fine, band); NF=j.numel()
    xc = torch.linspace(0,1,Nc)
    Op = -d2(xc,j,k) + KAPPA**2*val(xc,j,k)         # operator matrix (-u'' + kappa^2 u)
    n  = NF
    A_ = torch.zeros(Nc+2, n)
    A_[:Nc,:] = Op
    A_[Nc,:]  = math.sqrt(wb)*val(torch.zeros(1),j,k)      # BC at x=0
    A_[Nc+1,:]= math.sqrt(wb)*val(torch.ones(1),j,k)       # BC at x=1
    b = torch.zeros(Nc+2,1)
    b[:Nc,0] = torch.tensor(f_rhs(xc.numpy()))
    b[Nc,0]  = math.sqrt(wb)*u_exact(np.array([0.0]))[0]
    b[Nc+1,0]= math.sqrt(wb)*u_exact(np.array([1.0]))[0]
    s = A_.norm(dim=0).clamp_min(1e-30); An=A_/s
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb=torch.cat([b, torch.zeros(n,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    xt=torch.linspace(0,1,4000); pred=(val(xt,j,k)@c).numpy(); ue=u_exact(xt.numpy()); err=pred-ue
    return dict(NF=NF, relL2=np.linalg.norm(err)/np.linalg.norm(ue),
                MSE=np.mean(err**2), RMSE=math.sqrt(np.mean(err**2)), MAE=np.mean(np.abs(err)),
                Linf=np.max(np.abs(err)), relLinf=np.max(np.abs(err))/np.max(np.abs(ue)))

if __name__=="__main__":
    print("1D multiscale screened-Poisson  -u''+u = f,  u = sin(2 pi x) + exp(-((x-0.5)/0.02)^2)")
    print("AD-free wavelet operational-matrix W-PINN; the fine spike (w=0.02) is the hard part.\n")
    configs=[("coarse [0,1,2,3]",        (0,1,2,3),       ()),
             ("coarse [0,1,2,3,4,5]",    (0,1,2,3,4,5),   ()),
             ("[0-3]+band[6,7,8]@spike", (0,1,2,3),       (6,7,8))]
    hdr=f"{'basis':>26} {'NF':>4} | {'relL2':>10} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'Linf':>10} {'relLinf':>10}"
    print(hdr); print("-"*len(hdr))
    for name,co,fi in configs:
        m=solve(co, fi, band=(0.40,0.60))
        print(f"{name:>26} {m['NF']:>4d} | {m['relL2']:>10.2e} {m['MSE']:>10.2e} {m['RMSE']:>10.2e} "
              f"{m['MAE']:>10.2e} {m['Linf']:>10.2e} {m['relLinf']:>10.2e}", flush=True)
