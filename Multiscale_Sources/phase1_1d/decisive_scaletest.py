r"""
DECISIVE test: does adaptive MULTIRESOLUTION (banded wavelet W-PINN) beat fixed-width random
features (RBF-PIELM) as scales SEPARATE?  This settles whether paper #2's premise holds.

1D screened Poisson  -u'' + u = f  on [0,1], u = sin(2 pi x) + exp(-((x-0.5)/w)^2),  sweep w.
Both methods get the SAME number of basis functions (matched NF). RBF-PIELM is given the BEST of
several widths (best case for the competitor). Prediction: at large w both fine; as w -> small,
RBF-PIELM (single fixed width + random centres) cannot resolve the spike AND the background at a
fixed budget, while the banded wavelet adds finer levels only at the spike and holds.
"""
import math, numpy as np, torch
torch.set_default_dtype(torch.float64)
np.random.seed(0); torch.manual_seed(0)

def make(w):
    def u(x): return np.sin(2*np.pi*x)+np.exp(-((x-0.5)/w)**2)
    def f(x):
        upp_s=-(2*np.pi)**2*np.sin(2*np.pi*x); g=np.exp(-((x-0.5)/w)**2)
        upp_b=g*(4*(x-0.5)**2/w**4-2.0/w**2)
        return -(upp_s+upp_b)+u(x)
    return u,f

# ---- banded wavelet W-PINN ----
def wfam(coarse, fine, w, lo=0.,hi=1.,pad=0.5, rad=None):
    fam=[]
    for l in coarse:
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)),int(math.ceil((hi+pad)*j))+1): fam.append((j,float(k)))
    R=rad if rad else max(4*w,0.05)
    for l in fine:
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)),int(math.ceil((hi+pad)*j))+1):
            if abs(k/j-0.5)<=R: fam.append((j,float(k)))
    a=torch.tensor(fam); return a[:,0],a[:,1]
def wval(x,j,k): X=j[None,:]*x[:,None]-k[None,:]; return X*torch.exp(-X**2/2)
def wd2(x,j,k):  X=j[None,:]*x[:,None]-k[None,:]; return (j[None,:]**2)*X*(X**2-3)*torch.exp(-X**2/2)

def solve_wavelet(w, coarse=(0,1,2,3), fine=(4,5,6,7), wb=10., tik=1e-11, Nc=3000):
    u,f=make(w); j,k=wfam(coarse,fine,w); NF=j.numel()
    # local collocation refinement near the spike
    xu=np.linspace(0,1,Nc); R=max(4*w,0.05); h=max(w/6,3e-4)
    xl=np.arange(0.5-R,0.5+R+h,h); xl=xl[(xl>0)&(xl<1)]
    xc=np.unique(np.concatenate([xu,xl]));
    # area weights (1D): spacing/2 to neighbours
    xs=np.sort(xc); aw=np.zeros_like(xs); aw[1:-1]=(xs[2:]-xs[:-2])/2; aw[0]=(xs[1]-xs[0])/2; aw[-1]=(xs[-1]-xs[-2])/2
    xt=torch.tensor(xs); sw=torch.tensor(np.sqrt(aw))[:,None]
    Op=(-wd2(xt,j,k)+wval(xt,j,k))*sw
    A=torch.cat([Op, math.sqrt(wb)*wval(torch.tensor([0.,1.]),j,k)],0)
    b=torch.cat([torch.tensor(np.sqrt(aw)*f(xs)), math.sqrt(wb)*torch.tensor(u(np.array([0.,1.])))]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30); Aa=torch.cat([A/s,math.sqrt(tik)*torch.eye(NF)]); bb=torch.cat([b,torch.zeros(NF,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    xe=torch.linspace(0,1,8000); pred=(wval(xe,j,k)@c).numpy(); ue=u(xe.numpy())
    return NF, np.linalg.norm(pred-ue)/np.linalg.norm(ue)

# ---- RBF-PIELM (best of several widths) ----
def solve_rbf(w, M, betas=(20,50,100,300,1000,3000), wb=10., tik=1e-11, Nc=4000):
    u,f=make(w); xs=np.linspace(0,1,Nc)
    aw=np.full(Nc,1.0/Nc); xt=torch.tensor(xs); sw=torch.tensor(np.sqrt(aw))[:,None]
    best=1e9
    for beta in betas:
        C=torch.rand(M,dtype=torch.float64)
        def mats(x):
            x=torch.as_tensor(x).reshape(-1); r2=(x[:,None]-C[None,:])**2; phi=torch.exp(-beta*r2)
            return phi, (4*beta**2*r2-2*beta)*phi
        phi,phidd=mats(xs)
        Op=(-phidd+phi)*sw
        A=torch.cat([Op, math.sqrt(wb)*mats(np.array([0.,1.]))[0]],0)
        b=torch.cat([torch.tensor(np.sqrt(aw)*f(xs)), math.sqrt(wb)*torch.tensor(u(np.array([0.,1.])))]).unsqueeze(1)
        s=A.norm(dim=0).clamp_min(1e-30); Aa=torch.cat([A/s,math.sqrt(tik)*torch.eye(M)]); bb=torch.cat([b,torch.zeros(M,1)])
        c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
        xe=torch.linspace(0,1,8000); pred=(mats(xe.numpy())[0]@c).numpy(); ue=u(xe.numpy())
        best=min(best, np.linalg.norm(pred-ue)/np.linalg.norm(ue))
    return best

if __name__=="__main__":
    import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    print("DECISIVE: banded wavelet W-PINN vs RBF-PIELM (best width, matched NF) as spike width w -> 0\n")
    print(f"{'w':>7} | {'NF':>4} | {'wavelet relL2':>14} | {'RBF-PIELM relL2':>16} | winner")
    print("-"*64)
    ws=[0.20,0.10,0.05,0.02,0.01,0.005]; rows=[]
    for w in ws:
        NF,ew=solve_wavelet(w); er=solve_rbf(w,NF)
        rows.append(dict(w=w,NF=int(NF),wavelet=float(ew),rbf=float(er),ratio=float(er/ew)))
        print(f"{w:>7.3f} | {NF:>4d} | {ew:>14.3e} | {er:>16.3e} | {'wavelet' if ew<er else 'RBF-PIELM'}   ({er/ew:.1f}x)", flush=True)
    json.dump(rows, open("decisive_results.json","w"), indent=2); print("\nsaved decisive_results.json")

    # ---- diagram: relL2 vs spike width (log-log) ----
    W=[r["w"] for r in rows]; EW=[r["wavelet"] for r in rows]; ER=[r["rbf"] for r in rows]
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.loglog(W,EW,'o-',color='#1f77b4',lw=2.2,ms=8,label='banded wavelet W-PINN')
    ax.loglog(W,ER,'s--',color='#d62728',lw=2.2,ms=8,label='RBF-PIELM (best width, matched $N_F$)')
    ax.invert_xaxis()
    ax.set_xlabel('spike width $w$  (scale separation increases $\\rightarrow$)',fontsize=12)
    ax.set_ylabel('relative $L_2$ error',fontsize=12)
    ax.set_title('1D multiscale screened-Poisson: adaptive multiresolution vs. fixed-width features',fontsize=12)
    ax.grid(True,which='both',ls=':',alpha=0.5); ax.legend(fontsize=11,loc='lower left')
    for r in rows:
        ax.annotate(f"{r['ratio']:.0f}×", (r['w'], (r['wavelet']*r['rbf'])**0.5),
                    fontsize=8.5, ha='center', color='#555')
    plt.tight_layout(); plt.savefig("decisive.png",dpi=200); print("saved decisive.png")
