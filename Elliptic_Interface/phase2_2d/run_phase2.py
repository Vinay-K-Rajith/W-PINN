"""
Phase 2 - 2D elliptic interface problem (circular interface), Wavelet-PINN.

Domain Omega = (-1,1)^2.  Interface Gamma = { x^2+y^2 = r0^2 }.
    -div(a grad u) = f,   a = a_minus (inside disk), a_plus (outside).
Manufactured exact solution (homogeneous jumps [[u]]=0, [[a d_n u]]=0):
    u(x,y) = (x^2+y^2)/a_in  inside ;   (x^2+y^2)/a_out + C  outside,
    C = r0^2 (1/a_in - 1/a_out)  =>  f = -4 everywhere (constant), genuine flux-kink at r0.
    exact normal-deriv jump at r0:  [[d_n u]] = 2 r0 (1/a_out - 1/a_in).

Method: two tensor-product Gaussian-derivative wavelet expansions (one per subdomain),
coupled by interface conditions; solved AD-free by weighted least squares.
First-principles conditioning fixes: (1) parsimonious coarse levels, (2) column
normalization (Jacobi precond), (3) mild Tikhonov.  We report cond before/after.
"""
import time, math, numpy as np, torch
torch.set_default_dtype(torch.float64)
PI = math.pi

# ---------------------------------------------------------------- problem
r0 = 0.5
a_minus = 1.0
def make_problem(rho):
    a_in, a_out = a_minus, rho*a_minus
    C = r0**2 * (1.0/a_in - 1.0/a_out)
    def u_exact(x,y):
        r2 = x*x + y*y
        return np.where(r2 < r0*r0, r2/a_in, r2/a_out + C)
    def f(x,y): return -4.0*np.ones_like(x)
    jump_dn = 2*r0*(1.0/a_out - 1.0/a_in)        # exact [[d_n u]] at r0
    return dict(a_in=a_in, a_out=a_out, C=C, u_exact=u_exact, f=f, jump_dn=jump_dn)

# ---------------------------------------------------------------- wavelet family (2D tensor)
def build_1d(levels, lo=-1.0, hi=1.0, pad=0.5):
    out=[]
    for l in levels:
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            out.append((j,float(k)))
    return out

def build_family(levels):
    f1 = build_1d(levels)
    fam = [(jx,kx,jy,ky) for (jx,kx) in f1 for (jy,ky) in f1]
    arr = torch.tensor(fam, dtype=torch.float64)
    return arr[:,0],arr[:,1],arr[:,2],arr[:,3]

def W_mats(x, y, jx,kx,jy,ky):
    x=torch.as_tensor(x,dtype=torch.float64).reshape(-1)
    y=torch.as_tensor(y,dtype=torch.float64).reshape(-1)
    X=jx[None,:]*x[:,None]-kx[None,:]
    Y=jy[None,:]*y[:,None]-ky[None,:]
    E=torch.exp(-(X**2+Y**2)/2)
    Psi = X*Y*E
    lap = -(jx[None,:]**2)*X*Y*(3-X**2)*E - (jy[None,:]**2)*X*Y*(3-Y**2)*E
    dx  = jx[None,:]*(1-X**2)*Y*E
    dy  = jy[None,:]*(1-Y**2)*X*E
    return Psi, lap, dx, dy

# ---------------------------------------------------------------- collocation
def make_points(K=90, M=240, seed=0):
    g=np.linspace(-1,1,K)
    Xg,Yg=np.meshgrid(g,g); xs=Xg.ravel(); ys=Yg.ravel()
    r=np.sqrt(xs**2+ys**2)
    inside = r < r0-1e-9; outside = (r>r0+1e-9)&(np.abs(xs)<1)&(np.abs(ys)<1)
    th=np.linspace(0,2*PI,M,endpoint=False)
    xg=r0*np.cos(th); yg=r0*np.sin(th)                 # interface points
    nb=np.linspace(-1,1,K)
    bx=np.concatenate([nb,nb,-np.ones(K),np.ones(K)])
    by=np.concatenate([-np.ones(K),np.ones(K),nb,nb])  # square boundary (all in Omega+)
    return (xs[inside],ys[inside]),(xs[outside],ys[outside]),(xg,yg),(bx,by),(np.cos(th),np.sin(th))

# ---------------------------------------------------------------- solver
def solve(prob, levels, w_iface=10.0, w_bc=10.0, tik=1e-8, precond=True):
    jx,kx,jy,ky = build_family(levels); NF=jx.numel()
    (xm,ym),(xp,yp),(xg,yg),(bx,by),(nx,ny) = make_points()
    Pm,Lm,_,_ = W_mats(xm,ym,jx,kx,jy,ky)
    Pp,Lp,_,_ = W_mats(xp,yp,jx,kx,jy,ky)
    Pgm,_,dxg,dyg = W_mats(xg,yg,jx,kx,jy,ky)          # interface (same family, both sides)
    nxv=torch.tensor(nx); nyv=torch.tensor(ny)
    dnG = nxv[:,None]*dxg + nyv[:,None]*dyg            # normal-derivative matrix on Gamma
    Pb,_,_,_ = W_mats(bx,by,jx,kx,jy,ky)

    n = 2*NF+2
    def block(rows, cM=None,cP=None,bM=None,bP=None):
        B=torch.zeros(rows,n)
        if cM is not None: B[:, :NF]=cM
        if cP is not None: B[:, NF:2*NF]=cP
        if bM is not None: B[:, 2*NF]=bM
        if bP is not None: B[:, 2*NF+1]=bP
        return B
    ai,ao=prob["a_in"],prob["a_out"]
    rows=[]; rhs=[]
    rows.append(block(len(xm), cM=-ai*Lm)); rhs.append(torch.tensor(prob["f"](xm,ym)))
    rows.append(block(len(xp), cP=-ao*Lp)); rhs.append(torch.tensor(prob["f"](xp,yp)))
    rows.append(math.sqrt(w_iface)*block(len(xg), cP=Pgm,cM=-Pgm,bP=1.0,bM=-1.0)); rhs.append(torch.zeros(len(xg)))
    rows.append(math.sqrt(w_iface)*block(len(xg), cP=ao*dnG,cM=-ai*dnG));           rhs.append(torch.zeros(len(xg)))
    ub=torch.tensor(prob["u_exact"](bx,by))
    rows.append(math.sqrt(w_bc)*block(len(bx), cP=Pb,bP=1.0)); rhs.append(math.sqrt(w_bc)*ub)
    A=torch.cat(rows,0); b=torch.cat(rhs,0).unsqueeze(1)

    cond_raw=float(torch.linalg.cond(A).item())
    if precond:
        s=A.norm(dim=0).clamp_min(1e-30); An=A/s
    else:
        s=torch.ones(n); An=A
    cond_pc=float(torch.linalg.cond(An).item())
    # Tikhonov on normalized system
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(n)],0)
    bb=torch.cat([b, torch.zeros(n,1)],0)
    t0=time.time()
    y=torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1)
    dt=time.time()-t0
    theta=y/s
    cM,cP=theta[:NF],theta[NF:2*NF]; bM,bP=theta[2*NF],theta[2*NF+1]

    # evaluate on test grid
    gt=np.linspace(-0.999,0.999,220); Xt,Yt=np.meshgrid(gt,gt); xt=Xt.ravel(); yt=Yt.ravel()
    Pt,_,_,_=W_mats(xt,yt,jx,kx,jy,ky)
    um=(Pt@cM+bM).numpy(); up=(Pt@cP+bP).numpy()
    rr=xt*xt+yt*yt; pred=np.where(rr<r0*r0,um,up)
    ue=prob["u_exact"](xt,yt)
    relL2=np.linalg.norm(pred-ue)/np.linalg.norm(ue); emax=np.max(np.abs(pred-ue))
    # predicted normal-deriv jump on Gamma (mean)
    dnm=(dnG@cM).numpy(); dnp=(dnG@cP).numpy()
    jump_pred=float(np.mean(dnp-dnm)); jerr=abs(jump_pred-prob["jump_dn"])
    return dict(NF=NF,cond_raw=cond_raw,cond_pc=cond_pc,relL2=relL2,emax=emax,
                jump_pred=jump_pred,jerr=jerr,dt=dt)

if __name__=="__main__":
    levels=[0,1,2]
    print(f"levels={levels}")
    prob=make_problem(10.0)
    print("\n-- conditioning fix (rho=10) --")
    for pc in [False, True]:
        r=solve(prob, levels, precond=pc, tik=0.0 if not pc else 1e-10)
        tag="precond+tik" if pc else "raw"
        print(f"  {tag:>12}: NF/side={r['NF']}  cond_raw={r['cond_raw']:.1e}  cond_used={r['cond_pc']:.1e}  relL2={r['relL2']:.3e}  jumpErr={r['jerr']:.2e}")

    print("\n-- contrast sweep (precond+tik, levels=[0,1,2]) --")
    print(f"{'rho':>7} | {'relL2':>10} {'maxErr':>10} {'jump_pred':>10} {'jump_err':>10} {'cond_used':>10} {'t(s)':>6}")
    print("-"*78)
    for rho in [10.,100.,1000.,10000.]:
        p=make_problem(rho); r=solve(p, levels, precond=True, tik=1e-10)
        print(f"{rho:>7.0f} | {r['relL2']:>10.3e} {r['emax']:>10.3e} {r['jump_pred']:>10.4f} {r['jerr']:>10.2e} {r['cond_pc']:>10.1e} {r['dt']:>6.2f}")
