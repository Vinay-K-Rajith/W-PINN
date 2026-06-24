"""
Phase 2 closeout (1/2): COMPLEX-GEOMETRY (flower) interface -> mesh-free demonstration.

Level-set construction (homogeneous jumps automatic, fully analytic):
    phi(x,y) = x^2+y^2 - r0^2 + eps*(x^3 - 3 x y^2)     (harmonic cubic => Laplacian = 4)
    interface Gamma = {phi=0}  (3-lobed flower),  inside = {phi<0}
    u = phi/a   in each region  =>  [[u]]=0 on Gamma, [[a d_n u]]=0 everywhere, f = -Delta phi = -4.
    normal n = grad(phi)/|grad(phi)|;  exact kink [[d_n u]] = |grad phi| (1/a_out - 1/a_in)  (VARIES along Gamma).

Same decomposed wavelet solver as the circle; only the geometry (point classification,
interface points, normals) changes -> shows the method is mesh-free (no remeshing).
"""
import math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run_phase2 import build_family, W_mats
torch.set_default_dtype(torch.float64)

r0, eps, a_minus = 0.5, 0.30, 1.0
def phi(x,y):  return x*x+y*y - r0*r0 + eps*(x**3 - 3*x*y*y)
def gradphi(x,y): return (2*x + 3*eps*(x*x - y*y), 2*y - 6*eps*x*y)

def make_problem(rho):
    a_in,a_out = a_minus, rho*a_minus
    def u_exact(x,y):
        p=phi(x,y); return np.where(p<0, p/a_in, p/a_out)
    def f(x,y): return -4.0*np.ones_like(x)
    return dict(a_in=a_in,a_out=a_out,u_exact=u_exact,f=f)

def interface_points(M=300):
    th=np.linspace(0,2*np.pi,M,endpoint=False); rs=np.full(M,r0)
    for _ in range(60):                                  # Newton: phi(r,theta)=0
        c3=np.cos(3*th); g=rs*rs-r0*r0+eps*rs**3*c3; gp=2*rs+3*eps*rs*rs*c3
        rs=rs-g/gp
    xg=rs*np.cos(th); yg=rs*np.sin(th)
    gx,gy=gradphi(xg,yg); nrm=np.sqrt(gx*gx+gy*gy)
    return xg,yg,gx/nrm,gy/nrm,nrm                        # nrm = |grad phi| on Gamma

def make_points(K=110,M=300):
    g=np.linspace(-1,1,K); Xg,Yg=np.meshgrid(g,g); xs=Xg.ravel(); ys=Yg.ravel()
    p=phi(xs,ys); inside=p<-1e-6; outside=(p>1e-6)&(np.abs(xs)<1)&(np.abs(ys)<1)
    xg,yg,nx,ny,nrm=interface_points(M)
    nb=np.linspace(-1,1,K)
    bx=np.concatenate([nb,nb,-np.ones(K),np.ones(K)]); by=np.concatenate([-np.ones(K),np.ones(K),nb,nb])
    return (xs[inside],ys[inside]),(xs[outside],ys[outside]),(xg,yg,nx,ny,nrm),(bx,by)

def solve(prob, levels=[0,1,2,3], wi=10., wb=10., tik=1e-10):
    jx,kx,jy,ky=build_family(levels); NF=jx.numel()
    (xm,ym),(xp,yp),(xg,yg,nx,ny,nrm),(bx,by)=make_points()
    Pm,Lm,_,_=W_mats(xm,ym,jx,kx,jy,ky); Pp,Lp,_,_=W_mats(xp,yp,jx,kx,jy,ky)
    Pg,_,dxg,dyg=W_mats(xg,yg,jx,kx,jy,ky)
    nxv=torch.tensor(nx); nyv=torch.tensor(ny); dnG=nxv[:,None]*dxg+nyv[:,None]*dyg
    Pb,_,_,_=W_mats(bx,by,jx,kx,jy,ky)
    n=2*NF+2; ai,ao=prob["a_in"],prob["a_out"]
    def blk(rows,cM=None,cP=None,bM=None,bP=None):
        B=torch.zeros(rows,n)
        if cM is not None:B[:,:NF]=cM
        if cP is not None:B[:,NF:2*NF]=cP
        if bM is not None:B[:,2*NF]=bM
        if bP is not None:B[:,2*NF+1]=bP
        return B
    A=torch.cat([blk(len(xm),cM=-ai*Lm),blk(len(xp),cP=-ao*Lp),
                 math.sqrt(wi)*blk(len(xg),cP=Pg,cM=-Pg,bP=1.,bM=-1.),
                 math.sqrt(wi)*blk(len(xg),cP=ao*dnG,cM=-ai*dnG),
                 math.sqrt(wb)*blk(len(bx),cP=Pb,bP=1.)],0)
    ub=torch.tensor(prob["u_exact"](bx,by))
    b=torch.cat([torch.tensor(prob["f"](xm,ym)),torch.tensor(prob["f"](xp,yp)),
                 torch.zeros(len(xg)),torch.zeros(len(xg)),math.sqrt(wb)*ub]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30); An=A/s
    Aa=torch.cat([An,math.sqrt(tik)*torch.eye(n)]); bb=torch.cat([b,torch.zeros(n,1)])
    theta=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    cM,cP=theta[:NF],theta[NF:2*NF]; bM,bP=theta[2*NF],theta[2*NF+1]
    # metrics
    gt=np.linspace(-0.999,0.999,240); Xt,Yt=np.meshgrid(gt,gt); xt=Xt.ravel(); yt=Yt.ravel()
    Pt,_,_,_=W_mats(xt,yt,jx,kx,jy,ky)
    pred=np.where(phi(xt,yt)<0,(Pt@cM+bM).numpy(),(Pt@cP+bP).numpy()); ue=prob["u_exact"](xt,yt)
    relL2=np.linalg.norm(pred-ue)/np.linalg.norm(ue)
    jump_pred=(dnG@cP).numpy()-(dnG@cM).numpy(); jump_exact=nrm*(1/ao-1/ai)
    jerr=np.max(np.abs(jump_pred-jump_exact))
    return dict(NF=NF,relL2=relL2,jerr=jerr,coeff=(cM,cP,bM,bP),fam=(jx,kx,jy,ky),
                pts=(xg,yg),test=(Xt,Yt,pred.reshape(Xt.shape),ue.reshape(Xt.shape)))

if __name__=="__main__":
    print("flower interface (eps=%.2f), mesh-free wavelet W-PINN" % eps)
    print(f"{'rho':>7} | {'relL2':>11} {'maxKinkErr':>11}")
    print("-"*38)
    R={}
    for rho in [10.,100.,1000.]:
        r=solve(make_problem(rho)); R[rho]=r
        print(f"{rho:>7.0f} | {r['relL2']:>11.3e} {r['jerr']:>11.2e}")
    # figure at rho=1000
    r=R[1000.]; Xt,Yt,PR,UE=r['test']; xg,yg=r['pts']
    fig,ax=plt.subplots(1,3,figsize=(14.5,4.4))
    c0=ax[0].contourf(Xt,Yt,UE,40,cmap='viridis'); ax[0].plot(xg,yg,'w.',ms=1)
    ax[0].set_title('(a) exact, flower $\\Gamma$ ($\\rho$=1000)'); plt.colorbar(c0,ax=ax[0])
    c1=ax[1].contourf(Xt,Yt,PR,40,cmap='viridis'); ax[1].plot(xg,yg,'w.',ms=1)
    ax[1].set_title('(b) wavelet W-PINN (mesh-free)'); plt.colorbar(c1,ax=ax[1])
    c2=ax[2].contourf(Xt,Yt,np.log10(np.abs(PR-UE)+1e-16),40,cmap='magma'); ax[2].plot(xg,yg,'w.',ms=1)
    ax[2].set_title('(c) log10 |error|'); plt.colorbar(c2,ax=ax[2])
    for a in ax: a.set_aspect('equal')
    plt.tight_layout(); plt.savefig("phase2_flower.png",dpi=120); print("saved phase2_flower.png")
