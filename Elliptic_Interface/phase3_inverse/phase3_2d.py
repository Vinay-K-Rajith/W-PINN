"""
Phase 3 execution - 2D inverse elliptic-interface problems with W-PINN.
Strategy (validated in 1D): OUTER-loop over physical params + INNER AD-free least-squares forward solve.
Level-set manufactured truth: u = phi/a per region, f = -Delta phi.
  circle:  phi = x^2+y^2 - R^2                          (f=-4)
  flower:  phi = x^2+y^2 - r0^2 + eps(x^3-3xy^2)         (f=-4, harmonic cubic)
Experiments:
  (A) inverse CONTRAST  : geometry known, recover rho=a^+/a^-
  (B) inverse LOCATION  : contrast known, recover circle radius R
  (C) inverse SHAPE     : recover flower (r0, eps) jointly
  (D) robustness frontier: radius recovery vs #data and noise
Efficiency: wavelet matrices at the fixed grid precomputed once; only point-classification +
interface matrices change per candidate geometry.
"""
import sys, os, math, time, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase2_2d"))
from scipy.optimize import minimize
from run_phase2 import build_family, W_mats
torch.set_default_dtype(torch.float64); np.random.seed(0)

a_minus = 1.0
levels = [0,1,2]
jx,kx,jy,ky = build_family(levels); NF = jx.numel()

# fixed collocation
K=90; g=np.linspace(-1,1,K); Xg,Yg=np.meshgrid(g,g); XS=Xg.ravel(); YS=Yg.ravel()
nb=np.linspace(-1,1,K)
BX=np.concatenate([nb,nb,-np.ones(K),np.ones(K)]); BY=np.concatenate([-np.ones(K),np.ones(K),nb,nb])
# precompute wavelet matrices ONCE (geometry-independent points)
P_all,L_all,_,_ = W_mats(XS,YS,jx,kx,jy,ky)
P_b,_,_,_ = W_mats(BX,BY,jx,kx,jy,ky)
insq = (np.abs(XS)<1)&(np.abs(YS)<1)

# ---------------- geometry helpers ----------------
def phi_circle(x,y,R): return x*x+y*y-R*R
def grad_circle(x,y,R): return 2*x, 2*y
def iface_circle(R,M=240):
    th=np.linspace(0,2*np.pi,M,endpoint=False); xg=R*np.cos(th); yg=R*np.sin(th)
    return xg,yg,np.cos(th),np.sin(th)

def phi_flower(x,y,r0,eps): return x*x+y*y-r0*r0+eps*(x**3-3*x*y*y)
def grad_flower(x,y,r0,eps): return 2*x+3*eps*(x*x-y*y), 2*y-6*eps*x*y
def iface_flower(r0,eps,M=240):
    th=np.linspace(0,2*np.pi,M,endpoint=False); rs=np.full(M,r0)
    for _ in range(60):
        c3=np.cos(3*th); gg=rs*rs-r0*r0+eps*rs**3*c3; gp=2*rs+3*eps*rs*rs*c3; rs=rs-gg/gp
    xg=rs*np.cos(th); yg=rs*np.sin(th); gx,gy=grad_flower(xg,yg,r0,eps); nrm=np.hypot(gx,gy)
    return xg,yg,gx/nrm,gy/nrm

# ---------------- inner forward solve (AD-free) ----------------
def forward(rho, phi_grid, iface, ub, fval=-4.0, wi=10.,wb=10.,tik=1e-10):
    xg,yg,nx,ny = iface
    inside = phi_grid<-1e-9; outside=(phi_grid>1e-9)&insq
    Lm=L_all[inside]; Lp=L_all[outside]
    Pg,_,dxg,dyg = W_mats(xg,yg,jx,kx,jy,ky); dnG=torch.tensor(nx)[:,None]*dxg+torch.tensor(ny)[:,None]*dyg
    n=2*NF+2
    def blk(rows,cM=None,cP=None,bM=None,bP=None):
        B=torch.zeros(rows,n)
        if cM is not None:B[:,:NF]=cM
        if cP is not None:B[:,NF:2*NF]=cP
        if bM is not None:B[:,2*NF]=bM
        if bP is not None:B[:,2*NF+1]=bP
        return B
    nin,nout,nif,nbc=inside.sum(),outside.sum(),len(xg),len(ub)
    A=torch.cat([blk(nin,cM=-a_minus*Lm), blk(nout,cP=-rho*Lp),
                 math.sqrt(wi)*blk(nif,cP=Pg,cM=-Pg,bP=1.,bM=-1.),
                 math.sqrt(wi)*blk(nif,cP=rho*dnG,cM=-a_minus*dnG),
                 math.sqrt(wb)*blk(nbc,cP=P_b,bP=1.)],0)
    b=torch.cat([torch.full((nin,),fval),torch.full((nout,),fval),torch.zeros(nif),torch.zeros(nif),
                 math.sqrt(wb)*torch.tensor(ub)]).unsqueeze(1)
    s=A.norm(dim=0).clamp_min(1e-30)
    th=(torch.linalg.lstsq(torch.cat([A/s,math.sqrt(tik)*torch.eye(n)]),
                           torch.cat([b,torch.zeros(n,1)]),driver="gelsd").solution.squeeze(1))/s
    return th[:NF],th[NF:2*NF],th[2*NF],th[2*NF+1]

def predict_at(c,b,P): return (P@c+b).numpy()

# ---------------- truth / data ----------------
def truth_circle(R,rho):
    a_out=rho*a_minus
    def u(x,y): p=phi_circle(x,y,R); return np.where(p<0,p/a_minus,p/a_out)
    return u
def truth_flower(r0,eps,rho):
    a_out=rho*a_minus
    def u(x,y): p=phi_flower(x,y,r0,eps); return np.where(p<0,p/a_minus,p/a_out)
    return u

def make_data(uf, N, noise):
    xd=np.random.rand(N)*2-1; yd=np.random.rand(N)*2-1
    keep=(np.abs(xd)<0.97)&(np.abs(yd)<0.97); xd,yd=xd[keep],yd[keep]
    ud=uf(xd,yd)*(1+noise*np.random.randn(len(xd)))
    Pd,_,_,_=W_mats(xd,yd,jx,kx,jy,ky)
    return xd,yd,ud,Pd

# ================= experiments =================
def expA_contrast(rho_true=8.0,R=0.5,N=25,noise=1e-3):
    uf=truth_circle(R,rho_true); ub=uf(BX,BY)
    xd,yd,ud,Pd=make_data(uf,N,noise); phid=phi_circle(xd,yd,R); phg=phi_circle(XS,YS,R); ifc=iface_circle(R)
    def J(rho):
        cM,cP,bM,bP=forward(rho,phg,ifc,ub)
        pr=np.where(phid<0,predict_at(cM,bM,Pd),predict_at(cP,bP,Pd)); return np.sum((pr-ud)**2)
    a,b=1.2,40.; gr=(math.sqrt(5)-1)/2; c,d=b-gr*(b-a),a+gr*(b-a); fc,fd=J(c),J(d)
    for _ in range(40):
        if fc<fd: b,d,fd=d,c,fc; c=b-gr*(b-a); fc=J(c)
        else: a,c,fc=c,d,fd; d=a+gr*(b-a); fd=J(d)
    rh=(a+b)/2; return rh, abs(rh-rho_true)/rho_true

def expB_radius(R_true=0.5,rho=10.0,N=25,noise=1e-3):
    uf=truth_circle(R_true,rho); ub=uf(BX,BY)
    xd,yd,ud,Pd=make_data(uf,N,noise)
    def J(R):
        phg=phi_circle(XS,YS,R); phid=phi_circle(xd,yd,R); ifc=iface_circle(R)
        cM,cP,bM,bP=forward(rho,phg,ifc,ub)
        pr=np.where(phid<0,predict_at(cM,bM,Pd),predict_at(cP,bP,Pd)); return np.sum((pr-ud)**2)
    a,b=0.2,0.85; gr=(math.sqrt(5)-1)/2; c,d=b-gr*(b-a),a+gr*(b-a); fc,fd=J(c),J(d)
    for _ in range(40):
        if fc<fd: b,d,fd=d,c,fc; c=b-gr*(b-a); fc=J(c)
        else: a,c,fc=c,d,fd; d=a+gr*(b-a); fd=J(d)
    Rh=(a+b)/2; return Rh, abs(Rh-R_true)/R_true

def expC_shape(r0_true=0.5,eps_true=0.30,rho=10.0,N=60,noise=1e-3):
    uf=truth_flower(r0_true,eps_true,rho); ub=uf(BX,BY)
    xd,yd,ud,Pd=make_data(uf,N,noise)
    def J(p):
        r0,eps=p
        if r0<0.2 or r0>0.8 or abs(eps)>0.6: return 1e3
        phg=phi_flower(XS,YS,r0,eps); phid=phi_flower(xd,yd,r0,eps); ifc=iface_flower(r0,eps)
        cM,cP,bM,bP=forward(rho,phg,ifc,ub)
        pr=np.where(phid<0,predict_at(cM,bM,Pd),predict_at(cP,bP,Pd)); return np.sum((pr-ud)**2)
    res=minimize(J,[0.4,0.1],method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-12,maxiter=400))
    r0h,epsh=res.x; return (r0h,epsh),(abs(r0h-r0_true)/r0_true,abs(epsh-eps_true)/abs(eps_true))

if __name__=="__main__":
    t0=time.time()
    print("=== (A) inverse CONTRAST (circle, R=0.5 known) ===")
    print(f"{'rho_true':>9} {'noise':>7} | {'rho_hat':>9} {'rel_err':>9}")
    for rt in [4.,8.,20.]:
        for ns in [0.,1e-2]:
            rh,re=expA_contrast(rho_true=rt,noise=ns); print(f"{rt:>9.1f} {ns:>7.0e} | {rh:>9.4f} {re:>9.2e}")

    print("\n=== (B) inverse LOCATION (circle radius, rho=10 known) ===")
    print(f"{'R_true':>7} {'noise':>7} | {'R_hat':>8} {'rel_err':>9}")
    for ns in [0.,1e-2]:
        Rh,re=expB_radius(noise=ns); print(f"{0.5:>7.2f} {ns:>7.0e} | {Rh:>8.4f} {re:>9.2e}")

    print("\n=== (C) inverse SHAPE (flower r0,eps jointly, rho=10) ===")
    (r0h,epsh),(e0,ee)=expC_shape(noise=1e-3)
    print(f"true (r0,eps)=(0.50,0.30) -> hat=({r0h:.4f},{epsh:.4f})  relerr r0={e0:.2e} eps={ee:.2e}")

    print("\n=== (D) radius-recovery robustness frontier ===")
    print(f"{'N':>4} {'noise':>7} | {'R_relerr':>9}")
    for N in [10,20,50]:
        for ns in [0.,1e-2,5e-2]:
            _,re=expB_radius(N=N,noise=ns); print(f"{N:>4} {ns:>7.0e} | {re:>9.2e}")
    print(f"\ntotal {time.time()-t0:.0f}s")
