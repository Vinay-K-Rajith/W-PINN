"""Phase 2 analysis: (A) accuracy<->conditioning frontier vs basis richness;
(B) 2D summary figure (exact, predicted, error map, radial slice showing the kink)."""
import math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run_phase2 import make_problem, build_family, W_mats, make_points, solve, r0

torch.set_default_dtype(torch.float64)

# ---------- (A) frontier: richer level sets vs accuracy & conditioning ----------
print("== basis-richness frontier (rho=1000, precond+tik) ==")
print(f"{'levels':>14} | {'NF/side':>8} {'relL2':>11} {'cond_used':>10}")
print("-"*52)
prob=make_problem(1000.0)
import sys
for levels in [[0,1],[0,1,2],[0,1,2,3]]:
    r=solve(prob, levels, precond=True, tik=1e-10)
    print(f"{str(levels):>14} | {r['NF']:>8} {r['relL2']:>11.3e} {r['cond_pc']:>10.1e}"); sys.stdout.flush()

# ---------- (B) figure ----------
levels=[0,1,2,3]; rho=1000.0; prob=make_problem(rho)
jx,kx,jy,ky=build_family(levels); NF=jx.numel()
# re-solve to get coefficients (reuse solve internals via a thin recompute)
(xm,ym),(xp,yp),(xg,yg),(bx,by),(nx,ny)=make_points()
Pm,Lm,_,_=W_mats(xm,ym,jx,kx,jy,ky); Pp,Lp,_,_=W_mats(xp,yp,jx,kx,jy,ky)
Pgm,_,dxg,dyg=W_mats(xg,yg,jx,kx,jy,ky)
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
wi,wb=10.,10.
A=torch.cat([blk(len(xm),cM=-ai*Lm),blk(len(xp),cP=-ao*Lp),
             math.sqrt(wi)*blk(len(xg),cP=Pgm,cM=-Pgm,bP=1.,bM=-1.),
             math.sqrt(wi)*blk(len(xg),cP=ao*dnG,cM=-ai*dnG),
             math.sqrt(wb)*blk(len(bx),cP=Pb,bP=1.)],0)
ub=torch.tensor(prob["u_exact"](bx,by))
b=torch.cat([torch.tensor(prob["f"](xm,ym)),torch.tensor(prob["f"](xp,yp)),
             torch.zeros(len(xg)),torch.zeros(len(xg)),math.sqrt(wb)*ub]).unsqueeze(1)
s=A.norm(dim=0).clamp_min(1e-30); An=A/s
Aa=torch.cat([An,math.sqrt(1e-10)*torch.eye(n)]); bb=torch.cat([b,torch.zeros(n,1)])
theta=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
cM,cP=theta[:NF],theta[NF:2*NF]; bM,bP=theta[2*NF],theta[2*NF+1]

gt=np.linspace(-0.999,0.999,260); Xt,Yt=np.meshgrid(gt,gt); xt=Xt.ravel(); yt=Yt.ravel()
Pt,_,_,_=W_mats(xt,yt,jx,kx,jy,ky)
um=(Pt@cM+bM).numpy(); up=(Pt@cP+bP).numpy()
rr=xt*xt+yt*yt; pred=np.where(rr<r0*r0,um,up); ue=prob["u_exact"](xt,yt)
err=np.abs(pred-ue)
PR=pred.reshape(Xt.shape); UE=ue.reshape(Xt.shape); ER=err.reshape(Xt.shape)

# radial slice along y=0
xs=np.linspace(-0.999,0.999,800); ys=np.zeros_like(xs)
Ps,_,_,_=W_mats(xs,ys,jx,kx,jy,ky)
us=np.where(xs*xs<r0*r0,(Ps@cM+bM).numpy(),(Ps@cP+bP).numpy())
ues=prob["u_exact"](xs,ys)

fig,ax=plt.subplots(1,4,figsize=(19,4.3))
th=np.linspace(0,2*np.pi,200)
c0=ax[0].contourf(Xt,Yt,UE,40,cmap='viridis'); ax[0].plot(r0*np.cos(th),r0*np.sin(th),'w--',lw=1)
ax[0].set_title(f'(a) exact  ($\\rho$={int(rho)})'); plt.colorbar(c0,ax=ax[0])
c1=ax[1].contourf(Xt,Yt,PR,40,cmap='viridis'); ax[1].plot(r0*np.cos(th),r0*np.sin(th),'w--',lw=1)
ax[1].set_title('(b) wavelet W-PINN'); plt.colorbar(c1,ax=ax[1])
c2=ax[2].contourf(Xt,Yt,np.log10(ER+1e-16),40,cmap='magma'); ax[2].plot(r0*np.cos(th),r0*np.sin(th),'w--',lw=1)
ax[2].set_title('(c) log10 |error|'); plt.colorbar(c2,ax=ax[2])
ax[3].plot(xs,ues,'k',lw=2,label='exact'); ax[3].plot(xs,us,'--',color='tab:green',label='wavelet')
ax[3].axvline(r0,color='gray',ls=':'); ax[3].axvline(-r0,color='gray',ls=':')
ax[3].set_title('(d) slice y=0 (kinks at $\\pm r_0$)'); ax[3].set_xlabel('x'); ax[3].legend()
for a in ax[:3]: a.set_aspect('equal')
plt.tight_layout(); plt.savefig("phase2_summary.png",dpi=120)
print("\nsaved phase2_summary.png   relL2=%.3e"%(np.linalg.norm(pred-ue)/np.linalg.norm(ue)))
