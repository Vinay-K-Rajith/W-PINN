"""Phase 3 headline figure: interface-SHAPE recovery from sparse data + robustness frontier."""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import phase3_2d as P
torch.set_default_dtype(torch.float64); np.random.seed(1)

# ---- one flower shape recovery, capturing data points + recovered params ----
r0t,epst,rho,N,noise = 0.5,0.30,10.0,60,1e-3
uf=P.truth_flower(r0t,epst,rho); ub=uf(P.BX,P.BY)
xd,yd,ud,Pd=P.make_data(uf,N,noise)
from scipy.optimize import minimize
def J(p):
    r0,eps=p
    if r0<0.2 or r0>0.8 or abs(eps)>0.6: return 1e3
    phg=P.phi_flower(P.XS,P.YS,r0,eps); phid=P.phi_flower(xd,yd,r0,eps); ifc=P.iface_flower(r0,eps)
    cM,cP,bM,bP=P.forward(rho,phg,ifc,ub)
    pr=np.where(phid<0,P.predict_at(cM,bM,Pd),P.predict_at(cP,bP,Pd)); return np.sum((pr-ud)**2)
res=minimize(J,[0.4,0.1],method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-12,maxiter=400))
r0h,epsh=res.x
print(f"recovered (r0,eps)=({r0h:.4f},{epsh:.4f}) from N={len(xd)} pts, noise={noise}")

def curve(r0,eps,M=400):
    th=np.linspace(0,2*np.pi,M); rs=np.full(M,r0)
    for _ in range(80):
        c3=np.cos(3*th); gg=rs*rs-r0*r0+eps*rs**3*c3; gp=2*rs+3*eps*rs*rs*c3; rs=rs-gg/gp
    return rs*np.cos(th), rs*np.sin(th)
xt,yt=curve(r0t,epst); xh,yh=curve(r0h,epsh)

# robustness frontier (from phase3_2d expB_radius)
Ns=[10,20,50]; noises=[0.,1e-2,5e-2]
frontier={N:[P.expB_radius(N=N,noise=ns)[1] for ns in noises] for N in Ns}

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(xt,yt,'k-',lw=2.5,label='true interface')
ax[0].plot(xh,yh,'--',color='tab:green',lw=2,label='recovered')
ax[0].scatter(xd,yd,s=14,c='tab:red',alpha=.7,label=f'{len(xd)} sparse data pts')
ax[0].set_aspect('equal'); ax[0].set_xlim(-1,1); ax[0].set_ylim(-1,1)
ax[0].set_title(f'(a) interface SHAPE recovery\n true(0.50,0.30) → ({r0h:.3f},{epsh:.3f})'); ax[0].legend(loc='upper right',fontsize=8)
for N in Ns: ax[1].semilogy(noises,frontier[N],'o-',label=f'N={N} data pts')
ax[1].set_xlabel('measurement noise (relative)'); ax[1].set_ylabel('radius recovery rel. error')
ax[1].set_title('(b) robustness frontier'); ax[1].legend(); ax[1].grid(True,which='both',alpha=.3)
plt.tight_layout(); plt.savefig("phase3_summary.png",dpi=130); print("saved phase3_summary.png")
