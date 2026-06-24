"""Phase 1 summary figure: kink resolution, error profiles, and contrast robustness."""
import math, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run_phase1 import build_family, W_mats, make_problem, beta, a_minus
from phase1_sweep import lstsq_decomposed, x_test, jv, kv, NF

torch.set_default_dtype(torch.float64)

# (a)+(b): rho=1000 solution + error profiles (decomposed-wavelet vs single-expansion)
rho=1000.0; prob=make_problem(rho); ue=prob["u_exact"](x_test)
pred_dec,_,_,_ = lstsq_decomposed(prob)

# single-expansion least squares (one global smooth expansion, residual+BC only)
xm=np.linspace(0,beta,602)[1:-1]; xp=np.linspace(beta,1,602)[1:-1]
xa=np.sort(np.concatenate([xm,xp]))
P,_,D2 = W_mats(xa,jv,kv); P0,_,_=W_mats([0.],jv,kv); P1,_,_=W_mats([1.],jv,kv)
av=torch.tensor(np.where(xa<beta,a_minus,prob["a_plus"]))
n=NF+1; A=torch.zeros(len(xa)+2,n)
A[:len(xa),:NF]=-(av[:,None]*D2);
A[len(xa),:NF]=P0; A[len(xa),NF]=1; A[len(xa)+1,:NF]=P1; A[len(xa)+1,NF]=1
b=torch.zeros(len(xa)+2); b[:len(xa)]=torch.tensor(prob["f"](xa)); b[len(xa)]=prob["u0"]; b[len(xa)+1]=prob["u1"]
th=torch.linalg.lstsq(A,b.unsqueeze(1),driver="gelsd").solution.squeeze(1)
pred_sing=(W_mats(x_test,jv,kv)[0]@th[:NF]+th[NF]).numpy()

# (c): contrast curves (recompute quickly)
rhos=[10,100,1000,10000]; werr=[];
for r in rhos:
    pr=make_problem(float(r)); pd,_,_,_=lstsq_decomposed(pr)
    werr.append(np.linalg.norm(pd-pr["u_exact"](x_test))/np.linalg.norm(pr["u_exact"](x_test)))
xpinn=[1.198e-4,2.206e-4,6.330e-4,8.875e-3]   # from sweep run

fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot(x_test,ue,'k',lw=2,label='exact'); ax[0].plot(x_test,pred_dec,'--',color='tab:green',label='decomposed wavelet')
ax[0].plot(x_test,pred_sing,':',color='tab:red',lw=2,label='single expansion')
ax[0].axvline(beta,color='gray',ls=':',alpha=.6); ax[0].set_title(f'(a) solution, $\\rho$=1000 (kink at x={beta})'); ax[0].legend(); ax[0].set_xlabel('x')
ax[1].semilogy(x_test,np.abs(pred_dec-ue)+1e-16,color='tab:green',label='decomposed wavelet')
ax[1].semilogy(x_test,np.abs(pred_sing-ue)+1e-16,color='tab:red',label='single expansion')
ax[1].axvline(beta,color='gray',ls=':',alpha=.6); ax[1].set_title('(b) pointwise |error|'); ax[1].legend(); ax[1].set_xlabel('x')
ax[2].loglog(rhos,werr,'o-',color='tab:green',label='decomposed wavelet (LSQ, AD-free)')
ax[2].loglog(rhos,xpinn,'s-',color='tab:blue',label='decomposed MLP / XPINN')
ax[2].set_title('(c) rel. $L^2$ vs contrast $\\rho$'); ax[2].set_xlabel('contrast $a^+/a^-$'); ax[2].legend(); ax[2].grid(True,which='both',alpha=.3)
plt.tight_layout(); plt.savefig("phase1_summary.png",dpi=130)
print("saved phase1_summary.png")
print("wavelet relL2 vs rho:", [f"{e:.2e}" for e in werr])
