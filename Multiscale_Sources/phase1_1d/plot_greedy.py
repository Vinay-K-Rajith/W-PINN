r"""Figure 1: automatic residual-greedy refinement localizes the spike and beats the oracle."""
import numpy as np, torch, matplotlib.pyplot as plt
import greedy_residual as G
torch.set_default_dtype(torch.float64)

g = G.greedy(verbose=False)
jD,kD,lvl,sel = g['jD'],g['kD'],g['lvl'],g['sel']
c = g['c']

# panel data
xt=torch.linspace(0,1,4000); pred=(G.val(xt,jD[sel],kD[sel])@c).numpy(); ue=G.u_exact(xt.numpy())
centers=np.array([float(kD[i]/jD[i]) for i in sel]); levels=np.array([lvl[i] for i in sel])
H=g['history']; NF=[h['NF'] for h in H]; rl2=[h['relL2'] for h in H]; res=[h['res'] for h in H]

fig,ax=plt.subplots(1,3,figsize=(15,4.2))

# (a) solution
ax[0].plot(xt.numpy(),ue,'k-',lw=2,label='exact')
ax[0].plot(xt.numpy(),pred,'r--',lw=1.2,label='greedy (relL2=%.1e)'%g['final']['relL2'])
ax[0].axvline(G.X0,color='b',ls=':',alpha=.5); ax[0].set_title('(a) solution: smooth + w=0.02 spike')
ax[0].set_xlabel('x'); ax[0].legend(fontsize=9)

# (b) WHERE it refined: selected wavelets, center vs level (finest cluster at x0)
sc=ax[1].scatter(centers,levels,c=levels,cmap='viridis',s=18)
ax[1].axvline(G.X0,color='r',ls='--',label='true spike x0=%.2f'%G.X0)
ax[1].set_xlabel('wavelet center  k/j'); ax[1].set_ylabel('level  l')
ax[1].set_title('(b) discovered refinement (NOT given x0)'); ax[1].legend(fontsize=9)
plt.colorbar(sc,ax=ax[1],label='level')

# (c) convergence vs oracle / uniform / coarse
ax[2].semilogy(NF,rl2,'o-',label='greedy relL2')
ax[2].semilogy(NF,res,'s--',color='gray',label='residual indicator')
ax[2].axhline(4.29e-4,color='g',ls=':',label='oracle band (NF=124)')
ax[2].axhline(2.51e-2,color='orange',ls=':',label='uniform fine (NF=133)')
ax[2].axhline(1.16e-1,color='red',ls=':',label='coarse (NF=35)')
ax[2].set_xlabel('NF (functions)'); ax[2].set_title('(c) auto-refinement convergence')
ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('greedy.png',dpi=140); print('wrote greedy.png')

# quantify localization: fraction of FINEST-level atoms within +-0.1 of x0
for L in (6,7,8):
    m=levels==L
    if m.sum():
        near=np.abs(centers[m]-G.X0)<=0.10
        print(f"  level {L}: {m.sum():3d} atoms, {100*near.mean():.0f}% within 0.1 of x0")
