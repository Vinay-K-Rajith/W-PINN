"""
Phase 2 closeout (2/2): in-2D baseline head-to-head on the CIRCLE benchmark.
Compares the decomposed wavelet W-PINN against:
  - vanilla single 2D MLP PINN (AD)         -> standard baseline, cannot represent the kink
  - decomposed 2D MLP / XPINN (AD)          -> strongest competitor
at rho=10 and rho=1000, mirroring the 1D crossover.
"""
import time, math, numpy as np, torch
from run_phase2 import make_problem, solve as wavelet_solve, r0
torch.set_default_dtype(torch.float64)

# ---- point sets for the MLPs (reuse circle geometry) ----
def pts(K=70,M=160):
    g=np.linspace(-1,1,K); X,Y=np.meshgrid(g,g); xs=X.ravel(); ys=Y.ravel(); r=np.sqrt(xs**2+ys**2)
    ins=r<r0-1e-9; out=(r>r0+1e-9)&(np.abs(xs)<1)&(np.abs(ys)<1)
    th=np.linspace(0,2*np.pi,M,endpoint=False); xg=r0*np.cos(th); yg=r0*np.sin(th)
    nb=np.linspace(-1,1,K); bx=np.concatenate([nb,nb,-np.ones(K),np.ones(K)]); by=np.concatenate([-np.ones(K),np.ones(K),nb,nb])
    return (xs[ins],ys[ins]),(xs[out],ys[out]),(xg,yg,np.cos(th),np.sin(th)),(bx,by)

class MLP(torch.nn.Module):
    def __init__(s,w=64,d=4):
        super().__init__(); L=[torch.nn.Linear(2,w),torch.nn.Tanh()]
        for _ in range(d-1):L+=[torch.nn.Linear(w,w),torch.nn.Tanh()]
        L+=[torch.nn.Linear(w,1)]; s.net=torch.nn.Sequential(*L)
        for m in s.net:
            if isinstance(m,torch.nn.Linear): torch.nn.init.xavier_normal_(m.weight); torch.nn.init.zeros_(m.bias)
    def forward(s,p): return s.net(p)

def lap(u,p):
    g=torch.autograd.grad(u,p,torch.ones_like(u),create_graph=True)[0]
    ux,uy=g[:,:1],g[:,1:2]
    uxx=torch.autograd.grad(ux,p,torch.ones_like(ux),create_graph=True)[0][:,:1]
    uyy=torch.autograd.grad(uy,p,torch.ones_like(uy),create_graph=True)[0][:,1:2]
    return g,uxx+uyy

def testgrid():
    g=np.linspace(-0.999,0.999,200); X,Y=np.meshgrid(g,g); return X.ravel(),Y.ravel()

def relL2(pred,prob,xt,yt):
    ue=prob["u_exact"](xt,yt); return np.linalg.norm(pred-ue)/np.linalg.norm(ue)

def vanilla(prob,epochs=6000,lr=1e-3):
    (xm,ym),(xp,yp),(xg,yg,nx,ny),(bx,by)=pts()
    xc=np.concatenate([xm,xp]); yc=np.concatenate([ym,yp])
    P=torch.tensor(np.stack([xc,yc],1),requires_grad=True)
    av=torch.tensor(np.where(xc**2+yc**2<r0*r0,prob["a_in"],prob["a_out"])).reshape(-1,1)
    fv=torch.tensor(prob["f"](xc,yc)).reshape(-1,1)
    Pb=torch.tensor(np.stack([bx,by],1)); ub=torch.tensor(prob["u_exact"](bx,by)).reshape(-1,1)
    net=MLP(); opt=torch.optim.Adam(net.parameters(),lr=lr)
    for _ in range(epochs):
        opt.zero_grad(); u=net(P); _,L=lap(u,P)
        loss=(-av*L-fv).pow(2).mean()+10*(net(Pb)-ub).pow(2).mean(); loss.backward(); opt.step()
    xt,yt=testgrid(); pr=net(torch.tensor(np.stack([xt,yt],1))).detach().numpy().ravel()
    return relL2(pr,prob,xt,yt)

def xpinn(prob,epochs=6000,lr=1e-3):
    (xm,ym),(xp,yp),(xg,yg,nx,ny),(bx,by)=pts(); ai,ao=prob["a_in"],prob["a_out"]
    Pm=torch.tensor(np.stack([xm,ym],1),requires_grad=True); Pp=torch.tensor(np.stack([xp,yp],1),requires_grad=True)
    fm=torch.tensor(prob["f"](xm,ym)).reshape(-1,1); fp=torch.tensor(prob["f"](xp,yp)).reshape(-1,1)
    G=torch.tensor(np.stack([xg,yg],1),requires_grad=True); nrm=torch.tensor(np.stack([nx,ny],1))
    Pb=torch.tensor(np.stack([bx,by],1)); ub=torch.tensor(prob["u_exact"](bx,by)).reshape(-1,1)
    nm,npp=MLP(),MLP(); opt=torch.optim.Adam(list(nm.parameters())+list(npp.parameters()),lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        uM=nm(Pm); _,LM=lap(uM,Pm); uP=npp(Pp); _,LP=lap(uP,Pp)
        rM=(-ai*LM-fm).pow(2).mean(); rP=(-ao*LP-fp).pow(2).mean()
        uMg=nm(G); gMg=torch.autograd.grad(uMg,G,torch.ones_like(uMg),create_graph=True)[0]
        uPg=npp(G); gPg=torch.autograd.grad(uPg,G,torch.ones_like(uPg),create_graph=True)[0]
        cont=(uPg-uMg).pow(2).mean(); flux=(ao*(gPg*nrm).sum(1)-ai*(gMg*nrm).sum(1)).pow(2).mean()
        bc=(npp(Pb)-ub).pow(2).mean()
        (rM+rP+10*(cont+flux)+10*bc).backward(); opt.step()
    xt,yt=testgrid(); rr=xt*xt+yt*yt
    pm=nm(torch.tensor(np.stack([xt,yt],1))).detach().numpy().ravel()
    pp=npp(torch.tensor(np.stack([xt,yt],1))).detach().numpy().ravel()
    return relL2(np.where(rr<r0*r0,pm,pp),prob,xt,yt)

if __name__=="__main__":
    print(f"{'rho':>6} | {'wavelet (LSQ)':>13} {'t':>5} | {'XPINN-MLP':>10} {'t':>5} | {'vanilla-MLP':>11} {'t':>5}")
    print("-"*70)
    for rho in [10.,1000.]:
        p=make_problem(rho)
        t=time.time(); rw=wavelet_solve(p,[0,1,2,3])['relL2']; tw=time.time()-t
        t=time.time(); rx=xpinn(p); tx=time.time()-t
        t=time.time(); rv=vanilla(p); tv=time.time()-t
        print(f"{rho:>6.0f} | {rw:>13.3e} {tw:>5.1f} | {rx:>10.3e} {tx:>5.1f} | {rv:>11.3e} {tv:>5.1f}")
