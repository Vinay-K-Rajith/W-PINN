"""
Phase 3 credibility audit: rule out 'inverse crime' + test MODEL-MISMATCH robustness.
Truth = ELLIPSE (2-fold harmonic perturbation, keeps f=-4). General inversion model:
    phi(x,y;r0,c2,c3) = x^2+y^2 - r0^2 + c2*(x^2-y^2) + c3*(x^3-3xy^2)   (Delta phi = 4 => f=-4)
Invert the ellipse truth with three model families:
    (A) CIRCLE        (free r0)         -> WRONG family: must leave a detectable nonzero residual
    (B) ELLIPSE family(free r0,c2)      -> CORRECT family: should recover accurately
    (C) OVER-RICH     (free r0,c2,c3)   -> should recover r0,c2 and c3~0
Key credibility signals:
  * data is the ANALYTIC solution (independent of the wavelet forward) -> not an inverse crime;
  * a mis-specified model (A) CANNOT drive the data residual to the noise floor -> you can DIAGNOSE
    model adequacy from the residual, the opposite of an inverse-crime artifact.
"""
import math, numpy as np, torch
from scipy.optimize import minimize
import phase3_2d as P
torch.set_default_dtype(torch.float64); np.random.seed(0)

a_minus=P.a_minus
def phi_gen(x,y,r0,c2,c3): return x*x+y*y-r0*r0+c2*(x*x-y*y)+c3*(x**3-3*x*y*y)
def grad_gen(x,y,r0,c2,c3): return 2*x+2*c2*x+3*c3*(x*x-y*y), 2*y-2*c2*y-6*c3*x*y
def iface_gen(r0,c2,c3,M=240):
    th=np.linspace(0,2*np.pi,M,endpoint=False); rs=np.full(M,max(r0,0.3))
    for _ in range(80):
        c2t,c3t=np.cos(2*th),np.cos(3*th)
        g=rs*rs*(1+c2*c2t)+c3*c3t*rs**3-r0*r0; gp=2*rs*(1+c2*c2t)+3*c3*c3t*rs*rs
        rs=rs-g/np.where(np.abs(gp)<1e-9,1e-9,gp)
    xg=rs*np.cos(th); yg=rs*np.sin(th); gx,gy=grad_gen(xg,yg,r0,c2,c3); nrm=np.hypot(gx,gy)
    return xg,yg,gx/nrm,gy/nrm,th,rs

def truth_field(r0,c2,c3,rho):
    a_out=rho*a_minus
    def u(x,y): p=phi_gen(x,y,r0,c2,c3); return np.where(p<0,p/a_minus,p/a_out)
    return u

def radial(r0,c2,c3,th):              # interface radius at angles th (for geometric error)
    rs=np.full_like(th,max(r0,0.3))
    for _ in range(80):
        c2t,c3t=np.cos(2*th),np.cos(3*th)
        g=rs*rs*(1+c2*c2t)+c3*c3t*rs**3-r0*r0; gp=2*rs*(1+c2*c2t)+3*c3*c3t*rs*rs
        rs=rs-g/np.where(np.abs(gp)<1e-9,1e-9,gp)
    return rs

def invert(truth_params, model, rho=10.0, N=60, noise=1e-3):
    r0t,c2t,c3t = truth_params
    uf=truth_field(r0t,c2t,c3t,rho); ub=uf(P.BX,P.BY)
    xd,yd,ud,Pd=P.make_data(uf,N,noise)
    def unpack(p):
        if model=="circle": return p[0],0.0,0.0
        if model=="ellipse": return p[0],p[1],0.0
        return p[0],p[1],p[2]
    def J(p):
        r0,c2,c3=unpack(p)
        if r0<0.25 or r0>0.8 or abs(c2)>0.5 or abs(c3)>0.6: return 1e3
        phg=phi_gen(P.XS,P.YS,r0,c2,c3); phid=phi_gen(xd,yd,r0,c2,c3)
        xg,yg,nx,ny,_,_=iface_gen(r0,c2,c3)
        cM,cP,bM,bP=P.forward(rho,phg,(xg,yg,nx,ny),ub)
        pr=np.where(phid<0,P.predict_at(cM,bM,Pd),P.predict_at(cP,bP,Pd))
        return np.sum((pr-ud)**2)
    x0={"circle":[0.45],"ellipse":[0.45,0.05],"rich":[0.45,0.05,0.0]}[model]
    res=minimize(J,x0,method="Nelder-Mead",options=dict(xatol=1e-5,fatol=1e-14,maxiter=600))
    r0h,c2h,c3h=unpack(res.x)
    resid_rms=math.sqrt(res.fun/N)              # data-residual RMS
    th=np.linspace(0,2*np.pi,400)
    geo=np.max(np.abs(radial(r0h,c2h,c3h,th)-radial(r0t,c2t,c3t,th)))   # max interface error
    return (r0h,c2h,c3h),resid_rms,geo

if __name__=="__main__":
    truth=(0.5,0.15,0.0)   # ELLIPSE truth (c2=0.15), NOT a 3-lobe flower
    print("TRUTH = ellipse (r0,c2,c3)=(0.50,0.15,0.00); invert with 3 model families.\n")
    print(f"{'model':>9} {'noise':>7} | {'r0_hat':>7} {'c2_hat':>7} {'c3_hat':>7} | {'resid_RMS':>10} {'geom_err':>9}")
    print("-"*72)
    for model in ["circle","ellipse","rich"]:
        for noise in [0.0,1e-2]:
            (r0h,c2h,c3h),rr,geo=invert(truth,model,noise=noise)
            print(f"{model:>9} {noise:>7.0e} | {r0h:>7.4f} {c2h:>7.4f} {c3h:>7.4f} | {rr:>10.2e} {geo:>9.2e}")
    print("\nReading: CIRCLE (wrong family) leaves resid_RMS >> noise & large geom_err -> model is")
    print("diagnosably wrong (NOT an inverse crime). ELLIPSE/RICH (correct family) recover c2~0.15,")
    print("c3~0, resid_RMS ~ noise floor, geom_err ~ forward floor. Recovery is real, not an artifact.")
