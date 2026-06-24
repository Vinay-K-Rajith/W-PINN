r"""
Phase-2 core experiment -- AUTOMATIC residual-driven greedy multiresolution refinement.

Same 1D multiscale screened-Poisson as run_phase1_multiscale.py:
    -u''(x) + kappa^2 u(x) = f(x),  x in [0,1],   u = sin(2 pi x) + A exp(-((x-x0)/w)^2),  w=0.02.

NEW IDEA (the paper's Figure 1):
  Instead of MANUALLY banding the finest wavelets around the KNOWN spike location x0=0.5
  ("oracle"), we DISCOVER where/at-what-scale to refine from the PDE RESIDUAL alone.

  This is matching pursuit / weak-greedy on the PDE-OPERATOR dictionary:
    - dictionary atoms = wavelets psi_{j,k} across levels l=0..Lmax and all translates;
    - each atom's "operator column" is  a_i(x) = (-psi'' + kappa^2 psi)(x)  on the collocation grid;
    - given the current selection S and LSQ solution c_S, the residual is  r = A_S c_S - b ;
    - SCORE each NOT-yet-selected atom by its correlation with the residual,
          score_i = |a_i . r| / ||a_i||      (the standard greedy / OMP selection rule),
      add the top-batch, re-solve the column-normalised Tikhonov LSQ, repeat.
  The atom most correlated with the residual sits exactly where the equation is least satisfied
  -> the greedy automatically localizes the spike and climbs to the finest level only THERE.

  AD-free is what makes this cheap: the residual and all atom-residual correlations are matrix
  products of pre-stored operator columns -- no autodiff, no retraining, one small LSQ per step.

Theory hooks (for the write-up): weak-greedy / OMP a-posteriori residual indicator; structured
multiresolution dictionary => best-N-term (nonlinear approximation) optimality, which single-scale
random-feature dictionaries provably lack.
"""
import math, numpy as np, torch
torch.set_default_dtype(torch.float64)

KAPPA = 1.0
A, X0, W = 1.0, 0.5, 0.02

def u_exact(x): return np.sin(2*np.pi*x) + A*np.exp(-((x - X0)/W)**2)
def f_rhs(x):
    upp_smooth = -(2*np.pi)**2*np.sin(2*np.pi*x)
    g = np.exp(-((x - X0)/W)**2)
    upp_spike = A*g*(4*(x - X0)**2/W**4 - 2.0/W**2)
    return -(upp_smooth + upp_spike) + KAPPA**2*u_exact(x)

def val(x, j, k):
    X=j[None,:]*x[:,None]-k[None,:]; return X*torch.exp(-X**2/2)
def d2(x, j, k):
    X=j[None,:]*x[:,None]-k[None,:]; return (j[None,:]**2)*X*(X**2-3)*torch.exp(-X**2/2)

# ---- full candidate dictionary: levels 0..Lmax, all translates in [lo-pad, hi+pad] ----------
def dictionary(Lmax=8, lo=0.0, hi=1.0, pad=0.5):
    js, ks, lvl = [], [], []
    for l in range(Lmax+1):
        j=2.0**l
        for k in range(int(math.floor((lo-pad)*j)), int(math.ceil((hi+pad)*j))+1):
            js.append(j); ks.append(float(k)); lvl.append(l)
    return (torch.tensor(js), torch.tensor(ks), np.array(lvl))

def build(sel, jD, kD, Nc=1200, wb=10.0):
    """Assemble the column-normalised, Tikhonov-augmented LSQ for the SELECTED atoms."""
    j, k = jD[sel], kD[sel]; NF=len(sel)
    xc = torch.linspace(0,1,Nc)
    Op = -d2(xc,j,k) + KAPPA**2*val(xc,j,k)
    A_ = torch.zeros(Nc+2, NF)
    A_[:Nc,:]=Op
    A_[Nc,:] = math.sqrt(wb)*val(torch.zeros(1),j,k)
    A_[Nc+1,:]=math.sqrt(wb)*val(torch.ones(1),j,k)
    b = torch.zeros(Nc+2,1)
    b[:Nc,0]=torch.tensor(f_rhs(xc.numpy()))
    b[Nc,0] =math.sqrt(wb)*u_exact(np.array([0.0]))[0]
    b[Nc+1,0]=math.sqrt(wb)*u_exact(np.array([1.0]))[0]
    return A_, b, j, k, xc

def solve_sel(sel, jD, kD, tik=1e-12, **kw):
    A_, b, j, k, xc = build(sel, jD, kD, **kw)
    n=len(sel)
    s=A_.norm(dim=0).clamp_min(1e-30); An=A_/s
    Aa=torch.cat([An, math.sqrt(tik)*torch.eye(n)]); bb=torch.cat([b, torch.zeros(n,1)])
    c=(torch.linalg.lstsq(Aa,bb,driver="gelsd").solution.squeeze(1))/s
    return c, A_, b, xc

def metrics(c, j, k):
    xt=torch.linspace(0,1,4000); pred=(val(xt,j,k)@c).numpy(); ue=u_exact(xt.numpy()); err=pred-ue
    return dict(relL2=float(np.linalg.norm(err)/np.linalg.norm(ue)),
                Linf=float(np.max(np.abs(err))), relLinf=float(np.max(np.abs(err))/np.max(np.abs(ue))))

# ---------------------------- the greedy refinement loop -------------------------------------
def greedy(Lmax=8, start_levels=(0,1,2,3), batch=6, max_NF=130, tol=1e-4, verbose=True):
    jD, kD, lvl = dictionary(Lmax)
    Ndict=len(lvl)
    sel = [i for i in range(Ndict) if lvl[i] in start_levels]   # coarse background, level<=3
    history=[]
    # precompute operator columns of the WHOLE dictionary on the collocation grid (AD-free, once)
    xc=torch.linspace(0,1,1200)
    OpAll = (-d2(xc,jD,kD) + KAPPA**2*val(xc,jD,kD))            # (Nc, Ndict)
    fcol  = torch.tensor(f_rhs(xc.numpy()))                     # collocation RHS
    colnorm = OpAll.norm(dim=0).clamp_min(1e-30)
    step=0
    while True:
        c, A_, b, _ = solve_sel(sel, jD, kD)
        j, k = jD[sel], kD[sel]
        m = metrics(c, j, k)
        # PDE residual on collocation rows only (drop the 2 BC rows)
        r = (A_[:1200,:] @ c) - fcol
        res_ind = float(r.norm()/fcol.norm())
        history.append(dict(step=step, NF=len(sel), relL2=m['relL2'], relLinf=m['relLinf'],
                            res=res_ind, maxlvl=int(lvl[sel].max())))
        if verbose:
            print(f"  step {step:2d}  NF={len(sel):4d}  relL2={m['relL2']:.3e}  "
                  f"res_ind={res_ind:.3e}  maxlvl={int(lvl[sel].max())}", flush=True)
        if res_ind < tol or len(sel) >= max_NF: break
        # SCORE every not-yet-selected atom by correlation with the residual (greedy/OMP rule)
        score = (OpAll.t() @ r).abs() / colnorm                # (Ndict,)
        score_np = score.numpy().copy()
        score_np[sel] = -np.inf                                # exclude already-selected
        order = np.argsort(score_np)[::-1]
        add = [int(i) for i in order[:batch]]
        sel = sorted(set(sel) | set(add))
        step += 1
    # what did it discover?
    fine_sel = [i for i in sel if lvl[i] >= 4]
    centers = sorted(set(round(float(kD[i]/jD[i]),3) for i in fine_sel))
    return dict(sel=sel, jD=jD, kD=kD, lvl=lvl, c=c, history=history,
                fine_centers=centers, final=history[-1])

if __name__=="__main__":
    print("AUTOMATIC residual-greedy multiresolution refinement (spike location x0 is NOT given)\n")
    print("Greedy run:")
    g = greedy()
    print(f"\nGREEDY discovered fine-level (l>=4) wavelet centers at: {g['fine_centers']}")
    print(f"  (true spike is at x0={X0}; finest level used = {g['final']['maxlvl']})")

    # ----- baselines for the comparison table -----
    jD, kD, lvl = g['jD'], g['kD'], g['lvl']
    def sel_levels(levels): return [i for i in range(len(lvl)) if lvl[i] in levels]
    def sel_oracle():   # manual band [6,7,8] in [0.40,0.60] on top of [0..3] -- KNOWS x0
        s=[i for i in range(len(lvl)) if lvl[i] in (0,1,2,3)]
        for i in range(len(lvl)):
            if lvl[i] in (6,7,8) and 0.40 <= float(kD[i]/jD[i]) <= 0.60: s.append(i)
        return sorted(set(s))
    rows=[]
    for name, sel in [("coarse [0,1,2,3]",        sel_levels((0,1,2,3))),
                      ("uniform fine [0..5]",      sel_levels((0,1,2,3,4,5))),
                      ("oracle band[6,7,8]@spike", sel_oracle())]:
        c,_,_,_ = solve_sel(sel, jD, kD); m=metrics(c, jD[sel], kD[sel])
        rows.append((name, len(sel), m['relL2'], m['relLinf']))
    gf=g['final']; rows.append(("GREEDY (auto, no x0)", gf['NF'], gf['relL2'], gf['relLinf']))

    print(f"\n{'method':>26} {'NF':>5} | {'relL2':>10} {'relLinf':>10}")
    print("-"*58)
    for name,NF,rl2,rli in rows:
        print(f"{name:>26} {NF:>5d} | {rl2:>10.2e} {rli:>10.2e}")

    import json
    json.dump(dict(rows=rows, history=g['history'], fine_centers=g['fine_centers'],
                   x0=X0, maxlvl=gf['maxlvl']),
              open("greedy_results.json","w"), indent=2)
    print("\nwrote greedy_results.json")
