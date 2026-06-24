"""
Gap 3 -- CONVERGENCE / REFINEMENT study for the gear interface (paper-standard).
================================================================================
Extends the rho=10 banded-vs-coarse ablation in results.md to a full refinement
sequence: keep the global coarse basis [0,1,2,3] fixed and progressively add the
finest wavelet levels ONLY in the teeth annulus (the banded refinement), recording
relL2, L-inf, interface kink error, the basis size N_F, the post-normalisation
condition number, and the solve time at each step.

This produces the convergence figure (relL2 and conditioning vs N_F) and the
refinement table a CAMWA referee expects: it shows (i) the geometric error decay
from adaptive multiresolution refinement and (ii) the operational-matrix
conditioning ceiling that eventually limits it (parsimony, not more levels, is the
lever) -- reported honestly rather than hidden.

Run:  python convergence_study.py     (writes convergence_results.json, convergence.png)
"""
import json, time, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from run_gear import banded_family, solve
torch.set_default_dtype(torch.float64)

RHO = 10.0          # matches the existing ablation in results.md (worst-case contrast for this gear)

# refinement sequence: global coarse [0,1,2,3], then add banded finest levels one at a time
STEPS = [
    ("coarse [0,1,2,3] global", (0, 1, 2, 3), ()),
    ("+ band[4]",               (0, 1, 2, 3), (4,)),
    ("+ band[4,5]",             (0, 1, 2, 3), (4, 5)),
    ("+ band[4,5,6]",           (0, 1, 2, 3), (4, 5, 6)),
]

if __name__ == "__main__":
    rows = []
    hdr = f"{'basis':>26} | {'N_F':>6} {'relL2':>10} {'Linf':>10} {'kinkErr':>10} {'cond':>10} {'sec':>6}"
    print(f"gear convergence study, rho={RHO:.0f}\n"); print(hdr); print("-"*len(hdr))
    for name, coarse, fine in STEPS:
        fam = banded_family(coarse=coarse, fine=fine)
        t0 = time.time(); m, _, _ = solve(RHO, fam, want_cond=True); dt = time.time() - t0
        rows.append(dict(basis=name, NF=int(m["NF"]), relL2=m["relL2"], Linf=m["Linf"],
                         kink=m["kink"], cond=m["cond"], sec=dt))
        print(f"{name:>26} | {m['NF']:>6d} {m['relL2']:>10.2e} {m['Linf']:>10.2e} "
              f"{m['kink']:>10.2e} {m['cond']:>10.2e} {dt:>6.1f}", flush=True)

    with open("convergence_results.json", "w") as fh:
        json.dump(dict(rho=RHO, steps=rows), fh, indent=2)

    # ---- convergence figure: relL2 (and conditioning) vs N_F ----
    NF = [r["NF"] for r in rows]; L2 = [r["relL2"] for r in rows]; CN = [r["cond"] for r in rows]
    labels = [r["basis"].replace(" global", "") for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax[0].loglog(NF, L2, "o-", lw=2, ms=7, color="#1f77b4")
    for x, y, t in zip(NF, L2, labels):
        ax[0].annotate(t, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax[0].set_xlabel("$N_F$ (number of basis functions)"); ax[0].set_ylabel("relative $L^2$ error")
    ax[0].set_title(f"(a) banded refinement convergence ($\\rho$={RHO:.0f})"); ax[0].grid(True, which="both", alpha=.3)
    ax[1].loglog(NF, CN, "s-", lw=2, ms=7, color="#d62728")
    for x, y, t in zip(NF, CN, labels):
        ax[1].annotate(t, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax[1].set_xlabel("$N_F$"); ax[1].set_ylabel("condition number cond($\\tilde A$)")
    ax[1].set_title("(b) operational-matrix conditioning"); ax[1].grid(True, which="both", alpha=.3)
    plt.tight_layout(); plt.savefig("convergence.png", dpi=160)
    print("\nsaved convergence_results.json, convergence.png")
