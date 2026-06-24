"""
rho < 1 characterization (soundness)
=====================================================================================================
Contrast rho = a_out / a_in.  rho > 1 = stiff/insulating inclusions (done in studies.py);  rho < 1 =
inclusions MORE conductive than the matrix.  A referee asks whether the method is sound across the
WHOLE contrast range, not just rho >= 1.  We sweep rho across both regimes on the 5-inclusion forward
problem (manufactured O(1) fields, inhomogeneous jumps) and report relL2 + worst per-inclusion error.

Run: python rho_sweep.py
"""
import json, time, numpy as np
from multi_inclusion import solve, build_basis, INCL5

if __name__ == "__main__":
    basis = build_basis(INCL5)
    print(f"5 inclusions, sweeping rho = a_out/a_in across rho<1 and rho>1  (a_in=1 fixed)\n")
    print(f"{'rho':>8} | {'relL2':>10} {'Linf':>10} {'ujump':>10} {'fjump':>10} | {'worst per-incl':>14}")
    print("-"*72)
    rows = []
    for rho in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]:
        a_in = [1.0]*5; a_out = rho
        t0 = time.time(); m, _ = solve(INCL5, a_in, a_out, basis=basis); dt = time.time() - t0
        worst = max(m['per_incl'])
        rows.append(dict(rho=rho, relL2=m['relL2'], Linf=m['Linf'], ujump=m['ujump'],
                         fjump=m['fjump'], worst_incl=worst, sec=dt))
        print(f"{rho:>8.3f} | {m['relL2']:>10.2e} {m['Linf']:>10.2e} {m['ujump']:>10.2e} "
              f"{m['fjump']:>10.2e} | {worst:>14.2e}", flush=True)
    with open("rho_sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print("\n-> rho_sweep_results.json")
