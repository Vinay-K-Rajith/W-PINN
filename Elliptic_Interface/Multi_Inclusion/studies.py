"""
Paper-grade studies for the multi-inclusion W-PINN.  Writes tables to stdout + studies_results.json.
  (1) main   : 5 well-separated inclusions, contrast sweep rho in {10,100,1000}, per-inclusion relL2
  (2) hetero : each inclusion its OWN contrast in one solve (composite-material case)
  (3) ablate : global-coarse basis  vs  + per-inclusion radius-scaled fine band (refinement matters)
  (4) topo   : m = 2..6 inclusions -> relL2 stays flat (decomposition scales past two subdomains)
Run: python studies.py
"""
import json, time, numpy as np
from multi_inclusion import solve, build_basis, POOL6, INCL5

OUT = {}


def main_sweep():
    print("\n=== (1) main: 5 well-separated inclusions, contrast sweep ===")
    basis = build_basis(INCL5)
    print(f"NF_out={basis[0][0].numel()}  NF_in={[f[0].numel() for f in basis[1]]}  "
          f"NF_total={basis[0][0].numel()+sum(f[0].numel() for f in basis[1])}")
    rows = []
    for k, rho in enumerate([10.0, 100.0, 1000.0]):
        t0 = time.time()
        m, _ = solve(INCL5, [1.0]*5, rho, basis=basis, want_cond=(k == 0))
        dt = time.time() - t0
        rows.append(dict(rho=rho, **{k: v for k, v in m.items() if k != 'NFi'}, sec=dt))
        print(f"rho={rho:>6.0f}  relL2={m['relL2']:.2e}  Linf={m['Linf']:.2e}  "
              f"ujump={m['ujump']:.2e}  fjump={m['fjump']:.2e}  "
              f"per-incl=[{' '.join(f'{p:.1e}' for p in m['per_incl'])}]  "
              f"cond={m['cond']:.1e}  {dt:.1f}s", flush=True)
    OUT['main'] = rows


def hetero():
    print("\n=== (2) hetero: each inclusion its own contrast (one solve) ===")
    rho_i = [10.0, 100.0, 1000.0, 50.0, 500.0]      # contrast a_out/a_in,i per inclusion
    a_out = 1000.0; a_in = [a_out/r for r in rho_i]
    basis = build_basis(INCL5)
    t0 = time.time(); m, _ = solve(INCL5, a_in, a_out, basis=basis); dt = time.time() - t0
    print(f"rho_i={rho_i}")
    print(f"relL2={m['relL2']:.2e}  Linf={m['Linf']:.2e}  ujump={m['ujump']:.2e}  fjump={m['fjump']:.2e}")
    for i, (ri, p) in enumerate(zip(rho_i, m['per_incl'])):
        print(f"   inclusion {i}  rho={ri:>6.0f}  per-incl relL2={p:.2e}")
    OUT['hetero'] = dict(rho_i=rho_i, **{k: v for k, v in m.items() if k != 'NFi'}, sec=dt)


def ablate():
    print("\n=== (3) ablate: global-coarse basis vs + per-inclusion fine band (rho=1000) ===")
    rows = []
    for name, kw in [("coarse-only [0,1,2,3]", dict(band_cells=0.0)),
                     ("coarse + radius-scaled band", dict())]:
        basis = build_basis(INCL5, **kw)
        NF = basis[0][0].numel() + sum(f[0].numel() for f in basis[1])
        t0 = time.time(); m, _ = solve(INCL5, [1.0]*5, 1000.0, basis=basis); dt = time.time() - t0
        rows.append(dict(name=name, NF=NF, relL2=m['relL2'], Linf=m['Linf'], fjump=m['fjump'], sec=dt))
        print(f"{name:>28}  NF={NF:>5}  relL2={m['relL2']:.2e}  Linf={m['Linf']:.2e}  "
              f"fjump={m['fjump']:.2e}  {dt:.1f}s", flush=True)
    OUT['ablate'] = rows


def topo():
    print("\n=== (4) topo: m=2..6 inclusions, rho=100 (decomposition scales) ===")
    rows = []
    for m_ in range(2, 7):
        incl = POOL6[:m_]
        basis = build_basis(incl)
        NF = basis[0][0].numel() + sum(f[0].numel() for f in basis[1])
        t0 = time.time(); mt, _ = solve(incl, [1.0]*m_, 100.0, basis=basis); dt = time.time() - t0
        worst = max(mt['per_incl'])
        rows.append(dict(m=m_, NF=NF, relL2=mt['relL2'], worst_incl=worst, Linf=mt['Linf'], sec=dt))
        print(f"m={m_}  NF={NF:>5}  relL2={mt['relL2']:.2e}  worst-incl={worst:.2e}  "
              f"Linf={mt['Linf']:.2e}  {dt:.1f}s", flush=True)
    OUT['topo'] = rows


if __name__ == "__main__":
    t = time.time()
    main_sweep(); hetero(); ablate(); topo()
    with open("studies_results.json", "w") as f:
        json.dump(OUT, f, indent=2, default=float)
    print(f"\nAll studies done in {time.time()-t:.0f}s -> studies_results.json")
