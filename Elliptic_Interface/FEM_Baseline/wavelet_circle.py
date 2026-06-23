"""
Wavelet W-PINN on the SAME circle problem as fem_circle.py -> head-to-head accuracy + wall-time.
Problem: -div(a grad u)=f on (-1,1)^2, circle r0=0.5, u=phi/a (phi=x^2+y^2-r0^2), f=-Delta phi=-4,
homogeneous jumps, Dirichlet = exact u on the square.  Identical to the conforming-FEM reference.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Multi_Inclusion"))
from multi_inclusion import solve_physical, build_basis, matrix_precomp

R0 = 0.5
phi = lambda x, y: x*x + y*y - R0*R0
F = lambda x, y: np.full_like(np.asarray(x, float), -4.0)   # f = -Delta phi = -4

if __name__ == "__main__":
    incl = [(0.0, 0.0, R0)]
    gt = np.linspace(-0.999, 0.999, 300); Xt, Yt = np.meshgrid(gt, gt)
    xt = Xt.ravel(); yt = Yt.ravel()
    print(f"wavelet W-PINN, circle r0={R0}, same problem as fem_circle.py\n")
    print(f"{'rho':>7} {'NF':>6} {'relL2':>11} {'Linf':>11} {'build+solve_s':>13}")
    for rho in [10.0, 1000.0]:
        a_in, a_out = [1.0], rho
        G = lambda x, y: phi(x, y)/a_out                    # boundary lies in the matrix
        basis = build_basis(incl, coarse=(0, 1, 2, 3), band_cells=0.0, n_in_levels=3, alpha=4.0)
        NF = basis[0][0].numel() + sum(b[0].numel() for b in basis[1])
        t0 = time.time()
        pred = solve_physical(incl, a_in, a_out, F, G, basis=basis, K=140, M=400)
        dt = time.time() - t0
        u = pred(xt, yt)
        ue = np.where(phi(xt, yt) < 0, phi(xt, yt)/a_in[0], phi(xt, yt)/a_out)
        err = u - ue
        relL2 = np.linalg.norm(err)/np.linalg.norm(ue); Linf = np.max(np.abs(err))
        print(f"{rho:>7.0f} {NF:>6} {relL2:>11.3e} {Linf:>11.3e} {dt:>13.2f}", flush=True)
