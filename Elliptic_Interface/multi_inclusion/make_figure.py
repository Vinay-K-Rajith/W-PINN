"""Figure for the multi-inclusion case: exact / W-PINN prediction / log10|error|, all m interfaces
drawn.  Run: python make_figure.py  ->  multi_inclusion.png"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from multi_inclusion import solve, build_basis, INCL5

incl = INCL5; rho = 1000.0
basis = build_basis(incl)
m, fields = solve(incl, [1.0]*len(incl), rho, basis=basis, want_fields=True)
Xt, Yt, PR, UE, _ = fields
print(f"figure: rho={rho:.0f}  relL2={m['relL2']:.2e}  Linf={m['Linf']:.2e}")

th = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.4))
def circles(a):
    for (cx, cy, r) in incl:
        a.plot(cx + r*np.cos(th), cy + r*np.sin(th), 'w-', lw=1.0)
c0 = ax[0].contourf(Xt, Yt, UE, 40, cmap='viridis'); circles(ax[0])
ax[0].set_title(f'(a) exact, {len(incl)} inclusions ($\\rho$={rho:.0f})'); plt.colorbar(c0, ax=ax[0])
c1 = ax[1].contourf(Xt, Yt, PR, 40, cmap='viridis'); circles(ax[1])
ax[1].set_title('(b) wavelet W-PINN (mesh-free, block basis)'); plt.colorbar(c1, ax=ax[1])
c2 = ax[2].contourf(Xt, Yt, np.log10(np.abs(PR - UE) + 1e-16), 40, cmap='magma')
for (cx, cy, r) in incl:
    ax[2].plot(cx + r*np.cos(th), cy + r*np.sin(th), 'c-', lw=0.8)
ax[2].set_title('(c) $\\log_{10}$|error|'); plt.colorbar(c2, ax=ax[2])
for a in ax: a.set_aspect('equal')
plt.tight_layout(); plt.savefig("multi_inclusion.png", dpi=130); print("saved multi_inclusion.png")
