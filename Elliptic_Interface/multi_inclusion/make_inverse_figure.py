"""Figure for the multi-parameter inverse: the measured field with TRUE inclusions (white solid) and
RECOVERED inclusions (red dashed), sensors overlaid.  Run: python make_inverse_figure.py"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import inverse as I

I.model_setup()
res = np.load("inverse_results.npy", allow_pickle=True)
rr = res[-1]                                            # noise = 1% run
theta = rr['sol']; (sx, sy) = None, None

# rebuild the data field for the background (full field at TRUE geometry)
g = np.linspace(-0.98, 0.98, 220); X, Y = np.meshgrid(g, g)
tt = np.array([v for c in I.TRUE for v in c])
U = I.model_predict(tt, X.ravel(), Y.ravel(), full=True).reshape(X.shape)
sx, sy = I.make_sensors(300, 0)

th = np.linspace(0, 2*np.pi, 200)
fig, ax = plt.subplots(1, 1, figsize=(6.2, 5.4))
c = ax.contourf(X, Y, U, 40, cmap='viridis'); plt.colorbar(c, ax=ax, label='u (data field)')
ax.plot(sx, sy, 'w.', ms=1.5, alpha=0.5, label='sensors (matrix)')
for j, (cx, cy, r) in enumerate(I.TRUE):
    ax.plot(cx+r*np.cos(th), cy+r*np.sin(th), 'w-', lw=2.0, label='true' if j == 0 else None)
for j in range(I.M_INCL):
    cx, cy, r = theta[3*j:3*j+3]
    ax.plot(cx+r*np.cos(th), cy+r*np.sin(th), 'r--', lw=2.0, label='recovered' if j == 0 else None)
ax.set_aspect('equal'); ax.set_title('Multi-parameter inverse: recover (cx,cy,r) x4\n'
              f"from interior data, $\\rho$={I.A_OUT:.0f}, 1% noise  "
              f"(mean |pos err|={np.mean(rr['pos_err']):.1e})")
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout(); plt.savefig("inverse.png", dpi=130); print("saved inverse.png")
