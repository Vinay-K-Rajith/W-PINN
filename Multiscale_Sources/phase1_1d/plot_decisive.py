"""Polished diagram of the 1D decisive scale-separation sweep (reads decisive_results.json)."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("decisive_results.json"))
W=[r["w"] for r in R]; EW=[r["wavelet"] for r in R]; ER=[r["rbf"] for r in R]

fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.loglog(W, EW, 'o-', color='#1f77b4', lw=2.4, ms=9, label='banded wavelet W-PINN (adaptive multiresolution)')
ax.loglog(W, ER, 's--', color='#d62728', lw=2.4, ms=9, label='RBF-PIELM (best of 6 widths, matched $N_F$)')
ax.invert_xaxis()
ax.set_xlabel('spike width  $w$    (scale separation increases  $\\rightarrow$)', fontsize=12)
ax.set_ylabel('relative $L_2$ error', fontsize=12)
ax.set_title('1D multiscale screened-Poisson:\nadaptive multiresolution vs. fixed-width random features', fontsize=12.5)
ax.grid(True, which='both', ls=':', alpha=0.5)
ax.legend(fontsize=10.5, loc='upper right')
# advantage annotations, offset to the LEFT of the wavelet markers (no overlap)
for r in R:
    ax.annotate(f"{r['ratio']:.0f}×", (r['w'], r['wavelet']), textcoords="offset points",
                xytext=(16, -2), fontsize=9, color='#1f77b4', weight='bold')
plt.tight_layout(); plt.savefig("decisive.png", dpi=200); print("saved decisive.png")
