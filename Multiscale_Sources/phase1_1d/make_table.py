"""Render the 1D decisive scale-separation results (decisive_results.json) as a table image."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("decisive_results.json"))
cols = ["spike width $w$", "$N_F$ (matched)", "banded wavelet\nrel $L_2$", "RBF-PIELM\n(best width) rel $L_2$", "wavelet\nadvantage"]
rows = [[f"{r['w']:.3f}", f"{r['NF']}", f"{r['wavelet']:.2e}", f"{r['rbf']:.2e}", f"{r['ratio']:.0f}×"] for r in R]

fig, ax = plt.subplots(figsize=(10.0, 0.6 + 0.5*(len(rows)+1)))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#999999")
    if r == 0:
        cell.set_facecolor("#1f3b57"); cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#eaf2fb" if c < 2 else "#ffffff")
        if c == 4: cell.set_text_props(weight="bold", color="#1f77b4")
ax.set_title("1D multiscale screened-Poisson: adaptive multiresolution vs. fixed-width features\n"
             "(both matched $N_F$; RBF-PIELM given its best of 6 widths)",
             fontsize=12, weight="bold", pad=14)
fig.tight_layout(); fig.savefig("table_decisive.png", dpi=220, bbox_inches="tight"); print("saved table_decisive.png")
