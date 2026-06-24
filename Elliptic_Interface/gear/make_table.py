"""Render the gear error metrics as publication-quality table images (table.png, table_ablation.png)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- main contrast-sweep metrics (banded basis, NF=4764) ----
cols = ["$\\rho=a^+/a^-$", "rel $L_2$", "MSE", "RMSE", "MAE", "$L_\\infty$", "rel $L_\\infty$", "kink err"]
rows = [
    ["10",   "1.68e-03", "2.96e-08", "1.72e-04", "8.90e-05", "1.15e-03", "3.80e-03", "1.00e-02"],
    ["100",  "4.98e-04", "1.80e-09", "4.24e-05", "1.86e-05", "5.63e-04", "1.86e-03", "1.21e-02"],
    ["1000", "1.02e-03", "7.47e-09", "8.64e-05", "6.58e-05", "5.06e-04", "1.67e-03", "2.11e-02"],
]

def render(cols, rows, title, fname, figw=11.0):
    fig, ax = plt.subplots(figsize=(figw, 0.5 + 0.42*(len(rows)+1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celldata().items() if hasattr(tbl, "get_celldata") else tbl.get_celld().items():
        cell.set_edgecolor("#888888")
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f4f6")
        if c == 0 and r > 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    fig.tight_layout(); fig.savefig(fname, dpi=220, bbox_inches="tight"); print("saved", fname)

render(cols, rows,
       "Gear interface: error vs. contrast (banded wavelet W-PINN, $N_F=4764$)",
       "table.png")

# ---- ablation: global coarse basis vs banded refinement (rho=10) ----
cols2 = ["basis", "$N_F$", "rel $L_2$", "MSE", "$L_\\infty$", "kink err"]
rows2 = [
    ["[0,1,2,3] global",  "2500", "2.25e-02", "5.30e-06", "1.14e-02", "6.89e-02"],
    ["[0-3] + band[4,5]", "4764", "1.68e-03", "2.96e-08", "1.15e-03", "1.00e-02"],
]
render(cols2, rows2,
       "Banded multiresolution refinement vs. global coarse basis ($\\rho=10$)",
       "table_ablation.png", figw=9.0)
