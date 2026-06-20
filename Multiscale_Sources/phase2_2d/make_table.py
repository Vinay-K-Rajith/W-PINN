"""Render Phase-2 metrics (results.json) as a publication-quality table image (table.png)."""
import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = json.load(open("results.json"))
# fixed row order (methods); tolerate missing baselines (QUICK runs)
order = ["W-PINN coarse [0,1,2]", "W-PINN medium [0,1,2,3]", "W-PINN banded [0,1,2,3]+[4,5]@bumps",
         "RBF-PIELM (matched NF)", "vanilla PINN (4k Adam)"]
SHORT = {"W-PINN coarse [0,1,2]": "W-PINN coarse",
         "W-PINN medium [0,1,2,3]": "W-PINN medium",
         "W-PINN banded [0,1,2,3]+[4,5]@bumps": "W-PINN banded (adaptive)",
         "RBF-PIELM (matched NF)": "RBF-PIELM (best width)",
         "vanilla PINN (4k Adam)": "vanilla PINN (4k Adam)"}
rows, names = [], [k for k in order if k in R]
cols = ["method", "$N_F$ / params", "rel $L_2$", "MSE", "RMSE", "MAE", "$L_\\infty$", "rel $L_\\infty$"]
for k in names:
    m = R[k]
    rows.append([SHORT.get(k, k), f"{m['NF']}",
                 f"{m['relL2']:.2e}", f"{m['MSE']:.2e}", f"{m['RMSE']:.2e}",
                 f"{m['MAE']:.2e}", f"{m['Linf']:.2e}", f"{m['relLinf']:.2e}"])

fig, ax = plt.subplots(figsize=(13.5, 0.5 + 0.5*(len(rows)+1)))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.7)
tbl.auto_set_column_width(col=list(range(len(cols))))
best = min(range(len(names)), key=lambda i: R[names[i]]["relL2"])     # highlight best relL2
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#999999")
    if r == 0:
        cell.set_facecolor("#1f3b57"); cell.set_text_props(color="white", weight="bold")
    else:
        is_wpinn = names[r-1].startswith("W-PINN")
        cell.set_facecolor("#eaf2fb" if is_wpinn else "#fdeee3")
        if r-1 == best: cell.set_text_props(weight="bold")
        if c == 0: cell.set_text_props(weight="bold")
ax.set_title("2D multiscale screened-Poisson: W-PINN (banded) vs. baselines",
             fontsize=13, weight="bold", pad=12)
fig.tight_layout(); fig.savefig("table.png", dpi=220, bbox_inches="tight"); print("saved table.png")
