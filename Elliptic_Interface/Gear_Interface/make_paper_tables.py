"""Render the three new gap-closing studies as publication-quality table images + figures.
Consumes convergence_results.json, baselines_results.json, robustness_results.json.
Run after the study scripts finish:  python make_paper_tables.py
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

def render(cols, rows, title, fname, figw=11.0):
    fig, ax = plt.subplots(figsize=(figw, 0.5 + 0.42*(len(rows)+1)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.6)
    cells = tbl.get_celld()
    for (r, c), cell in cells.items():
        cell.set_edgecolor("#888888")
        if r == 0:
            cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f4f6")
        if c == 0 and r > 0:
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    fig.tight_layout(); fig.savefig(fname, dpi=220, bbox_inches="tight"); print("saved", fname)

def e(v): return f"{v:.2e}"

# ---------- Gap 3: convergence / refinement ----------
try:
    C = json.load(open("convergence_results.json"))
    cols = ["basis", "$N_F$", "rel $L_2$", "$L_\\infty$", "kink err", "cond($\\tilde A$)"]
    rows = [[s["basis"], str(s["NF"]), e(s["relL2"]), e(s["Linf"]), e(s["kink"]), e(s["cond"])]
            for s in C["steps"]]
    render(cols, rows, f"Banded multiresolution refinement: convergence ($\\rho={C['rho']:.0f}$)",
           "table_convergence.png", figw=11.0)
except FileNotFoundError:
    print("convergence_results.json not found -- skipping")

# ---------- Gap 1: head-to-head baselines ----------
try:
    B = json.load(open("baselines_results.json"))
    rhos = sorted(B.keys(), key=lambda x: int(x))
    def grab(method, rho, key):
        d = B[rho][method]; return d[key]
    cols = ["method"] + sum([[f"rel $L_2$ ($\\rho$={r})", f"$L_\\infty$ ($\\rho$={r})"] for r in rhos], [])
    NFW = B[rhos[0]]["NF"]
    label = {"wavelet": f"wavelet W-PINN (banded, $N_F$={NFW})",
             "rbf_pielm_best": f"RBF-PIELM (best width, $N_F$={NFW})",
             "xpinn": "XPINN-MLP (Adam 8k)",
             "vanilla": "vanilla MLP (Adam 8k)"}
    rows = []
    for m in ["wavelet", "rbf_pielm_best", "xpinn", "vanilla"]:
        row = [label[m]]
        for r in rhos:
            row += [e(grab(m, r, "relL2")), e(grab(m, r, "Linf"))]
        rows.append(row)
    render(cols, rows, "Gear interface: head-to-head on the identical problem", "table_baselines.png", figw=12.5)
except FileNotFoundError:
    print("baselines_results.json not found -- skipping")

# ---------- Gap 2: Helmholtz + extreme contrast ----------
try:
    R = json.load(open("robustness_results.json"))
    cols = ["$k$", "$\\rho$", "rel $L_2$", "$L_\\infty$", "kink err"]
    rows = [[f"{h['k']:.0f}", f"{h['rho']:.0f}", e(h["relL2"]), e(h["Linf"]), e(h["kink"])]
            for h in R["helmholtz"]]
    render(cols, rows, "Helmholtz gear interface  $-\\nabla\\!\\cdot(a\\nabla u)-k^2u=f$  (banded wavelet)",
           "table_helmholtz.png", figw=10.0)

    cols2 = ["$\\rho=a^+/a^-$", "rel $L_2$", "$L_\\infty$", "kink err"]
    # prepend the published moderate-contrast rows for a full 1e-3..1e6 picture
    base = {"10": ("1.68e-03", "1.15e-03", "1.00e-02"),
            "100": ("4.98e-04", "5.63e-04", "1.21e-02"),
            "1000": ("1.02e-03", "5.06e-04", "2.11e-02")}
    ext = {f"{x['rho']:.0e}": (e(x["relL2"]), e(x["Linf"]), e(x["kink"])) for x in R["extreme_contrast"]}
    order = [("1e-03", ext.get("1e-03")), ("10", base["10"]), ("100", base["100"]),
             ("1000", base["1000"]), ("1e+04", ext.get("1e+04")), ("1e+05", ext.get("1e+05")),
             ("1e+06", ext.get("1e+06"))]
    rows2 = [[lbl, v[0], v[1], v[2]] for lbl, v in order if v is not None]
    render(cols2, rows2, "Gear interface: extreme-contrast robustness (banded wavelet W-PINN)",
           "table_contrast.png", figw=9.0)
except FileNotFoundError:
    print("robustness_results.json not found -- skipping")

print("done.")
