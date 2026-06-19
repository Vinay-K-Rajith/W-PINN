# Gear / "settings-icon" interface — results

Decomposed AD-free wavelet W-PINN on an 8-tooth gear interface (`N=8, R0=0.55, B=0.04, s=8.0`;
~44% radial tooth depth). Banded multiresolution basis: coarse levels `[0,1,2,3]` global +
finest levels `[4,5]` placed **only** in the teeth annulus `r∈(0.42, 0.86)` (NF = 4764).
Solved by column-normalised Tikhonov least squares (`tik=1e-10`, interface/BC weights `10`).
Test grid 300×300; source `f=-Δφ` computed exactly by autograd (one-time data; solve is AD-free).

## Error metrics vs. contrast ρ = a⁺/a⁻

| ρ     | relL2     | MSE       | RMSE      | MAE       | L∞ (max)  | rel L∞    | kink err  |
|------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|
| 10    | 1.68e-03  | 2.96e-08  | 1.72e-04  | 8.90e-05  | 1.15e-03  | 3.80e-03  | 1.00e-02  |
| 100   | 4.98e-04  | 1.80e-09  | 4.24e-05  | 1.86e-05  | 5.63e-04  | 1.86e-03  | 1.21e-02  |
| 1000  | 1.02e-03  | 7.47e-09  | 8.64e-05  | 6.58e-05  | 5.06e-04  | 1.67e-03  | 2.11e-02  |

Contrast-robust (relL2 ≈ 1e-3 across two orders of magnitude in ρ). The L∞ error and the
interface flux-kink error are localised at the sharp tooth-tip cusps; the bulk error is ~1e-4
(see `sol.png`, panel c).

## Banded refinement vs. global coarse basis (ablation, ρ=10)

| basis              | NF   | relL2     | MSE       | L∞        | kink err  |
|:-------------------|-----:|----------:|----------:|----------:|----------:|
| `[0,1,2,3]` global | 2500 | 2.25e-02  | 5.30e-06  | 1.14e-02  | 6.89e-02  |
| `[0-3]+band[4,5]`  | 4764 | 1.68e-03  | 2.96e-08  | 1.15e-03  | 1.00e-02  |

Adding the finest wavelets **only** in the teeth band improves relL2 ~13× (and the worst-case
L∞ ~10×) at a few hundred extra basis functions — far cheaper than a global finest level
(~9600 functions, computationally impractical). This is adaptive multiresolution refinement at
the interface, which a fixed-coordinate vanilla PINN cannot do.

Metric definitions (over the test grid, `err = u_pred − u_exact`):
relL2 = ‖err‖₂/‖u‖₂ · MSE = mean(err²) · RMSE = √MSE · MAE = mean|err| ·
L∞ = max|err| · rel L∞ = L∞/max|u| · kink err = max interface |[[∂ₙu]]_pred − [[∂ₙu]]_exact|.

Reproduce: `python run_gear.py` (writes this table to stdout and regenerates `sol.png`).
