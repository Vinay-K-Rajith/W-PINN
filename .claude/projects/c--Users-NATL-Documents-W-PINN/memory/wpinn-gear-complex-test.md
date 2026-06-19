---
name: wpinn-gear-complex-test
description: The "much harder than flower" gear/settings-icon interface test + banded-refinement finding
metadata:
  type: project
---

Goal (2026-06-19): a MUCH more complex interface than the 3-lobe flower, per user request = a **gear / "settings-icon" shape** (8 sharp teeth). New folder Elliptic_Interface/Gear_Interface/ (canonical layout pending; prototype = _prototype_gear.py). Part of [[wpinn-application-decision]] paper #1 — closes the "harder geometry / credibility" gap.

**Geometry finding (first-principles):** a HARMONIC phi (which gave the flower exact f=-4) CANNOT make real teeth — r^N is suppressed for r<1, so harmonic gears top out at ~3% tooth depth (just ripples); pushing amplitude makes the level set self-intersect. FIX: a globally-smooth WINDOWED level set
  phi = (x^2+y^2) - R0^2 + B*Re((x+iy)^N)/R0^N * exp(-s*((x^2+y^2)-R0^2)^2)
The Gaussian window restores full tooth amplitude at r=R0 (depth set by B, no r^N suppression) and decays away (no runaway). Still smooth => u=phi/a keeps homogeneous jumps [[u]]=0,[[a d_n u]]=0. Cost: source f=-Delta phi is now spatially VARYING — computed EXACTLY by autograd on phi at the fixed collocation points (one-time problem data; the wavelet SOLVE stays AD-free). Tuned valid params: N=8, R0=0.55, B=0.04, s=8.0 -> single closed curve, tips r~0.785 (margin to box wall), valley r~0.525, ~44% depth.

**KEY RESULT — banded multiresolution refinement:** the gear is genuinely hard for the coarse global basis. Levels [0,1,2,3] global (NF=2500): relL2 ~2.25e-2, contrast-robust (rho=10 & 1000 both ~2.2e-2), Linf ~1e-2, kinkErr ~7e-2. Adding finest wavelets ONLY in the teeth annulus (band(0.42,0.86), levels [4,5]; NF=4764) drops relL2 to **1.70e-3 (~13x better)**, MSE 3e-8, Linf 1.2e-3, kinkErr 1e-2 at rho=10. This is the methodological headline: ADAPTIVE multiresolution refinement at the interface — cheap (few hundred extra fns vs ~9600 for a global finest level) and exactly what wavelets are for; a vanilla PINN can't do it. Full global level-4 tensor product is impractical (~21600x19200 dense SVD, killed after 8min).

PACKAGED (2026-06-19, user choices: "lock ~1e-3, package now" + "clean runner + sol.png"). Folder Elliptic_Interface/Gear_Interface/ = run_gear.py (self-contained: geometry+autograd f, banded family, AD-free LSQ solve, full metric suite, figure) + sol.png. Scratch (_prototype/_scan*) deleted. FINAL banded[4,5] (NF=4764) contrast sweep, full metric SUITE per user request:
  rho=10:   relL2 1.68e-3, MSE 2.96e-8, RMSE 1.72e-4, MAE 8.90e-5, Linf 1.15e-3, relLinf 3.80e-3, kinkErr 1.00e-2
  rho=100:  relL2 4.98e-4, ... Linf 5.63e-4, kinkErr 1.21e-2
  rho=1000: relL2 1.02e-3, MSE 7.47e-9, Linf 5.06e-4, relLinf 1.67e-3, kinkErr 2.11e-2
Contrast-robust (~1e-3 across 2 orders of rho). sol.png = 3-panel (exact | W-PINN | log10|error|); error localised at the 8 tooth tips (the sharp cusps), bulk ~1e-4. Each banded solve ~270s.

CAVEATS: 1.0e-3 < flower's 5e-6 — the gear is honestly harder (sharp tooth-tip cusps); could push with band[4,5,6] at more cost (deferred). Solve ~270s.

PROMOTED TO FULL CANONICAL LAYOUT (2026-06-19, user req "match canonical form"): Gear_Interface/ now mirrors Flower_Interface exactly = config.py (windowed gear geometry + autograd helpers lap_grad), EllipticInterface.py (problem def; variable f_in/f_out via autograd, jump_dn), Wfamily.py (BANDED family COARSE[0,1,2,3]+FINE[4,5] in BAND(0.42,0.86) + op-matrices), Model.py (copy), WPINN_GearInterface.ipynb (4 cells like flower; executed via `python -m nbconvert`, reproduces relL2=1.015e-3 at rho=1000), sol.png. PLUS run_gear.py (self-contained runner), results.md (error table), make_table.py -> table.png + table_ablation.png (paper-ready 220dpi table images). Note: jupyter not on bash PATH; use `python -m nbconvert`.

RELATIVE ERROR IN %: relL2 ~0.1% (0.05% best @rho=100, 0.17% worst @rho=10); relLinf (worst point, at tooth-tip cusp) ~0.17-0.38%. Global-coarse ablation was 2.3%. My rating: GOOD/publishable PINN-class result (not classical/FEM 1e-6 regime); beats measured XPINN 0.63% on the EASIER circle; frame vs PINN baselines not FEM; emphasize contrast-robustness; be upfront error localises at cusps (motivates banded refinement).
