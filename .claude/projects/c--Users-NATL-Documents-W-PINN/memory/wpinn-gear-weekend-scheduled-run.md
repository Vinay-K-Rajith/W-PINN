---
name: wpinn-gear-weekend-scheduled-run
description: Unattended weekend job (scheduled 2026-06-20) finishing the 3 CAMWA gap studies + auto git push
metadata:
  type: project
---

On 2026-06-20 (Sat) set up the three CAMWA gap-closing studies for the gear interface (paper #1,
see [[wpinn-gear-complex-test]]) and scheduled unattended completion since the user left for the
weekend, back Monday. New scripts in `Elliptic_Interface/Gear_Interface/`:
  - `convergence_study.py` -> convergence_results.json + convergence.png (Gap 3: relL2/cond vs band levels)
  - `baselines_gear.py` -> baselines_results.json (Gap 1: wavelet vs single-scale RBF-PIELM via IDENTICAL
    decomposed assembly, vs XPINN-MLP, vs vanilla-MLP, on the SAME gear problem, rho=10 & 1000)
  - `helmholtz_robustness.py` -> robustness_results.json (Gap 2: Helmholtz gear -div(a grad u)-k^2 u=f,
    k=2/4/8; + extreme contrast rho 1e-3..1e6)
  - `make_paper_tables.py` -> table_convergence/baselines/helmholtz/contrast.png (220dpi)
  - minor backward-compatible edit to `run_gear.py` solve(): added `want_cond` -> cond in metrics.

Partial live results before scheduling (all matched published numbers): convergence coarse[0123] relL2
2.25e-2 cond 3.4e19 -> +band[4] 3.37e-3 -> +band[4,5] 1.68e-3 cond 4.8e20 (band[4,5,6] = conditioning-
ceiling test, was still running). Baselines rho=10: wavelet 1.68e-3 vs best single-scale RBF-PIELM ~9e-2
(eps=5) = ~53x win (thesis confirmed on the gear).

**Unattended mechanism:** Windows Scheduled Task `WPINN_weekend_run` (one-time, fired 2026-06-20 12:30)
runs `weekend_run.ps1` -> runs each study only if its JSON is missing (preserves finished ones) ->
make_paper_tables.py -> `git add -A` + commit + `git push origin main` (HTTPS, GCM creds cached, verified).
Self-deletes the task; writes `WEEKEND_DONE.txt` + `weekend_run.log`. Independent of Claude/VS Code windows.

**MONDAY TODO:** check `Elliptic_Interface/Gear_Interface/WEEKEND_DONE.txt` (push exit=0 means pushed) and
`weekend_run.log`; then write `results.md` sections for the 3 studies + the literature comparison table
(DCSNN JCP 2022, cusp-capturing PINN JCP 2023, I-PINN CMAME 2024 -- cite their own-benchmark numbers,
attributed; the rigorous head-to-head is the own-run baselines). Still need the convergence figure +
tables wired into results.md. CAMWA accept odds discussed: ~50% base rate, topic in-scope.
