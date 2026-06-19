# Memory Index

- [W-PINN application decision](wpinn-application-decision.md) — CHOSEN novel application: elliptic interface (discontinuous-coefficient) problems; first-principles formulation + 3-phase plan
- [W-PINN Phase 1 results](wpinn-phase1-results.md) — Phase 1 PASS: decomposed wavelet exact & contrast-robust, beats XPINN ~6 orders at high contrast; conditioning is the Phase 2 obstacle
- [W-PINN Phase 2 results](wpinn-phase2-results.md) — Phase 2 100% PASS: 2D circle+flower(mesh-free) relL2~1e-6 contrast-robust; conditioning solved by parsimony; beats XPINN ~140x-5000x in 2D
- [W-PINN Phase 3 prep](wpinn-phase3-prep.md) — inverse-contrast PoC PASSES via outer-loop+inner-LSQ (rho to ~1e-9 noiseless); joint-Adam fails; strategy transfers to interface-location recovery
- [W-PINN gear complex test](wpinn-gear-complex-test.md) — harder-than-flower gear/settings-icon interface; windowed level set for deep teeth; banded refinement drops error ~13x (2.2e-2->1.7e-3); paper #2 = integro-differential/nonlocal (AD-free wavelet op-matrix encodes integrals)
- Paper #2 folder+plan created (2026-06-19): Integro_Differential/integro_differential_wpinn_plan.md — W-PINN for integro-differential/nonlocal PDEs; thesis = operational matrix encodes the INTEGRAL (AD-free genuinely wins, can't autodiff an integral); new machinery = integral op-matrix G[m,i]=∫K(x_m,t)ψ_i(t)dt
- [Paper #2 integro verdict](wpinn-paper2-integro-verdict.md) — PARKED: 1D Volterra IDE mechanism works but lane crowded (PINNIES/Rahimkhani/peridynamic-horizon)
- [Paper #2 CHOSEN: multiscale](wpinn-paper2-multiscale-chosen.md) — after exhaustive search, field saturated for single-feature linear problems (PIELM/XPIELM/SEPINN own linear-solve + enrichment); W-PINN's durable edge = MULTIRESOLUTION ADAPTIVITY. App = multi-scale localized-feature problems + inverse recovery of UNKNOWN number of sources at unknown scales (sparsity). Folder Multiscale_Sources/; Phase 0+1 PASS (banded refinement 270x better than coarse on a 2-scale problem)

## Packaging (2026-06-17)
- Canonical-layout folders (config/Wfamily/Model/<Problem>/WPINN_*.ipynb/sol.png) built + verified: Elliptic_Interface/Circular_Interface (relL2 1.5e-5), Flower_Interface (4.7e-5, mesh-free), Inverse (shape recovery (0.4999,0.3008) vs (0.5,0.30)). Exploratory phase1_1d/phase2_2d/phase3_inverse kept as working record. Novelty narrowed: mechanism×application (AD-free wavelet operational-matrix W-PINN on interface problems + inverse shape recovery); MLP interface-PINN space crowded. Target CAMWA/AMC (Elsevier, elsarticle.cls). Viva prep: Elliptic_Interface/VIVA_QA.md.
- [W-PINN fractional prior-art](wpinn-fractional-prior-art.md) — why fractional was rejected: crowded prior art, and AD-free isn't the real selling point (smoothing is)
