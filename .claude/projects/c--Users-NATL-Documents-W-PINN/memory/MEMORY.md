# Memory Index

- [W-PINN application decision](wpinn-application-decision.md) — CHOSEN novel application: elliptic interface (discontinuous-coefficient) problems; first-principles formulation + 3-phase plan
- [W-PINN Phase 1 results](wpinn-phase1-results.md) — Phase 1 PASS: decomposed wavelet exact & contrast-robust, beats XPINN ~6 orders at high contrast; conditioning is the Phase 2 obstacle
- [W-PINN Phase 2 results](wpinn-phase2-results.md) — Phase 2 100% PASS: 2D circle+flower(mesh-free) relL2~1e-6 contrast-robust; conditioning solved by parsimony; beats XPINN ~140x-5000x in 2D
- [W-PINN Phase 3 prep](wpinn-phase3-prep.md) — inverse-contrast PoC PASSES via outer-loop+inner-LSQ (rho to ~1e-9 noiseless); joint-Adam fails; strategy transfers to interface-location recovery

## Packaging (2026-06-17)
- Canonical-layout folders (config/Wfamily/Model/<Problem>/WPINN_*.ipynb/sol.png) built + verified: Elliptic_Interface/Circular_Interface (relL2 1.5e-5), Flower_Interface (4.7e-5, mesh-free), Inverse (shape recovery (0.4999,0.3008) vs (0.5,0.30)). Exploratory phase1_1d/phase2_2d/phase3_inverse kept as working record. Novelty narrowed: mechanism×application (AD-free wavelet operational-matrix W-PINN on interface problems + inverse shape recovery); MLP interface-PINN space crowded. Target CAMWA/AMC (Elsevier, elsarticle.cls). Viva prep: Elliptic_Interface/VIVA_QA.md.
- [W-PINN fractional prior-art](wpinn-fractional-prior-art.md) — why fractional was rejected: crowded prior art, and AD-free isn't the real selling point (smoothing is)
