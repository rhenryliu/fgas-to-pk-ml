# Test-batch declaration (GR6/B9 + amendment 6 I4.1) — fixed BEFORE any test evaluation

Committed prior to any test-fold number being computed for any dual-codec
configuration. The test batch comprises exactly three registered `dual_vae`
configurations, all on the pinned context (tag 20260706, R < 10 crop, pinned
SplitSpec) and all rung-3 (MDN mapping, K = 3):

1. **Arm V** — `codec_x: vae` (latent_dim_x = 3, beta_x = 0.01; the H3.4
   selection), `codec_y: vae` (latent_dim_y = 3, beta_y = 1e-4).
   Run config: `scripts/configs/run/dual_vae_vae_vae.yaml`.
2. **Arm P** — `codec_x: pca` (d = 4), `codec_y: vae` (as above).
   Run config: `scripts/configs/run/dual_vae_pca_vae.yaml`.
3. **(pca, pca)** — `codec_x: pca` (d = 4), `codec_y: pca` (d = 3, matching
   the selected VAE-Y dimensionality). Run config:
   `scripts/configs/run/dual_vae_pca_pca.yaml`.

Rules in force:

- The **primary configuration** is designated from val evidence only (the
  I3 dominance criterion on the dual-track Stage 4 ladders) BEFORE any test
  number is seen; test results report, never re-designate.
- Stage 5 runs the primary through `scripts/run.py`; Stage 5b evaluates the
  other two declared configurations (its fairness constraints and
  full-metric reporting — global + suppressed-regime + per-curve + coverage
  — stand). Each configuration touches the test fold exactly once.
- No configuration is selected, tuned, or modified in response to a test
  number.

Primary designation (entered 2026-07-06 after the I3 criterion was applied
to the val-fold ladders, BEFORE any test evaluation of any declared
configuration): **Arm P — `dual_vae_pca_vae.yaml`**, by full I3 dominance
(global, all suppressed thresholds, both coverage distances; see
`dual_vae_stage4_gate_report.md`). Arm V and (pca, pca) proceed as the
Stage 5b arms.

---

# Second and final test batch (amendment 7, J1) — fixed BEFORE either run

Committed prior to any test-fold number existing for either configuration.
Exactly two new configurations; each evaluated on test exactly once, only
after its val-fold gates clear; no other configuration touches test; the
amendment-6 primary designation is unaffected regardless of outcomes:

4. **`dual_vae_ft`** — the Stage 6 two-phase fine-tune of Arm V
   (codec_x = vae ld=3 beta_x=0.01, codec_y = vae ld=3 beta_y=1e-4, frozen
   decoder-Y incl. obs_logvar; MDN K=3 re-fit in Phase 2). Val gates first
   (G4.2 in full; calibration-regression vs Arm V; OOD ||mu1'||; G4.3).
   Run config: `scripts/configs/run/dual_vae_ft.yaml`.
5. **`vib_regressor`** — the CVAE comparison row on the corrected pinned
   context (existing run config, pinned split; J4). Point-accuracy metrics
   only in the notes; its sampling spread carries no valid uncertainty
   semantics.

Stage 6 recoverability outcome: no expectation encoded (J2.4).
