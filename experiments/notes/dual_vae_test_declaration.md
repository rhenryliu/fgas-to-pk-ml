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

Primary designation: **PENDING** — to be inserted here after the I3
criterion is applied to the val-fold ladder results (and before any test
run). No expectation about the outcome is encoded (I6.2).
