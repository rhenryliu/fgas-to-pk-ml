# DRAFT test-batch declaration — cvae noise-head variant (2026-07-09)

**Status: DRAFT — not in force.** Written per the task instruction to name the
single configuration that would be tested, BEFORE any deliberate test-fold
evaluation of it; the maintainer signs off (and re-dates) this declaration
before any test number is computed or read. Follows the GR6/B9 + amendment-6
I4.1 rules of `dual_vae_test_declaration.md`.

Exactly one configuration:

7. **`cvae_noise`** — the Sohn-style conditional VAE with the learned per-k
   observation-noise head (`obs_noise_head: true`; run config
   `scripts/configs/run/cvae_noise.yaml`, an exact copy of the committed
   `cvae.yaml` plus the flag) on the pinned context
   (`scripts/configs/data/config_dual_vae.yaml`, tag 20260708, R < 10 crop,
   conditioned on the 5 cosmological parameters, pinned split seed 10, model
   seed 0). Purpose: the calibration comparison row against the declared
   `cvae` row (third declaration, item 6) — same architecture, same frozen
   hyperparameters, aleatoric head on. Full-metric reporting (global +
   suppressed-regime + per-curve + coverage). One test evaluation, through
   `scripts/run.py` / `run_training`.

Rules in force (unchanged from the standing declarations):

- Selection and tuning are val-fold only and are already complete: the val
  comparison is recorded in `cvae_noise_head_val_comparison.md`. Test results
  report, never re-designate; no configuration is selected, tuned, or modified
  in response to a test number. The amendment-6 primary designation
  (Arm P, `dual_vae_pca_vae.yaml`) is unaffected regardless of outcome.
- Encoded expectation, recorded as mechanism (not outcome): at the frozen
  operating point the head is under-converged (see the val note's diagnosis),
  so **over**-coverage of the central intervals is the documented expectation
  on test, mirroring the val fold.

Transparency note: the standard `run_training` machinery mechanically computed
its held-out(test) diagnostic block for the two val-stage run records
(`20260709T094306Z__dbe723b7`, `20260709T094429Z__2890276e`), and
`scripts/run.py` prints a headline test RMSE. Those numbers were not consulted
for any selection or reporting; the deliberate, declared test evaluation of
item 7 (full metrics, coverage included) has NOT been run and awaits sign-off.
