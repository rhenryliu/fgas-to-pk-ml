# Dual-VAE programme — CLOSED (amendment 7, J7.2; 2026-07-06)

Stages 1-6 complete. The spec plus all seven maintainer amendments are the
historical record (`docs/dual_vae_staged_spec.md`); the consolidated
documentation of the final state is `docs/dual_vae_doc.md`.

- **Recommended model:** `dual_vae`, primary configuration (PCA-x d=4,
  VAE-y ld=3 beta=1e-4, MDN K=3); test run
  `20260706T215627Z__cfc56666__43ea387`.
- **Test budget:** six declared configurations across three declarations,
  each evaluated exactly once (three in the amendment-6 batch;
  `vib_regressor` in the amendment-7 batch; the genuine `cvae` in a
  maintainer-requested post-close-out addendum, 2026-07-08 — most accurate
  in the table at 0.01254 but 23-26 pp under-covered, run
  `20260708T231014Z__fdbfd21b__864001d`); `dual_vae_ft`'s declared
  evaluation was conditional on val gates and did not run (calibration
  regression).

**Open items carried forward** (a future programme, not an amendment):

1. Per-halo dataset (the codec verdict may not transfer to noisier per-halo
   profiles — recorded caution in `dual_vae_doc.md` section 7).
2. Input-noise propagation into the predictive spread (reconnects at the
   F0.5 boundary documented in `composite_samples`).
3. Paired-DMO suppression target (the within-hydro proxy's
   feedback-correlated bias at k >~ 5 h/Mpc; `env.json` provenance note in
   every run).
4. NPE-in-observable-space via `sbi`.
5. The parked noise-head hypothesis for F-1 (`dual_vae_findings.md`).
6. Rebuilds of the snap82 / other-rank variants under the corrected
   statistic (stale-variant warning in `known_issues.md`).

Per J7.3: any further work is a new programme.
