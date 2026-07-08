# Codec ablation (Stage 5b deliverable, amendments B8 + 6/I4.2)

The dual-track Stage 4 already delivered the codec comparison on val; per
I4.2, 5b is the **test-set evaluation and write-up** of the declared
non-primary configurations, not a re-fit. Fairness constraints held: every
arm re-fit its full pipeline (codecs + mapping) on its own code space inside
the one-process `dual_vae` plugin, identical folds (pinned SplitSpec),
identical seeds (0), identical hyperparameter recipe (hidden 128 x 2, VAE
epochs 5000, map_epochs 2000, MDN K=3), identical evaluation context
(tag 20260706, R < 10 crop, nd 2, k [0.5, 5.0]). Each configuration touched
the test fold exactly once (`dual_vae_test_declaration.md`).

## Arms (test fold; full metrics as mandated — a comparison missing the
## suppressed-regime numbers is invalid)

| metric | **(pca, vae) — primary** | (vae, vae) — Arm V | (pca, pca) |
|---|---|---|---|
| run_id (all `__43ea387`) | `20260706T215627Z__cfc56666` | `20260706T220156Z__0e252abf` | `20260706T220238Z__6457b899` |
| global RMSE | **0.03129** | 0.04881 | 0.03486 |
| SP<0.95 (n=2670) | **0.04990** | 0.07702 | 0.05450 |
| SP<0.9 (n=1421) | **0.06173** | 0.09760 | 0.06720 |
| SP<0.8 (n=541) | **0.07661** | 0.13944 | 0.08837 |
| per-curve median / p90 / max | 0.0084 / 0.0520 / 0.1324 | 0.0151 / 0.0698 / 0.2661 | 0.0122 / 0.0637 / 0.1497 |
| coverage 0.68 / 0.90 / 0.95 | 0.670 / 0.865 / 0.920 | 0.689 / 0.907 / 0.934 | 0.662 / 0.887 / 0.934 |
| worst test sims | 305, 598, 109 | 598, 822, 977 | 305, 977, 598 |
| ||code|| median / p99 / max | 0.31 / 0.94 / 1.07 | 4.91 / 13.78 / 15.79 | 0.31 / 0.94 / 1.07 |

## Adjudication

1. **The x-codec drives the ablation.** Swapping the x-side from VAE to PCA
   improves every accuracy metric (global 1.56x, deep-suppression 1.82x) —
   the G4.0b gap, first seen at the probe and confirmed by the dual-track
   ladder on val, holds on test with full metrics. The F-1 deficit is a real
   property of the current VAE-X representation, not an artifact of any
   probe or fold.
2. **The y-codec choice is second-order for accuracy but favours the VAE.**
   (pca, vae) beats (pca, pca) by ~10% globally and ~13% at SP<0.8: the
   learned y-manifold decodes better than a linear 3-component one, and its
   learned noise head calibrates slightly better at 68% (0.670 vs 0.662).
   The VAE-Y's role in the architecture (Stage 2's replication results,
   latents, obs_sigma) is therefore retained at no accuracy cost — it is
   the x-side where reconstruction-trained VAE codes underperform.
3. **Calibration is healthy in all three arms** (within ~3.5 pp of nominal
   everywhere); no arm reproduces the CVAE under-coverage — now a
   same-context measurement, not a historical precedent: the genuine `cvae`
   on this exact context covers 0.428/0.716 at nominal 0.68/0.95
   (`20260708T231014Z__fdbfd21b__864001d`; third test declaration).
4. **OOD:** the PCA x-codes are norm-bounded on test; Arm V's VAE codes
   reach 15.8 (attenuated echo of the CVAE precedent) without creating
   unique worst-curves. The shared hard sims (598, 977, 305) fail in every
   arm; **V2 evidence check:** they are NOT parameter-corner outliers (598
   and 977 carry a single mildly extreme parameter each; 305 several), but
   they ARE the suite's most deeply suppressed targets — min SP(k) of
   0.438 / 0.620 / 0.545, depth percentiles 0.1 / 3.1 / 1.4 of 1024. The
   correct label is therefore "the sparse deep-suppression tail of the
   target distribution, hard in every arm", not codec-induced and not an
   unevidenced "physics-hard".
5. **Tail coverage (V3):** all three arms under-cover consistently at 95%
   (0.920 / 0.934 / 0.934, i.e. ~2-3 pp low) while being near-nominal at
   68% — the signature of slightly light tails, plausibly from the
   diagonal-covariance MDN mixtures; observed and noted, no fix in this
   programme.

**G5b: PASS** — report complete; both mandatory arms evaluated under
identical conditions. The comparison IS the deliverable; no accuracy
threshold applies.
