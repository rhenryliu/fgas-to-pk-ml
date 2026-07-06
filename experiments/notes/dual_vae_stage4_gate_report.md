# Stage 4.0 resolution (amendment 6, 2026-07-06): dual-track ladder

The maintainer resolved the G4.0b STOP (which stands below as recorded
history) with a dual-track Stage 4: Arm V (codec_x = vae, ld=3, beta=0.01)
and Arm P (codec_x = pca, d=4), full ladder each, primary designation by the
I3 criterion fixed **before** the ladder numbers exist (val rung-3 dominance
on global RMSE + suppressed-regime RMSE at all N >= 30 thresholds + pooled
coverage distance at 68%/95%; split verdict -> STOP). The B9 test batch is
declared as Arm V rung-3, Arm P rung-3, and (pca, pca).

## I1 — suppressed-regime breakdown of the probe arms (context, not a gate)

Deterministic refit of the three arms under the identical protocol
(`scripts/stage4_0_i1_breakdown.py`); every recomputed global mean matches
the recorded value exactly. Val y-space RMSE, truth-masked:

| arm | global | SP<0.95 (n=1240) | SP<0.9 (n=671) | SP<0.8 (n=234) |
|---|---|---|---|---|
| ceiling (full profile) | 0.03635 | 0.05660 | 0.07273 | 0.10181 |
| pca_scores_d4 | 0.03229 | 0.04949 | 0.06360 | **0.07971** |
| selected VAE (ld=3, b=0.01) | 0.05349 | 0.08795 | 0.11461 | 0.17432 |

The probe-level picture is uniform across regimes: pca_scores_d4 leads the
VAE arm at every threshold (2.2x at SP<0.8) and — notably — beats the
unrestricted ceiling in the deep-suppression regime (0.0797 vs 0.1018),
suggesting the whitened 4-dim compression acts as a useful regularizer for
the probe in the tail. Recorded as context; the I3 criterion decides the
primary from the ladder, not from this table.

---

# [historical STOP record — resolved by amendment 6] Dual-VAE Stage 4.0 gate report — G4.0a PASS, G4.0b STOP (ratio 1.66 > 1.25)

Amendment-5 probe protocol (H3), recorded run
`20260706T205815Z__e90f52b8__a6dab80` (implementation commit `a6dab80`
preceded it). Fixed rung-2 MLP probe (hidden 128 x 2, 2000 epochs, lr 1e-3,
internal holdout best-epoch restore), 3 seeds per arm, metric = mean val
y-space RMSE through the frozen VAE-Y decoder
(`20260706T190601Z__8a715039__8522dd4`). Tag 20260706, R < 10 crop, pinned
folds. Arm figure: `figures/2026-07/07-06/*__probe_selection__probe_arms.png`.

## Arm table (mean over 3 seeds; per-seed in the run summary)

| arm | probe val y-RMSE |
|---|---|
| **ceiling: full 16-bin profile** | **0.03635** |
| pca_scores d=2 | 0.05680 |
| pca_scores d=3 | 0.03805 |
| **pca_scores d=4 (best reference)** | **0.03229** |
| pca_scores d=6 | 0.03559 |
| best VAE candidate: ld=3, beta=0.01 (`20260706T200039Z__3080064e__8cbe5b2`) | 0.05349 |
| (VAE candidates span) | 0.05349 - 0.06391 |

## Gate verdicts

- **G4.0a — architecture viability: PASS.** Ceiling 0.03635 <
  `pca_linear` 0.038537 (`20260706T185703Z__a760845e__a4ba5aa`). The
  unrestricted nonlinear two-stage pathway beats the simplest direct
  baseline — by 6%.
- **G4.0b — codec adequacy: STOP.** Selected candidate (H3.4 tiebreak:
  ld=3, beta=0.01, probe 0.05349) / best reference (pca_scores_d4, 0.03229)
  = **1.656 > 1.25**. Structural tripwire; maintainer decision required.

## H3.6 findings (reported, no gates, no encoded expectations)

1. **F-1 re-measurement — the deficit is real, not a probe artifact.**
   Under the nonlinear probe, VAE-X codes still trail PCA scores at every
   d >= 3: ratios 1.41 (d=3), 1.82 (d=4), 1.52 (d=6); parity again only at
   d=2 (0.99), where both arms are information-starved. The linear-probe
   caveat is discharged: **the VAE-X objective (reconstruction + rate,
   heteroscedastic NLL) genuinely discards mu2-relevant information that a
   plain linear projection retains.** A plausible mechanism, flagged as
   interpretation: the learned per-bin obs_logvar downweights exactly the
   low-variance profile directions that carry feedback information, whereas
   PCA keeps them (cf. F-2 in `dual_vae_findings.md`).
2. **Ceiling-vs-selected gap:** 1.47x (0.05349 / 0.03635) — the information
   lost to the chosen x-compression, pre-mapping.
3. **Ceiling-vs-`mlp_regressor` gap:** 1.88x (0.03635 / 0.019378) — the cost
   of routing through the two-stage architecture at all (z-space objective +
   frozen decoder), even with no x-compression. Together with G4.0a's thin
   6% margin, this quantifies how much the F0.5 posterior-mean design
   sacrifices relative to direct regression on this data — the number
   Stage 6's fine-tune (and the Stage 5b ablation) exist to interrogate.
4. Seed spread is small for the decisive arms (pca_d4: 0.0321-0.0325;
   ceiling: 0.0354-0.0376); pca_d3/d6 show one high seed each (0.042),
   consistent with occasional probe-fit variance — the 3-seed mean is doing
   its job.

## Status

STOPPED per H3.5 (G4.0b > 1.25). Not proceeding to the Stage 4 ladder. The
natural options are the maintainer's to choose, but the arm table makes one
observation unavoidable: **pca_scores_d4 is currently the best available
x-codec under the pipeline's own selection instrument**, and the
codec-agnostic `dual_vae` plugin (B7) already supports `codec_x: "pca"`
end-to-end — the Stage 5b machinery could adjudicate a (pca_x, vae_y)
configuration with full metrics whenever the spec permits it as more than an
ablation arm.
