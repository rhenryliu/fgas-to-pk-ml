# Dual-VAE Stage 2 gate report — GATE FAILED, STOPPED

Staged spec v2, Stage 2 (VAE-Y on SP(k); Lin et al. 2026, arXiv:2509.01881,
methodological replication). Implementation committed at `df58268` (component +
stage script) and `ade9e26` (CLI capacity overrides) before the corresponding
runs, so all records carry clean git SHAs. Per GR5 and G2.1's failure clause,
**work stops at this gate**; no Stage 3 work was started.

## Evidence: three recorded rounds (12 runs)

All rounds: pinned context (`config_dual_vae.yaml`), pinned split, train-fold
fit, val-fold scores, seed 0, beta = 1 with the learned per-bin noise head
(F0.2), 5 cosmological params as context (F0.4). "PCA" is PCA-on-Y at the
matched dimension on the identical folds (Stage 1, bitwise-reproducible).

Round 1 — 1000 epochs, hidden 128 x 2 (undertrained: best_epoch at schedule end):

| ld | val recon RMSE | PCA | A_j (train) | run_id |
|---|---|---|---|---|
| 2 | 0.00551 | 0.00287 | [93.3, 0.78] | `20260705T211326Z__a0c1f0fa__df58268` |
| 3 | 0.00505 | 0.00134 | [101.2, 0.00, 0.80] | `20260705T211327Z__cece324a__df58268` |
| 4 | 0.00473 | 0.00100 | [0.0, 0.0, 78.3, 0.71] | `20260705T211328Z__9f14f9ca__df58268` |

Round 2 — 5000 epochs, hidden 128 x 2 (**converged**: best_epoch ~2000/5000;
the best round, its ld=3 run is the selected VAE-Y):

| ld | val recon RMSE | PCA | A_j (train) | run_id |
|---|---|---|---|---|
| 2 | 0.00475 | 0.00287 | [2012.7, 10.6] | `20260705T212217Z__9e5ab437__df58268` |
| 3 | **0.00451** | 0.00134 | [1966.6, 0.0, 6.3] | `20260705T212219Z__3e6622fe__df58268` (selected) |
| 4 | 0.00460 | 0.00100 | [0.0, 0.0, 1721.7, 9.4] | `20260705T212220Z__bcf9747d__df58268` |

Round 3 — capacity check: 5000 epochs, hidden 256 x 3 layers, lr 5e-4
(**strictly worse** — capacity is not the bottleneck):

| ld | val recon RMSE | PCA | A_j (train) | run_id |
|---|---|---|---|---|
| 2 | 0.01069 | 0.00287 | [0.40, 153.7] | `20260705T213234Z__49b7435e__ade9e26` |
| 3 | 0.00926 | 0.00134 | [0.52, 0.0, 159.5] | `20260705T213236Z__bce9e1dc__ade9e26` |
| 4 | 0.00732 | 0.00100 | [0.68, 0.0, 184.4, 0.0] | `20260705T213236Z__72b445ad__ade9e26` |

## Gate verdicts

**G2.1 — VAE-Y <= PCA-on-Y at matched dimension: FAIL.** Best VAE-Y (round 2)
is 1.65x worse than PCA at dim 2 (0.00475 vs 0.00287) and 3.4x worse at dim 3
(0.00451 vs 0.00134). The plateau at ~0.0045 across ld in {2, 3, 4} and across
schedules, with best_epoch well inside the schedule and a capacity increase
making things worse, indicates an ELBO fixed point rather than an optimization
failure: the fitted obs_sigma (raw 0.0007-0.0059 per k bin) absorbs exactly the
residual the decoder does not explain, the KL charges ~1.3-1.7 nats for the
second (weakly-active) dimension whose posterior blur (sigma_q ~ 0.3-0.5)
limits mean-decode precision, and training reconstructs from sampled z rather
than mu. In raw terms the VAE-Y still captures ~99.7% of the val target's
0.083 spread — but a plain linear code does strictly better at every matched
dimension, which is precisely what this gate exists to catch.

**G2.2 — activity 1 <= A_j <= 10 on every retained dimension: FAIL.** The
dominant dimension is delta-like in every converged run (A_0 ~ 1700-2000,
posterior essentially deterministic); the second active dimension sits at the
band's edge (6.3-10.6); extra dimensions collapse outright (A ~ 0, KL ~ 0.005
nats). No configuration produced a latent with all retained dimensions inside
the band. Per-dimension traces: `*__latent_traces.png` figures for all 12 runs
(figures/2026-07/07-05/, gitignored, regenerable).

**G2.3 — Lin-replication checks (selected run `20260705T212219Z__3e6622fe__df58268`): reported; one outright disagreement.**

1. *Reconstruction vs PCA at matched dimension (their Fig. 12 analogue):*
   **disagree.** Lin et al. report VAE reconstructions competitive with or
   better than PCA at matched dimension; here PCA wins at every dimension by
   1.7-3.4x. Plausible sources of the difference, flagged for human review,
   not resolved: this target (the Gen 2 P_total/P_DM proxy over k in
   [0.5, 5.0]) is close to linear in two factors (Stage 1: two PCA components
   reach 3.5% of the target spread), leaving little for a nonlinear code to
   add; and this implementation trains at beta = 1 with a learned noise head
   whereas their training details differ.
2. *Dimensionality behaviour across {2, 3, 4} (their Fig. 5 analogue):*
   **agree.** Adding a 3rd/4th dimension does not add active dimensions: the
   active count stays at 2 (one dominant + one weak), the extras collapse to
   KL ~ 0. The effective latent dimensionality of SP(k) given the cosmological
   context is ~2 on this data, matching their qualitative finding of a small
   number of meaningful latents.
3. *Latent-parameter correlations over all 35 SB35 parameters (their Fig. 1
   analogue; val fold):* **agree.** The active dimensions organize by feedback,
   not cosmology: mu2[0] correlates most with VariableWindVelFactor (+0.30),
   BlackHoleRadiativeEfficiency (-0.30), VariableWindSpecMomentum (+0.29),
   WindDumpFactor (-0.25); mu2[2] with VariableWindVelFactor (+0.39) and
   WindFreeTravelDensFac (-0.38). Cosmological correlations are weaker
   (|r| <= 0.21), as expected since the model is conditioned on the 5
   cosmological parameters. Matrix and figure:
   `latent_param_correlations` in the selected run's summary and
   `*__latent_param_corr.png`.

One outright disagreement (< 2), so G2.3 itself does not trigger the
human-review halt — but G2.1 and G2.2 do.

**G2.4 — reduction-convention audit: PASS.** Every run summary carries the
GR7 statement (NLL summed over k bins, KL summed over latent dimensions, both
batch-averaged; ~0.5/exp(obs_logvar_b) per bin vs beta = 1 per latent
dimension); `metrics.jsonl` traces are consistent with it.

## Status and options (maintainer decision required — spec veto rule)

Stopped per GR5. The spec's own reading of a G2.1 failure is "the VAE adds
nothing over a linear code" for reconstruction on this target. Options that
would need a human edit to the spec before any further Stage 2+ work:

1. Accept the finding and re-scope: SP(k) here is ~2-factor linear; a PCA (or
   probabilistic PCA) y-code with the same z1 -> z2 ladder would preserve the
   architecture with a code that G2.1 shows is strictly better.
2. Revisit the G2.2 band: with a learned noise head at beta = 1, a
   high-precision dominant dimension (A >> 10) may be the *correct* behaviour
   for a near-deterministic target; the [1, 10] band imports Lin et al.'s
   regime, which this target may simply not inhabit.
3. Deliberate ELBO changes (beta < 1, fixed obs noise, or IWAE-style bounds)
   — these leave the F0.2 resolved decision and would need it re-opened.
4. Proceed-with-caveat is NOT taken unilaterally: G2.1's failure clause is
   explicit ("stop").

The Lin-replication *scientific* findings (2-factor latent structure organized
by wind/AGN feedback) are positive results independent of the gate outcome.
