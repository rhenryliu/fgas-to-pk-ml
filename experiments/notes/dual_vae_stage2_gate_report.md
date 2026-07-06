# Dual-VAE Stage 2 gate report — tag 20260706 (Branch A): G2.1' FAIL, STOPPED

**C3 conditional status resolved (E5.3): the definitive G2.1' evaluation
against the clean-data bar is a FAIL.** Full 12-run sweep re-run on tag
`20260706` (implementation commit `b45bf84` preceded the runs).

- **E5.2 consistency check: PASS, exactly.** Every configuration reproduced
  its 20260702 numbers bitwise (selected ld=4 beta=0.001: val recon RMSE
  0.001969, per-dim KL [4.68, 4.06, 13.07, 6.95], best_epoch 2515) — as it
  must, since VAE-Y consumes y and X_params only and E3.2 asserted both
  unchanged. No data anomaly.
- **G2.1' — codec adequacy: FAIL.** Selected val recon RMSE 0.001969 > bar
  0.0017012 (10% of the clean-data best baseline, `mlp` 0.017012, run
  `20260706T061104Z__8fc38d33__3e94bcc`). The codec did not get worse — the
  end-to-end baseline got 2.5x better once the corrupted X was fixed, so the
  same y-codec truncation error is now 11.6% of the best pipeline error.
  Runs: `20260706T064012Z...064020Z__*__b45bf84` (12); selected
  `20260706T064020Z__3ff03e76__b45bf84`.
- **G2.2' — no collapse: PASS** (as before). **G2.4: PASS.** **G2.3**
  verdicts carry over verbatim (identical fits and correlations on identical
  y): disagree / partially agree / partially agree.

**Stopped per GR5.** Stage 3 (E6) not started on this gate state. Options
that need a maintainer edit: (a) accept 11.6% and relax G21_FRACTION or fix
an absolute bar; (b) extend the resolved sweep grid (F0.2/F0.3: e.g.
beta 1e-4 and/or latent_dim_y in {6, 8} — the beta trend 0.01 -> 0.001 gave
0.00240 -> 0.00197 at ld=4, one more decade may close the 16% gap, but the
grid is spec-pinned); (c) accept that a y-codec at ~12% of pipeline error is
adequate for the ladder's diagnostic purpose and re-scope the gate.

---

# [SUPERSEDED — G2.1' bar was provisional; resolved above] Stage 2 gate report — tag 20260702, amended gates: PASS

Re-run under the 2026-07-05 spec amendment (`docs/dual_vae_staged_spec.md`):
tag `20260702`, beta swept per B1, gates G2.1'/G2.2' per B2/B3, posterior-mean
decoding per B4 (which the superseded 20260617 runs below also used — their
`reconstruct()` decoded the posterior mean, so B4 is a confirmation, not a
correction), B5 truncation diagnostic in every summary. Implementation commits
`e863c23` (sweep + amended gates) and `fbd692e` (trace-figure fix) preceded
the recorded runs. The original 20260617 report is kept below, superseded.

**Run-integrity note.** The first full 12-run invocation (implementation
`e863c23`, HEAD at launch `fbd692e`... run at 22:26-22:51 UTC) completed all
12 fits but crashed in the trace-figure code (stale reference to the removed
activity band) after writing only the first record
(`20260705T225134Z__d81e73e0__fbd692e`, ld=2 beta=1.0, now redundant). The fix
was committed and the sweep re-run in full; the 12 records below are the
authoritative set.

## Sweep (12 runs, 5000 epochs, hidden 128 x 2, seed 0, val fold, raw scale)

| ld | beta | val recon RMSE | PCA @ dim | per-dim KL (train) | collapse_ok | run_id (20260705T...__e863c23) |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 0.00475 | 0.00287 | 3.99, 1.73 | yes | `232027Z__d81e73e0` |
| 2 | 0.1 | 0.00385 | 0.00287 | 5.37, 3.47 | yes | `232028Z__406776c8` |
| 2 | 0.01 | 0.00408 | 0.00287 | 7.54, 5.12 | yes | `232029Z__3daae21e` |
| 2 | 0.001 | 0.00346 | 0.00287 | 11.37, 6.45 | yes | `232030Z__d6ac6880` |
| 3 | 1.0 | 0.00451 | 0.00134 | 4.00, 0.008, 1.58 | **no** | `232030Z__efd14e87` |
| 3 | 0.1 | 0.00317 | 0.00134 | 6.16, 2.37, 3.82 | yes | `232031Z__0a6c739a` |
| 3 | 0.01 | 0.00203 | 0.00134 | 10.30, 3.89, 4.63 | yes | `232032Z__9dc14b78` |
| 3 | 0.001 | 0.00200 | 0.00134 | 13.33, 4.41, 5.37 | yes | `232033Z__c9198e95` |
| 4 | 1.0 | 0.00460 | 0.00100 | 0.014, 0.012, 3.97, 1.76 | **no**\* | `232033Z__417bb958` |
| 4 | 0.1 | 0.00248 | 0.00100 | 2.93, 0.46, 6.62, 4.10 | yes | `232034Z__f744bbed` |
| 4 | 0.01 | 0.00240 | 0.00100 | 3.83, 2.91, 9.47, 4.82 | yes | `232035Z__2ba53ce0` |
| 4 | **0.001** | **0.00197** | 0.00100 | 4.68, 4.06, 13.07, 6.95 | yes | `232035Z__592313b6` **(selected)** |

\* ld=4 beta=1.0 prints `collapse_ok=True` in its record because 0.014/0.012
just clear the 0.01-nat bar; treated as effectively collapsed here for the
dimensionality discussion (the two dims carry ~0.01 nats each).

## Gate verdicts (amended)

**G2.1' — codec adequacy: PASS.** Selected (ld=4, beta=0.001) val
reconstruction RMSE 0.001969 <= bar 0.0042921 (10% of the best Stage 1
cross-modal val RMSE, `mlp` 0.042921, run `20260705T221753Z__e18a8464__760d8d6`;
same tag, per A3). The codec contributes at most ~1/20 of the best current
end-to-end error in quadrature terms.

**G2.2' — no collapse: PASS.** Selected run per-dim KL (train fold, best
epoch): 4.68 / 4.06 / 13.07 / 6.95 nats, all >= 0.01. A_j (documented, not
gated, per B3): 5394 / 2381 / 55677 / 11831 — delta-like posteriors, as
expected at beta = 0.001 on a near-deterministic target.

**B5 — suppressed-regime truncation error (selected run, val).**
Global 0.00197; SP<0.95: 0.00307 (1240 bins); SP<0.9: 0.00379 (671);
SP<0.8: 0.00462 (234). The codec truncation error is 1.6-2.3x worse in the
deeply-suppressed bins than globally — heterogeneous exactly as B5
anticipated; same pattern as PCA-on-Y (0.0014/0.0015/0.0021 at n=4).

**G2.3 — Lin-replication verdicts (re-issued on tag 20260702):**

1. *Reconstruction vs PCA at matched dimension (Fig. 12 analogue) +
   beta-sweep behaviour:* **disagree** (finding, not gated, per B2). PCA wins
   at every matched dimension at every beta (best VAE 0.00197 vs PCA 0.00100
   at dim 4). The beta sweep confirms the maintainer's rate-distortion
   reading: recon improves monotonically as beta falls (at ld=4:
   0.00460 -> 0.00248 -> 0.00240 -> 0.00197 for beta 1 -> 0.001), closing the
   PCA gap from 4.6x to 2.0x without reaching it. The remaining gap at
   beta=0.001 is no longer a rate penalty; it is the decode-from-sampled-z
   training blur plus finite optimization.
2. *Dimensionality behaviour (Fig. 5 analogue):* **partially agree.** The
   collapse pattern is beta-dependent: at beta = 1 the extra dimensions
   collapse outright (the 20260617 finding reproduces on this tag); at the
   selected beta = 0.001 all four dimensions carry active KL, but the recon
   gain saturates (2 dims 0.00346 -> 3 dims 0.00200 -> 4 dims 0.00197):
   a genuinely useful 3rd dimension, a marginal 4th. Effective
   dimensionality of SP(k) given cosmology is ~3 at the operating beta.
3. *Latent-parameter correlations over all 35 SB35 parameters (Fig. 1
   analogue; val fold):* **partially agree.** Feedback parameters remain
   prominent (mu2[1]: BlackHoleFeedbackFactor +0.22, ThermalWindFraction
   -0.20; mu2[3]: VariableWindSpecMomentum +0.25, WindDumpFactor -0.24), but
   at the selected low beta the latents also carry residual cosmology
   (mu2[2]: Omega0 +0.43, OmegaBaryon -0.33) despite the 5-parameter
   conditioning — the higher-rate code absorbs cosmological dependence the
   context did not fully explain. All correlations are modest (|r| <= 0.43),
   consistent with information spread across dimensions rather than one
   parameter per latent.

One outright disagreement (< 2): no human-review halt from G2.3.

**G2.4 — reduction-convention audit: PASS** (GR7 string in all 12 summaries;
metrics.jsonl traces consistent).

## Selection carried into Stages 3-5

`latent_dim_y = 4`, `beta_y = 0.001`, hidden 128, n_layers 2, epochs 5000,
lr 1e-3, weight_decay 1e-4, batch 128, anneal_epochs -> epochs // 4,
internal val_frac 0.1, seed 0. Frozen VAE-Y checkpoint:
`<scratch>/models/20260705T232035Z__592313b6__e863c23/model.joblib`.

---

# [SUPERSEDED — tag 20260617, pre-amendment gates] Stage 2 gate report — GATE FAILED, STOPPED

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
