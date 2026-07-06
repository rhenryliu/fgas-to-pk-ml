# Stage 3.0 — X-matrix forensics: VERDICT = CORRUPTION (builder-statistic defect, both tags)

Recorded diagnostics run: `20260706T052008Z__fee13aa6__8aba03c` (full tables
in its `summary.json`; census figure
`figures/2026-07/07-06/*__x_forensics__census.png`). Implementation commit
`8aba03c` preceded the run. All numbers below are from that record.

## Verdict, one paragraph

The pathological X values are **traceable to a construction defect in this
repo's own data-assembly statistic, not to the 17 -> 20 re-binning and not to
genuine physical tails**: `build_fgas_spk_dataset` computes the stack as a
**mean of per-halo ratios** `Delta Sigma_ionized / Delta Sigma_total / f_b`
([builder.py:118](../../src/fgas_spk/builder.py#L118), no near-zero guard),
and the per-halo denominator `Delta Sigma_total` — an *excess* surface
density — legitimately crosses zero for individual halos at R >~ 0.7 Mpc/h.
One near-zero cell produces an unbounded ratio that survives both nanmeans
and poisons the entire (sim, nd, bin) value. The defect is present in **both
tags** (the old tag's -555/+237 extremes have the same origin); the new grid
merely samples the singular region more densely. Per the amendment's
taxonomy this is **corruption** ("pathological values traceable to ... another
upstream defect"), with the defect sitting in the builder statistic rather
than the upstream profile files (which look healthy: median
|Delta Sigma_total| ~ 1e5-1e6 at the offending bins).

## Evidence

**3.0.4 — the smoking gun (source-denominator audit, 8 worst offenders).**
Rebuilding each offender's stack from the lindajin per-halo profiles
reproduces the served value exactly, and in every case the extreme is driven
by 1-2 cells with |Delta Sigma_total| 4-6 orders of magnitude below the bin
median. Worst three:

| sim | bin (R) | rebuilt stack | median \|dSigma_tot\| at bin | culprit cell |
|---|---|---|---|---|
| 429 | R=3.00 | -8539.8 | 9.5e5 | xz, halo 49: dSigma_tot = **-0.246**, per-halo ratio **-1.59e6** |
| 399 | R=4.38 | -1100.5 | 1.9e6 | yz, halo 57: dSigma_tot = **-0.384**, ratio **-2.05e5** |
| 117 | R=6.41 | -363 | 4.5e5 | xy, halo 36: dSigma_tot = **-1.96**, ratio **-6.77e4** |

**3.0.1/3.0.2 — census and offenders.** New tag (served X, nd index 2): 992
hard-implausible cells (outside [-0.5, 2.0]) across **635 of 1024 sims**
(train/val/test membership 433/63/139 — the val fold is majority-affected);
robust-|z|>10: 626 cells / 467 sims. Old tag: 478 hard cells / 393 sims.
Offenders start at R ~ 0.74 Mpc/h on the new grid (bins 2+, counts 15-74 per
bin) and at R ~ 3.7 on the old, tracking where individual-halo
Delta Sigma_total zero-crossings begin on each grid. No non-finite values, no
exact zeros, medians are physically sensible (0.31 at 0.1 Mpc/h rising to
0.92 at 20 Mpc/h) — the bulk is healthy; the corruption is point-like.

**f_gas definition (3.0.2 confirmation).** The sidecar stamps
`fgas: Delta Sigma_ionized / Delta Sigma_total / f_b, dimensionless`. As a
ratio of excess surface densities this quantity is NOT mathematically bounded
— the provisional [-0.5, 2.0] range is not a physical bound of the summary
statistic as currently defined (it *would* be a reasonable range for a
ratio-of-stacked-profiles definition; see Recommendation). Maintainer
confirmation of the intended bound remains open.

**3.0.3 — PCA poisoning confirmed.** Train-fold PCA on the new tag: leading
component EVR 0.982 with **99.9% of its score variance carried by 5 rows,
all offenders**; all four leading components point at <= 5 offender rows
(shares 0.98-0.999). Old tag: EVR spectrum [0.45, 0.36, 0.07, ...], same
domination pattern in weaker form. This is precisely the mechanism that made
both G3.1' arms fail equally (~0.073) and voided the ratio test — and it
means **every PCA-on-X number in Stages 1-3, both tags, is
offender-dominated**.

**3.0.5 — mlp attribution (diagnostic only, not a recorded baseline).**
Excluding hard-tier offender sims (train and val, same folds otherwise): val
RMSE **0.0403** vs 0.0429 on all rows — the new-tag mlp improvement is *not*
an artifact of the offenders (a discriminative model is robust to input
tails; they carry little gradient weight). Excluding the robust-z tier:
0.0467 (dropping 467 sims changes the val composition; diagnostic only).

## Consequences

1. G3.1''s mutual failure and VAE-X's epoch-0 restore are both explained by
   point-like corruption, consistent with the maintainer's reading (c)/(d).
2. **Exclusion lists are not a viable branch-A resolution**: 62% of sims have
   at least one hard cell. A radial crop would need R <~ 0.7 Mpc/h (2 bins)
   to dodge the singular region on the new grid — that guts the profile.
   The fix has to be at the *statistic* level.
3. Every X-involving number is tainted at the ~point-corruption level on both
   tags; y and params are clean. Stage 2's VAE-Y stands (C3 note in place).

## Recommendation (Branch A, concrete)

Fix the builder statistic and regenerate the dataset under a new tag:
compute the stack as a **ratio of stacked profiles** — nanmean (or nansum)
`Delta Sigma_ionized` over halos and projections, likewise
`Delta Sigma_total`, divide once, then apply `f_b` — so the denominator is
the stacked total (~1e5-1e6, never near zero) rather than a per-halo value.
This is also closer to what an observer measures (stacked lensing over a
stacked sample). **Which definition is scientifically intended
(mean-of-ratios vs ratio-of-means) is a physics decision (CLAUDE.md rule 3)
and is NOT resolved here** — the reference notebook's choice should be
checked before the builder is touched; if mean-of-ratios is intended, a
guarded/trimmed variant needs a maintainer-specified rule. Per 3.0.6 this
stops here for the maintainer's Branch decision.

STOPPED per 3.0.6. No Stage 3 resumption, no Stage 4, no builder edit.
