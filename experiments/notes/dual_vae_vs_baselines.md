# dual_vae vs baselines (Stage 5 comparison note)

Context: tag 20260706 (ratio-of-stacks f_gas), R < 10 Mpc/h crop, nd index 2,
k in [0.5, 5.0], pinned SplitSpec (0.7/0.1/0.2, seed 100). All dual-codec
numbers from the three declared test runs
(`dual_vae_test_declaration.md`; each configuration touched test exactly
once); baseline numbers from the F4 Stage 1 records. Primary configuration:
**Arm P, (codec_x, codec_y) = (pca d=4, vae ld=3 beta=1e-4), MDN K=3**,
designated on val evidence before any test number existed.

## Gate verdicts (Stage 5 / 5b)

- **G5.1 PASS.** Test evaluated exactly once per declared configuration;
  records, checkpoints, and figures exist (two runs' post-record figure pass
  crashed on a `train.py` history-plot bug — fixed in-repo — and their
  figures were regenerated from the saved checkpoints; no quantity was
  recomputed). Registered-pipeline sanity: the plugin's val RMSEs reproduce
  the staged Stage 4 ladder **exactly** (Arm P 0.032698, Arm V 0.054312) —
  the one-process `dual_vae` fit and the staged pipeline do not silently
  differ.
- **G5.2 PASS** (this note; every claim carries a run_id).
- **G5b PASS** (`dual_vae_codec_ablation.md`; both mandatory arms evaluated
  under identical conditions with full metrics).

## Accuracy (test fold; suppressed-regime truth-masked)

| model | test RMSE | SP<0.95 | SP<0.9 | SP<0.8 | run_id |
|---|---|---|---|---|---|
| `mlp` | 0.01403 | 0.02166 | 0.02689 | 0.03391 | `20260706T185716Z__bcdaaee9__a4ba5aa` |
| `mlp_regressor` | 0.01434 | 0.02258 | 0.02741 | 0.03153 | `20260706T185713Z__c62b6ed0__a4ba5aa` |
| `vib_regressor` (accuracy only, see below) | 0.01489 | 0.02216 | 0.02748 | 0.03305 | `20260706T224755Z__e1ecad20__3fc7da0` |
| **`dual_vae` primary (pca, vae)** | **0.03129** | 0.04990 | 0.06173 | 0.07661 | `20260706T215627Z__cfc56666__43ea387` |
| `pca_linear` | 0.03242 | 0.04365 | 0.05285 | 0.07395 | `20260706T185703Z__a760845e__a4ba5aa` |
| `dual_vae` (pca, pca) | 0.03486 | 0.05450 | 0.06720 | 0.08837 | `20260706T220238Z__6457b899__43ea387` |
| `dual_vae` Arm V (vae, vae) | 0.04881 | 0.07702 | 0.09760 | 0.13944 | `20260706T220156Z__0e252abf__43ea387` |

The primary composite sits between the direct MLPs (2.2x better globally)
and edges `pca_linear` globally while trailing it slightly in the suppressed
regime. **`vib_regressor` row restriction (amendment 7, J4.2):** the CVAE
comparison row is the vib_regressor on the corrected pinned context
(declared second test batch; per-curve median/p90/max 0.0083/0.0238/0.0600)
and reports **point-accuracy metrics only** — no coverage or calibration
entries, because its sampling spread carries no valid uncertainty semantics
(the documented reason for its renaming from cvae). Historical CVAE numbers
predate the Branch-A definition change and remain non-comparable (A3/E7.3).

## Stage 6 outcome (amendment 7, J2): recoverable accuracy, unacceptable calibration

`dual_vae_ft` (the two-phase fine-tune of Arm V; val-gate record
`20260706T224653Z__cb248494__7270c4c`) answered the F-1 recoverability
question decisively on val: RMSE **0.02262** — better than Arm P (0.0327)
and the Stage 4.0 ceiling (0.0364) — so the VAE-X deficit **is
objective-induced and recoverable under task supervision**. But pooled
coverage collapsed to 0.540 / 0.792 (a 10.7 / 10.2 pp regression vs Arm V,
far beyond the 3 pp materiality threshold; G4.2 fails outright) and the
encoder blow-up mode returned (||mu1'|| max 20.0). Per the Stage 6 rule
that a calibration regression is not a tolerable trade-off, `dual_vae_ft`
was **not adopted and never touched test** (its J1 declaration was
conditional on val gates). The published sentence: task supervision buys
back the accuracy the reconstruction objective discards, and pays for it in
calibration — the frozen-pipeline primary remains the recommended model.

## Calibration and probabilistic behaviour (test fold)

Primary: coverage 0.670 / 0.865 / 0.920 at nominal 0.68 / 0.90 / 0.95 —
within 1 pp at 68%, ~3-3.5 pp under at 90/95%. Arm V: 0.689 / 0.907 / 0.934.
(pca, pca): 0.662 / 0.887 / 0.934 — the truncation-residual obs_sigma stand-in
calibrates comparably to the learned noise head at these levels. None of the
three shows the CVAE-style under-coverage failure. Spread semantics per F0.5:
mapping density + decoder-Y observation noise only.

## OOD behaviour (test fold, `latent_norms` from the runner)

Primary and (pca, pca) share the PCA x-codes: ||mu1|| median 0.314, p99
0.937, max 1.065 — no blow-up. Arm V's VAE codes: median 4.91, p99 13.78,
max 15.79 — echoing the historical CVAE OOD precedent (norms 17-22) in
attenuated form; its worst test curves (sims 598, 822, 977) overlap the
other arms' worst (305, 598, 977, 109), so the extreme norms are not
creating unique failures, but the margin is thinner.

## The architecture-cost paragraph (required by amendment 6, I4.3)

The measured price of the probabilistic low-dimensional design, stated
without softening: at the Stage 4.0 probe, the unrestricted two-stage
pathway (full profile -> MLP -> mu2 -> frozen decoder-Y) is **1.88x** worse
than direct `mlp_regressor`; the ladder's own version — Arm P rung-2 vs
`mlp_regressor` on val — is **1.67x** (0.03245 vs 0.019378), and the primary
rung-3 on test is **2.18x** `mlp_regressor` (0.03129 vs `mlp_regressor`'s
test RMSE **0.01434**, run `20260706T185713Z__c62b6ed0__a4ba5aa`; same tag,
same crop, same pinned-split test fold — verified same-fold provenance, V1).
The val-to-test asymmetry behind the ratio widening (1.69x on val -> 2.18x
on test) is real, not a mixed-fold artifact: `mlp_regressor` improves 26%
from val to test (0.0194 -> 0.0143) while the primary improves only 4%
(0.0327 -> 0.0313) — the test fold happens to be easier for the direct
regressor; fold-difficulty variance of this size is consistent with the
~100-sim val fold. Routing
through a 3-dimensional y-manifold with a z-space objective costs roughly a
factor of two in raw accuracy on this data. The architecture's contribution
is what the direct regressors do not provide: a calibrated predictive
distribution (coverage above), manifold-constrained outputs (every
prediction decodes from the SP(k) manifold), and interpretable latents
(the Stage 2 latent-parameter correlations; z2 organized by feedback).
Whether that trade is worth a 2x accuracy price is a scientific judgement
for the paper, informed by these numbers — not resolved here.
