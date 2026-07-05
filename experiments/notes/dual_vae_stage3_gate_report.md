# Dual-VAE Stage 3 gate report — letter-pass, substantive FAIL: STOPPED

Tag 20260702, amended gates (B6). Implementation commit `67c11df` preceded the
runs. Frozen VAE-Y: `20260705T232035Z__592313b6__e863c23` (Stage 2 selected).

## Runs (beta sweep at latent_dim_x = 2, 5000 epochs, seed 0)

| beta | val recon RMSE (raw) | PCA-on-X @ 2 | per-dim KL | best_epoch | G3.1' ratio | run_id |
|---|---|---|---|---|---|---|
| 1.0 | 9.752 | 8.705 | 0.016, 0.033 | **0** | 0.9982 | `20260705T233320Z__31ced299__67c11df` (selected) |
| 0.1 | 9.752 | 8.705 | 0.016, 0.033 | **0** | 0.9982 | `20260705T233322Z__e66461e4__67c11df` |
| 0.01 | 14.688 | 8.705 | 4.52, 4.10 | 14 | 1.1282 | `20260705T233323Z__d4cf2638__67c11df` |
| 0.001 | 14.691 | 8.705 | 4.52, 4.11 | 14 | 1.1285 | `20260705T233324Z__02aaf8b6__67c11df` |

(beta = 1.0 and 0.1 restore identical epoch-0 weights: the anneal makes
``beta_eff(0) = 0`` for both, and epoch 0 was the best internal-val epoch for
both, so the restored models coincide.)

## Formal gate outcomes — and why they do not mean what they say

- **G3.1' (<= 1.05 x PCA arm): letter-PASS at 0.9982.** But both arms sit at
  y-space val RMSE ~0.073 — worse than every Stage 1 baseline (mlp 0.0429,
  mlp_regressor 0.0504, pca_linear 0.0614). The gate's ratio is uninformative
  when both arms are equally inadequate: a 2-dimensional code of the raw
  20-bin f_gas profile carries too little usable information regardless of
  codec, and the "VAE codes" being compared are those of an **untrained
  encoder** (see below).
- **G3.2' (per-dim KL >= 0.01): letter-PASS at [0.016, 0.033].** These are
  epoch-0 KLs of a barely-perturbed random initialisation, not evidence of a
  learned, non-collapsed code.
- **G3.3 (OOD reference statistics): present** (in every run summary), but
  they describe an untrained encoder's codes (val ``||mu1||``: median 0.17,
  p99 1.10, max 9.49) and are not a usable OOD baseline for later stages.

**Substantive verdict: the Stage 3 goal — "a validated low-dimensional
representation of the observable" — was not achieved.** Proceeding to Stage 4
on these frozen codes would launder a degenerate codec through formally
passing gates. Stopped per the spirit of GR5; runs and evidence committed per
GR8's failure protocol.

## Diagnosis (from the committed traces, run `...31ced299`'s metrics.jsonl)

Classic heavy-tail failure of a Gaussian likelihood, the known f_gas landmine
(Stage 1 report, observation 1; 20-bin re-binned grid changes nothing
qualitative). Train NLL descends monotonically (28.3 -> -22.9 over 5000
epochs) while the internal-holdout NLL explodes monotonically (50 -> 9793):
the model shrinks per-bin ``obs_logvar`` where its training rows are quiet,
and held-out rows with 50-300 sigma tail values (raw f_gas values to +236 /
-555 at R >~ 1 Mpc/h) then incur astronomical NLL. The learned noise head
does what it can — fitted raw-scale ``obs_sigma`` spans 0.17 to 336 across
radial bins — but a **per-bin** variance cannot pay for **per-example**
tails. Best-epoch restore therefore correctly selects epoch ~0: under this
likelihood, on this data, "no training" genuinely generalises best. No
hyperparameter (lr, batch size, epochs, val_frac) changes this: the
train/holdout divergence is monotone from epoch ~1.

## Blocked, and the options (maintainer decision required; spec veto rule)

Every candidate fix touches a resolved decision, so none was taken
unilaterally (binding rule 4):

1. **Robust likelihood for VAE-X** — Student-t NLL with learned per-bin scale
   (and fixed or learned dof), amending F0.2 for the X usage only. The most
   direct treatment of the mechanism; keeps the pinned data untouched.
2. **Variance-stabilising input transform** — model ``asinh(X / s)`` (scale
   ``s`` recorded) inside the X codec, reporting recon RMSE on both scales.
   Amends F0.2's raw-scale reporting for X; keeps the Gaussian NLL.
3. **Radial crop** — e.g. ``radial_range_mpch: [0.1, ~3]``, where the
   profiles are tame; the outer bins are projection-noise-dominated. This
   changes the pinned DataConfig **and the observable definition** — a
   physics decision (rule 3), explicitly not mine to make.
4. **Recorded winsorisation** of X at a stated quantile — a data
   modification; least principled, listed for completeness.
5. **Accept the letter-pass and proceed** — produces a composite whose X arm
   is untrained; Stage 4/5 numbers would be valid but scientifically empty,
   and Stage 5b would ablate codecs on a degenerate X side. Not recommended.

Stage 4 implementation (mapping rungs, composite helpers, stage script) is
complete, synthetically verified, and committed, but **no Stage 4 runs were
launched** — they require an honest frozen VAE-X.

The Y-side results (Stage 2, amended gates: PASS) are unaffected.
