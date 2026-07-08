# The dual-codec SP(k) emulator: documentation and results

This document is self-contained: it describes the code and results as they
now exist. The historical record of instructions and amendments lives in
`docs/dual_vae_staged_spec.md` (cited here once, as provenance). Every number
below carries its run_id or points to the committed note that does; runs live
under `experiments/runs/<run_id>/`.

## 1. Executive summary

**What exists.** Two registered models in `fgas_spk.models.REGISTRY`:

- **`dual_vae`** — a codec-agnostic composite that predicts the matter
  power-spectrum suppression SP(k) from a stacked gas-fraction profile
  f_gas(R): an x-codec compresses the profile to a low-dimensional code, a
  mixture density network maps codes to the y-codec's latent space, and the
  y-codec's decoder produces the SP(k) curve. Both codecs are selectable per
  run (`codec_x` / `codec_y` in {"vae", "pca"}). `predict_samples` gives a
  calibrated predictive distribution.
- **`dual_vae_ft`** — a two-phase task-supervised fine-tune of the all-VAE
  configuration. It is registered and reproducible but **not adopted**: it
  recovers accuracy at an unacceptable calibration cost (section 6,
  amendment 7).

**The primary configuration** is `(codec_x, codec_y) = (PCA d=4, VAE ld=3
beta=1e-4)` with an MDN (K=3) mapping — config
`scripts/configs/run/dual_vae_pca_vae.yaml`.

**Headline test results** (tag 20260706, R < 10 Mpc/h crop, pinned split;
suppressed-regime RMSE is truth-masked at TRUE SP(k) < t; full table and
restrictions in `experiments/notes/dual_vae_vs_baselines.md`):

| model | test RMSE | SP<0.8 | coverage 68/95 | run_id |
|---|---|---|---|---|
| `cvae` (spread invalid: 23-26 pp under) | 0.01254 | 0.02806 | 0.428 / 0.716 | `20260708T231014Z__fdbfd21b__864001d` |
| `mlp` (direct) | 0.01403 | 0.03391 | — | `20260706T185716Z__bcdaaee9__a4ba5aa` |
| `mlp_regressor` (direct) | 0.01434 | 0.03153 | — | `20260706T185713Z__c62b6ed0__a4ba5aa` |
| `vib_regressor` (accuracy only) | 0.01489 | 0.03305 | n/a by design | `20260706T224755Z__e1ecad20__3fc7da0` |
| **`dual_vae` primary (pca, vae)** | **0.03129** | **0.07661** | **0.670 / 0.920** | `20260706T215627Z__cfc56666__43ea387` |
| `pca_linear` (direct) | 0.03242 | 0.07395 | — | `20260706T185703Z__a760845e__a4ba5aa` |
| `dual_vae` (pca, pca) | 0.03486 | 0.08837 | 0.662 / 0.934 | `20260706T220238Z__6457b899__43ea387` |
| `dual_vae` (vae, vae) | 0.04881 | 0.13944 | 0.689 / 0.934 | `20260706T220156Z__0e252abf__43ea387` |

**What it is for, honestly framed.** The composite provides a calibrated
probabilistic SP(k) prediction (coverage within ~1 pp of nominal at 68%,
~2-3 pp under at 95%), outputs constrained to the learned SP(k) manifold,
and interpretable latents (the y-latents organize by feedback physics). Its
measured cost is roughly a factor of two in raw accuracy against direct
regressors (2.18x vs `mlp_regressor` on test; 1.67x at the ladder on val) —
the price of routing through a 3-dimensional y-manifold with a
posterior-mean z-space objective. The head-to-head against the architecture
it replaces makes the trade concrete: the genuine `cvae` is the single most
accurate model in the table (0.01254) but under-covers by 23-26 pp at every
nominal level — its spread is unusable as uncertainty — while `dual_vae` is
the only model in the table with usable coverage. Whether calibration is
worth ~2.5x in raw accuracy is a scientific judgement; the numbers above are
the inputs.

## 2. Quickstart

Environment: `module load python && conda activate fgas-ml`, from the repo
root, with `$FGAS_SCRATCH_ROOT` set.

**Train + evaluate the primary configuration** (fits everything in one
process; writes a run record, a scratch checkpoint, and figures):

```bash
python scripts/run.py \
    --config-data scripts/configs/data/config_dual_vae.yaml \
    --config-run  scripts/configs/run/dual_vae_pca_vae.yaml
```

**Load a committed checkpoint and predict** (the primary's checkpoint):

```python
import joblib, numpy as np
from fgas_spk import DataConfig, load_training_data
from fgas_spk.paths import fill_data_root

td = load_training_data(fill_data_root(
    DataConfig.from_yaml("scripts/configs/data/config_dual_vae.yaml")))
model = joblib.load("/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml/models/"
                    "20260706T215627Z__cfc56666__43ea387/model.joblib")

y_hat = model.predict(td.X, None, td.X_params)          # (n, 36)
# X: (n, 16) cropped f_gas profile; X_params: (n, 5) cosmological params.

samples = model.predict_samples(td.X[:8], None, td.X_params[:8],
                                n_samples=200, seed=0)  # (200, 8, 36)
# F0.5 spread semantics (from the docstring): the spread is the mapping
# density plus decoder-Y observation noise and deliberately EXCLUDES the
# encoder-X posterior spread and input measurement noise.

mu1 = model.latents(td.X, None, td.X_params)            # (n, 4) x-codes
mu2 = model.latents_y(td.y, None, td.X_params)          # (n, 3) y-codes
sigma = model.obs_sigma()                               # (36,) per-k noise
```

**Switch codecs** via `model_params` (no code changes):

```yaml
model_params:
  codec_x: vae   # or pca
  codec_y: pca   # or vae
```

**Re-run any stage or gate** (all re-runnable; see each script's docstring):

```bash
python scripts/stage1_pca_dimensionality.py            # PCA baselines
python scripts/stage1_baseline_table.py                # baseline table + E4.2
python scripts/stage2_vae_y.py                         # VAE-Y sweep + gates
python scripts/stage3_vae_x.py --vae-y-run-id <id>     # VAE-X grid
python scripts/stage3_0_forensics.py                   # X-matrix forensics
python scripts/f33_census.py                           # cropped-slice census
python scripts/stage4_0_probe_selection.py --vae-y-run-id <id>   # codec probe
python scripts/stage4_latent_map.py --codec-x pca --pca-dim 4 \
       --vae-y-run-id <id>                             # mapping ladder
python scripts/stage6_finetune.py                      # fine-tune val gates
```

**Rebuild the dataset** (slow path, ~1024 source files):

```bash
python scripts/rebuild_dataset_20260706.py
```

The build **hard-fails** on any non-finite f_gas value anywhere, and on any
value outside `fgas_sanity_range` ([-1, 3], production default) **within**
the modelled radial window; outside the window, extremes are censused into
`__meta__` and warned. A guard failure prints the offending
(sim, nd, bin, value) cells and rejects the build.

## 3. Architecture

**Codecs.** `GaussianVAE` (`fgas_spk/models/dual_vae_components.py`): an MLP
encoder/decoder pair with the five cosmological CAMELS parameters as context
in both, a Gaussian NLL with a learned per-bin observation-noise head
(`obs_logvar`, exposed as `obs_sigma()` on the raw scale), beta-weighted KL
with linear warm-up, internal holdout with best-epoch restore. `PcaCodec`:
PCA scores as codes, inverse transform as decode, train-fold truncation-
residual std standing in for `obs_sigma` — so any codec pair supports the
full probabilistic interface.

**Mapping ladder.** `LatentMapRidge` (rung 1), `LatentMapMLP` (rung 2), and
`LatentMapMDN` (rung 3; mixture NLL, seeded sampling). The composite
prediction path is `x -> encode_x -> mapping -> decode_y`; rung 3 defines
the predictive distribution (`composite_samples`: MDN draws decoded through
the y-decoder plus per-bin observation noise).

**Primary and why.** The x-codec was chosen by a fixed nonlinear-probe
protocol plus a full dual-track ladder, decided by a criterion fixed before
the numbers existed (dominance on global RMSE, all suppressed thresholds,
and both coverage distances). The PCA-d4 arm dominated every component
(section 6, amendments 5-6). The y-side VAE earns its place: it beats a
PCA y-codec by ~10% globally and ~13% in deep suppression at equal
calibration, and carries the interpretable latents.

**`dual_vae_ft`** differs in two phases after the standard fit: (1) the
encoder-X and rung-2 MLP are fine-tuned jointly with a y-space NLL through
the frozen decoder-Y (all decoder parameters frozen, `obs_logvar` included);
(2) the encoder is frozen and the MDN re-fit on the new codes. Not adopted
(calibration regression; section 6).

## 4. Data

**The statistic.** f_gas(R) is the **ratio of stacked profiles**: nanmean of
Delta Sigma_ionized over the selected halos x 3 projections per radial bin,
divided by the same nanmean of Delta Sigma_total, normalized by the
per-simulation f_b. The original builder computed a **mean of per-halo
ratios** instead; a single halo whose Delta Sigma_total crossed zero (generic
at R >~ 0.7 Mpc/h) produced an unbounded ratio that poisoned the whole stack
(values to -8540), corrupting 62% of simulations at some radius. The
forensics that located this, cell by cell against the source profiles, are in
`experiments/notes/dual_vae_stage3_forensics.md`.

**The discriminator.** Datasets built with the corrected statistic carry an
`fgas_definition` key in their `__meta__` (and sidecar YAML). **Absent stamp
= old statistic = do not train on it.** Tag history: `20260706` is the only
valid tag; everything earlier (`20260616`, `20260617`, `20260702`, both
snapshots, all ranks) is superseded **by definition change**, not merely by
tag. The snap82 and other-rank variants have not been rebuilt (warning in
`experiments/notes/known_issues.md`).

**The crop.** The modelled radial scope is R < 10 comoving Mpc/h
(`radial_range_mpch: [0, 10]`, 16 of 20 bins): the stacked denominator
itself legitimately decays toward zero at the outermost radii, making the
statistic intrinsically ill-conditioned there. The boundary value 10 is a
preliminary, maintainer-owned physics ruling. The retained slice spans
[0.041, 1.25] — comfortably inside the production sanity range [-1, 3]
(ruling and revisit trigger recorded in the builder docstring and
`known_issues.md`).

## 5. Metrics and where things live

**Metric families.** Global val/test RMSE (raw SP(k) scale);
suppressed-regime RMSE (pooled over bins where the TRUE SP(k) < t, t in
{0.95, 0.9, 0.8} — always truth-masked, never prediction-masked); pooled and
per-k-bin coverage of central predictive intervals; per-curve RMSE rankings
with worst-simulation lists; OOD code norms (||mu1|| median/p99/max);
latent-parameter Pearson correlations against all 35 SB35 parameters.

**Directory map.** Run records and the greppable ledger:
`experiments/runs/<run_id>/` and `experiments/runs.jsonl`. Figures: dated
tree under `figures/` (gitignored, regenerable), run_id in the filename.
Notes (all under `experiments/notes/`): `dual_vae_baseline_table.md`,
`dual_vae_stage1_gate_report.md` … `dual_vae_stage4_gate_report.md`,
`dual_vae_stage3_forensics.md`, `dual_vae_findings.md`,
`dual_vae_test_declaration.md`, `dual_vae_vs_baselines.md`,
`dual_vae_codec_ablation.md`, `known_issues.md`, and this document.

## 6. Decision log — the seven amendments, consolidated

| # | trigger | decision | consequence |
|---|---|---|---|
| 1 | Original G2.1/G2.2 failed (VAE-Y < PCA; activity band) | Data tag correction + beta became a hyperparameter; gates reformed (G2.1'/G2.2') | Rate-distortion sweep; VAE-Y selected honestly |
| 2 | VAE-X epoch-0 pathology | Stage 3.0 forensics ordered before any likelihood fix; gates hardened (floor, G3.4) | Corruption found, not tails |
| 3 | Forensics verdict: builder defect | Branch A: ratio-of-stacks fix, guard, stamp, rebuild `20260706` | Baselines improved 2-3x on clean data |
| 4 | G2.1' failed only because the baseline improved | Quadrature adequacy gate; R < 10 crop; two-tier guard; one capped beta decade | Stage 2 passed at 0.49% floor inflation |
| 5 | G3.1' unreachable by construction (linear probe) | Gate retired; findings F-1/F-2 promoted; Stage 4.0 nonlinear-probe codec selection | G4.0a viability passed; G4.0b tripwire fired |
| 6 | G4.0b ratio 1.66 | Dual-track ladder, primary criterion fixed ex ante; test batch declared | Arm P (PCA-x) primary by full dominance |
| 7 | Programme close | Stage 6 fine-tune; closing verifications; CVAE row; sanity-range ruling; this document | Stage 6 not adopted; programme closed |

**Scientific findings along the way** (details and run_ids in the notes):

- **Lin et al. replication (Stage 2, VAE-Y):** reconstruction-vs-PCA
  *disagrees* (PCA wins at matched dimension at every beta; the completed
  rate-distortion curve saturates at ~0.0019 vs PCA-3's 0.0013, so the gap
  is not rate-limited); dimensionality *partially agrees* (collapse is
  beta-dependent; effective dimensionality ~3 at the operating beta);
  latent-parameter correlations *partially agree* (feedback parameters
  prominent — wind and AGN — with residual cosmology at low beta).
  Selected VAE-Y: ld=3, beta=1e-4, val recon RMSE 0.00192
  (`20260706T190601Z__8a715039__8522dd4`).
- **F-1:** VAE-X codes carry 31-48% less mu2-relevant information than PCA
  scores at matched dimension under a linear probe — and the deficit
  *survives the nonlinear probe* (1.41-1.82x at d >= 3): the
  reconstruction+rate objective genuinely discards task-relevant structure.
  A noise-head mechanism is recorded as a parked, untested hypothesis.
- **F-2:** reconstruction-optimal is not downstream-optimal (the
  best-reconstruction VAE-X config had the worst downstream adequacy).
- **The corruption incident:** a mean-of-ratios statistic with an unguarded
  near-zero denominator, present in every pre-20260706 tag, found by
  cell-level forensics against the source profiles and fixed at the
  statistic level (exclusion or cropping alone could not have worked).
- **Architecture cost:** the two-stage design costs ~1.7-2.2x vs direct
  regression (probe 1.88x; ladder 1.67x; test 2.18x with both provenances
  verified same-fold).
- **Codec verdict:** PCA-x + VAE-y. The x-side deficit is the VAE
  objective's, not the architecture's; the y-side VAE is strictly useful.
- **Stage 6 recoverability verdict:** task supervision recovers the F-1
  deficit and more (val RMSE 0.0226, beating the frozen-codec ceiling
  0.0364) but collapses calibration (coverage 0.540/0.792, a ~10.5 pp
  regression) and revives encoder blow-up (||mu1'|| max 20). Per the
  no-calibration-regression rule, `dual_vae_ft` never touched test
  (`20260706T224653Z__cb248494__7270c4c`).

## 7. Known limitations and deferred work

- **Predictive spread excludes** the encoder-X posterior spread and input
  measurement noise (F0.5, by design); input-noise propagation reconnects at
  the `composite_samples` docstring boundary when it is taken up.
- **The suppression target is the within-hydro proxy** P_total/P_DM, biased
  at k >~ 5 h/Mpc in a feedback-correlated way; the paired-DMO target
  remains the known to-do (recorded verbatim in every run's `env.json`).
- **Per-halo dataset:** deferred; the codec verdict (PCA-x beats VAE-x) was
  measured on stacked, low-noise profiles and may not transfer to noisier
  per-halo inputs.
- **NPE-in-observable-space** (`sbi`) remains a separate deferred track.
- **The parked noise-head hypothesis** (F-1 mechanism) is recorded, testable,
  and deliberately not pursued here.
- **Single-snapshot scope** (snap74, z=0.47); snap82 exists only under the
  superseded statistic.
- **`vib_regressor` restriction:** its sampling spread has no uncertainty
  semantics; its row is accuracy-only, permanently. The genuine `cvae` row
  (added post-close-out under a third test declaration) is the opposite
  case: semantically meaningful spread, measured 23-26 pp under-coverage on
  this context — accuracy-valid, calibration-invalid. A `cvae` with a
  learned noise head (its known missing piece) is untested here and belongs
  to future work.
- **95% tail under-coverage** (~2-3 pp across all arms; diagonal-MDN light
  tails) — observed, not fixed.

## 8. File map (added or changed by this programme)

- `src/fgas_spk/models/dual_vae_components.py` — GaussianVAE, PcaCodec, the
  three mapping rungs, composite helpers.
- `src/fgas_spk/models/dual_vae.py` / `dual_vae_ft.py` — the two registered
  models. `models/__init__.py` — their registration.
- `src/fgas_spk/builder.py` — ratio-of-stacks statistic, two-tier guard,
  `FGAS_DEFINITION` + `fgas_meta_stamp`.
- `src/fgas_spk/schema.py` — additive `extra_meta` on `save_dataset`.
- `src/fgas_spk/train.py` — mixed-schema history figure fix.
- `scripts/stage1_pca_dimensionality.py`, `stage1_baseline_table.py`,
  `stage2_vae_y.py`, `stage3_vae_x.py`, `stage3_0_forensics.py`,
  `f33_census.py`, `stage4_0_probe_selection.py`, `stage4_0_i1_breakdown.py`,
  `stage4_latent_map.py`, `stage6_finetune.py`,
  `rebuild_dataset_20260706.py` — the stage and gate scripts (all
  re-runnable).
- `scripts/configs/data/config_dual_vae.yaml` — the pinned context.
- `scripts/configs/run/dual_vae_{vae_vae,pca_vae,pca_pca,ft}.yaml` — the four
  composite configs; `pca_reference.yaml` — split pinned.
- `tests/test_dual_vae_components.py`, `test_dual_vae_plugin.py`,
  `test_builder_fgas_statistic.py` — the new test suites.
- `docs/dual_vae_staged_spec.md` — the spec + all seven amendments
  (historical record); `docs/dual_vae_doc.md` — this document.
