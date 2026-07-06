# Staged spec (v2, amended): dual-VAE architecture for f_gas → SP(k)

**Repository:** `rhenryliu/fgas-to-pk-ml` (src layout, package `fgas_spk`)
**Prepared for:** Claude Code execution, stage by stage
**Supersedes:** dual_vae_staged_spec.md (v1)
**File provenance:** the v2 spec was circulated outside the repo; it is checked
in here for the first time together with the 2026-07-05 maintainer amendment
(bottom of this file). Where the amendment conflicts with the original text,
**the amendment wins**.

---

## Purpose

Implement and evaluate a dual-VAE architecture as an alternative to the Sohn-style
CVAE:

```
Training (stages 2-4, on frozen predecessors):
  y = SP(k)     --encoder_y--> z2 --decoder_y--> y_hat     (VAE-Y, trained alone)
  x = f_gas(R)  --encoder_x--> z1 --decoder_x--> x_hat     (VAE-X, trained alone)
  (mu1, mu2) posterior-mean pairs --> mapping M: z1 -> z2   (ladder: ridge/MLP/MDN)

Prediction (composite):
  x --encoder_x--> mu1 --M--> p(z2|z1) --decoder_y--> p(y_hat)
```

VAE-Y doubles as a **methodological replication of Lin et al. (2026,
arXiv:2509.01881) on this project's own data** (decision F0.1 = (b)): same model
class in spirit (VAE with conditioning on cosmological parameters, latent-activity
and dimensionality checks, latent-parameter correlation analysis), applied to the
Gen 2 SP(k) proxy target. No Gen 1 data acquisition. The latent spaces z1 and z2
are first-class scientific products.

## Resolved decisions (do not re-ask; veto requires a human edit to this file)

- **F0.1 = (b):** methodological replication on this project's data only.
- **F0.2 (as amended 2026-07-05, see B1 and D2):** both VAEs use a Gaussian NLL
  likelihood with a learned per-bin observation-noise parameter `obs_logvar` of
  shape (n_bins,), initialized to 0.0 (unit variance on the standardised scale,
  i.e. one per-bin standard deviation on the raw scale), exposed via an
  `obs_sigma()` method on the raw target scale. (Original text described this as
  "the same noise-head design as the repo's CVAE" -- a spec error: the repo CVAE
  has no noise head; that was a planned fix, not landed code. The head is defined
  on its own terms above.) **beta is a tuned hyperparameter**: both `GaussianVAE`
  usages sweep beta in {1, 0.1, 0.01, 0.001}, selected by val reconstruction RMSE
  (posterior-mean decode) subject to the no-collapse criterion (B3). No
  beta-TCVAE decomposition; no cyclical annealing (linear KL warm-up as in the
  existing cvae/vib plugins is fine).
- **F0.3:** latent_dim_x = latent_dim_y = 2 as the first configuration; both are
  independent `model_params` keys tunable per run. Stage 2 additionally sweeps
  latent_dim_y in {2, 3, 4} for the Lin-style dimensionality check.
- **F0.4:** conditioning is config-driven through the existing DataConfig
  machinery: `include_camels_params: true`,
  `camels_param_names: [Omega0, sigma8, OmegaBaryon, HubbleParam, n_s]`
  (the `config_camels.yaml` pattern). Models consume `X_params` as context in
  both encoder and decoder, exactly as the existing cvae/vib plugins do. No
  astrophysical parameters.
- **F0.5:** the z1 -> z2 mapping is trained on posterior means (mu1, mu2), not
  samples. Consequence (record in the composite's docstring): the composite's
  predictive spread = mapping density + decoder-Y observation noise, and
  deliberately EXCLUDES encoder-X posterior spread and input measurement noise.
  Input-noise propagation is a later, separate addition.
- **F0.6:** noise augmentation for VAE-X deferred. G3.3's OOD reference
  statistics are the cheap protection retained now.
- **F0.7:** single-snapshot training only (snap74 configs as pinned below). No
  snapshot stacking or conditioning; a joint-snapshot model is out of scope.
- **F0.8:** registry name for the composite: `dual_vae`. Component classes are
  NOT registry entries (see F0.9). Mapping rungs are internal, selected by
  `model_params`.
- **F0.9 (integration architecture, hybrid):** the standalone VAEs cannot honestly
  implement `ProfileToSpk` (`predict(X) -> SP(k)` is meaningless for a y-only
  autoencoder), so:
  - Stages 2-4 use **dedicated stage scripts** under `scripts/` that bypass
    `run_training` but reuse `load_training_data`, `grouped_split`, and
    `write_run_record` directly, so every exploration run is recorded and
    ledgered without polluting the registry.
  - Stage 5 registers a single `dual_vae` plugin whose `fit()` retrains all
    three components end-to-end in one process using the stage-2/3/4-selected
    hyperparameters, giving one-config reproducibility through `scripts/run.py`.
- **Rung-3 density estimator:** mixture density network (MDN) over z2 given z1.
  A normalizing flow is out of scope at this dimensionality.

## Ground rules (apply to every stage)

- **GR1.** Follow repo conventions and CLAUDE.md: Google-style docstrings,
  Canadian English, GELU activations, lazy torch imports (importing
  `fgas_spk.models` stays torch-free; heavy deps import inside methods),
  `pick_device()` for device selection, two-config-per-run design.
- **GR2.** Run-record schema changes are **additive only**. Every recorded run
  carries the suppression-definition note in `env.json` (the existing
  `write_run_record` default handles this; do not override it).
- **GR3.** All splits use the existing `grouped_split` by `sim_index` with the
  pinned `SplitSpec` below. Never split by row.
- **GR4.** Fixed seeds recorded per run. Every gate evaluation is a re-runnable
  script (under `scripts/` or a stage-specific diagnostics module), not a one-off.
- **GR5.** Do not proceed past a failed gate. Report failure + evidence and stop.
- **GR6 (as amended 2026-07-05, see B9).** Nothing is tuned against the test
  split in Stages 1-4. Test data is evaluated exactly once per registered model
  configuration, with all configurations declared before any test evaluation
  runs, batched at Stage 5 / Stage 5b. No configuration is selected, tuned, or
  modified in response to a test number.
- **GR7. Loss-scaling consistency (known past bug class).** Within each ELBO, the
  reconstruction NLL and the KL term must share one reduction convention (both
  per-element means or both sums). State the convention and the effective
  per-dimension weighting in the module docstring and in each run's summary.
- **GR8. Commit discipline (checkpoints per stage).**
  1. Commit the stage's implementation BEFORE launching its training runs, so
     `write_run_record` captures a clean (non-dirty) git SHA.
  2. After gates pass, commit the run records, figures list, and the gate report
     in a second commit.
  Message convention: `stage N: <short summary>` and
  `stage N records: <run_ids> [gates GN.x pass]`. On a gate failure, still commit
  the records with `[gate GN.x FAIL]` so the evidence is checkpointed, then stop.

## Pinned evaluation context (identical for every stage and every model)

- **DataConfig:** the `config_camels.yaml` pattern -- suite
  `CAMELS-IllustrisTNG-L50n512-SB35`, snapshot 74 (z = 0.47), `source: lindajin`,
  `rank: m500`, tag `'20260702'` (amended 2026-07-05 per A1; was `'20260617'`),
  `number_density_indices: [2]`, `target_mode: k_range`, `k_range: [0.5, 5.0]`
  (the proxy-sound range), `include_camels_params: true` with the five
  cosmological names per F0.4. Write the stage DataConfig once as
  `scripts/configs/data/config_dual_vae.yaml` and reuse it verbatim everywhere.
- **SplitSpec:** `train_frac: 0.7, val_frac: 0.1, test_frac: 0.2, seed: 100` --
  matching the existing cvae / vib_regressor / mlp_regressor run configs so all
  comparisons share folds.
- **EvaluationSpec:** `suppressed_thresholds: [0.95, 0.9, 0.8]` (runner default).

---

## Stage 1 — PCA dimensionality baselines and baseline-table refresh

**Goal:** linear reference points for both modalities, and a shared baseline
metric table, before any VAE exists. All later gates are relative to these.

Tasks:
1. New stage script `scripts/stage1_pca_dimensionality.py`: load the pinned
   DataConfig via `load_training_data`, apply `grouped_split` with the pinned
   SplitSpec, and fit sklearn PCA separately on X (f_gas) and on Y (SP(k)) using
   the train fold only. Produce, for n_components = 1..10 on the val fold:
   explained-variance-ratio curves and per-dimension reconstruction RMSE (raw
   target scale). Write a run record per modality (RunConfig model names
   `"pca_x_recon"` / `"pca_y_recon"` -- record-only labels, not registry entries)
   with the tables in `summary.json`, and save the two figures via `figure_dir`.
   [Amended per B5: the PCA-on-Y baseline additionally reports the
   suppressed-regime truncation error -- val reconstruction RMSE restricted to
   bins where TRUE SP(k) < t for t in {0.95, 0.9, 0.8}.]
2. Refresh the cross-modal baseline table on the pinned context: run
   `scripts/run.py` for `pca_linear`, `mlp_regressor`, and plain `mlp` with the
   pinned DataConfig and their existing run configs edited to the pinned
   SplitSpec. Collect val-split global RMSE and suppressed-regime RMSE into a
   short markdown table committed as
   `experiments/notes/dual_vae_baseline_table.md`, with the run_ids.

**Gate G1:**
- G1.1: PCA artifacts + run records exist and re-run reproducibly. Pass = yes.
- G1.2: Baseline table exists with run_ids for all three reference models on the
  pinned context. Pass = yes. (Reference points, not thresholds.)

Commit per GR8.

---

## Stage 2 — VAE-Y: autoencode SP(k) (Lin et al. methodological replication)

**Goal:** a validated low-dimensional generative model of SP(k), plus the
Lin-style checks adapted to this data.

Tasks:
1. Implement `src/fgas_spk/models/dual_vae_components.py` containing a single
   reusable `GaussianVAE` component class (torch, lazily imported), used for both
   modalities: MLP encoder emitting (mu, logvar), reparameterized sampling, MLP
   decoder, optional context vector concatenated into both encoder and decoder
   inputs (the existing plugins' `_context` pattern), Gaussian NLL with the
   learned per-bin `obs_logvar` head per F0.2, `obs_sigma()` accessor, linear KL
   warm-up (`anneal_epochs` as in the cvae plugin), internal `val_frac` holdout
   for best-epoch restore, `history` list of per-epoch dicts. Standardize inputs
   internally as the existing plugins do. Per-epoch history must include: total
   loss, recon NLL, total KL, per-dimension KL, and the per-dimension activity
   statistic A_j = Var(mu_j over the batch) / mean(sigma_j^2) (Lin et al.
   Eq. F21 analogue).
2. Stage script `scripts/stage2_vae_y.py`: train `GaussianVAE` on the train-fold
   Y with `X_params` context, for latent_dim_y in {2, 3, 4} (seeded, one run
   record each; RunConfig model label `"vae_spk"`, hyperparameters in
   `model_params`). [Amended per B1/C3: crossed with beta in
   {1, 0.1, 0.01, 0.001}.] Evaluate on the val fold: reconstruction RMSE (raw
   scale, posterior-mean decode per B4), final per-dimension KL and A_j, and
   fitted `obs_sigma()` per k-bin. [Amended per B5: plus the suppressed-regime
   truncation error.]
3. Lin-replication checks (val fold), written into the selected run's summary
   and as figures:
   (i) VAE-Y reconstruction RMSE vs the Stage 1 PCA-on-Y curve at matched
       dimension (their Fig. 12 analogue) -- [amended per B2: reported as a
       finding, not gated; the beta-sweep behaviour is part of the evidence];
   (ii) dimensionality behaviour across the {2, 3, 4} sweep: does a 3rd/4th
        dimension carry active KL, or collapse / duplicate (their Fig. 5
        analogue)? Report per-dimension KL and A_j evidence;
   (iii) Pearson correlations of posterior means mu2 against all 35 SB35
         parameters (their Fig. 1 bottom-panel analogue; requires a second data
         load with `camels_param_indices: null` for the correlation analysis
         only -- the model itself stays conditioned on the 5 cosmological
         parameters). Candidate paper figure.
4. Select latent_dim_y (and beta, per amended F0.2) by val reconstruction RMSE
   subject to the no-collapse criterion; record the selection and its run_id in
   the gate report.

**Gate G2 (as amended 2026-07-05, B2/B3):**
- G2.1' (replaces G2.1): selected VAE-Y val reconstruction RMSE (posterior-mean
  decode, raw target scale) <= 10% of the best Stage 1 cross-modal val RMSE on
  the current pinned context. The VAE-vs-PCA reconstruction comparison at
  matched dimension is reported as a finding, no longer gated.
- G2.2' (replaces G2.2): no retained latent dimension is collapsed,
  operationally: per-dim KL >= 0.01 nats at best epoch. The [1, 10] activity
  band is dropped. Delta-like posteriors (large A_j) are permitted and expected
  in this near-deterministic regime; document them, with the A_j values, in the
  run summary rather than gating on them.
- G2.3: the three replication checks reported, each labelled
  agree / partially agree / disagree with Lin et al.'s qualitative findings, one
  paragraph of evidence each. Disagreement is diagnostic, not failure (the
  target definition and data generation differ); two or more outright
  disagreements halt for human review.
- G2.4: GR7 reduction-convention audit present in the summary and consistent.

Commit per GR8.

---

## Stage 3 — VAE-X: autoencode f_gas

**Goal:** a validated low-dimensional representation of the observable, plus the
OOD reference statistics.

Tasks:
1. Stage script `scripts/stage3_vae_x.py`: reuse `GaussianVAE` on the train-fold
   X with the same context, latent_dim_x = 2 first (sweep {2, 3, 4, 6} only if
   G3.1' fails at 2). [Amended per B6: crossed with the F0.2 beta sweep.]
   RunConfig model label `"vae_fgas"`. Same diagnostics as Stage 2
   (reconstruction RMSE vs Stage 1 PCA-on-X reported as a finding,
   per-dimension KL, A_j, obs_sigma per R-bin).
2. OOD reference statistics on the val fold: the distribution of ||mu1|| and
   per-dimension mu1 ranges, written into the run summary. These are the
   detection baseline for the encoder-blow-up failure mode (the CVAE precedent:
   two test simulations with latent norms ~17-22 dominated test MSE).
3. Latent-parameter correlation analysis for mu1 (as in Stage 2 task 3(iii)) --
   whether f_gas latents organize by the same feedback parameters as SP(k)
   latents is itself a candidate paper point.

**Gate G3 (as amended 2026-07-05, B6):**
- G3.1' (downstream adequacy, replaces G3.1): on the train fold, fit two ridge
  regressions to the frozen mu2 targets -- one from mu1 (VAE-X codes), one from
  PCA-on-X scores at the same dimension. Decode both val predictions through
  the frozen decoder-Y to y-space. Gate: VAE-X-arm val y-space RMSE <= 1.05 x
  PCA-x-scores-arm val y-space RMSE (guardrail margin for seed-level noise).
  The VAE-X-vs-PCA-on-X reconstruction comparison is reported as a finding, not
  gated.
- G3.2': collapse detection only, as B3 (per-dim KL >= 0.01 nats at best epoch).
- G3.3: OOD reference statistics present in the run summary.

Commit per GR8.

---

## Stage 4 — Latent mapping ladder z1 → z2

**Goal:** the probabilistic core. The Stage 2/3 VAEs are frozen throughout.

Tasks:
1. Stage script `scripts/stage4_latent_map.py`: build (mu1, mu2) pairs from the
   frozen encoders on the train fold (per F0.5); same for val.
2. **Rung 1 -- ridge regression** z1 -> z2 (sklearn). Diagnostic baseline: if
   linear-in-latent already matches the direct `mlp_regressor` after decoding,
   the VAEs have effectively linearized the map (reportable either way).
3. **Rung 2 -- small MLP** z1 -> z2 (torch, GELU, same conventions).
4. **Rung 3 -- MDN** modelling p(z2 | z1): small MLP emitting mixture weights,
   means, and diagonal scales for K components (K in `model_params`, default 3;
   report sensitivity to K in {1, 3, 5} on the val fold).
5. Composite evaluation per rung, implemented as a helper (this becomes the
   Stage 5 plugin's predict path): x -> encoder_x posterior mean mu1 -> map ->
   decoder_y. Point prediction: rungs 1-2 decode M(mu1); rung 3 decodes the
   MDN mixture mean. Predictive distribution (rung 3 only): sample
   z2 ~ p(z2|mu1), decode each, add decoder-Y observation noise per bin from
   `obs_sigma()`; per-k-bin intervals from these y-space samples.
   One run record per rung (model labels `"latent_map_ridge"` /
   `"latent_map_mlp"` / `"latent_map_mdn"`), with the frozen component run_ids
   recorded in the summary (additive provenance).

**Gate G4:**
- G4.1: composite val global RMSE and suppressed-regime RMSE tabulated for all
  three rungs against the Stage 1 baseline table, appended to
  `experiments/notes/dual_vae_baseline_table.md`. Additionally report the rung-2
  vs `mlp_regressor` gap as the measured information-bottleneck cost (a number
  to report, not a threshold).
- G4.2: rung-3 calibration on val: empirical coverage at nominal 68% and 95%,
  per k-bin and pooled. Pass = pooled coverage within +/-10 percentage points of
  nominal at both levels. Outside the band: stop and diagnose before Stage 5 (no
  predictive spread reaches paper figures uncalibrated).
- G4.3: per-curve RMSE ranking on val by `sim_index`; cross-reference the worst
  curves against the G3.3 ||mu1|| reference and report whether composite
  failures coincide with extreme encoder-X codes. Analysis present = pass;
  findings reported either way.

Commit per GR8.

---

## Stage 5 — Register `dual_vae` and evaluate once on test

**Goal:** one registered, protocol-conforming model; one honest test evaluation
through the standard runner.

Tasks:
1. Implement `src/fgas_spk/models/dual_vae.py`, registered as `dual_vae`,
   conforming to `ProfileToSpk` and the construction contract
   (`Model(seed=..., **model_params)`, tolerant of unexpected keys):
   - `fit(training_data)` trains VAE-Y, then VAE-X, then the selected mapping
     rung end-to-end in one process (F0.9 hybrid), using hyperparameters from
     `model_params` (populated from the Stage 2-4 selections). Internal
     holdouts as in the components. `history` concatenates the three phases'
     per-epoch traces with a `phase` key.
   - `predict(X, X_cond, X_params)` -> point prediction as defined in Stage 4
     task 5.
   - `predict_samples(...)` (rung-3 configurations only) -> y-space samples per
     Stage 4 task 5, seeded and MPS-safe following the cvae plugin's pattern.
     Docstring must state the F0.5 spread semantics (mapping density +
     decoder-Y observation noise; excludes encoder-X posterior spread and input
     measurement noise).
   - `latents(X, ...)` -> mu1 (so the runner's latent-norm diagnostics apply);
     an additional non-protocol `latents_y(y, ...)` -> mu2 for inspection.
   - `obs_sigma()` -> decoder-Y per-bin noise on the raw target scale.
   Add the plugin import to `fgas_spk/models/__init__.py`; keep the package
   import torch-free.
   [Amended per B7: the plugin is codec-agnostic -- `model_params` keys
   `codec_x` and `codec_y`, each in {"vae", "pca"}, both defaulting to "vae".
   The "pca" codec: PCA scores as codes, PCA inverse transform as decode, and
   per-bin train-fold reconstruction-residual standard deviations serving the
   obs_sigma() role (honestly interpreted as code-truncation error) so
   predict_samples remains well-defined for any codec pair. Registry name stays
   `dual_vae`; the docstring states the codec-agnostic design.]
2. Write `scripts/configs/run/dual_vae.yaml` mirroring the existing run-config
   style, with the pinned SplitSpec and the selected hyperparameters, fully
   commented.
3. One run through `scripts/run.py` with the pinned DataConfig. The runner's
   standard diagnostics (per-curve, suppressed-regime, coverage, latent norms,
   figures) constitute the test evaluation -- executed once (GR6 as amended:
   batched with Stage 5b, all configurations declared first).
4. Comparison note `experiments/notes/dual_vae_vs_baselines.md` (~1 page):
   accuracy, calibration, and OOD behaviour vs `pca_linear`, `mlp_regressor`,
   plain `mlp`, and the current CVAE numbers, with the G4.1 bottleneck-cost
   figure stated and every claim pointing at a committed run_id or artifact.
5. If not already done in passing: confirm `model.history` per-epoch traces are
   threaded into `write_run_record` for all new models (existing repo to-do).

**Gate G5:**
- G5.1: test evaluation executed exactly once; run record, checkpoint,
  and figures committed; the composite's sanity checks vs the Stage 4 val-fold
  results are consistent (the retrained composite should reproduce the staged
  results within seed-level variation -- a large discrepancy is a
  stop-and-report condition, since it implies the staged and registered
  pipelines silently differ).
- G5.2: comparison note exists; every claim traceable to an artifact.

Commit per GR8.

---

## Stage 5b (added 2026-07-05, B8) — codec ablation

After Stage 5:
- Arms: (vae, vae) -- already evaluated in Stage 5 -- and (pca, pca) mandatory;
  the two cross arms (pca_x, vae_y) and (vae_x, pca_y) optional if cheap.
- Fairness constraints: each arm re-fits the FULL mapping ladder on its own
  code space (a mapping trained on one codec's mu-space is never reused for
  another); identical folds, seeds, hyperparameter grids, and evaluation
  context; every arm reports global RMSE, suppressed-regime RMSE, per-curve
  RMSE ranking, and (rung 3) coverage. A comparison missing the
  suppressed-regime numbers is invalid.
- Deliverable: `experiments/notes/dual_vae_codec_ablation.md`, every claim
  traceable to a run_id.
- Gate G5b: report complete; both mandatory arms evaluated under identical
  conditions. No accuracy threshold -- the comparison IS the deliverable.

---

## Stage 6 (optional; only after G5) — End-to-end fine-tune

Fine-tune encoder-X + mapping with a y-prediction NLL through **frozen**
decoder-Y (decoder-Y stays frozen to preserve the y-manifold constraint).
Register separately as `dual_vae_ft`; the frozen-pipeline `dual_vae` remains
untouched as the diagnosable baseline. Re-run the full Stage 5 evaluation for
`dual_vae_ft`, including G4.2 coverage in full -- fine-tuning can silently break
calibration, and a calibration regression relative to `dual_vae` is a
stop-and-report condition, not a tolerable trade-off.

---

## Explicit non-goals

- No per-halo dataset work (sequenced after box-mean validation).
- No paired-DMO target (deferred; env.json provenance note stands).
- No Gen 1 CAMELS data acquisition (F0.1 = (b)).
- No input-noise propagation / noise augmentation (F0.6 deferred; the F0.5
  spread-semantics docstring marks where it reconnects).
- No NPE-in-observable-space via `sbi` (separate deferred track).
- No snapshot stacking or multi-snapshot conditioning (F0.7).
- No breaking changes to the run-record schema, splits, or existing registered
  models.

---
---

# Maintainer amendment (2026-07-05): G2 resolution + data tag correction

This is the maintainer decision required by GR5 and the spec veto rule,
recorded verbatim. Where this amendment conflicts with the original spec, this
amendment wins.

Summary of the decision: the G2.1/G2.2 failures are accepted as a **spec design
error**, not an implementation failure. G2.1 conflated a replication finding
(does a Lin-style VAE beat PCA on this target?) with the actual gating question
(is the y-codec adequate for the pipeline?), and G2.2 imported Lin et al.'s
activity band without their tuned KL-scaling regime (their lambda was tuned over
[1e-4, 1e-1]; beta = 1 was never their operating point). The VAE codec path
continues. PCA is NOT swapped in as the codec; it enters later as a controlled
ablation (new Stage 5b).

## A. Data tag correction (do this first; it invalidates all prior numbers)

A1. The pinned context used tag `20260617`. The current data variant is
    `20260702`. Edit `scripts/configs/data/config_dual_vae.yaml` to
    `tag: '20260702'`. All gates from now on are evaluated on this tag only.

A2. **Re-run, do not rewrite.** Existing run records and ledger lines for the
    `20260617` runs remain untouched (append-only provenance). Re-run Stage 1 in
    full on the new tag: the PCA-on-X / PCA-on-Y dimensionality baselines and
    the `pca_linear` / `mlp_regressor` / `mlp` baseline refresh. Regenerate
    `experiments/notes/dual_vae_baseline_table.md` with the new-tag numbers and
    run_ids, keeping the old table under a clearly-marked
    "superseded (tag 20260617)" heading.

A3. Every cross-run comparison from here on must be same-tag. No gate may
    compare a new-tag number against an old-tag number.

A4. Confirm whether anything else in the dataset changed between tags beyond
    content (shapes, k-grid, radial grid, parameter stamping). Report the
    dataset `__meta__` diff in the Stage 1 re-run gate report. If shapes
    changed, say so before proceeding to Stage 2.

## B. Spec amendments

B1. **F0.2 amended — beta becomes a tuned hyperparameter.** The fixed
    beta = 1.0 rationale ("principled with the learned noise head") applies to
    conditional models p(y|x), not to unconditional compression autoencoders,
    where beta prices information rate and beta = 1 admits a rate–distortion
    fixed point. Both `GaussianVAE` usages now sweep
    beta in {1, 0.1, 0.01, 0.001}, selected by val reconstruction RMSE
    (posterior-mean decode) subject to the no-collapse criterion (B3). The
    noise head, obs_sigma() accessor, and all other F0.2 provisions stand.

B2. **G2.1 replaced by G2.1' (codec adequacy, runtime-relative).**
    G2.1': selected VAE-Y val reconstruction RMSE (posterior-mean decode, raw
    target scale) <= 10% of the best Stage 1 cross-modal val RMSE on the
    current pinned context. The VAE-vs-PCA reconstruction comparison at matched
    dimension is REPORTED AS A FINDING (part of the Lin replication verdicts),
    no longer gated.

B3. **G2.2 replaced by G2.2' (collapse detection only).**
    G2.2': no retained latent dimension is collapsed, operationally: per-dim
    KL >= 0.01 nats at best epoch. The [1, 10] activity band is dropped.
    Delta-like posteriors (large A_j) are permitted and expected in this
    near-deterministic regime; document them, with the A_j values, in the run
    summary rather than gating on them.

B4. **Reporting standard: posterior-mean decode.** All reported reconstruction
    RMSEs use posterior-mean decoding. Confirm which convention the completed
    `20260617` Stage 2 runs used; if sampled decoding was in the reported
    numbers, note that in the gate report (a reporting-hygiene note, no re-run
    of old-tag results).

B5. **New standard codec diagnostic — suppressed-regime truncation error.**
    For every codec run (VAE and, in Stage 5b, PCA): val reconstruction RMSE
    restricted to bins where TRUE SP(k) < t for t in {0.95, 0.9, 0.8}
    (ground-truth-masked, never prediction-masked), alongside the global
    number. Purpose: codec truncation error may be heterogeneous, concentrated
    in the rare deeply-suppressed curves that the global RMSE averages away.
    Add this to the Stage 1 PCA-on-Y baselines during the A2 re-run as well.

B6. **Stage 3 gates amended preemptively (same logic as B2/B3).**
    - G3.1' (downstream adequacy, replaces G3.1): on the train fold, fit two
      ridge regressions to the frozen mu2 targets — one from mu1 (VAE-X codes),
      one from PCA-on-X scores at the same dimension. Decode both val
      predictions through the frozen decoder-Y to y-space. Gate: VAE-X-arm val
      y-space RMSE <= 1.05 x PCA-x-scores-arm val y-space RMSE (guardrail
      margin for seed-level noise). The VAE-X-vs-PCA-on-X reconstruction
      comparison is reported as a finding, not gated.
    - G3.2' : collapse detection only, as B3.
    - G3.3 (OOD reference statistics) unchanged.
    - F0.2-as-amended (beta sweep) applies to VAE-X.

B7. **Composite is codec-agnostic.** `dual_vae` gains `model_params` keys
    `codec_x` and `codec_y`, each in {"vae", "pca"}, both defaulting to "vae".
    The "pca" codec: PCA scores as codes, PCA inverse transform as decode, and
    per-bin train-fold reconstruction-residual standard deviations serving the
    obs_sigma() role (honestly interpreted as code-truncation error) so
    predict_samples remains well-defined for any codec pair. Registry name
    stays `dual_vae` (defaults make it accurate); docstring states the
    codec-agnostic design.

B8. **New Stage 5b — codec ablation (this is the PCA comparison, moved to the
    end of the pipeline where it belongs).** After Stage 5:
    - Arms: (vae, vae) — already evaluated in Stage 5 — and (pca, pca)
      mandatory; the two cross arms (pca_x, vae_y) and (vae_x, pca_y) optional
      if cheap.
    - Fairness constraints: each arm re-fits the FULL mapping ladder on its own
      code space (a mapping trained on one codec's mu-space is never reused for
      another); identical folds, seeds, hyperparameter grids, and evaluation
      context; every arm reports global RMSE, suppressed-regime RMSE,
      per-curve RMSE ranking, and (rung 3) coverage. A comparison missing the
      suppressed-regime numbers is invalid.
    - Deliverable: `experiments/notes/dual_vae_codec_ablation.md`, every claim
      traceable to a run_id.
    - Gate G5b: report complete; both mandatory arms evaluated under identical
      conditions. No accuracy threshold — the comparison IS the deliverable.

B9. **GR6 amended to permit Stage 5b.** Test data is evaluated exactly once per
    registered model configuration, with all configurations declared before any
    test evaluation runs, batched at Stage 5 / Stage 5b. No configuration is
    selected, tuned, or modified in response to a test number. Everything else
    about GR6 (no tuning against test in Stages 1–4) stands.

## C. Resume plan (in order)

C1. Commit the spec amendment (GR8 discipline continues throughout).
C2. Execute A1–A4 (tag correction; Stage 1 re-run incl. B5 diagnostic; new
    baseline table; meta diff report). Re-evaluate Gate G1 on the new tag.
C3. Re-run Stage 2 on the new tag under the amended spec: sweep
    latent_dim_y in {2, 3, 4} x beta in {1, 0.1, 0.01, 0.001} (12 short runs;
    seeded; one run record each). Select per B1. Evaluate G2.1', G2.2', G2.3
    (replication verdicts re-issued on the new tag, now including the
    VAE-vs-PCA finding per B2 and the beta-sweep behaviour as evidence in
    verdict (i)), and G2.4. The latent–parameter correlation analysis (task
    3(iii)) is re-run on the new tag.
C4. Update `experiments/notes/dual_vae_stage2_gate_report.md`: mark the
    20260617 report superseded (keep it), append the new-tag report.
C5. On G2' pass, proceed to Stage 3 under the B6 amendments. Stages 4 and 5
    proceed as originally specified except where amended above (B7, B9), then
    Stage 5b.

## D. Housekeeping

D1. The two pre-existing `tests/test_fgas_spk_builder.py` failures
    (`KeyError: 'r12_mpch'`) stay out of this work: do not fix them in this
    line of commits. Open a note in the ledger or an issue and leave them for a
    separate branch.
D2. The stale F0.2 cross-reference ("same noise-head design as the repo's
    CVAE") is acknowledged as a spec error — the repo CVAE has no noise head;
    that was a planned fix, not landed code. Your self-contained implementation
    was the correct reading. Fix the spec text to describe the head on its own
    terms while amending F0.2 per B1.
D3. Scientific framing guard, for every note and report from here on: the
    finding "SP(k) given cosmology is ~2-factor and nearly linear" concerns the
    y-manifold's intrinsic geometry. It does NOT imply the f_gas -> SP(k) MAP
    is linear (the map's genuine nonlinearity is established separately by the
    mlp-vs-pca_linear scatter behaviour). Do not let any summary sentence
    conflate the two.

---
---

# Maintainer amendment 2 (2026-07-05): Stage 3.0 forensics + gate hardening

This is the maintainer decision required by GR5 for the Stage 3 halt, recorded
verbatim. **Do not amend F0.2, do not modify the VAE-X likelihood, and do not
launch Stage 4.** All five options in the Stage 3 gate report are deferred:
they treat the symptom while assuming the input data is legitimate, and that
assumption is now the thing under test.

Rationale, in brief. (a) f_gas is a fraction-like quantity; fitted per-bin
obs_sigma up to 336 and per-example 50-300 sigma residuals describe values no
gas-fraction profile can physically contain -- heavy projection-noise tails
produce ~5-20 sigma outliers, not 300. (b) The A4 diff isolates the suspect:
between tags, X was re-binned 17 -> 20 by a new upstream code path while y is
bitwise-identical -- the one modality that exploded is the one a new code path
produces, and this pipeline has precedent for upstream bugs (the halo-ordering
correction in the builder). (c) The PCA arm of G3.1' failing equally (~0.073
both arms) is diagnostic: PCA uses no likelihood, so "Gaussian NLL is
too thin-tailed" cannot explain it, whereas a poisoned input matrix rotating
the leading components explains both arms at once. (d) The discriminative/
generative asymmetry (mlp fine, VAE-X diverging from epoch 1) is exactly the
signature of bad rows a generative model must pay for in reconstruction NLL.
Consequence: **every new-tag number involving X is provisional** -- including
mlp's 0.0429 and therefore the G2.1' bar. Stage 2's VAE-Y results themselves
are insulated (VAE-Y consumes y and params only).

## Stage 3.0 — X-matrix forensics (new sub-stage; blocks Stage 3 resumption)

All diagnostics run on the raw (pre-standardization) X as served by
`load_training_data` under the pinned config, unless stated otherwise. One
run-record-per-analysis is not required; a single recorded diagnostics run with
a committed note is sufficient (GR8 applies).

3.0.1 **Value census, both tags.** Per-radial-bin quantiles of X (min, 0.1%,
      1%, 50%, 99%, 99.9%, max), plus counts of non-finite values and exact
      zeros, on tag 20260702 AND tag 20260617. The grids differ, so compare
      (i) the pooled value distribution across all bins and (ii) per-bin
      profiles side by side. The question: did the value distribution change
      character between tags, and where.

3.0.2 **Offender census.** Flag cells (sim_index, nd, bin) by two tiers:
      - hard-implausibility: value outside [-0.5, 2.0] -- a PROVISIONAL range;
        confirm the f_gas definition (and hence the true physical bound) from
        the dataset `__meta__` and builder documentation, state what you find,
        and flag the range for maintainer confirmation in the report;
      - distributional: robust per-bin z-score (median/MAD) with |z| > 10.
      Classify the spatial pattern: a few isolated simulations (cf. the known
      OOD precedent, sims 919/1000), a systematic outer-bin effect, or
      scattered. Report which folds the offending simulations fall in.

3.0.3 **PCA poisoning check.** Compare PCA-on-X leading components across tags:
      loading profiles, explained-variance spectra, and whether any leading
      component's scores are dominated by the offender rows from 3.0.2. A
      component that "points at" a handful of rows confirms the mechanism that
      voided G3.1'.

3.0.4 **Rebinning code inspection.** Locate the upstream re-binning code path
      that produced the 20-point grid (builder and/or the upstream
      SimulationStacker products). If inspectable in-repo, look specifically
      for empty-annulus / division-by-near-zero / sentinel-fill edge cases at
      the flagged bins. If the code path is not accessible from this repo,
      instead produce a compact offender report (sim ids, nd indices, radial
      bins, values) formatted for an upstream bug report, as a committed
      artifact.

3.0.5 **Attribution of the mlp improvement (diagnostic only, not a recorded
      baseline).** Recompute the mlp val RMSE with the 3.0.2 offender rows
      excluded, same folds otherwise. Purpose: measure how much of the
      0.0617 -> 0.0429 improvement survives cleaning, i.e. whether the "sharp
      improvement on re-binned profiles" is real signal or an artifact of the
      same pathology. Label the result diagnostic-only in the note.

3.0.6 **Deliverable and STOP.** Commit
      `experiments/notes/dual_vae_stage3_forensics.md` with the evidence and a
      verdict: one of
      - **corruption** (pathological values traceable to the re-binning or
        another upstream defect),
      - **genuine heavy tails** (values physically admissible under the
        confirmed f_gas definition; distribution heavy-tailed by nature),
      - **mixed / inconclusive**,
      plus a recommended branch from Section B. Then STOP for maintainer
      decision. Do not proceed down any branch autonomously -- Branch A in
      particular involves upstream coordination and physics-adjacent exclusion
      choices that are maintainer calls (CLAUDE.md rule 3).

## B. Pre-specified branches (executed only after the maintainer's reply)

B-A **Corruption.** Resolution is upstream fix (new data tag; maintainer
    coordinates) or recorded exclusion (a `sim_ids` exclusion list and/or a
    radial crop via `radial_range_mpch`, justified by data quality, with the
    forensics note as provenance). Then: re-run Stage 1 on the corrected data
    (third baseline-table regeneration, superseded headings as before),
    re-evaluate G2.1' against the regenerated bar (no VAE-Y retrain needed
    unless y changes -- it should not), and re-run the Stage 3 sweep. **No F0.2
    amendment.**

B-B **Genuine heavy tails.** Amend F0.2 for the X modality: primary fix is an
    input transform (asinh with per-bin robust scaling) applied inside the
    VAE-X preprocessing AND to the PCA-on-X diagnostics, since a transform
    repairs both the likelihood and every PCA-based comparison arm; Student-t
    likelihood is the fallback if the transform alone is insufficient. Record
    in the amendment that the long-run robustness choice should be driven by
    the real DESI+ACT+HSC f_gas noise properties, not CAMELS artifacts.

B-C **Mixed.** Maintainer specifies the combination; do not guess.

## C. Gate hardening (applies from now on, regardless of branch)

C1 **G3.1' floor condition (anti-vacuous-pass).** The ratio test is valid only
   if the reference arm clears an absolute floor: the PCA-x-scores ridge arm's
   val y-space RMSE must be <= the Stage 1 `pca_linear` baseline val RMSE on
   the same tag and folds. If the floor fails, G3.1' fails regardless of the
   ratio -- a mutual-failure ratio of ~1 must never pass again.

C2 **G3.4 (new, generic training-sanity gate).** Any selected model whose
   best-epoch restore is epoch 0, or whose internal-holdout objective never
   improves over initialization, is an automatic gate failure for that
   configuration, whatever other criteria say.

C3 **Stage 2 status marked conditional.** The Stage 2 gate report gains a note:
   G2.1' was evaluated against a bar (best cross-modal baseline 0.00429) that
   is provisional pending the Stage 3.0 verdict; the gate is re-evaluated in
   Branch A. The VAE-Y model itself and the G2.2'/G2.3/G2.4 conclusions are
   unaffected (y bitwise-identical across tags).

## D. Housekeeping

D1 Not launching Stage 4 on the epoch-0 codes was the correct call; Stage 4
   remains parked exactly as committed (34166de) and untouched until an honest
   VAE-X exists.
D2 GR8 commit discipline continues; the forensics note commit is tagged
   `[stage 3.0 verdict: <corruption|heavy-tails|mixed>]`.

---
---

# Maintainer amendment 3 (2026-07-06): Branch A — builder statistic fix + rebuild

Recorded verbatim. Resolves the Stage 3.0 STOP: verdict **corruption**
accepted, located in this repo's builder statistic (mean of per-halo ratios
with an unguarded near-zero denominator), present in both tags; re-binning
exonerated.

**Definition ruling (CLAUDE.md rule 3, resolved by maintainer):** the correct
statistic is the ratio of stacked profiles — the established convention of the
maintainer's measurements paper and data-side pipeline (stack ΔΣ_ionized over
the selected halos, stack ΔΣ_total likewise, divide once, normalize by f_b).
Note the identity: ratio of per-halo *means* equals ratio of per-halo *sums*
over the same halo set, so "mean then ratio" and "sum then ratio" are the same
estimator; implement whichever is cleaner.

**Not in scope:** no F0.2 amendment, no VAE-X likelihood or preprocessing
change, no gate changes. The hardened gates (G3.1' floor, G3.4, C3) stand
exactly as committed — running the fixed data through unchanged gates is the
experiment. Upstream (lindajin) products need no changes and no bug report.

E1. **Builder fix.** E1.1 ratio-of-stacked-means in `build_fgas_spk_dataset`
    (pool halos x projections per bin, divide once, / f_b). E1.2 verify the
    pooling convention against the reference notebook first; STOP on conflict.
    E1.3 docstring states the statistic and why.
E2. **Guards, provenance stamp, tests.** E2.1 build-time HARD FAIL on
    non-finite or out-of-`fgas_sanity_range` values (default [-1.0, 3.0],
    PROVISIONAL); realized per-bin min/max recorded in `__meta__`. E2.2
    `fgas_definition` stamp in `__meta__` (additive; the discriminator against
    old-statistic datasets). E2.3 regression tests (near-zero-denominator cell;
    stamp + realized range). r12_mpch failures stay out of scope (D1).
E3. **Rebuild** tag `'20260706'` (pinned variant). E3.2 identity assertions
    vs 20260702: spk, k grid, radial grid, params bitwise-identical; ONLY fgas
    may differ; STOP on any other difference. E3.3 repin config; A3 same-tag
    rule applies.
E4. **Stage 1 re-run** (third regeneration; superseded headings kept). E4.2
    new deliverable: clean-data re-verification of the map-nonlinearity
    evidence (val-fold pred-vs-true scatters for pca_linear / mlp /
    mlp_regressor, global + suppressed RMSE, predicted-value ranges) —
    reported finding, no gate. E4.3 re-evaluate G1.
E5. **Stage 2 re-run** (full sweep; runtime-relative G2.1' bar). E5.2 must
    reproduce the 20260702 results to seed-level variation (y and params
    unchanged) — material deviation is STOP-and-report. E5.3 resolve the C3
    conditional status.
E6. **Stage 3 re-run** with unchanged likelihood and the hardened gates. On
    G3' pass: unpark Stage 4 (34166de) and proceed through Stages 4, 5, 5b
    without further maintainer stops, subject to GR5. If Stage 3 fails on
    clean data, STOP per GR5 (the deferred options return, with tails then
    known to be genuine).
E7. **Housekeeping.** E7.1 resolution addendum to the forensics note. E7.2 the
    3.0.5 diagnostic (0.0403) predicts nothing about the new baselines; do not
    cite it as an expectation. E7.3 ledger note: all pre-20260706 X-dependent
    results are superseded by definition change; the stamp is the
    discriminator.

---
---

# Maintainer amendment 4 (2026-07-06): quadrature adequacy gate + radial crop + beta extension

Recorded verbatim; resolves the E5 STOP. The G2.1' failure is judged an
artifact of the gate's operationalization, not of the codec: the
10%-of-baseline fraction was a proxy for the actual adequacy criterion (codec
error subdominant to the pipeline floor in quadrature); the codec did not
regress — the bar moved because the baseline improved on clean data. The
Stage 3 options menu remains closed.

F1. **G2.1' reformed to the quadrature criterion.** Let `base` = best
    same-tag, same-config cross-modal val RMSE from the current Stage 1
    table, `codec` = selected VAE-Y val reconstruction RMSE (posterior-mean
    decode, raw scale). Gate: sqrt(base^2 + codec^2) / base - 1 <= 0.01,
    equivalently codec <= base * sqrt(1.01^2 - 1) ~= 0.1418 * base.
    Semantics: the y-codec may inflate the achievable pipeline floor by at
    most 1%. (Derivation: if the pipeline's irreducible error is `base` and
    the codec contributes independently, the floor with the codec is the
    quadrature sum; requiring <= 1% inflation gives the bound.) The former
    10% linear fraction is superseded as a proxy. Runtime-relative; no
    absolute bar. F1.3 audit: no other budget-fraction criteria exist in the
    spec (G3.1' is a comparative guardrail, not a budget criterion — left).
F2. **Radial crop (maintainer physics ruling).** Modelled radial range
    cropped to R < 10 in the stored grid's native units (comoving Mpc/h,
    verified from the sidecar `units` block and the `radii_mpch` key).
    Implemented via the DataConfig `radial_range_mpch` load-time crop; no
    rebuild — the crop is modelling scope, not a data change. Rationale:
    ratio-of-stacks cured the per-halo zero-crossing defect, but the stacked
    denominator itself legitimately decays toward zero at the outermost
    radii, making the statistic intrinsically ill-conditioned there; the
    outermost bins are excluded pending a physics assessment of the
    usable-signal boundary against the real-data stacks. The boundary value
    10 is preliminary and maintainer-owned.
F3. **Guard redesign (builder; NO rebuild of 20260706).** Two-tier guard
    parameterized by `modelled_range` (native radial units, set to the F2
    crop): non-finite hard-fails anywhere; within modelled_range hard-fail
    outside `fgas_sanity_range` (default [-1, 3]); outside modelled_range
    census (count/min/max per bin) into `__meta__`, warn, don't fail. F3.2:
    20260706 is NOT rebuilt (the guard changes no values); its stamped
    (-15, 130) range reflects the documented escape hatch; two-tier semantics
    apply from the next build. F3.3: census the pinned (nd-2, cropped) slice
    of the current build against [-1, 3] BEFORE anything trains; any retained
    violation is a STOP (do not widen, do not move the crop autonomously).
    F3.4: tests updated for two-tier behaviour.
F4. **Stage 1 fourth regeneration** under the cropped config (PCA baselines
    with B5, baseline refresh, table regenerated; superseded headings keyed
    by tag AND radial range). F4.2 re-issue the map-nonlinearity finding on
    cropped data (the citable version). F4.3 re-evaluate G1.
F5. **Stage 2 — capped beta extension, then the reformed gate.** F5.1 extend
    the sweep by exactly one decade: beta = 1e-4 at latent_dim_y in {2,3,4}
    (three runs; same seeds/protocol). **Cap: one-time, bounded
    evidence-completion of the rate-distortion curve; no further beta
    extensions in any later amendment cycle.** F5.2 no retrain of existing
    configurations (y/params unchanged by the crop; the 20260706 sweep
    stands); select per B1 over the full sweep including the new decade.
    F5.3 evaluate reformed G2.1' against the F4 best cross-modal RMSE;
    re-evaluate G2.2'/G2.4 if the selection changed; update G2.3 verdict (i)
    with the completed rate-distortion curve. F5.4 optional finding run:
    latent_dim_y = 6 at the selected beta (not gated; excluded from codec
    selection). F5.5 update the Stage 2 gate report.
F6. **Stage 3 and onward.** F6.1 on G2' pass: Stage 3 on the cropped config,
    hardened gates, G3.1' floor recomputed from the F4 pca_linear baseline;
    no VAE-X likelihood/preprocessing changes (E6.3 logic stands). F6.2 on
    G3' pass: unpark Stage 4 and proceed through Stages 4, 5, 5b without
    further maintainer stops, subject to GR5.
F7. **Housekeeping.** F7.1 notes citing the old narrow-band-collapse scatter
    or historical headline RMSEs (0.0666 / 0.078 line) gain superseded
    pointers to the F4.2 finding. F7.2 forensics note gains a final addendum
    (guard redesign, crop ruling, F3.3 census outcome). F7.3 no expectations
    about post-crop baselines are encoded anywhere; report what the data
    shows.

---
---

# Maintainer amendment 5 (2026-07-06): retire G3.1' — probe-based codec selection at Stage 4

Recorded verbatim; resolves the Stage 3 STOP. G3.1' is retired as
**structurally malformed**: its C1 floor compared a two-stage pathway
(x -> linear ridge -> latent bottleneck, z-space objective -> nonlinear
decode) against a direct full-rank linear map (`pca_linear`) — not
like-for-like. Evidence: linear-probe ceiling from the complete profile
0.0422 > floor 0.0385 while the oracle sits at 0.00192; the gate was
unreachable for any codec. The nonlinear probe the gate lacked is Stage 4's
rung 2, so codec selection and adequacy move there. No new Stage 3 machinery;
no retraining — the 16 committed Stage 3 configurations are consumed as
frozen candidates. [H5.1: the amendment-2 C1 floor is hereby excised from the
hardened-gates set; G3.4 and the other C-series hardening remain in force.]

H1. **Retire G3.1' and preserve its findings.** Stage 3 gate set is now
    G3.2' / G3.3 / G3.4; the VAE-X-vs-PCA-on-X reconstruction comparison
    remains a reported finding. Promoted findings (numbers + run_ids in
    `experiments/notes/`): F-1 (VAE-X codes 31-48% less mu2-informative than
    PCA-x scores at d >= 3 under an identical LINEAR probe; parity at d=2
    low beta; caveat: a linear probe cannot read nonlinearly-coded
    information — H3 re-measures under the nonlinear probe) and F-2 (the B1
    best-X-recon selection rule anti-correlates with downstream adequacy:
    1.48 worst vs 0.989 best ratio). H1.4: one-line sanity note attributing
    the full-profile-ridge (0.0422) vs 6-dim-PCA-arm (0.0413) inversion
    after a lambda-grid check.
H2. **Stage 3 output redefined:** every grid configuration passing G3.2' and
    G3.4 enters the candidate set carried to Stage 4 (run_ids listed in the
    Stage 3 gate report). VAE-Y selection unaffected (ld=3, beta=1e-4).
H3. **New Stage 4.0 — probe-based codec selection and adequacy.**
    H3.1 fixed probe: the Stage 4 rung-2 MLP recipe, frozen in the script,
    no per-arm tuning, 3 seeds per arm; metric = mean over seeds of val
    y-space RMSE after decoding probe outputs through the frozen VAE-Y
    decoder (targets mu2, fit in z-space, judged in y-space; the objective
    mismatch is inherent to F0.5, identical across arms, and is what Stage 6
    later addresses). H3.2 arms: (a) ceiling = full cropped profile;
    (b) reference = PCA-x scores d in {2, 3, 4, 6}; (c) candidates = mu1
    from every Stage 3 candidate. H3.3 **G4.0a (STOP on fail):** the ceiling
    arm must beat the current Stage 1 `pca_linear` val RMSE (runtime-
    relative) — else the dual-codec architecture is unviable on this data.
    H3.4 selection: best candidate probe metric; tiebreak smaller latent
    dimension, then better X-reconstruction. H3.5 **G4.0b (graded):**
    ratio = selected / best reference arm; <= 1.05 clean pass;
    (1.05, 1.25] proceed with documented caveat (5b adjudicates);
    > 1.25 STOP. Constants maintainer-set and amendable. H3.6 reported
    findings (no gates, no encoded expectations): ceiling-vs-selected gap,
    ceiling-vs-`mlp_regressor` gap, and the F-1 re-measurement under the
    nonlinear probe.
H4. **Ladder proceeds on the selected codec:** rungs 1-3 and gates G4.1 /
    G4.2 / G4.3 unchanged (rung 2 may reuse the selected candidate's probe
    fits if seeds/protocol match). On G4 pass: Stages 5 and 5b per spec, no
    further maintainer stops, subject to GR5; the mandatory 5b (pca, pca)
    arm gives any G4.0b-documented gap its definitive full-metric
    comparison.
H5. **Housekeeping.** Spec C1 floor excised (pointer above); Stage 3 gate
    report updated supersede-and-append (the FAIL stands as history); no
    expectations about probe outcomes encoded anywhere.

---
---

# Maintainer amendment 6 (2026-07-06): G4.0b resolution — dual-track ladder

Recorded verbatim; resolves the Stage 4.0 STOP. The tripwire is resolved by
running the full Stage 4 ladder on BOTH codec arms and deciding the primary
configuration by a criterion fixed here, before the ladder numbers exist: the
probe metric that fired the tripwire is a single global y-space RMSE, and the
project's own history (the pca_linear suppressed-regime lesson) forbids
letting a single global number decide a model choice; symmetrically, the F-1
nonlinear-probe result forbids ignoring the deficit. G4.0b's role is served;
no further codec gates are added.

I1. **Immediate:** suppressed-regime breakdown (TRUE SP(k) < t, t in
    {0.95, 0.9, 0.8}, truth-masked, with per-threshold N) of the Stage 4.0
    ceiling, pca_scores_d4, and selected-VAE arms, appended to the Stage 4.0
    gate report. Context for the record; the I3 criterion, not this table,
    decides the primary.
I2. **Dual-track Stage 4.** Arm V: codec_x = vae (ld=3, beta=0.01, the H3.4
    selection). Arm P: codec_x = pca, d = 4. Both consume the frozen VAE-Y
    (ld=3, beta=1e-4). Full ladder per arm (rungs 1-3, identical folds/
    seeds/protocols); G4.1 two-armed, G4.2 and G4.3 per arm (PCA score norms
    serve the ||mu1|| role for Arm P). Mappings re-fit per arm on its own
    code space; nothing reused across arms.
I3. **Primary criterion (fixed now, applied after).** On val rung-3: Arm X
    is primary if at least as good as the other on ALL of — global RMSE;
    suppressed-regime RMSE at every threshold with N >= 30; pooled-coverage
    distance from nominal at both 68% and 95%. Seed-level ties count as "at
    least as good". Neither dominates -> STOP per GR5 with the two-armed
    table (maintainer scientific choice). The non-primary arm is retained as
    a first-class ablation arm.
I4. **Test declaration (B9), fixed now:** the test batch is Arm V rung-3,
    Arm P rung-3, and (pca, pca). Primary designation on val evidence BEFORE
    any test number is seen; test reports, never re-designates. Stage 5 runs
    the primary through scripts/run.py; Stage 5b's mandatory arms are the
    other two declared configurations (a test-set evaluation and write-up,
    not a re-fit); 5b fairness and full-metric reporting stand. The Stage 5
    comparison note gains a required paragraph on the H3.6 architecture-cost
    finding (two-stage ceiling vs direct mlp_regressor, 1.88x at the probe;
    report the ladder's own version), framed as the measured price of the
    probabilistic low-dimensional design; no softening of the number.
I5. **Parked:** the noise-head interpretation of F-1 is recorded as a
    testable hypothesis, explicitly NOT pursued here (no new VAE-X
    objectives/losses/preprocessing). Stage 6, if it runs, runs on Arm V
    regardless of primary, tests whether the F-1 deficit is
    objective-induced and recoverable under task supervision, and re-clears
    G4.2 in full.
I6. **Housekeeping:** Stage 4.0 report supersede-and-append (the STOP stands
    as history); no expectations about I1, the ladders, or the I3 verdict
    encoded anywhere.
