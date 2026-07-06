# Dual-VAE Stage 1 gate report — tag 20260706, R < 10 Mpc/h crop (amendment 4)

Fourth regeneration, per F4. Same tag and statistic as the previous section;
modelled scope cropped to R < 10 Mpc/h (F2 maintainer physics ruling; units
verified from the sidecar `units` block: comoving Mpc/h). 16 of 20 radial
bins retained (0.1 .. 9.364). F3.3 pre-training census on the retained
nd-2 slice: **PASS** — span [0.041, 1.250] against the provisional sanity
range [-1, 3], zero violations (all previous extremes, incl. the [-1.32,
4.70] pair, sat in the excluded R > 10 bins).

## Gate G1 (cropped) — PASS

- **G1.1**: `pca_x_recon` `20260706T185651Z__4b5b02ea__a4ba5aa`,
  `pca_y_recon` `20260706T185652Z__b2db7ddd__a4ba5aa`; bitwise-reproducible
  re-invocation (`...185655Z` pair). PCA-on-X (cropped): 0.0521 (n=1) ->
  0.0122 (n=10). PCA-on-Y bitwise-unchanged (y untouched by the crop).
- **G1.2**: table regenerated (three superseded tables kept, keyed by tag and
  radial range). Val RMSE: `mlp_regressor` **0.019378**
  (`20260706T185713Z__c62b6ed0__a4ba5aa`, now the best cross-modal baseline),
  `mlp` 0.019910 (`20260706T185716Z__bcdaaee9__a4ba5aa`), `pca_linear`
  0.038537 (`20260706T185703Z__a760845e__a4ba5aa`). The crop slightly
  *helped* mlp_regressor (0.0204 -> 0.0194) and slightly *hurt* the plain mlp
  (0.0170 -> 0.0199) — the excluded outer bins evidently carried some usable
  signal for the unbottlenecked model; reported as found (F7.3).

**Runtime-relative references (this table):** F1 quadrature G2.1' bar =
sqrt(1.01^2 - 1) x 0.019378 = **0.002747**; G3.1' C1 floor = `pca_linear` =
**0.038537**.

## F4.2 — map-nonlinearity finding (cropped; THE citable version)

Holds on the modelled scope: `mlp_regressor` beats `pca_linear` by 2.0x
globally (0.0194 vs 0.0385) and in deep suppression (0.0542 vs 0.1072 at
SP<0.8). Predicted-range widths: pca_linear 0.67 of true, mlp_regressor 0.77,
mlp 0.73 — genuine but moderate linear under-coverage, consistent with the
uncropped E4.2 finding. Scatter:
`figures/2026-07/07-06/stage1_baselines__20260706__e42_pred_vs_true_val.png`.

---

# [SUPERSEDED — full radial grid, pre-crop] Stage 1 gate report — tag 20260706 (Branch A, corrected f_gas statistic)

Third regeneration, per amendment 3 (E4). Tag `20260706` carries the
ratio-of-stacked-profiles f_gas (definition stamped in `__meta__`); E3.2
identity assertions passed (only fgas changed vs 20260702). Implementation
commits `e3b55e6`/`3e94bcc` preceded the runs.

## Gate G1 (tag 20260706) — PASS

- **G1.1**: `pca_x_recon` `20260706T061039Z__129449f5__3e94bcc`, `pca_y_recon`
  `20260706T061040Z__85a299a1__3e94bcc`; bitwise-reproducible on re-invocation
  (`...061043Z` pair). PCA-on-X is now a sane compression problem: val recon
  RMSE 0.0526 (n=1) -> 0.0174 (n=10), values O(0.01-1) — the corrected X has
  no pathological tail for PCA to chase. PCA-on-Y bitwise-identical to both
  prior tags (y unchanged), B5 truncation table unchanged.
- **G1.2**: baseline table regenerated (both prior tables under superseded
  headings). Val RMSE: `mlp` **0.017012** (`20260706T061104Z__8fc38d33__3e94bcc`),
  `mlp_regressor` 0.020370 (`20260706T061100Z__ee4485c8__3e94bcc`),
  `pca_linear` 0.038203 (`20260706T061051Z__1f810c16__3e94bcc`). All three
  improve 2-3x over the corrupted-X numbers — the corrupted cells were
  genuinely destroying usable signal.

**Runtime-relative references fixed here:** G2.1' bar = 0.1 x 0.017012 =
**0.0017012**; G3.1' C1 floor = `pca_linear` = **0.038203**.

## E4.2 — map-nonlinearity re-verification (finding, no gate)

Re-established, in tempered form. The f_gas -> SP(k) map remains genuinely
nonlinear on clean data: `mlp` beats `pca_linear` by 2.2x globally (0.0170 vs
0.0382) and by 2.5x in the deep-suppression regime (0.0433 vs 0.1076 at
SP<0.8). However, the historical "pca_linear collapses to a narrow band"
evidence was partly a contamination artifact: on clean X, `pca_linear` spans
0.69 of the true val range ([0.610, 1.052] vs true [0.461, 1.105]) — genuine
under-coverage of the extremes, but not the near-constant collapse seen on
corrupted X. Scatters: `figures/2026-07/07-05/stage1_baselines__20260706__e42_pred_vs_true_val.png`.
Any paper text citing the old scatter behaviour should be revised accordingly
(D3 framing guard applies: this is evidence about the MAP, not the y-manifold).

---

# [SUPERSEDED — tag 20260702, superseded f_gas statistic] Stage 1 gate report (amendment A2)

Spec amendment of 2026-07-05 (`docs/dual_vae_staged_spec.md`, commit `5d6fe55`)
repinned the context from tag `20260617` to `20260702` and added the B5
suppressed-regime truncation diagnostic. This section is the authoritative
Stage 1 gate evaluation; the original 20260617 report is kept below, marked
superseded (amendment A3: no cross-tag comparisons).

## A4 — dataset diff, tag 20260617 -> 20260702

From the embedded `__meta__` and the arrays themselves:

- **Shapes changed (X modality only):** the radial grid was re-binned from 17
  points (0.117-20 Mpc/h, dense inner sampling) to 20 points (0.1-20 Mpc/h,
  near-uniform inner spacing then log-spaced outer); `fgas` / `fgas_std` are
  (1024, 5, 20) accordingly. The new grid does **not** contain the old radii
  (a re-binning, not an extension), so X values are not comparable across tags.
- **Y unchanged:** `suppression` (1024, 255) is bitwise-identical between tags;
  the k grid is identical. `camels_params` is bitwise-identical.
- **Parameter stamping added:** the new tag stamps `camels_param_names`
  (all 35, matching the registry order); the old tag had none.
- Meta fields changed: `created_utc`, `n_radii` 17 -> 20, `tag`, plus the
  `arrays`/`camels_param_names` entries above. No change to suite, snapshot,
  redshift, source, rank, or provenance fields.

Since shapes changed, this is stated here (and was reported to the maintainer)
before any Stage 2 work on the new tag. Nothing in the stage scripts hardcodes
widths; all dimensions are inferred at load/fit time.

## Gate G1.1 (new tag) — PASS

- `pca_x_recon`: run `20260705T221711Z__b05fb53d__760d8d6` (20 radial bins).
  Val recon RMSE n=1..10: 8.7166, 8.7052, 8.7041, 8.7007, 8.6969, 8.2501,
  8.2291, 8.2224, 8.2026, 8.1872. Still heavy-tail dominated (see the
  superseded report's observation 1, which carries over qualitatively).
- `pca_y_recon`: run `20260705T221713Z__d5b9b15c__760d8d6`. Val recon RMSE
  n=1..10 identical to the 20260617 numbers (0.012134, 0.0028746, 0.0013369,
  0.0010034, ...) — expected, since the suppression content is bitwise
  identical; a useful cross-tag consistency check (reported, not gated).
  New B5 suppressed-regime truncation error (val, truth-masked): at n=2,
  RMSE(SP<0.95) = 0.00387, RMSE(SP<0.9) = 0.00433, RMSE(SP<0.8) = 0.00494
  (n_bins 1240/671/234; at n=3: 0.00185/0.00191/0.00235) — the truncation
  error is indeed ~1.5-1.7x larger in the deeply-suppressed bins than
  globally (0.00287 at n=2), confirming the heterogeneity B5 was added to
  expose. Full per-n table in the run summary.
- Reproducibility: second invocation (`20260705T221715Z__b05fb53d__760d8d6`,
  `20260705T221716Z__d5b9b15c__760d8d6`) bitwise-identical tables.

## Gate G1.2 (new tag) — PASS

`experiments/notes/dual_vae_baseline_table.md` regenerated on tag 20260702
(old table kept under the superseded heading). Val-fold global RMSE:
`mlp` **0.042921** (`20260705T221753Z__e18a8464__760d8d6`, now the best
cross-modal baseline — the re-binned profiles are markedly more informative
for the plain MLP), `mlp_regressor` 0.050388
(`20260705T221749Z__5e0c03f0__760d8d6`), `pca_linear` 0.061429
(`20260705T221739Z__c9583454__760d8d6`).

**G2.1' reference (fixed here):** best Stage 1 cross-modal val RMSE on this
tag = 0.042921 (`mlp`), so the Stage 2 codec-adequacy bar is
0.1 x 0.042921 = **0.0042921** raw-scale val reconstruction RMSE.

Framing guard (amendment D3): the low PCA-on-Y dimensionality concerns the
y-manifold's intrinsic geometry only; the f_gas -> SP(k) *map* is genuinely
nonlinear (mlp vs pca_linear: 0.0429 vs 0.0614 on this tag). The two claims
are kept distinct in all notes.

---

# [SUPERSEDED — tag 20260617] Dual-VAE Stage 1 gate report

Staged spec v2, Stage 1 (PCA dimensionality baselines and baseline-table
refresh). Implementation committed at `fe6a28c` before any run (GR8), so every
run record below carries a clean git SHA.

## Gate G1.1 — PCA artifacts + run records exist and re-run reproducibly: PASS

Canonical runs (with figures):

- `pca_x_recon` (f_gas(R), 17 radial bins): run
  `20260705T204259Z__b36e6a96__fe6a28c`. Val-fold reconstruction RMSE (raw
  scale), n = 1..10: 7.9048, 7.9028, 5.7277, 5.7199, 5.7193, 5.6973, 0.67891,
  0.63529, 0.4918, 0.27343.
- `pca_y_recon` (SP(k), 36 k bins in [0.5, 5.0] h/Mpc): run
  `20260705T204300Z__0234cc3a__fe6a28c`. Val-fold reconstruction RMSE (raw
  scale), n = 1..10: 0.012134, 0.0028746, 0.0013369, 0.0010034, 0.00089262,
  0.00084598, 0.00074704, 0.00069928, 0.00066817, 0.0006474.

Reproducibility: a second invocation (runs `20260705T204352Z__b36e6a96__fe6a28c`
and `20260705T204352Z__0234cc3a__fe6a28c`) produced **bitwise-identical**
`val_recon_rmse`, `val_r2`, and explained-variance tables for both modalities
(deterministic full-SVD PCA on the seeded grouped split). Checkpoints live on
scratch under `models/<run_id>/model.joblib`; figures (gitignored, regenerable):

- `figures/2026-07/07-05/20260705T204259Z__b36e6a96__pca_x_recon__pca_dimensionality.png`
- `figures/2026-07/07-05/20260705T204300Z__0234cc3a__pca_y_recon__pca_dimensionality.png`

## Gate G1.2 — baseline table with run_ids for all three reference models: PASS

`experiments/notes/dual_vae_baseline_table.md`, generated by
`scripts/stage1_baseline_table.py` on the pinned context. Val-fold global RMSE:
`pca_linear` 0.057267 (`20260705T204410Z__86702b77__fe6a28c`), `mlp_regressor`
0.050471 (`20260705T204419Z__939cfe69__fe6a28c`), `mlp` 0.061675
(`20260705T204423Z__b90ff578__fe6a28c`). Suppressed-regime val RMSE per
threshold is in the table. All three share the pinned SplitSpec (asserted by
the script before running).

## Observations for later stages (reported, not acted on)

1. **The f_gas profiles are heavy-tailed on the raw scale.** Median values are
   O(0.3-0.9) across radii, but 875/1024 rows exceed 1 somewhere, with extremes
   of +236.6 and -555.3 at R >~ 1 Mpc/h (e.g. sims 392, 787, 557, 171). The
   raw-scale PCA-on-X RMSE is dominated by these tails — hence the large
   values and the cliff between n=6 (5.70) and n=7 (0.68), where components
   start spending capacity on outlier directions. This is a property of the
   data (the same tail pathology behind the CVAE OOD precedent), not of the
   fit; the G3.1 comparison at matched dimension stays fair because the VAE-X
   gate uses the same raw-scale val-fold RMSE. Expect the Stage 3 obs_sigma
   per R-bin and the G3.3 ||mu1|| reference statistics to carry the load here
   (decision F0.6 keeps noise augmentation deferred).
2. **SP(k) is very low-dimensional linearly.** Two components already
   reconstruct the val fold to 0.0029 RMSE (~3.5% of the val target's 0.083
   standard deviation),
   and n=3 reaches 0.0013. The Lin-style latent_dim_y sweep in {2, 3, 4}
   (decision F0.3) brackets this well; G2.1 (VAE-Y <= PCA-on-Y at matched
   dimension) is a demanding but appropriate bar.
3. Baseline ordering: `mlp_regressor` (0.0505) < `pca_linear` (0.0573) <
   `mlp` (0.0617) on val global RMSE, but `pca_linear` is best in the
   suppressed regime at every threshold. Reference points, not thresholds.
