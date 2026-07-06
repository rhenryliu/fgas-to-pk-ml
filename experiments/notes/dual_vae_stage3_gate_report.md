# Stage 3 resolution (amendment 5, 2026-07-06): G3.1' retired; candidate set to Stage 4.0

The maintainer retired G3.1' as **structurally malformed** on the evidence
below (the FAIL stands as recorded history). Stage 3's gate set is now
G3.2' / G3.3 / G3.4 — **all PASS** on the 16-run grid — and Stage 3's output
is the **candidate set** (H2.1): every grid configuration passed both G3.2'
and G3.4, so all 16 enter Stage 4.0's probe-based selection:

`20260706T200035Z__923fa7f0`, `200036Z__cf35add9`, `200037Z__e72e3fab`,
`200037Z__fc11d46c`, `200038Z__75e83299`, `200039Z__a58053de`,
`200039Z__3080064e`, `200040Z__f4536019`, `200041Z__d478d3d0`,
`200041Z__0325e0a7`, `200043Z__fd693065`, `200043Z__794d86af`,
`200044Z__a43ec084`, `200045Z__7a4a4ba0`, `200046Z__50a0f2fd`,
`200047Z__2f14b4a6` (all `__8cbe5b2`).

The failed gate's two real findings are promoted to
`experiments/notes/dual_vae_findings.md` (F-1, F-2, with the H1.4 lambda
sanity note). Codec selection and adequacy now live in Stage 4.0
(`scripts/stage4_0_probe_selection.py`; gates G4.0a / G4.0b). VAE-Y
selection is unaffected (ld=3, beta=1e-4).

---

# [historical FAIL record — gate since retired] Dual-VAE Stage 3 gate report — tag 20260706, R < 10 crop: G3.1' FAIL (gate unreachable by construction), STOPPED

Full grid per the spec's own widening rule ("sweep {2, 3, 4, 6} only if
G3.1' fails at 2" — it did): 16 runs, latent_dim_x in {2, 3, 4, 6} x beta in
{1, 0.1, 0.01, 0.001}, cropped clean data, frozen VAE-Y
`20260706T190601Z__8a715039__8522dd4`. Implementation commit `8cbe5b2`
(hardened gates C1/C2 live). Runs `20260706T200035Z...200047Z__*__8cbe5b2`
(the earlier ld=2-only invocation `20260706T192023Z...192026Z` is the same
configuration set, superseded by this self-contained grid).

## The health story first: VAE-X is fixed

Every configuration now trains honestly on the corrected, cropped data:
interior best-epochs (301-569 of 5000), no collapsed dimensions at any
(ld, beta), G3.4 sanity PASS, G3.3 OOD statistics present. X-recon RMSE
tracks PCA at matched dimension within 2-28% (e.g. ld=2 beta=1: 0.0330 vs
0.0324; ld=6 beta=1: 0.0251 vs 0.0195). The Stage 3.0 -> Branch A chain
(corruption diagnosis -> statistic fix -> crop) fully resolved the original
epoch-0 pathology.

## Gate verdicts (selected: ld=6, beta=1.0, `20260706T200044Z__a43ec084__8cbe5b2`)

- **G3.2' no-collapse: PASS** (min per-dim KL 0.64 nats).
- **G3.3 OOD stats: PASS** (in every summary).
- **G3.4 training sanity: PASS** (best_epoch 569).
- **G3.1' downstream adequacy: FAIL** — floor AND ratio:
  - Floor: the PCA-x-scores arm plateaus at 0.0411-0.0413 (d = 4 and 6)
    against the 0.0385 `pca_linear` floor. It never clears it at any
    dimension.
  - Ratio: the VAE arm trails the PCA arm by 31-48% at d >= 3 (selected:
    1.48). Only at d=2 with beta <= 0.01 do the arms match (0.99).

## Why: the floor is unreachable by construction (decisive diagnostic)

Two probe ceilings, computed against the frozen VAE-Y on the same folds:

| probe | val y-space RMSE |
|---|---|
| oracle: true mu2 -> decoder-Y | **0.00192** (the codec is superb) |
| ridge from the FULL 16-bin X -> mu2 -> decoder-Y | **0.0422** |
| best any-dim PCA-scores ridge arm (d=4) | 0.0411 |
| G3.1' floor (`pca_linear`, direct 25-comp linear map to y) | 0.0385 |

Even the complete profile, linearly probed into mu2, cannot clear the floor.
The binding constraint is the **linearity of the ridge probe** — the
f_gas -> mu2 map is nonlinear (consistent with the F4.2 finding that the
f_gas -> SP(k) map is nonlinear at 2x) — not the x-codec's dimension or
quality. The C1 floor, as operationalized (a ridge-probe arm vs a direct
linear baseline), conflates probe expressiveness with codec adequacy: no
x-codec, however perfect, can pass it on this data. Note Stage 4's actual
mapping ladder is exactly the nonlinear probe this gate lacks (rungs 2-3).

## Findings for the maintainer (reported, not acted on)

1. **Gate design:** G3.1' is the second gate whose operationalization broke
   on contact with clean data (after the G2.1' fraction). A repaired form
   needs either a nonlinear probe (e.g. the Stage 4 MLP rung as the arm) or
   a floor derived from the same two-stage linear pathway (e.g. the full-X
   ridge ceiling 0.0422) rather than from a direct-regression baseline.
2. **Real representation finding inside the failure:** at matched dimension
   under the same linear probe, VAE-X codes are 31-48% less mu2-informative
   than raw PCA scores at d >= 3 — the VAE spends capacity on
   reconstruction-relevant but downstream-irrelevant structure. At d=2,
   low beta, they match. This is a genuine, reportable property, separable
   from the gate-design problem.
3. **Selection-criterion mismatch:** B1 selects by X-recon RMSE (best:
   ld=6 beta=1), but downstream adequacy anti-correlates with it (that
   config has the WORST ratio, 1.48; the best ratio in-grid is
   ld=2 beta=0.001 at 0.989). If the pipeline's goal is the composite, the
   Stage 3 selection rule optimizes the wrong objective.

STOPPED per GR5 / F6.1 (E6.3 logic: the Stage 3 options menu stays closed;
its reopening — or a gate repair — is a maintainer edit). Stage 4 remains
parked; no composite was trained.

---

# [SUPERSEDED - corrupted-X era] # Dual-VAE Stage 3 gate report — letter-pass, substantive FAIL: STOPPED

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
