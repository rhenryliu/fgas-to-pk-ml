# cvae observation-noise head — val-fold comparison (2026-07-09)

The `cvae` plugin (`src/fgas_spk/models/cvae.py`) gained a config-toggleable
learned per-k observation-noise head (`obs_noise_head`, default **false**),
the fix decided for its documented under-coverage (`docs/dual_vae_doc.md` §7:
`predict_samples` drew latent-prior samples only, with no aleatoric term). The
head mirrors `GaussianVAE` (`dual_vae_components.py`): a learnable `obs_logvar`
of shape `(n_k,)` (init 0.0, clamped ±8 in the loss), Gaussian-NLL
reconstruction on the standardised scale (train **and** internal `val_loss`,
terminal beta), optimiser + best-epoch snapshot/restore membership, seeded
per-draw observation noise in `predict_samples` (before the standardisation is
inverted; same CPU-generator pattern, MPS-safe), and an `obs_sigma()` accessor
on the raw SP(k) scale. `predict` is untouched. Note `beta` semantics differ
between the two modes (MSE → NLL rescales the reconstruction term against the
KL); see the module docstring.

New run config: `scripts/configs/run/cvae_noise.yaml` — an exact copy of the
committed `cvae.yaml` plus `obs_noise_head: true`. Every hyperparameter is
frozen at the committed `cvae.yaml` values per the maintainer's policy.

## Toggle-off regression (gate 1) — PASS

- Seeded synthetic fits (curve and `single_k`, CPU): predictions and samples
  **bit-identical** before vs after the edit.
- Pinned-context reproduction of run `20260708T231014Z__fdbfd21b__864001d`
  (its recorded split, seed 100; committed `cvae.yaml` model params; cuda;
  tag 20260708 arrays, bit-identical to the 20260706 build it trained on):
  val RMSE **0.02091613934152134** — exact match, and val-fold predictions
  bit-identical between the pre- and post-edit code. Val-fold only; the test
  fold was not predicted.
- Config hash of the committed `(config_dual_vae.yaml, cvae.yaml)` pair is
  `dbe723b7` before and after the edit (a constructor default absent from the
  YAML never enters `model_params`, hence not the hash).

Aside: the earlier cvae ledger row `20260709T073706Z__9a3dea51__2bd967d`
(val 0.0389) is **not** a committed-pair baseline — its recorded data config
had `radial_range_mpch: null` and `include_camels_params: false`, i.e. not the
pinned dual-VAE context.

## Unit tests (gate 2) and synthetic calibration (gate 3) — PASS

`tests/test_cvae.py` extended with a noise-head section: off-toggle
equivalence of seeded predictions/samples vs a default construction; on-toggle
training; `obs_sigma()` shape `(n_k,)`, strictly positive, moved off its init;
per-example sample spread strictly wider than the latent-only spread from the
same fitted weights; seeded reproducibility; the `single_k` path (`obs_sigma`
shape `(1,)`); a closed-form NLL + clamp check; and the decisive synthetic
gate — on `y = f(x) + eps` with known homoscedastic per-k sigma (0.10, 0.20),
`obs_sigma()` recovered the truth within [0.8, 1.3]× per k bin and the
held-out central-68% interval covered within [0.58, 0.78]. Full file passes
(see the test run in the work log).

## Real-data val comparison (gate 4)

Context: `scripts/configs/data/config_dual_vae.yaml` (tag 20260708, R < 10
crop, conditioned on the 5 cosmological parameters), pinned split
(0.7/0.1/0.2, seed 10), model seed 0, cuda, via `scripts/run.py`. All numbers
below are **val-fold** (102 curves × 36 k bins), computed from the run
checkpoints; per the task's protocol no test-fold number was consulted or used
for anything (the standard runner mechanically writes its held-out-test
diagnostics into every run record — those blocks were left unread).

| val fold                    | cvae (head off)                       | cvae_noise (head on)                  |
|-----------------------------|---------------------------------------|---------------------------------------|
| run record                  | `20260709T094306Z__dbe723b7__731a4e7` | `20260709T094429Z__2890276e__731a4e7` |
| val RMSE                    | 0.017665                              | **0.016359** (−7.4%)                  |
| coverage @ 0.68             | 0.430 (−25.0 pp)                      | 0.971 (**+29.1 pp**)                  |
| coverage @ 0.90             | 0.614 (−28.6 pp)                      | 0.991 (+9.1 pp)                       |
| coverage @ 0.95             | 0.685 (−26.5 pp)                      | 0.995 (+4.5 pp)                       |
| mean predictive spread      | 0.0063                                | 0.0511 (latent-only: 0.0035)          |
| KL / latent_gap @ best epoch| 0.277 / 1.475 (best 102)              | 0.109 / 0.814 (best **199**)          |

Both runs were made with a dirty working tree (the edit itself, pre-commit);
`env.json` records the flag.

**Against the reporting targets:** val RMSE is comfortably within ~10% of the
twin (it is better). Coverage is **not** within ~5 pp: the head flips the
model from ~25 pp under-coverage to ~29 pp **over**-coverage at the 68% level
(much closer at 90/95%).

## Diagnosis of the over-coverage: the noise head is rate-limited, not mis-specified

`obs_sigma()` rises monotonically from 0.0075 (k = 0.5) to 0.078 (k = 5.0) —
i.e. ≈ 0.74 × y_std at every k, because `obs_logvar` only descended from its
0.0 init to ≈ −0.59 in 200 epochs. The standardised per-k point residuals
(train 0.11–0.41, val 0.18–0.44) imply an optimum near `obs_logvar ≈ −4 to
−1.6`. Mechanism: once `e^{obs_logvar} ≫ err²` the NLL gradient w.r.t. each
`obs_logvar_k` saturates at ~0.5, and Adam then moves the parameter ≈ lr per
step — 200 epochs × 6 batches × 5e-4 ≈ 0.6, exactly the descent observed. The
recorded trace confirms truncation, not convergence: `val_loss` was still
falling steeply at the final epoch and `best_epoch = 199`. (The synthetic
calibration gate converged because it ran 300 epochs × 13 batches at lr 1e-3,
and its true standardised sigma is much larger.)

The maintainer's sweep exception was **not** triggered — no posterior/prior KL
collapse (KL 0.109, latent_gap 0.81 > 0) and val RMSE better than the twin —
and a beta sweep would not address a reconstruction-term descent-rate limit in
any case, so no sweep was run. **Flagged for the maintainer** (hyperparameters
are frozen; any of these is a deliberate change): more epochs, a larger lr for
the `obs_logvar` parameter only (a separate optimiser param-group, the usual
fix), or a warm start of `obs_logvar` at the log residual variance. Expected
effect: `obs_sigma` shrinking toward the per-k residual RMSE, pulling 68%
coverage down toward nominal from above.

## Files

- `src/fgas_spk/models/cvae.py` — the head (all behaviour documented in the
  module/class docstrings).
- `scripts/configs/run/cvae_noise.yaml` — the noise-head variant config.
- `tests/test_cvae.py` — the new noise-head section.
- `experiments/notes/cvae_noise_test_declaration_draft.md` — DRAFT test
  declaration (not in force; maintainer sign-off required).
