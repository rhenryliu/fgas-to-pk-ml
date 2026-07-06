"""Stage 1 of the dual-VAE programme: refresh the cross-modal baseline table.

Runs the three reference models -- ``pca_linear``, ``mlp_regressor``, and the
plain ``mlp`` -- through the standard runner (:func:`fgas_spk.train.run_training`,
the same path as ``scripts/run.py``) on the pinned dual-VAE evaluation context,
then collects the **val-fold** global RMSE and suppressed-regime RMSE into
``experiments/notes/dual_vae_baseline_table.md``, with run_ids. These are the
reference points (not thresholds) that Gate G4.1 later appends the composite
rungs to.

Two deliberate details:

* The runner's own ``suppressed_rmse`` diagnostic is computed on the held-out
  (test) fold, so this script recomputes it on the **val** fold -- the staged
  spec compares everything on val until Stage 5 (GR6). The runner does also
  record test-fold numbers in each run record, as it does for every run; they
  are reported nowhere here and are not used for any selection.
* Each run config's split is asserted equal to the pinned SplitSpec before
  running, so a drifted YAML fails loudly instead of silently comparing on
  different folds (GR3).

Usage::

    python scripts/stage1_baseline_table.py \\
        [--config-data scripts/configs/data/config_dual_vae.yaml] \\
        [--scratch-root PATH]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np

from fgas_spk.experiment import load_configs
from fgas_spk.loader import load_training_data
from fgas_spk.paths import fill_data_root, repo_root
from fgas_spk.train import grouped_split, run_training

from stage1_pca_dimensionality import PINNED_SPLIT

DEFAULT_DATA_CONFIG = "scripts/configs/data/config_dual_vae.yaml"

# The three reference models and their run-config YAMLs (paths repo-relative).
BASELINE_RUN_CONFIGS = (
    ("pca_linear", "scripts/configs/run/pca_reference.yaml"),
    ("mlp_regressor", "scripts/configs/run/mlp_regressor.yaml"),
    ("mlp", "scripts/configs/run/mlp.yaml"),
)

# Suppressed-regime thresholds of the pinned EvaluationSpec.
SUPPRESSED_THRESHOLDS = (0.95, 0.9, 0.8)

NOTE_PATH = "experiments/notes/dual_vae_baseline_table.md"

# Amendment A2 (2026-07-05): the original table was produced on tag 20260617,
# which the amendment superseded with 20260702. The old numbers are preserved
# verbatim below (append-only provenance) and re-emitted by every regeneration;
# no gate may compare against them (amendment A3: same-tag comparisons only).
SUPERSEDED_APPENDIX = """
## Superseded (tag 20260706, FULL radial grid — pre-crop) — do not compare against these

Produced 2026-07-06 06:11Z on tag `20260706` with the full 20-bin radial
grid, before amendment 4 (F2) cropped the modelled scope to R < 10 Mpc/h.
Same tag and statistic as the current table; different modelled scope. Kept
for provenance only.

| model | run_id | val RMSE | val RMSE (SP<0.95) | val RMSE (SP<0.9) | val RMSE (SP<0.8) |
|---|---|---|---|---|---|
| `pca_linear` | `20260706T061051Z__1f810c16__3e94bcc` | 0.038203 | 0.054479 (n=1240) | 0.070513 (n=671) | 0.1076 (n=234) |
| `mlp_regressor` | `20260706T061100Z__ee4485c8__3e94bcc` | 0.02037 | 0.029619 (n=1240) | 0.038467 (n=671) | 0.057557 (n=234) |
| `mlp` | `20260706T061104Z__8fc38d33__3e94bcc` | 0.017012 | 0.024276 (n=1240) | 0.030958 (n=671) | 0.043275 (n=234) |

## Superseded (tag 20260702, superseded per-halo-ratio f_gas statistic) — do not compare against these

Produced 2026-07-05 22:17Z on tag `20260702`. Superseded by the Branch-A
DEFINITION change (ratio-of-stacked-profiles, tag `20260706`): the X matrix
of this tag carried the per-halo-ratio corruption diagnosed in
`dual_vae_stage3_forensics.md`. Kept for provenance only.

| model | run_id | val RMSE | val RMSE (SP<0.95) | val RMSE (SP<0.9) | val RMSE (SP<0.8) |
|---|---|---|---|---|---|
| `pca_linear` | `20260705T221739Z__c9583454__760d8d6` | 0.061429 | 0.086437 (n=1240) | 0.10619 (n=671) | 0.16568 (n=234) |
| `mlp_regressor` | `20260705T221749Z__5e0c03f0__760d8d6` | 0.050388 | 0.07638 (n=1240) | 0.096441 (n=671) | 0.14279 (n=234) |
| `mlp` | `20260705T221753Z__e18a8464__760d8d6` | 0.042921 | 0.064242 (n=1240) | 0.078452 (n=671) | 0.10965 (n=234) |

## Superseded (tag 20260617, superseded f_gas statistic) — do not compare against these

Produced 2026-07-05 20:44Z on dataset tag `20260617` (17 radial bins), before
amendment A1 repinned the context to tag `20260702` (20 radial bins, re-binned
f_gas profiles; SP(k) content identical). Kept for provenance only.

| model | run_id | val RMSE | val RMSE (SP<0.95) | val RMSE (SP<0.9) | val RMSE (SP<0.8) |
|---|---|---|---|---|---|
| `pca_linear` | `20260705T204410Z__86702b77__fe6a28c` | 0.057267 | 0.076948 (n=1240) | 0.098986 (n=671) | 0.16107 (n=234) |
| `mlp_regressor` | `20260705T204419Z__939cfe69__fe6a28c` | 0.050471 | 0.083179 (n=1240) | 0.10746 (n=671) | 0.16809 (n=234) |
| `mlp` | `20260705T204423Z__b90ff578__fe6a28c` | 0.061675 | 0.09848 (n=1240) | 0.12847 (n=671) | 0.20395 (n=234) |
"""


def val_metrics(
    model, td, val_mask: np.ndarray, thresholds=SUPPRESSED_THRESHOLDS
) -> dict:
    """Val-fold global RMSE and suppressed-regime RMSE for a fitted model.

    Same definitions as the runner's held-out diagnostics, applied to the val
    fold: global RMSE pools all (row, k bin) squared errors; the suppressed
    RMSE at threshold ``t`` pools only the bins where the TRUE SP(k) < ``t``.

    Args:
        model: A fitted model exposing ``predict(X, X_cond, X_params)``.
        td: The full loaded :class:`~fgas_spk.loader.TrainingData`.
        val_mask (np.ndarray): Boolean row mask of the val fold.
        thresholds (Iterable[float]): TRUE SP(k) thresholds ``t``.

    Returns:
        dict: ``{"rmse": float, "suppressed": {str(t): {"rmse": float | None,
            "n_bins": int}}}``.
    """
    y_true = np.asarray(td.y[val_mask], dtype=float)
    y_pred = np.asarray(
        model.predict(
            td.X[val_mask],
            td.X_cond[val_mask] if td.X_cond is not None else None,
            td.X_params[val_mask] if td.X_params is not None else None,
        ),
        dtype=float,
    ).reshape(y_true.shape)
    se = (y_pred - y_true) ** 2

    suppressed: dict = {}
    for t in thresholds:
        mask = y_true < t
        n_bins = int(np.count_nonzero(mask))
        suppressed[str(t)] = {
            "rmse": float(np.sqrt(np.mean(se[mask]))) if n_bins else None,
            "n_bins": n_bins,
        }
    return {
        "rmse": float(np.sqrt(np.mean(se))),
        "suppressed": suppressed,
        # E4.2: the predicted-value range diagnoses range-collapse (a linear
        # model predicting a narrow band) vs diagonal-tracking behaviour.
        "pred_range": [float(y_pred.min()), float(y_pred.max())],
        "true_range": [float(y_true.min()), float(y_true.max())],
        "_y_true": y_true,
        "_y_pred": y_pred,
    }


def _write_scatter_figure(rows: list[dict], label: str):
    """E4.2: side-by-side val-fold predicted-vs-true scatters for all models.

    matplotlib is imported lazily (Agg backend); returns None if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    from fgas_spk.paths import figure_dir

    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4.2),
                             sharex=True, sharey=True)
    for ax, row in zip(np.atleast_1d(axes), rows):
        yt, yp = row["_y_true"].ravel(), row["_y_pred"].ravel()
        ax.scatter(yt, yp, s=5, alpha=0.3)
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"{row['model']}\nval RMSE {row['rmse']:.4g}, "
                     f"pred range [{row['pred_range'][0]:.3f}, "
                     f"{row['pred_range'][1]:.3f}]", fontsize=9)
        ax.set_xlabel("true SP(k)")
    np.atleast_1d(axes)[0].set_ylabel("predicted SP(k)")
    fig.suptitle(f"{label}\nE4.2 val-fold pred vs true (clean data)")
    out = figure_dir() / f"{label}__e42_pred_vs_true_val.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _format_table(rows: list[dict], data_config, td) -> str:
    """Render the baseline-table markdown document.

    Args:
        rows (list[dict]): One entry per model with keys ``model``, ``run_id``
            and the :func:`val_metrics` output.
        data_config: The pinned, resolved DataConfig (for the context header).
        td: The loaded TrainingData (for shapes and the dataset path).

    Returns:
        str: The full markdown document.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    thresholds = [str(t) for t in SUPPRESSED_THRESHOLDS]

    header = (
        "# Dual-VAE baseline table (Stage 1)\n\n"
        f"Generated by `scripts/stage1_baseline_table.py` on {stamp}. "
        "Reference points for the dual-VAE gates (G2.1/G3.1 compare "
        "reconstructions; G4.1 appends the composite rungs here). These are "
        "reference points, not thresholds.\n\n"
        "## Pinned evaluation context\n\n"
        f"- DataConfig: `scripts/configs/data/config_dual_vae.yaml` "
        f"(suite `{data_config.suite}`, snap {data_config.snapshot}, "
        f"tag `{data_config.tag}`, nd indices "
        f"{data_config.number_density_indices}, radial range "
        f"{list(data_config.radial_range_mpch) if data_config.radial_range_mpch else 'full'} Mpc/h, "
        f"target `{data_config.target_mode}` "
        f"k in {list(data_config.k_range)} h/Mpc, params "
        f"{data_config.camels_param_names})\n"
        f"- Dataset: `{td.source_path}`\n"
        f"- Split: grouped by `sim_index`, train/val/test = "
        f"{PINNED_SPLIT.train_frac}/{PINNED_SPLIT.val_frac}/"
        f"{PINNED_SPLIT.test_frac}, seed {PINNED_SPLIT.seed}\n"
        f"- Shapes: X {td.X.shape}, y {td.y.shape}, X_params "
        f"{td.X_params.shape if td.X_params is not None else None}\n"
        "- All metrics below are **val-fold** (test is untouched until "
        "Stage 5, GR6; the runner records its usual test-fold diagnostics "
        "inside each run record, unused here). Suppressed-regime RMSE at "
        "threshold t pools the (row, k bin) entries with TRUE SP(k) < t.\n\n"
    )

    cols = " | ".join(f"val RMSE (SP<{t})" for t in thresholds)
    lines = [
        f"| model | run_id | val RMSE | {cols} |",
        "|---|---|---|" + "---|" * len(thresholds),
    ]
    for row in rows:
        supp = row["suppressed"]
        cells = []
        for t in thresholds:
            entry = supp[t]
            cells.append(
                f"{entry['rmse']:.5g} (n={entry['n_bins']})"
                if entry["rmse"] is not None else "-- (n=0)"
            )
        lines.append(
            f"| `{row['model']}` | `{row['run_id']}` | {row['rmse']:.5g} | "
            + " | ".join(cells) + " |"
        )

    # E4.2: clean-data re-verification of the map-nonlinearity evidence
    # (reported finding, no gate): predicted-value ranges expose the linear
    # model's range-collapse (or lack of it) on the corrected X.
    e42 = [
        "",
        "## E4.2 — map-nonlinearity re-verification (val fold, clean data; finding, no gate)",
        "",
        f"True SP(k) range on val: [{rows[0]['true_range'][0]:.4f}, "
        f"{rows[0]['true_range'][1]:.4f}]. Scatters: "
        "`*__e42_pred_vs_true_val.png` (figures tree, regenerable).",
        "",
        "| model | val RMSE | pred range | pred-range width / true width |",
        "|---|---|---|---|",
    ]
    true_width = rows[0]["true_range"][1] - rows[0]["true_range"][0]
    for row in rows:
        width = row["pred_range"][1] - row["pred_range"][0]
        e42.append(
            f"| `{row['model']}` | {row['rmse']:.5g} | "
            f"[{row['pred_range'][0]:.4f}, {row['pred_range'][1]:.4f}] | "
            f"{width / true_width:.2f} |"
        )
    return header + "\n".join(lines) + "\n" + "\n".join(e42) + "\n" + SUPERSEDED_APPENDIX


def main(argv: list[str] | None = None) -> int:
    """Run the three baselines on the pinned context and write the table.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Stage 1: run pca_linear / mlp_regressor / mlp on the "
        "pinned dual-VAE context and write the val-fold baseline table."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG,
                        help="Path to the pinned DataConfig YAML.")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root (else $FGAS_SCRATCH_ROOT).")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    td = None
    data_config_used = None
    for model_name, run_yaml in BASELINE_RUN_CONFIGS:
        data_config, run_config = load_configs(args.config_data, run_yaml)
        if run_config.model != model_name:
            raise ValueError(
                f"{run_yaml} declares model {run_config.model!r}, expected "
                f"{model_name!r}."
            )
        if run_config.split != PINNED_SPLIT:
            raise ValueError(
                f"{run_yaml} split {run_config.split} differs from the pinned "
                f"SplitSpec {PINNED_SPLIT}; edit the YAML (GR3)."
            )

        result = run_training(
            data_config, run_config, scratch_root_override=args.scratch_root
        )
        print(f"[{model_name}] run_id: {result.run_id}  "
              f"val_rmse: {result.summary.get('val_rmse'):.6g}")

        if td is None:
            data_config_used = fill_data_root(data_config)
            td = load_training_data(data_config_used)
            masks = grouped_split(td.sim_index, PINNED_SPLIT)

        model = joblib.load(result.checkpoint_path)
        metrics = val_metrics(model, td, masks["val"])
        # Consistency guard: our val RMSE must match the runner's.
        runner_val = result.summary.get("val_rmse")
        if runner_val is not None and not np.isclose(
            metrics["rmse"], runner_val, rtol=1e-6, atol=1e-9
        ):
            raise RuntimeError(
                f"[{model_name}] recomputed val RMSE {metrics['rmse']:.8g} != "
                f"runner's {runner_val:.8g}; split or checkpoint mismatch."
            )
        rows.append({"model": model_name, "run_id": result.run_id, **metrics})

    fig_path = _write_scatter_figure(
        rows, f"stage1_baselines__{data_config_used.tag}"
    )
    print(f"E4.2 scatter figure: {fig_path}")
    for row in rows:  # arrays served their purpose; keep the note JSON-free
        row.pop("_y_true"), row.pop("_y_pred")

    note_path = repo_root() / NOTE_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(_format_table(rows, data_config_used, td))
    print(f"baseline table written: {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
