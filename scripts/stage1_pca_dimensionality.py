"""Stage 1 of the dual-VAE programme: PCA dimensionality baselines.

Fits sklearn PCA separately on the f_gas(R) profiles (``X``) and on the SP(k)
suppression curves (``y``) of the **train fold only**, then evaluates, for
``n_components = 1..10``, the explained-variance-ratio curves and the val-fold
reconstruction RMSE (raw scale) of each modality. These linear reference points
are what the Stage 2/3 VAE gates (G2.1 / G3.1) compare against at matched
dimension.

Every invocation is recorded: one run record per modality, with record-only
model labels ``"pca_x_recon"`` / ``"pca_y_recon"`` (not registry entries), the
per-``n`` tables in ``summary.json``, a fitted-PCA checkpoint on scratch under
``<scratch_root>/models/<run_id>/``, and one two-panel figure per modality via
:func:`fgas_spk.paths.figure_dir`.

Conventions (stated here per GR7's spirit, though no ELBO is involved):

* PCA is fit on the **raw** arrays -- sklearn's PCA mean-centres internally but
  applies no per-bin rescaling -- so the reconstruction RMSE is on the raw
  target scale, directly comparable to the VAE gates' raw-scale RMSE.
* The reconstruction RMSE at ``n`` is the root of the mean squared error over
  **all** val-fold entries (rows x bins), reconstructing from the leading ``n``
  principal components of the train-fold fit.
* The split is the pinned grouped-by-``sim_index`` SplitSpec of the staged
  spec (0.7 / 0.1 / 0.2, seed 100); the train fold fits, the val fold scores.
  The test fold is never touched here (GR6).

Usage::

    python scripts/stage1_pca_dimensionality.py \\
        [--config-data scripts/configs/data/config_dual_vae.yaml] \\
        [--scratch-root PATH] [--no-figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.experiment import RunConfig, SplitSpec
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.paths import fill_data_root, figure_dir, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split

# Pinned evaluation context of the staged spec: every stage and every model
# compares on these folds. Do not tune these here.
PINNED_SPLIT = SplitSpec(train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=100)

# Component counts scanned for the dimensionality curves.
N_COMPONENTS_GRID = tuple(range(1, 11))

# Amendment B5: suppressed-regime truncation-error thresholds (TRUE SP(k) < t).
SUPPRESSED_THRESHOLDS = (0.95, 0.9, 0.8)

DEFAULT_DATA_CONFIG = "scripts/configs/data/config_dual_vae.yaml"


def _figure_label(run_id: str, model_name: str) -> str:
    """Return ``<timestamp>__<config-hash>__<model>`` for figure filenames.

    Mirrors :func:`fgas_spk.train._figure_label`: the trailing git-sha segment
    of ``run_id`` is replaced by the human-readable model label.

    Args:
        run_id (str): The run identifier, ``<ts>__<config-hash8>__<sha7>``.
        model_name (str): The record-only model label.

    Returns:
        str: The figure label.
    """
    return f"{run_id.rsplit('__', 1)[0]}__{model_name}"


def pca_dimensionality_table(
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    n_grid: tuple[int, ...],
    seed: int,
    suppressed_thresholds: tuple[float, ...] | None = None,
) -> tuple[object, dict]:
    """Fit PCA on the train fold and tabulate per-``n`` val reconstruction error.

    Args:
        train_arr (np.ndarray): Train-fold data, shape (n_train, n_features).
        val_arr (np.ndarray): Val-fold data, shape (n_val, n_features).
        n_grid (tuple[int, ...]): Component counts to evaluate; entries above
            ``min(n_features, n_train)`` are dropped (and reported as such).
        seed (int): Passed to PCA's ``random_state`` (inert for the
            deterministic full SVD used here; recorded for completeness).
        suppressed_thresholds (tuple[float, ...] | None): When given (the
            SP(k) modality), also report the amendment-B5 suppressed-regime
            truncation error: the val reconstruction RMSE restricted to the
            entries where the TRUE value < t, for each threshold t
            (ground-truth-masked, never prediction-masked). None skips it
            (the f_gas modality, where the thresholds have no meaning).

    Returns:
        tuple[object, dict]: The fitted :class:`sklearn.decomposition.PCA`
            (with ``max(n_grid)`` retained components) and the per-``n`` table:
            ``n_components``, ``val_recon_rmse`` (raw scale),
            ``val_r2`` (1 - SSE/SST, SST about the train mean),
            ``train_explained_variance_ratio`` (per component) and its
            cumulative sum; plus, with thresholds,
            ``suppressed_recon_rmse[str(t)] = {"rmse": [per n], "n_bins": int}``.
    """
    from sklearn.decomposition import PCA

    n_cap = min(train_arr.shape[0], train_arr.shape[1])
    n_used = [n for n in n_grid if n <= n_cap]
    n_max = max(n_used)

    pca = PCA(n_components=n_max, svd_solver="full", random_state=seed)
    pca.fit(train_arr)

    z_val = pca.transform(val_arr)                     # (n_val, n_max)
    sst = float(np.sum((val_arr - pca.mean_) ** 2))    # about the train mean

    thresholds = tuple(suppressed_thresholds or ())
    masks = {t: val_arr < t for t in thresholds}       # truth-masked (B5)

    val_rmse: list[float] = []
    val_r2: list[float] = []
    suppressed: dict[str, dict] = {
        str(t): {"rmse": [], "n_bins": int(masks[t].sum())} for t in thresholds
    }
    for n in n_used:
        recon = pca.mean_ + z_val[:, :n] @ pca.components_[:n]
        se = (val_arr - recon) ** 2
        sse = float(se.sum())
        val_rmse.append(float(np.sqrt(sse / val_arr.size)))
        val_r2.append(1.0 - sse / sst)
        for t in thresholds:
            mask = masks[t]
            suppressed[str(t)]["rmse"].append(
                float(np.sqrt(np.mean(se[mask]))) if mask.any() else None
            )

    table = {
        "n_components": n_used,
        "n_components_dropped": [n for n in n_grid if n > n_cap],
        "val_recon_rmse": val_rmse,
        "val_r2": val_r2,
        "train_explained_variance_ratio":
            [float(v) for v in pca.explained_variance_ratio_[:n_max]],
        "train_explained_variance_ratio_cumulative":
            [float(v) for v in np.cumsum(pca.explained_variance_ratio_[:n_max])],
    }
    if thresholds:
        table["suppressed_recon_rmse"] = suppressed
    return pca, table


def _write_figure(table: dict, label: str, modality_desc: str) -> Path | None:
    """Write the two-panel dimensionality figure for one modality.

    Left panel: per-component train explained-variance ratio (log-y) with its
    cumulative sum on a twin linear axis. Right panel: val-fold reconstruction
    RMSE (raw scale) versus the number of components. matplotlib is imported
    lazily (Agg backend); returns None if it is unavailable.

    Args:
        table (dict): The per-``n`` table from :func:`pca_dimensionality_table`.
        label (str): Figure label (goes in the filename and title).
        modality_desc (str): Short description for the axis titles.

    Returns:
        Path | None: The written path, or None without matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    n = table["n_components"]
    fig, (ax_evr, ax_rmse) = plt.subplots(1, 2, figsize=(10, 4))

    ax_evr.plot(n, table["train_explained_variance_ratio"], marker="o",
                color="C0", label="per component")
    ax_evr.set_yscale("log")
    ax_evr.set_xlabel("component")
    ax_evr.set_ylabel("train explained-variance ratio")
    ax_twin = ax_evr.twinx()
    ax_twin.plot(n, table["train_explained_variance_ratio_cumulative"],
                 marker=".", color="C1", linestyle="--", label="cumulative")
    ax_twin.set_ylabel("cumulative (linear)")
    lines, labels = ax_evr.get_legend_handles_labels()
    lines_tw, labels_tw = ax_twin.get_legend_handles_labels()
    ax_evr.legend(lines + lines_tw, labels + labels_tw, loc="center right")

    ax_rmse.plot(n, table["val_recon_rmse"], marker="o", color="C2")
    ax_rmse.set_yscale("log")
    ax_rmse.set_xlabel("n components")
    ax_rmse.set_ylabel(f"val reconstruction RMSE of {modality_desc}")

    fig.suptitle(f"{label}\nPCA dimensionality: {modality_desc}")
    out = figure_dir() / f"{label}__pca_dimensionality.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 1 PCA dimensionality baselines and record both runs.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Stage 1: PCA dimensionality baselines on X (f_gas) and "
        "Y (SP(k)), train-fold fit, val-fold scores, one run record each."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG,
                        help="Path to the pinned DataConfig YAML.")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root (else $FGAS_SCRATCH_ROOT).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed recorded on the RunConfig (the split "
                        "seed is pinned separately at 100).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip writing the figures.")
    args = parser.parse_args(argv)

    data_config = DataConfig.from_yaml(args.config_data)
    data_config = fill_data_root(data_config)
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    if not masks["val"].any():
        raise RuntimeError("Pinned split produced an empty val fold; cannot score.")

    # The B5 suppressed-regime truncation diagnostic applies to the SP(k)
    # modality only (the thresholds are SP(k) values; meaningless for f_gas).
    modalities = {
        "pca_x_recon": (td.X, "f_gas(R)", None),
        "pca_y_recon": (td.y, "SP(k)", SUPPRESSED_THRESHOLDS),
    }

    for model_label, (arr, desc, thresholds) in modalities.items():
        arr = np.asarray(arr, dtype=float)
        pca, table = pca_dimensionality_table(
            arr[masks["train"]], arr[masks["val"]], N_COMPONENTS_GRID, args.seed,
            suppressed_thresholds=thresholds,
        )

        run_config = RunConfig(
            model=model_label,  # record-only label, not a registry entry
            model_params={"n_components_grid": list(N_COMPONENTS_GRID),
                          "svd_solver": "full"},
            seed=args.seed,
            split=PINNED_SPLIT,
        )
        summary = {
            "stage": "dual_vae_stage1",
            "modality": desc,
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "n_test": int(masks["test"].sum()),
            "n_features": int(arr.shape[1]),
            "scale_convention": (
                "PCA fit on raw values (mean-centred internally, no per-bin "
                "rescaling); reconstruction RMSE pooled over all val-fold "
                "entries (rows x bins), raw scale."
            ),
            **table,
        }
        # Lean ledger line: the headline curve without the EVR tables.
        ledger_metrics = {
            "stage": "dual_vae_stage1",
            "modality": desc,
            "n_components": table["n_components"],
            "val_recon_rmse": table["val_recon_rmse"],
        }
        record = write_run_record(
            data_config,
            run_config,
            dataset_path=td.source_path,
            data_root=data_config.project_root,
            scratch_root=scratch_root,
            summary=summary,
            ledger_metrics=ledger_metrics,
            device="cpu",
        )

        ckpt_dir = Path(scratch_root) / "models" / record.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pca, ckpt_dir / "model.joblib")

        fig_path = None
        if not args.no_figures:
            fig_path = _write_figure(
                table, _figure_label(record.run_id, model_label), desc
            )

        print(f"[{model_label}] run_id      : {record.run_id}")
        print(f"[{model_label}] run_dir     : {record.run_dir}")
        print(f"[{model_label}] checkpoint  : {ckpt_dir / 'model.joblib'}")
        print(f"[{model_label}] figure      : {fig_path}")
        rmse_by_n = ", ".join(
            f"n={n}: {r:.5g}"
            for n, r in zip(table["n_components"], table["val_recon_rmse"])
        )
        print(f"[{model_label}] val RMSE    : {rmse_by_n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
