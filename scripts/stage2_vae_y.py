"""Stage 2 of the dual-VAE programme: VAE-Y, autoencoding SP(k).

Trains the reusable :class:`~fgas_spk.models.dual_vae_components.GaussianVAE`
on the **train-fold** SP(k) curves with the five cosmological CAMELS parameters
as context (decision F0.4), sweeping ``latent_dim_y`` over {2, 3, 4} (decision
F0.3) crossed with ``beta`` over {1, 0.1, 0.01, 0.001} (F0.2 as amended
2026-07-05, B1), and evaluates each on the val fold with posterior-mean
decoding (B4), including the B5 suppressed-regime truncation error. Gates are
the amended G2.1' (codec adequacy vs 10% of the best Stage 1 cross-modal val
RMSE) and G2.2' (collapse detection only); the VAE-vs-PCA comparison at
matched dimension is reported as a finding (B2). This doubles as the methodological
replication of Lin et al. (2026, arXiv:2509.01881) on this project's own data
(decision F0.1 = (b)); the checks below are their Figs. 12 / 5 / 1 analogues.

Per swept dimension (one seeded run record each, model label ``"vae_spk"``):

* val-fold reconstruction RMSE on the raw SP(k) scale, next to the Stage 1
  PCA-on-Y RMSE at the matched dimension (recomputed here on the identical
  folds via ``stage1_pca_dimensionality.pca_dimensionality_table`` -- Stage 1
  verified that table is bitwise reproducible);
* final per-dimension KL and activity ``A_j`` on the train and val folds;
* the fitted per-k-bin ``obs_sigma()`` (raw scale);
* per-epoch history (total loss, NLL, total KL, per-dimension KL, ``A_j``)
  into ``metrics.jsonl``, plus KL / activity trace figures.

The selected dimension (lowest val reconstruction RMSE subject to the activity
criterion 1 <= A_j <= 10 on every dimension, train fold) additionally gets the
Lin-replication artifacts: the RMSE-vs-dimension comparison figure, and the
Pearson correlations of its val-fold posterior means ``mu2`` against **all 35**
SB35 parameters (a second data load with the full parameter set -- the model
itself stays conditioned on the 5 cosmological parameters), as a figure and a
summary block.

GR7 (one reduction convention) is inherited from ``GaussianVAE``: NLL and KL
are both per-example sums over their own dimensions, batch-averaged; the
convention string is recorded in every run summary.

Usage::

    python scripts/stage2_vae_y.py \\
        [--config-data scripts/configs/data/config_dual_vae.yaml] \\
        [--scratch-root PATH] [--seed 0] [--epochs N] [--no-figures]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.experiment import RunConfig
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.models.dual_vae_components import GaussianVAE
from fgas_spk.paths import figure_dir, fill_data_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split, pick_device

from stage1_pca_dimensionality import (
    N_COMPONENTS_GRID,
    PINNED_SPLIT,
    _figure_label,
    pca_dimensionality_table,
)

DEFAULT_DATA_CONFIG = "scripts/configs/data/config_dual_vae.yaml"

# Stage 1 canonical PCA-on-Y run (the numbers recomputed here must match it).
STAGE1_PCA_Y_RUN_ID = "20260706T185652Z__b2db7ddd__a4ba5aa"

# The swept latent dimensionalities (decision F0.3).
LATENT_DIMS = (2, 3, 4)

# The swept terminal KL weights (F0.2 as amended 2026-07-05, B1: beta is a
# tuned hyperparameter, not fixed at 1).
BETAS = (1.0, 0.1, 0.01, 0.001)

# Shared VAE-Y hyperparameters; latent_dim and beta are added per sweep point.
# These are the Stage 2 selections that Stage 5's dual_vae plugin will inherit.
# epochs=5000 is the converged schedule established on the 20260617 rounds
# (best_epoch ~2000-2300 with best-epoch restore guarding overfit).
MODEL_PARAMS = {
    "hidden": 128,
    "n_layers": 2,
    "epochs": 5000,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "dropout": 0.0,
    "batch_size": 128,
    "anneal_epochs": None,  # -> epochs // 4, linear KL warm-up
    "val_frac": 0.1,      # internal holdout for best-epoch restore
}

# Gate G2.2' (amendment B3): collapse detection only. A dimension is collapsed
# when its per-dim KL (train fold, best-epoch weights) falls below this.
COLLAPSE_KL_MIN = 0.01

# Gate G2.1' as reformed by amendment 4 (F1): quadrature codec adequacy. With
# `base` the best same-tag, same-config cross-modal val RMSE from the current
# Stage 1 table and `codec` the selected VAE-Y val recon RMSE (posterior-mean
# decode, raw scale), the gate is sqrt(base^2 + codec^2)/base - 1 <= 0.01 --
# the y-codec may inflate the achievable pipeline floor by at most 1% --
# equivalently codec <= base * sqrt(1.01^2 - 1). The former 10% linear
# fraction is superseded as a proxy for exactly this criterion.
# BEST_BASELINE_VAL_RMSE must track the current Stage 1 table (override via
# --g21-reference): F4 cropped-config refresh, mlp_regressor, run
# 20260706T185713Z__c62b6ed0__a4ba5aa.
BEST_BASELINE_VAL_RMSE = 0.019378
G21_MAX_FLOOR_INFLATION = 0.01
G21_QUADRATURE_FACTOR = float(np.sqrt((1.0 + G21_MAX_FLOOR_INFLATION) ** 2 - 1.0))

# Amendment B5: suppressed-regime truncation thresholds (TRUE SP(k) < t).
SUPPRESSED_THRESHOLDS = (0.95, 0.9, 0.8)

REDUCTION_CONVENTION = (
    "ELBO reduction (GR7): Gaussian NLL summed over k bins and KL summed over "
    "latent dimensions, both averaged over the batch; effective weight "
    "~0.5/exp(obs_logvar_b) per bin vs beta=1 per latent dimension; "
    "standardised scale during training, RMSE/obs_sigma reported on the raw "
    "SP(k) scale."
)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root-mean-square difference over all elements."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _collapse_ok(kl_per_dim: list[float]) -> bool:
    """True when no dimension is collapsed (per-dim KL >= 0.01 nats; B3)."""
    return all(k >= COLLAPSE_KL_MIN for k in kl_per_dim)


def _suppressed_recon_rmse(
    y_true: np.ndarray, y_recon: np.ndarray, thresholds=SUPPRESSED_THRESHOLDS
) -> dict:
    """Truth-masked reconstruction RMSE where TRUE SP(k) < t (amendment B5).

    Args:
        y_true (np.ndarray): True SP(k), shape (n_examples, n_k).
        y_recon (np.ndarray): Reconstruction, same shape.
        thresholds (Iterable[float]): The SP(k) thresholds ``t``.

    Returns:
        dict: ``{str(t): {"rmse": float | None, "n_bins": int}}``.
    """
    se = (np.asarray(y_recon) - np.asarray(y_true)) ** 2
    out: dict = {}
    for t in thresholds:
        mask = np.asarray(y_true) < t
        n_bins = int(mask.sum())
        out[str(t)] = {
            "rmse": float(np.sqrt(np.mean(se[mask]))) if n_bins else None,
            "n_bins": n_bins,
        }
    return out


def _write_trace_figure(model: GaussianVAE, label: str) -> Path | None:
    """Two-panel per-epoch trace: per-dimension KL (log-y) and activity A_j.

    The activity panel marks the [1, 10] gate band. matplotlib is imported
    lazily (Agg backend); returns None if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    epochs = [row["epoch"] for row in model.history]
    n_dims = len(model.history[0]["kl_per_dim"])
    fig, (ax_kl, ax_act) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    for j in range(n_dims):
        ax_kl.plot(epochs, [r["kl_per_dim"][j] for r in model.history],
                   label=f"dim {j}")
        ax_act.plot(epochs, [r["activity"][j] for r in model.history])
    ax_kl.set_yscale("log")
    ax_kl.set_ylabel("KL per dimension")
    ax_kl.legend()
    ax_kl.set_title(f"{label}\nlatent traces (train fold)")
    ax_act.set_yscale("log")
    # A_j is documented, not gated (amendment B3); mark A = 1 for orientation.
    ax_act.axhline(1.0, color="C2", linewidth=1, linestyle=":", label="A = 1")
    ax_act.set_xlabel("epoch")
    ax_act.set_ylabel(r"activity $A_j$")
    ax_act.legend()
    out = figure_dir() / f"{label}__latent_traces.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_rmse_vs_dim_figure(
    results: list[dict], pca_table: dict, label: str
) -> Path | None:
    """VAE-Y val reconstruction RMSE at {2, 3, 4} over the PCA-on-Y curve.

    The Lin et al. Fig. 12 analogue on this data. matplotlib is imported
    lazily (Agg backend); returns None if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pca_table["n_components"], pca_table["val_recon_rmse"],
            marker=".", color="C0", label="PCA on Y (Stage 1)")
    betas = sorted({r["beta"] for r in results}, reverse=True)
    for i, beta in enumerate(betas):
        pts = [r for r in results if r["beta"] == beta]
        ax.plot([p["latent_dim"] for p in pts],
                [p["val_recon_rmse"] for p in pts],
                marker="o", linestyle="none", color=f"C{i + 1}",
                label=f"VAE-Y (beta={beta})")
    ax.set_yscale("log")
    ax.set_xlabel("latent dimension / n components")
    ax.set_ylabel("val reconstruction RMSE of SP(k)")
    ax.set_title(f"{label}\nVAE-Y vs PCA-on-Y (Lin Fig. 12 analogue)")
    ax.legend()
    out = figure_dir() / f"{label}__rmse_vs_dim.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_correlation_figure(
    corr: np.ndarray, param_names: list[str], label: str
) -> Path | None:
    """Heatmap of Pearson r(mu2_j, theta_p) over all 35 SB35 parameters.

    The Lin et al. Fig. 1 bottom-panel analogue. matplotlib is imported lazily
    (Agg backend); returns None if unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    n_dims, n_params = corr.shape
    fig, ax = plt.subplots(figsize=(max(10, 0.32 * n_params), 1.0 + 0.6 * n_dims))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels([f"mu2[{j}]" for j in range(n_dims)])
    for j in range(n_dims):
        for p in range(n_params):
            if abs(corr[j, p]) >= 0.3:
                ax.text(p, j, f"{corr[j, p]:.2f}", ha="center", va="center",
                        fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r (val fold)")
    ax.set_title(f"{label}\nmu2 vs all 35 SB35 parameters (Lin Fig. 1 analogue)")
    out = figure_dir() / f"{label}__latent_param_corr.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def latent_param_correlations(
    mu: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Pearson correlation of each latent dimension against each parameter.

    Args:
        mu (np.ndarray): Posterior means, shape (n_examples, n_dims).
        params (np.ndarray): Parameter values, shape (n_examples, n_params).

    Returns:
        np.ndarray: Correlation matrix, shape (n_dims, n_params).
    """
    mu = np.asarray(mu, dtype=float)
    params = np.asarray(params, dtype=float)
    # Zero-variance columns (a collapsed latent dimension, or a frozen
    # parameter) get unit scale so their correlation reads 0 rather than NaN.
    mu_std = np.where(mu.std(axis=0) < 1e-12, 1.0, mu.std(axis=0))
    pa_std = np.where(params.std(axis=0) < 1e-12, 1.0, params.std(axis=0))
    mu_c = (mu - mu.mean(axis=0)) / mu_std
    pa_c = (params - params.mean(axis=0)) / pa_std
    return (mu_c.T @ pa_c) / mu.shape[0]


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 2 VAE-Y sweep, replication checks, and run records.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Stage 2: VAE-Y on train-fold SP(k) with cosmological "
        "context; latent_dim_y sweep over {2, 3, 4}; Lin-replication checks."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG,
                        help="Path to the pinned DataConfig YAML.")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root (else $FGAS_SCRATCH_ROOT).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed (torch + internal holdout).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override MODEL_PARAMS['epochs'] (smoke tests / "
                        "convergence checks).")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Override MODEL_PARAMS['hidden'] (capacity checks).")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override MODEL_PARAMS['n_layers'] (capacity checks).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override MODEL_PARAMS['lr'].")
    parser.add_argument("--experiments-root", default=None,
                        help="Override the run-record tree (smoke tests; "
                        "default: <repo>/experiments).")
    parser.add_argument("--g21-reference", type=float,
                        default=BEST_BASELINE_VAL_RMSE,
                        help="Best Stage 1 cross-modal val RMSE on the pinned "
                        "context; the F1 quadrature bar is "
                        "sqrt(1.01^2 - 1) x this.")
    parser.add_argument("--betas", type=float, nargs="+", default=None,
                        help="Override the beta grid (e.g. --betas 1e-4 for "
                        "the F5.1 one-decade extension runs).")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=None,
                        help="Override the latent_dim_y grid (e.g. "
                        "--latent-dims 6 for the F5.4 finding run).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip writing figures.")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)
    device = pick_device()

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    y_tr, y_va = td.y[masks["train"]], td.y[masks["val"]]
    assert td.X_params is not None, "pinned config must load the 5 cosmo params"
    c_tr, c_va = td.X_params[masks["train"]], td.X_params[masks["val"]]

    # PCA-on-Y reference on the identical folds (matches the Stage 1 record).
    _, pca_table = pca_dimensionality_table(
        np.asarray(y_tr, dtype=float), np.asarray(y_va, dtype=float),
        N_COMPONENTS_GRID, seed=0,
    )
    pca_rmse_at = dict(zip(pca_table["n_components"], pca_table["val_recon_rmse"]))

    model_params = dict(MODEL_PARAMS)
    for key in ("epochs", "hidden", "n_layers", "lr"):
        value = getattr(args, key)
        if value is not None:
            model_params[key] = value

    # --- sweep: latent_dim x beta (F0.2 as amended, B1; grids overridable
    # for the F5.1 extension / F5.4 finding runs) ----------------------------
    latent_dims = tuple(args.latent_dims) if args.latent_dims else LATENT_DIMS
    betas = tuple(args.betas) if args.betas else BETAS
    results: list[dict] = []
    for ld in latent_dims:
        for beta in betas:
            model = GaussianVAE(
                latent_dim=ld, beta=beta, seed=args.seed, **model_params
            )
            model.fit(y_tr, c_tr)
            stats_tr = model.latent_stats(y_tr, c_tr)
            stats_va = model.latent_stats(y_va, c_va)
            y_va_recon = model.reconstruct(y_va, c_va)
            res = {
                "latent_dim": ld,
                "beta": beta,
                "model": model,
                "train_recon_rmse": _rmse(model.reconstruct(y_tr, c_tr), y_tr),
                # Posterior-mean decode throughout (amendment B4).
                "val_recon_rmse": _rmse(y_va_recon, y_va),
                "suppressed_recon_rmse": _suppressed_recon_rmse(y_va, y_va_recon),
                "pca_val_rmse_matched": pca_rmse_at[ld],
                "kl_per_dim_train": stats_tr["kl_per_dim"],
                "activity_train": stats_tr["activity"],
                "kl_per_dim_val": stats_va["kl_per_dim"],
                "activity_val": stats_va["activity"],
                "collapse_ok": _collapse_ok(stats_tr["kl_per_dim"]),
                "obs_sigma": [float(s) for s in model.obs_sigma()],
                "best_epoch": model.best_epoch,
            }
            results.append(res)
            print(f"[vae_spk ld={ld} beta={beta}] "
                  f"val recon RMSE {res['val_recon_rmse']:.5g} "
                  f"(PCA {res['pca_val_rmse_matched']:.5g}) "
                  f"KL={np.round(res['kl_per_dim_train'], 3).tolist()} "
                  f"A_j={np.round(res['activity_train'], 1).tolist()} "
                  f"collapse_ok={res['collapse_ok']} "
                  f"best_epoch={res['best_epoch']}")

    # --- selection (task 4 as amended, B1): lowest val recon RMSE
    # (posterior-mean decode) subject to the no-collapse criterion (B3);
    # fall back to lowest RMSE (gate G2.2' then fails, reported).
    eligible = [r for r in results if r["collapse_ok"]]
    pool = eligible if eligible else results
    selected = min(pool, key=lambda r: r["val_recon_rmse"])
    print(f"selected latent_dim_y = {selected['latent_dim']}, "
          f"beta = {selected['beta']} (collapse_ok={selected['collapse_ok']}, "
          f"eligible: {[(r['latent_dim'], r['beta']) for r in eligible]})")

    # --- Lin check (iii): mu2 vs ALL 35 SB35 parameters (val fold). The model
    # stays conditioned on the 5 cosmological params; only this analysis loads
    # the full parameter set.
    full_config = dataclasses.replace(data_config, camels_param_names=None)
    td_full = load_training_data(full_config)
    if not np.array_equal(td_full.sim_index, td.sim_index):
        raise RuntimeError("full-parameter load returned different rows")
    params_all_va = td_full.X_params[masks["val"]]
    assert td_full.param_names is not None
    mu2_va = selected["model"].encode(y_va, c_va)
    corr = latent_param_correlations(mu2_va, params_all_va)

    # --- run records (one per sweep point; the selected one carries the
    # replication blocks). Written after selection so summaries are complete.
    for res in results:
        is_selected = res is selected
        run_config = RunConfig(
            model="vae_spk",  # record-only label, not a registry entry
            model_params={"latent_dim": res["latent_dim"],
                          "beta": res["beta"], **model_params},
            seed=args.seed,
            split=PINNED_SPLIT,
        )
        summary = {
            "stage": "dual_vae_stage2",
            "component": "vae_y",
            "latent_dim": res["latent_dim"],
            "beta": res["beta"],
            "selected": is_selected,
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "decode_convention": "posterior-mean decode (amendment B4)",
            "val_recon_rmse": res["val_recon_rmse"],
            "train_recon_rmse": res["train_recon_rmse"],
            "suppressed_recon_rmse": res["suppressed_recon_rmse"],
            "pca_val_rmse_matched_dim": res["pca_val_rmse_matched"],
            "stage1_pca_y_run_id": STAGE1_PCA_Y_RUN_ID,
            "kl_per_dim_train": res["kl_per_dim_train"],
            "activity_train": res["activity_train"],
            "kl_per_dim_val": res["kl_per_dim_val"],
            "activity_val": res["activity_val"],
            "collapse_criterion": f"per-dim KL >= {COLLAPSE_KL_MIN} nats at "
                                  "best epoch, train fold (G2.2', B3)",
            "collapse_ok": res["collapse_ok"],
            # A_j documented, not gated (B3): delta-like posteriors are
            # permitted and expected in this near-deterministic regime.
            "obs_sigma_raw_scale": res["obs_sigma"],
            "best_epoch": res["best_epoch"],
            "reduction_convention": REDUCTION_CONVENTION,
        }
        if is_selected:
            summary["lin_replication"] = {
                "rmse_vs_pca": {
                    "vae_val_rmse": {
                        f"ld{r['latent_dim']}_beta{r['beta']}": r["val_recon_rmse"]
                        for r in results
                    },
                    "pca_val_rmse": {str(n): v for n, v in pca_rmse_at.items()},
                },
                "dimensionality_sweep": {
                    f"ld{r['latent_dim']}_beta{r['beta']}": {
                        "kl_per_dim_train": r["kl_per_dim_train"],
                        "activity_train": r["activity_train"],
                    }
                    for r in results
                },
                "latent_param_correlations": {
                    "fold": "val",
                    "param_names": list(td_full.param_names),
                    "pearson_r": [[float(v) for v in row] for row in corr],
                },
            }
        ledger_metrics = {
            "stage": "dual_vae_stage2",
            "latent_dim": res["latent_dim"],
            "beta": res["beta"],
            "selected": is_selected,
            "val_recon_rmse": res["val_recon_rmse"],
            "pca_val_rmse_matched_dim": res["pca_val_rmse_matched"],
            "collapse_ok": res["collapse_ok"],
        }
        record = write_run_record(
            data_config,
            run_config,
            dataset_path=td.source_path,
            data_root=data_config.project_root,
            scratch_root=scratch_root,
            summary=summary,
            ledger_metrics=ledger_metrics,
            metrics=res["model"].history,
            device=device,
            experiments_root=args.experiments_root,
        )
        res["run_id"] = record.run_id

        ckpt_dir = Path(scratch_root) / "models" / record.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(res["model"], ckpt_dir / "model.joblib")

        label = _figure_label(
            record.run_id,
            f"vae_spk_ld{res['latent_dim']}_b{res['beta']}",
        )
        if not args.no_figures:
            _write_trace_figure(res["model"], label)
            if is_selected:
                _write_rmse_vs_dim_figure(results, pca_table, label)
                _write_correlation_figure(
                    corr, list(td_full.param_names), label
                )
        print(f"[vae_spk ld={res['latent_dim']} beta={res['beta']}] "
              f"run_id: {record.run_id}"
              + ("  [SELECTED]" if is_selected else ""))

    # --- gate headline (F1 quadrature form) ---------------------------------
    base = args.g21_reference
    codec = selected["val_recon_rmse"]
    inflation = float(np.sqrt(base ** 2 + codec ** 2) / base - 1.0)
    g21_bar = G21_QUADRATURE_FACTOR * base
    print(json.dumps({
        "G2.1prime_codec_adequacy": bool(inflation <= G21_MAX_FLOOR_INFLATION),
        "G2.1prime_floor_inflation": inflation,
        "G2.1prime_bar": g21_bar,
        "G2.2prime_no_collapse": bool(selected["collapse_ok"]),
        "finding_vae_vs_pca_matched_dim": {
            "vae": selected["val_recon_rmse"],
            "pca": selected["pca_val_rmse_matched"],
        },
        "selected_latent_dim": selected["latent_dim"],
        "selected_beta": selected["beta"],
        "selected_run_id": selected["run_id"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
