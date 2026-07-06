"""Stage 4 of the dual-VAE programme: the latent mapping ladder z1 -> z2.

Builds ``(mu1, mu2)`` posterior-mean pairs from the **frozen** Stage 2/3
encoders (decision F0.5) on the train fold, fits the three mapping rungs, and
evaluates each rung's **composite** (x -> encoder-X -> map -> decoder-Y) on the
val fold:

* **Rung 1** (``latent_map_ridge``): sklearn RidgeCV, z1 -> z2. The
  linear-in-latent diagnostic baseline.
* **Rung 2** (``latent_map_mlp``): :class:`~fgas_spk.models.dual_vae_components.LatentMapMLP`.
* **Rung 3** (``latent_map_mdn``): :class:`~fgas_spk.models.dual_vae_components.LatentMapMDN`
  modelling p(z2 | z1); K = 3 is the recorded configuration, with a K in
  {1, 3, 5} val-fold sensitivity table in its summary.

Composite point predictions and (rung 3) predictive samples use the shared
helpers :func:`~fgas_spk.models.dual_vae_components.composite_predict` /
:func:`~fgas_spk.models.dual_vae_components.composite_samples` -- the same
functions the Stage 5 plugin's predict path uses. Per F0.5 the rung-3 spread
is the mapping density plus decoder-Y observation noise only.

Gates (original spec, unamended):

* **G4.1**: composite val global + suppressed-regime RMSE per rung, appended
  to ``experiments/notes/dual_vae_baseline_table.md``; the rung-2 vs
  ``mlp_regressor`` gap reported as the information-bottleneck cost.
* **G4.2**: rung-3 coverage at nominal 68% / 95%, per k-bin and pooled;
  pass = pooled within +/-10 percentage points at both levels.
* **G4.3**: per-curve val RMSE ranking by ``sim_index``, cross-referenced
  against the G3.3 ``||mu1||`` OOD reference.

Usage::

    python scripts/stage4_latent_map.py \\
        --vae-x-run-id <stage3-selected> --vae-y-run-id <stage2-selected> \\
        [--seed 0] [--epochs N] [--no-figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.experiment import RunConfig
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.models.dual_vae_components import (
    LatentMapMDN,
    LatentMapMLP,
    LatentMapRidge,
    composite_predict,
    composite_samples,
)
from fgas_spk.paths import figure_dir, fill_data_root, repo_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split, pick_device

from stage1_pca_dimensionality import PINNED_SPLIT, _figure_label
from stage2_vae_y import (
    DEFAULT_DATA_CONFIG,
    _rmse,
    _suppressed_recon_rmse,
)

# mlp_regressor on the pinned context (tag 20260706, R < 10 crop; F4 table);
# the G4.1 information-bottleneck-cost reference.
MLP_REGRESSOR_VAL_RMSE = 0.019378
MLP_REGRESSOR_RUN_ID = "20260706T185713Z__c62b6ed0__a4ba5aa"

# Rung hyperparameters (shared trunk conventions with the components).
MAP_PARAMS = {
    "hidden": 128,
    "n_layers": 2,
    "epochs": 2000,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "batch_size": 128,
    "val_frac": 0.1,
}
MDN_K_DEFAULT = 3
MDN_K_SENSITIVITY = (1, 3, 5)

COVERAGE_LEVELS = (0.68, 0.95)
COVERAGE_TOLERANCE_PP = 10.0
N_PREDICTIVE_SAMPLES = 400

NOTE_PATH = "experiments/notes/dual_vae_baseline_table.md"


def coverage_table(
    samples: np.ndarray, y_true: np.ndarray, levels=COVERAGE_LEVELS
) -> dict:
    """Empirical central-interval coverage, pooled and per k-bin.

    Args:
        samples (np.ndarray): Predictive samples, shape (S, N, n_k).
        y_true (np.ndarray): True values, shape (N, n_k).
        levels (Iterable[float]): Nominal central-interval levels.

    Returns:
        dict: Per level: ``pooled`` fraction, ``per_k`` list, and ``pass``
            per the +/-10 pp pooled tolerance.
    """
    out: dict = {}
    for level in levels:
        lo = np.quantile(samples, (1.0 - level) / 2.0, axis=0)
        hi = np.quantile(samples, (1.0 + level) / 2.0, axis=0)
        inside = (y_true >= lo) & (y_true <= hi)
        pooled = float(inside.mean())
        out[str(level)] = {
            "pooled": pooled,
            "per_k": [float(v) for v in inside.mean(axis=0)],
            "pass": bool(abs(pooled - level) <= COVERAGE_TOLERANCE_PP / 100.0),
        }
    return out


def per_curve_ranking(
    y_true: np.ndarray, y_pred: np.ndarray, sim_index: np.ndarray,
    mu1_norms: np.ndarray, top_n: int = 10,
) -> dict:
    """G4.3: per-curve RMSE ranking cross-referenced with ``||mu1||``.

    Args:
        y_true (np.ndarray): True SP(k), shape (N, n_k).
        y_pred (np.ndarray): Composite point predictions, same shape.
        sim_index (np.ndarray): Originating simulation ids, shape (N,).
        mu1_norms (np.ndarray): ``||mu1||_2`` per row, shape (N,).
        top_n (int): Number of worst curves listed.

    Returns:
        dict: Distribution stats, the worst curves (sim id, RMSE, ||mu1||,
            and its percentile within the val-fold norms), and the Pearson
            correlation between per-curve RMSE and ``||mu1||``.
    """
    rmse_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
    order = np.argsort(rmse_curve)[::-1][: min(top_n, rmse_curve.shape[0])]
    pct = 100.0 * np.argsort(np.argsort(mu1_norms)) / max(len(mu1_norms) - 1, 1)
    corr = float(np.corrcoef(rmse_curve, mu1_norms)[0, 1])
    return {
        "mean": float(rmse_curve.mean()),
        "median": float(np.median(rmse_curve)),
        "p90": float(np.percentile(rmse_curve, 90)),
        "max": float(rmse_curve.max()),
        "rmse_vs_mu1_norm_pearson_r": corr,
        "worst": [
            {
                "sim_index": int(sim_index[i]),
                "rmse": float(rmse_curve[i]),
                "mu1_norm": float(mu1_norms[i]),
                "mu1_norm_percentile": float(pct[i]),
            }
            for i in order
        ],
    }


def _write_coverage_figure(cov: dict, k: np.ndarray, label: str) -> Path | None:
    """Per-k coverage at both nominal levels with the +/-10 pp pooled band."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (level, entry) in enumerate(cov.items()):
        colour = f"C{i}"
        ax.plot(k, entry["per_k"], marker=".", color=colour,
                label=f"nominal {level} (pooled {entry['pooled']:.3f})")
        ax.axhline(float(level), color=colour, linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{label}\ncomposite coverage (val)")
    ax.legend()
    out = figure_dir() / f"{label}__composite_coverage.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_rmse_vs_norm_figure(g43: dict, label: str) -> Path | None:
    """G4.3 scatter stand-in: worst-curve markers over the rank correlation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    worst = g43["worst"]
    ax.scatter([w["mu1_norm"] for w in worst], [w["rmse"] for w in worst],
               color="C3", label=f"{len(worst)} worst val curves")
    for w in worst[:5]:
        ax.annotate(str(w["sim_index"]), (w["mu1_norm"], w["rmse"]), fontsize=7)
    ax.set_xlabel(r"$||\mu_1||_2$")
    ax.set_ylabel("per-curve RMSE")
    ax.set_title(f"{label}\nworst curves vs encoder-X code norm "
                 f"(r = {g43['rmse_vs_mu1_norm_pearson_r']:.2f} over all val)")
    ax.legend()
    out = figure_dir() / f"{label}__rmse_vs_mu1_norm.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 4 mapping ladder, gates, and run records.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Stage 4: fit the z1->z2 mapping ladder on frozen VAE "
        "codes and evaluate the composites on the val fold."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--codec-x", choices=("vae", "pca"), default="vae",
                        help="Amendment-6 dual track: Arm V (vae) loads the "
                        "frozen Stage 3 checkpoint; Arm P (pca) fits a "
                        "train-fold PcaCodec at --pca-dim.")
    parser.add_argument("--pca-dim", type=int, default=4,
                        help="PCA codec dimension for --codec-x pca (Arm P).")
    parser.add_argument("--vae-x-run-id", default=None,
                        help="Stage 3 selected vae_fgas run id (frozen); "
                        "required for --codec-x vae.")
    parser.add_argument("--vae-y-run-id", required=True,
                        help="Stage 2 selected vae_spk run id (frozen).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override MAP_PARAMS['epochs'] (smoke tests).")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-table-append", action="store_true",
                        help="Skip appending the G4.1 rows to the baseline "
                        "table (smoke tests).")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)
    device = pick_device()

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    x_tr, x_va = td.X[masks["train"]], td.X[masks["val"]]
    y_tr, y_va = np.asarray(td.y[masks["train"]], float), np.asarray(td.y[masks["val"]], float)
    assert td.X_params is not None
    c_tr, c_va = td.X_params[masks["train"]], td.X_params[masks["val"]]
    sim_va = td.sim_index[masks["val"]]

    models_root = Path(scratch_root) / "models"
    vae_y = joblib.load(models_root / args.vae_y_run_id / "model.joblib")
    if args.codec_x == "vae":
        if args.vae_x_run_id is None:
            raise SystemExit("--vae-x-run-id is required for --codec-x vae")
        vae_x = joblib.load(models_root / args.vae_x_run_id / "model.joblib")
        arm = "V"
        codec_x_desc = {"codec_x": "vae", "vae_x_run_id": args.vae_x_run_id}
    else:
        # Arm P: train-fold PCA codec (deterministic full SVD; the context is
        # ignored by PcaCodec). The mapping is re-fit on this code space (I2.3).
        from fgas_spk.models.dual_vae_components import PcaCodec

        vae_x = PcaCodec(latent_dim=args.pca_dim, seed=0)
        vae_x.fit(np.asarray(x_tr, dtype=float))
        arm = "P"
        codec_x_desc = {"codec_x": "pca", "pca_dim": args.pca_dim}

    # Frozen posterior-mean / code pairs (F0.5). For Arm P the code norms
    # serve the ||mu1|| OOD role (I2.2).
    mu1_tr, mu1_va = vae_x.encode(x_tr, c_tr), vae_x.encode(x_va, c_va)
    mu2_tr = vae_y.encode(y_tr, c_tr)
    mu1_norms_va = np.linalg.norm(mu1_va, axis=1)

    map_params = dict(MAP_PARAMS)
    if args.epochs is not None:
        map_params["epochs"] = args.epochs

    # --- fit the three rungs -------------------------------------------------
    ridge = LatentMapRidge(seed=args.seed)
    ridge.fit(mu1_tr, mu2_tr)

    mlp = LatentMapMLP(seed=args.seed, **map_params)
    mlp.fit(mu1_tr, mu2_tr)

    mdns = {}
    for k_mix in MDN_K_SENSITIVITY:
        mdn = LatentMapMDN(n_components=k_mix, seed=args.seed, **map_params)
        mdn.fit(mu1_tr, mu2_tr)
        mdns[k_mix] = mdn
    mdn_default = mdns[MDN_K_DEFAULT]

    # --- composite evaluation per rung ---------------------------------------
    rungs = {
        "latent_map_ridge": ridge,
        "latent_map_mlp": mlp,
        "latent_map_mdn": mdn_default,
    }
    results: dict[str, dict] = {}
    for name, rung in rungs.items():
        y_hat = composite_predict(vae_x, rung, vae_y, x_va, c_va)
        res = {
            "val_rmse": _rmse(y_hat, y_va),
            "suppressed_rmse": _suppressed_recon_rmse(y_va, y_hat),
            "per_curve": per_curve_ranking(y_va, y_hat, sim_va, mu1_norms_va),
        }
        results[name] = res
        print(f"[{name}] composite val RMSE {res['val_rmse']:.5g} "
              f"suppressed(0.8) "
              f"{res['suppressed_rmse']['0.8']['rmse']:.5g}")

    # Rung-3 predictive distribution: coverage (G4.2) + K sensitivity.
    samples = composite_samples(
        vae_x, mdn_default, vae_y, x_va, c_va,
        n_samples=N_PREDICTIVE_SAMPLES, seed=args.seed,
    )
    cov = coverage_table(samples, y_va)
    results["latent_map_mdn"]["coverage"] = cov
    k_sensitivity = {}
    for k_mix, mdn in mdns.items():
        y_hat_k = composite_predict(vae_x, mdn, vae_y, x_va, c_va)
        s_k = samples if k_mix == MDN_K_DEFAULT else composite_samples(
            vae_x, mdn, vae_y, x_va, c_va,
            n_samples=N_PREDICTIVE_SAMPLES, seed=args.seed,
        )
        cov_k = coverage_table(s_k, y_va)
        k_sensitivity[str(k_mix)] = {
            "val_rmse": _rmse(y_hat_k, y_va),
            "pooled_coverage": {lv: cov_k[lv]["pooled"] for lv in cov_k},
            "best_epoch": mdn.best_epoch,
        }
    results["latent_map_mdn"]["k_sensitivity"] = k_sensitivity
    print("G4.2 pooled coverage:",
          {lv: round(cov[lv]["pooled"], 3) for lv in cov},
          "pass:", all(cov[lv]["pass"] for lv in cov))

    # Bottleneck cost (G4.1): rung 2 vs the direct mlp_regressor.
    bottleneck = {
        "rung2_val_rmse": results["latent_map_mlp"]["val_rmse"],
        "mlp_regressor_val_rmse": MLP_REGRESSOR_VAL_RMSE,
        "mlp_regressor_run_id": MLP_REGRESSOR_RUN_ID,
        "gap": results["latent_map_mlp"]["val_rmse"] - MLP_REGRESSOR_VAL_RMSE,
        "ratio": results["latent_map_mlp"]["val_rmse"] / MLP_REGRESSOR_VAL_RMSE,
    }

    # --- run records ----------------------------------------------------------
    rung_model_params = {
        "latent_map_ridge": {"alphas": "logspace(-4, 3, 15)"},
        "latent_map_mlp": dict(map_params),
        "latent_map_mdn": {"n_components": MDN_K_DEFAULT, **map_params},
    }
    frozen = {
        "arm": arm,
        **codec_x_desc,
        "vae_y_run_id": args.vae_y_run_id,
    }
    for name, rung in rungs.items():
        res = results[name]
        run_config = RunConfig(
            model=name,  # record-only label, not a registry entry
            model_params=rung_model_params[name],
            seed=args.seed,
            split=PINNED_SPLIT,
        )
        summary = {
            "stage": "dual_vae_stage4",
            "rung": name,
            "frozen_components": frozen,
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "composite_val_rmse": res["val_rmse"],
            "composite_suppressed_rmse": res["suppressed_rmse"],
            "per_curve": res["per_curve"],
            "spread_semantics": (
                "F0.5: rung-3 spread = mapping density + decoder-Y observation "
                "noise; EXCLUDES encoder-X posterior spread and input noise."
            ),
        }
        if name == "latent_map_ridge":
            summary["ridge_alpha"] = ridge.alpha
        if name == "latent_map_mlp":
            summary["bottleneck_cost"] = bottleneck
            summary["best_epoch"] = mlp.best_epoch
        if name == "latent_map_mdn":
            summary["coverage"] = res["coverage"]
            summary["k_sensitivity"] = res["k_sensitivity"]
            summary["best_epoch"] = mdn_default.best_epoch
        ledger_metrics = {
            "stage": "dual_vae_stage4",
            "arm": arm,
            "rung": name,
            "composite_val_rmse": res["val_rmse"],
        }
        record = write_run_record(
            data_config,
            run_config,
            dataset_path=td.source_path,
            data_root=data_config.project_root,
            scratch_root=scratch_root,
            summary=summary,
            ledger_metrics=ledger_metrics,
            metrics=getattr(rung, "history", None) or None,
            device=device,
            experiments_root=args.experiments_root,
        )
        res["run_id"] = record.run_id

        ckpt_dir = Path(scratch_root) / "models" / record.run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(rung, ckpt_dir / "model.joblib")

        label = _figure_label(record.run_id, f"{name}_arm{arm}")
        if not args.no_figures:
            if name == "latent_map_mdn":
                _write_coverage_figure(res["coverage"], np.asarray(td.k), label)
            _write_rmse_vs_norm_figure(res["per_curve"], label)
        print(f"[{name}] run_id: {record.run_id}")

    # --- G4.1 table append -----------------------------------------------------
    if not args.no_table_append:
        lines = [
            "",
            f"## Stage 4 composite rungs — Arm {arm} "
            f"({codec_x_desc}) (appended; val fold)",
            "",
            f"Frozen y-codec: VAE-Y `{args.vae_y_run_id}`. Bottleneck cost "
            f"(rung 2 vs `mlp_regressor` {MLP_REGRESSOR_VAL_RMSE}): "
            f"gap {bottleneck['gap']:+.5g} (ratio {bottleneck['ratio']:.3f}).",
            "",
            "| rung | run_id | val RMSE | val RMSE (SP<0.95) | val RMSE (SP<0.9) | val RMSE (SP<0.8) |",
            "|---|---|---|---|---|---|",
        ]
        for name in rungs:
            res = results[name]
            sup = res["suppressed_rmse"]
            cells = " | ".join(
                f"{sup[t]['rmse']:.5g} (n={sup[t]['n_bins']})"
                for t in ("0.95", "0.9", "0.8")
            )
            lines.append(
                f"| `{name}` | `{res['run_id']}` | {res['val_rmse']:.5g} | {cells} |"
            )
        note = repo_root() / NOTE_PATH
        note.write_text(note.read_text() + "\n".join(lines) + "\n")
        print(f"G4.1 rows appended to {note}")

    print(json.dumps({
        "G4.2_coverage_pass": all(cov[lv]["pass"] for lv in cov),
        "G4.2_pooled": {lv: cov[lv]["pooled"] for lv in cov},
        "rung_val_rmse": {n: results[n]["val_rmse"] for n in rungs},
        "bottleneck_ratio": bottleneck["ratio"],
        "run_ids": {n: results[n]["run_id"] for n in rungs},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
