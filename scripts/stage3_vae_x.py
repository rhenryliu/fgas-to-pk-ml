"""Stage 3 of the dual-VAE programme: VAE-X, autoencoding f_gas(R).

Trains :class:`~fgas_spk.models.dual_vae_components.GaussianVAE` on the
**train-fold** f_gas(R) profiles with the five cosmological CAMELS parameters
as context (decision F0.4), sweeping ``beta`` over {1, 0.1, 0.01, 0.001}
(F0.2 as amended 2026-07-05, B1) at ``latent_dim_x = 2`` first; pass
``--latent-dims 2 3 4 6`` to widen the sweep only if G3.1' fails at 2 (spec
Stage 3 task 1). One seeded run record per configuration, model label
``"vae_fgas"``.

Gates are the amended set (B6):

* **G3.1' (downstream adequacy):** on the train fold, two ridge regressions
  are fitted to the frozen VAE-Y posterior-mean targets ``mu2`` -- one from
  the VAE-X codes ``mu1``, one from PCA-on-X scores at the same dimension.
  Both val predictions are decoded through the frozen decoder-Y to y-space;
  the gate is VAE-X-arm val y-space RMSE <= 1.05 x PCA-arm val y-space RMSE.
  The frozen VAE-Y is loaded from its Stage 2 checkpoint (``--vae-y-run-id``).
  The VAE-X-vs-PCA-on-X *reconstruction* comparison is reported as a finding,
  not gated.
* **G3.2':** collapse detection only (per-dim KL >= 0.01 nats at best epoch).
* **G3.3:** OOD reference statistics of ``||mu1||`` and per-dimension ``mu1``
  ranges (train and val folds) in every run summary -- the detection baseline
  for the encoder-blow-up failure mode.

The B5 suppressed-regime diagnostic applies to y-codecs (its thresholds are
SP(k) values); an f_gas reconstruction has no SP(k) bins, so it appears here
only inside the G3.1' y-space arm RMSEs, which are reported globally and
truth-masked. Selection among configurations follows B1: lowest val
reconstruction RMSE (posterior-mean decode, raw f_gas scale) subject to the
no-collapse criterion. The latent-parameter correlation analysis (Stage 2 task
3(iii) pattern) runs for the selected configuration.

GR7 is inherited from ``GaussianVAE``; the convention string is recorded in
every run summary.

Usage::

    python scripts/stage3_vae_x.py --vae-y-run-id <stage2-selected-run-id> \\
        [--latent-dims 2] [--seed 0] [--epochs N] [--no-figures]
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
from fgas_spk.paths import fill_data_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split, pick_device

from stage1_pca_dimensionality import (
    N_COMPONENTS_GRID,
    PINNED_SPLIT,
    _figure_label,
    pca_dimensionality_table,
)
from stage2_vae_y import (
    BETAS,
    COLLAPSE_KL_MIN,
    DEFAULT_DATA_CONFIG,
    MODEL_PARAMS,
    REDUCTION_CONVENTION,
    SUPPRESSED_THRESHOLDS,
    _collapse_ok,
    _rmse,
    _suppressed_recon_rmse,
    _write_correlation_figure,
    _write_trace_figure,
    latent_param_correlations,
)

# Stage 1 canonical PCA-on-X run on the pinned tag (finding reference).
STAGE1_PCA_X_RUN_ID = "20260706T061039Z__129449f5__3e94bcc"

# G3.1' guardrail: VAE-X arm <= this factor x PCA-x-scores arm (B6).
G31_GUARDRAIL = 1.05

# Amendment 2, C1 (anti-vacuous-pass floor): the ratio test is valid only if
# the PCA-x-scores arm's val y-space RMSE clears the Stage 1 `pca_linear`
# baseline on the same tag and folds (tag 20260706, Branch-A clean data:
# run 20260706T061051Z__1f810c16__3e94bcc). A mutual-failure ratio of ~1 must
# never pass again.
G31_FLOOR_PCA_LINEAR_VAL_RMSE = 0.038203


def ood_reference_stats(mu: np.ndarray) -> dict:
    """G3.3 OOD reference statistics for a set of latent codes.

    Args:
        mu (np.ndarray): Posterior means, shape (n_examples, latent_dim).

    Returns:
        dict: ``norm`` (median / p99 / max of ``||mu||_2``) and ``per_dim``
            (min / max per dimension).
    """
    norms = np.linalg.norm(np.asarray(mu, dtype=float), axis=1)
    return {
        "norm": {
            "median": float(np.median(norms)),
            "p99": float(np.percentile(norms, 99)),
            "max": float(np.max(norms)),
        },
        "per_dim": {
            "min": [float(v) for v in mu.min(axis=0)],
            "max": [float(v) for v in mu.max(axis=0)],
        },
    }


def downstream_adequacy(
    mu1_tr: np.ndarray,
    mu1_va: np.ndarray,
    x_tr: np.ndarray,
    x_va: np.ndarray,
    mu2_tr: np.ndarray,
    vae_y,
    c_va: np.ndarray,
    y_va: np.ndarray,
    latent_dim: int,
    seed: int,
) -> dict:
    """Evaluate the G3.1' two-arm downstream-adequacy comparison.

    Fits one :class:`~sklearn.linear_model.RidgeCV` per arm on the train fold
    -- VAE-X codes ``mu1`` -> ``mu2``, and PCA-on-X scores (same dimension,
    train-fold fit) -> ``mu2`` -- decodes both val predictions through the
    frozen decoder-Y, and reports val y-space RMSEs (global and B5
    truth-masked).

    Args:
        mu1_tr (np.ndarray): VAE-X train-fold codes, shape (n_tr, d).
        mu1_va (np.ndarray): VAE-X val-fold codes, shape (n_va, d).
        x_tr (np.ndarray): Train-fold raw profiles (for the PCA arm).
        x_va (np.ndarray): Val-fold raw profiles.
        mu2_tr (np.ndarray): Frozen VAE-Y train-fold codes (targets).
        vae_y: The frozen Stage 2 VAE-Y (its ``decode`` is used).
        c_va (np.ndarray): Val-fold context for decoder-Y.
        y_va (np.ndarray): Val-fold true SP(k) (for the RMSEs).
        latent_dim (int): The VAE-X latent dimension (PCA arm matches it).
        seed (int): PCA random_state (inert for full SVD; recorded).

    Returns:
        dict: Per-arm ``{"rmse": float, "suppressed": {...}}`` under keys
            ``vae_arm`` / ``pca_arm``, plus ``ratio`` and ``pass`` per the
            guardrail.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV

    alphas = np.logspace(-4, 3, 15)

    pca = PCA(n_components=latent_dim, svd_solver="full", random_state=seed)
    scores_tr = pca.fit_transform(x_tr)
    scores_va = pca.transform(x_va)

    arms = {}
    for name, feats_tr, feats_va in (
        ("vae_arm", mu1_tr, mu1_va),
        ("pca_arm", scores_tr, scores_va),
    ):
        ridge = RidgeCV(alphas=alphas).fit(feats_tr, mu2_tr)
        y_hat = vae_y.decode(ridge.predict(feats_va), c_va)
        arms[name] = {
            "rmse": _rmse(y_hat, y_va),
            "suppressed": _suppressed_recon_rmse(y_va, y_hat),
            "ridge_alpha": (
                [float(a) for a in np.atleast_1d(ridge.alpha_)]
            ),
        }
    ratio = arms["vae_arm"]["rmse"] / arms["pca_arm"]["rmse"]
    # C1 floor: the reference arm must itself be a usable predictor.
    floor_ok = arms["pca_arm"]["rmse"] <= G31_FLOOR_PCA_LINEAR_VAL_RMSE
    return {
        **arms,
        "ratio": float(ratio),
        "guardrail": G31_GUARDRAIL,
        "floor_reference_pca_linear": G31_FLOOR_PCA_LINEAR_VAL_RMSE,
        "floor_ok": bool(floor_ok),
        "pass": bool(floor_ok and ratio <= G31_GUARDRAIL),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 3 VAE-X sweep, gates, and run records.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Stage 3: VAE-X on train-fold f_gas(R) with cosmological "
        "context; beta sweep at latent_dim_x=2; amended gates G3.1'/G3.2'/G3.3."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG,
                        help="Path to the pinned DataConfig YAML.")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root (else $FGAS_SCRATCH_ROOT).")
    parser.add_argument("--vae-y-run-id", required=True,
                        help="Stage 2 selected vae_spk run id; its scratch "
                        "checkpoint provides the frozen mu2 targets and "
                        "decoder-Y for G3.1'.")
    parser.add_argument("--latent-dims", type=int, nargs="+", default=[2],
                        help="latent_dim_x values to sweep (default: 2; widen "
                        "to 2 3 4 6 only if G3.1' fails at 2).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Model seed (torch + internal holdout).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override MODEL_PARAMS['epochs'] (smoke tests).")
    parser.add_argument("--experiments-root", default=None,
                        help="Override the run-record tree (smoke tests).")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip writing figures.")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)
    device = pick_device()

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    x_tr = np.asarray(td.X[masks["train"]], dtype=float)
    x_va = np.asarray(td.X[masks["val"]], dtype=float)
    y_tr, y_va = td.y[masks["train"]], td.y[masks["val"]]
    assert td.X_params is not None, "pinned config must load the 5 cosmo params"
    c_tr, c_va = td.X_params[masks["train"]], td.X_params[masks["val"]]

    # Frozen Stage 2 VAE-Y: mu2 targets (train fold) + decoder-Y for G3.1'.
    vae_y_ckpt = Path(scratch_root) / "models" / args.vae_y_run_id / "model.joblib"
    vae_y = joblib.load(vae_y_ckpt)
    mu2_tr = vae_y.encode(y_tr, c_tr)

    # PCA-on-X reference on the identical folds (finding, not gated).
    _, pca_table = pca_dimensionality_table(
        x_tr, x_va, N_COMPONENTS_GRID, seed=0,
    )
    pca_rmse_at = dict(zip(pca_table["n_components"], pca_table["val_recon_rmse"]))

    model_params = dict(MODEL_PARAMS)
    if args.epochs is not None:
        model_params["epochs"] = args.epochs

    # --- sweep: latent_dim_x (default just 2) x beta ------------------------
    results: list[dict] = []
    for ld in args.latent_dims:
        for beta in BETAS:
            model = GaussianVAE(
                latent_dim=ld, beta=beta, seed=args.seed, **model_params
            )
            model.fit(x_tr, c_tr)
            stats_tr = model.latent_stats(x_tr, c_tr)
            stats_va = model.latent_stats(x_va, c_va)
            adequacy = downstream_adequacy(
                stats_tr["mu"], stats_va["mu"], x_tr, x_va, mu2_tr,
                vae_y, c_va, y_va, latent_dim=ld, seed=args.seed,
            )
            res = {
                "latent_dim": ld,
                "beta": beta,
                "model": model,
                "train_recon_rmse": _rmse(model.reconstruct(x_tr, c_tr), x_tr),
                "val_recon_rmse": _rmse(model.reconstruct(x_va, c_va), x_va),
                "pca_val_rmse_matched": pca_rmse_at[ld],
                "kl_per_dim_train": stats_tr["kl_per_dim"],
                "activity_train": stats_tr["activity"],
                "kl_per_dim_val": stats_va["kl_per_dim"],
                "activity_val": stats_va["activity"],
                "collapse_ok": _collapse_ok(stats_tr["kl_per_dim"]),
                "obs_sigma": [float(s) for s in model.obs_sigma()],
                "best_epoch": model.best_epoch,
                # Amendment 2, C2 (G3.4): an epoch-0 restore (or no internal
                # holdout improvement at all) is an automatic failure for this
                # configuration, whatever the other criteria say.
                "sanity_ok": model.best_epoch not in (None, 0),
                "adequacy": adequacy,
                "ood_train": ood_reference_stats(stats_tr["mu"]),
                "ood_val": ood_reference_stats(stats_va["mu"]),
            }
            results.append(res)
            print(f"[vae_fgas ld={ld} beta={beta}] "
                  f"val recon RMSE {res['val_recon_rmse']:.5g} "
                  f"(PCA {res['pca_val_rmse_matched']:.5g}) "
                  f"KL={np.round(res['kl_per_dim_train'], 3).tolist()} "
                  f"collapse_ok={res['collapse_ok']} "
                  f"G3.1' ratio={adequacy['ratio']:.4f} "
                  f"(vae {adequacy['vae_arm']['rmse']:.5g} / "
                  f"pca {adequacy['pca_arm']['rmse']:.5g}) "
                  f"best_epoch={res['best_epoch']}")

    # --- selection (B1 + amendment 2 C2): lowest val recon RMSE subject to
    # no-collapse AND the G3.4 training-sanity criterion.
    eligible = [r for r in results if r["collapse_ok"] and r["sanity_ok"]]
    pool = eligible if eligible else results
    selected = min(pool, key=lambda r: r["val_recon_rmse"])
    print(f"selected latent_dim_x = {selected['latent_dim']}, "
          f"beta = {selected['beta']} (collapse_ok={selected['collapse_ok']}, "
          f"sanity_ok={selected['sanity_ok']})")

    # --- correlations of mu1 vs ALL 35 SB35 parameters (val fold) -----------
    full_config = dataclasses.replace(data_config, camels_param_names=None)
    td_full = load_training_data(full_config)
    if not np.array_equal(td_full.sim_index, td.sim_index):
        raise RuntimeError("full-parameter load returned different rows")
    params_all_va = td_full.X_params[masks["val"]]
    assert td_full.param_names is not None
    mu1_va = selected["model"].encode(x_va, c_va)
    corr = latent_param_correlations(mu1_va, params_all_va)

    # --- run records ---------------------------------------------------------
    for res in results:
        is_selected = res is selected
        run_config = RunConfig(
            model="vae_fgas",  # record-only label, not a registry entry
            model_params={"latent_dim": res["latent_dim"],
                          "beta": res["beta"], **model_params},
            seed=args.seed,
            split=PINNED_SPLIT,
        )
        summary = {
            "stage": "dual_vae_stage3",
            "component": "vae_x",
            "latent_dim": res["latent_dim"],
            "beta": res["beta"],
            "selected": is_selected,
            "n_train": int(masks["train"].sum()),
            "n_val": int(masks["val"].sum()),
            "decode_convention": "posterior-mean decode (amendment B4)",
            "val_recon_rmse": res["val_recon_rmse"],
            "train_recon_rmse": res["train_recon_rmse"],
            "pca_val_rmse_matched_dim": res["pca_val_rmse_matched"],
            "stage1_pca_x_run_id": STAGE1_PCA_X_RUN_ID,
            "g31_prime_downstream_adequacy": res["adequacy"],
            "frozen_vae_y_run_id": args.vae_y_run_id,
            "kl_per_dim_train": res["kl_per_dim_train"],
            "activity_train": res["activity_train"],
            "kl_per_dim_val": res["kl_per_dim_val"],
            "activity_val": res["activity_val"],
            "collapse_criterion": f"per-dim KL >= {COLLAPSE_KL_MIN} nats at "
                                  "best epoch, train fold (G3.2', B3)",
            "collapse_ok": res["collapse_ok"],
            "g34_sanity_ok": res["sanity_ok"],
            "obs_sigma_raw_scale": res["obs_sigma"],
            "ood_reference": {"train": res["ood_train"], "val": res["ood_val"]},
            "best_epoch": res["best_epoch"],
            "reduction_convention": REDUCTION_CONVENTION,
        }
        if is_selected:
            summary["latent_param_correlations"] = {
                "fold": "val",
                "param_names": list(td_full.param_names),
                "pearson_r": [[float(v) for v in row] for row in corr],
            }
        ledger_metrics = {
            "stage": "dual_vae_stage3",
            "latent_dim": res["latent_dim"],
            "beta": res["beta"],
            "selected": is_selected,
            "val_recon_rmse": res["val_recon_rmse"],
            "g31_ratio": res["adequacy"]["ratio"],
            "g31_pass": res["adequacy"]["pass"],
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
            record.run_id, f"vae_fgas_ld{res['latent_dim']}_b{res['beta']}"
        )
        if not args.no_figures:
            _write_trace_figure(res["model"], label)
            if is_selected:
                _write_correlation_figure(corr, list(td_full.param_names), label)
        print(f"[vae_fgas ld={res['latent_dim']} beta={res['beta']}] "
              f"run_id: {record.run_id}"
              + ("  [SELECTED]" if is_selected else ""))

    # --- gate headline -------------------------------------------------------
    print(json.dumps({
        "G3.1prime_downstream_adequacy": selected["adequacy"]["pass"],
        "G3.1prime_ratio": selected["adequacy"]["ratio"],
        "G3.1prime_floor_ok": selected["adequacy"]["floor_ok"],
        "G3.2prime_no_collapse": bool(selected["collapse_ok"]),
        "G3.3_ood_stats_present": True,
        "G3.4_training_sanity": bool(selected["sanity_ok"]),
        "finding_vae_vs_pca_x_recon": {
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
