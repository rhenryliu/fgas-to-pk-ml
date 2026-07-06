"""Stage 6 val-fold gate evaluation for ``dual_vae_ft`` (amendment 7, J2.3).

Fits :class:`~fgas_spk.models.dual_vae_ft.DualVaeFt` on the train fold with
the exact ``dual_vae_ft.yaml`` configuration (so the later declared test run
through ``scripts/run.py`` refits the identical model deterministically) and
evaluates the J2.3 val-fold gates **before any test touch**:

* **G4.2 in full** — pooled coverage within +/-10 pp at 68% and 95%;
* **calibration-regression check** vs Arm V's staged rung-3 val coverage
  (0.7132 / 0.8941, run `20260706T214941Z__eae10d43__0908bc9`): a worsening
  of the distance-from-nominal by more than MATERIAL_PP at either level is a
  STOP (operationalization of "materially worse ... beyond seed-level
  variation": the observed seed/K spread was 1-2 pp; the threshold doubles
  it and is recorded here);
* **G3.4 sanity** on the phase-1 fine-tune (best epoch not 0/None);
* **OOD**: fine-tuned ``||mu1'||`` distribution against the Arm V VAE-X
  G3.3 reference (read from its Stage 3 record);
* **G4.3** per-curve analysis.

Also reports the J2.4 recoverability verdict inputs: val global and
suppressed-regime RMSE against Arm V, Arm P, the Stage 4.0 ceiling, and
``mlp_regressor`` (no encoded expectation). Exit 1 on any gate failure.

Usage::

    python scripts/stage6_finetune.py [--no-figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.experiment import RunConfig
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.models import REGISTRY
from fgas_spk.paths import fill_data_root, repo_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import _subset, grouped_split, pick_device

from stage1_pca_dimensionality import PINNED_SPLIT, _figure_label
from stage2_vae_y import DEFAULT_DATA_CONFIG, _rmse, _suppressed_recon_rmse
from stage4_latent_map import coverage_table, per_curve_ranking
from stage4_0_probe_selection import PCA_LINEAR_RUN_ID  # noqa: F401 (provenance)

RUN_YAML = "scripts/configs/run/dual_vae_ft.yaml"

# Arm V staged rung-3 val coverage (run 20260706T214941Z__eae10d43__0908bc9).
ARM_V_VAL_COVERAGE = {"0.68": 0.7132352941176471, "0.95": 0.8940631808278867}
# Materiality threshold for the calibration-regression STOP (pp; see docstring).
MATERIAL_PP = 3.0

# Arm V VAE-X Stage 3 record (its G3.3 OOD reference block is read from disk).
ARM_V_VAE_X_RUN_ID = "20260706T200039Z__3080064e__8cbe5b2"

# J2.4 recoverability references (val fold, same tag/crop/folds).
RECOVERABILITY_REFS = {
    "arm_V_rung3": (0.05431160240563648, "20260706T214941Z__eae10d43__0908bc9"),
    "arm_P_rung3": (0.032697704733729495, "20260706T215154Z__eae10d43__0908bc9"),
    "stage4_0_ceiling": (0.03635376826727498, "20260706T205815Z__e90f52b8__a6dab80"),
    "mlp_regressor": (0.019378, "20260706T185713Z__c62b6ed0__a4ba5aa"),
}


def main(argv: list[str] | None = None) -> int:
    """Fit dual_vae_ft, evaluate the J2.3 val gates, write the record.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: 0 when every val gate passes; 1 otherwise (STOP).
    """
    parser = argparse.ArgumentParser(description="Stage 6 val-gate evaluation.")
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    run_config = RunConfig.from_yaml(RUN_YAML)
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    train_td, val_td = _subset(td, masks["train"]), _subset(td, masks["val"])
    y_va = np.asarray(val_td.y, dtype=float)

    model = REGISTRY[run_config.model](seed=run_config.seed,
                                       **run_config.model_params)
    model.fit(train_td)

    # --- val-fold evaluation --------------------------------------------------
    y_hat = model.predict(val_td.X, val_td.X_cond, val_td.X_params)
    samples = model.predict_samples(
        val_td.X, val_td.X_cond, val_td.X_params, n_samples=400,
        seed=run_config.seed,
    )
    cov = coverage_table(samples, y_va)
    mu1_ft = model.latents(val_td.X, val_td.X_cond, val_td.X_params)
    norms = np.linalg.norm(mu1_ft, axis=1)
    per_curve = per_curve_ranking(y_va, y_hat, val_td.sim_index, norms)

    # Gates.
    g42 = all(cov[lv]["pass"] for lv in cov)
    g34 = model.ft_best_epoch not in (None, 0)
    regression = {}
    calibration_ok = True
    for lv, ref in ARM_V_VAL_COVERAGE.items():
        nominal = float(lv)
        dist_ft = abs(cov[lv]["pooled"] - nominal)
        dist_v = abs(ref - nominal)
        worsening_pp = 100.0 * (dist_ft - dist_v)
        regression[lv] = {"ft_pooled": cov[lv]["pooled"], "arm_v_pooled": ref,
                          "worsening_pp": worsening_pp}
        calibration_ok &= worsening_pp <= MATERIAL_PP

    arm_v_ood = json.loads(
        (repo_root() / "experiments" / "runs" / ARM_V_VAE_X_RUN_ID /
         "summary.json").read_text()
    )["ood_reference"]["val"]["norm"]
    ood_ft = {"median": float(np.median(norms)),
              "p99": float(np.percentile(norms, 99)),
              "max": float(np.max(norms))}

    recoverability = {
        "dual_vae_ft_val_rmse": _rmse(y_hat, y_va),
        "dual_vae_ft_suppressed": _suppressed_recon_rmse(y_va, y_hat),
        "references": {k: {"val_rmse": v[0], "run_id": v[1]}
                       for k, v in RECOVERABILITY_REFS.items()},
    }

    summary = {
        "stage": "dual_vae_stage6_valgate",
        "val_rmse": recoverability["dual_vae_ft_val_rmse"],
        "suppressed_rmse": recoverability["dual_vae_ft_suppressed"],
        "coverage": cov,
        "G4.2_pass": bool(g42),
        "calibration_regression": {
            "vs_arm_v_run": "20260706T214941Z__eae10d43__0908bc9",
            "material_pp": MATERIAL_PP, **regression,
            "pass": bool(calibration_ok),
        },
        "G3.4_ft_sanity": {"ft_best_epoch": model.ft_best_epoch,
                           "pass": bool(g34)},
        "ood": {"finetuned": ood_ft, "arm_v_reference": arm_v_ood},
        "per_curve": per_curve,
        "recoverability": recoverability,
    }
    record = write_run_record(
        data_config, run_config,
        dataset_path=td.source_path,
        data_root=data_config.project_root,
        scratch_root=scratch_root,
        summary=summary,
        ledger_metrics={
            "stage": "dual_vae_stage6_valgate",
            "val_rmse": summary["val_rmse"],
            "G4.2_pass": bool(g42),
            "calibration_ok": bool(calibration_ok),
            "ft_best_epoch": model.ft_best_epoch,
        },
        metrics=model.history,
        device=pick_device(),
        experiments_root=args.experiments_root,
    )
    ckpt_dir = Path(scratch_root) / "models" / record.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ckpt_dir / "model.joblib")

    print(json.dumps({
        "record": record.run_id,
        "G4.2_pass": bool(g42),
        "coverage_pooled": {lv: cov[lv]["pooled"] for lv in cov},
        "calibration_regression_pass": bool(calibration_ok),
        "regression_pp": {lv: round(regression[lv]["worsening_pp"], 2)
                          for lv in regression},
        "G3.4_ft_sanity": bool(g34),
        "ft_best_epoch": model.ft_best_epoch,
        "ood_ft_max": ood_ft["max"],
        "val_rmse": summary["val_rmse"],
        "recoverability_refs": {k: v[0] for k, v in RECOVERABILITY_REFS.items()},
    }, indent=2))
    return 0 if (g42 and calibration_ok and g34) else 1


if __name__ == "__main__":
    raise SystemExit(main())
