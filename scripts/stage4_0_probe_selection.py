"""Stage 4.0 of the dual-VAE programme: probe-based codec selection (H3).

Amendment 5 retired the linear-ridge G3.1' gate as structurally malformed;
this script is its replacement instrument. A **fixed nonlinear probe** — the
Stage 4 rung-2 MLP recipe, frozen here, no per-arm tuning, three seeds per
arm — is trained from each arm's features to the frozen VAE-Y posterior
means ``mu2`` on the train fold, and scored as the **mean over seeds of the
val y-space RMSE** after decoding the probe outputs through the frozen
decoder-Y (targets fit in z-space, judged in y-space; that objective
mismatch is inherent to the F0.5 posterior-mean design, identical across
arms, and is what Stage 6's fine-tune later addresses).

Arms (H3.2): (a) **ceiling** — the full cropped f_gas profile;
(b) **reference** — PCA-x scores at d in {2, 3, 4, 6} (train-fold fit);
(c) **candidates** — ``mu1`` from every Stage 3 candidate configuration
(the 16 frozen checkpoints listed in the Stage 3 gate report).

Gates:

* **G4.0a (STOP on fail):** the ceiling arm must beat the current Stage 1
  ``pca_linear`` val RMSE — if the unrestricted nonlinear two-stage pathway
  cannot beat the simplest direct baseline, the architecture is unviable on
  this data and no codec choice can rescue it.
* **G4.0b (graded):** ratio = selected-candidate probe RMSE / best
  reference-arm probe RMSE. <= 1.05 clean pass; (1.05, 1.25] proceed with a
  documented caveat (Stage 5b adjudicates with full metrics); > 1.25 STOP.
  Both constants are maintainer-set and amendable.

Selection (H3.4): best candidate probe metric; tiebreak smaller latent
dimension, then better X-reconstruction. Reported findings (H3.6, no gates,
no encoded expectations): ceiling-vs-selected gap, ceiling-vs-
``mlp_regressor`` gap, and the F-1 re-measurement under this nonlinear probe.

Usage::

    python scripts/stage4_0_probe_selection.py \\
        --vae-y-run-id 20260706T190601Z__8a715039__8522dd4 [--no-figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.experiment import RunConfig
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.models.dual_vae_components import LatentMapMLP
from fgas_spk.paths import figure_dir, fill_data_root, repo_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split, pick_device

from stage1_pca_dimensionality import PINNED_SPLIT, _figure_label
from stage2_vae_y import DEFAULT_DATA_CONFIG, _rmse
from stage4_latent_map import MAP_PARAMS, MLP_REGRESSOR_RUN_ID, MLP_REGRESSOR_VAL_RMSE

# The H2 candidate set: every Stage 3 grid configuration (all passed G3.2'
# and G3.4); run_ids as listed in the Stage 3 gate report.
CANDIDATE_RUN_IDS = (
    "20260706T200035Z__923fa7f0__8cbe5b2",
    "20260706T200036Z__cf35add9__8cbe5b2",
    "20260706T200037Z__e72e3fab__8cbe5b2",
    "20260706T200037Z__fc11d46c__8cbe5b2",
    "20260706T200038Z__75e83299__8cbe5b2",
    "20260706T200039Z__a58053de__8cbe5b2",
    "20260706T200039Z__3080064e__8cbe5b2",
    "20260706T200040Z__f4536019__8cbe5b2",
    "20260706T200041Z__d478d3d0__8cbe5b2",
    "20260706T200041Z__0325e0a7__8cbe5b2",
    "20260706T200043Z__fd693065__8cbe5b2",
    "20260706T200043Z__794d86af__8cbe5b2",
    "20260706T200044Z__a43ec084__8cbe5b2",
    "20260706T200045Z__7a4a4ba0__8cbe5b2",
    "20260706T200046Z__50a0f2fd__8cbe5b2",
    "20260706T200047Z__2f14b4a6__8cbe5b2",
)

# Fixed probe protocol (H3.1): the Stage 4 rung-2 recipe, frozen; 3 seeds.
PROBE_PARAMS = dict(MAP_PARAMS)
PROBE_SEEDS = (0, 1, 2)

# G4.0a floor: Stage 1 pca_linear on the pinned (cropped) context.
PCA_LINEAR_VAL_RMSE = 0.038537
PCA_LINEAR_RUN_ID = "20260706T185703Z__a760845e__a4ba5aa"

# G4.0b grading constants (maintainer-set, amendable).
G40B_CLEAN = 1.05
G40B_CAVEAT = 1.25

REFERENCE_DIMS = (2, 3, 4, 6)


def probe_arm(
    feats_tr: np.ndarray,
    feats_va: np.ndarray,
    mu2_tr: np.ndarray,
    vae_y,
    c_va: np.ndarray,
    y_va: np.ndarray,
) -> dict:
    """Run the fixed three-seed probe on one arm.

    Args:
        feats_tr (np.ndarray): Train-fold features, shape (n_tr, d).
        feats_va (np.ndarray): Val-fold features, shape (n_va, d).
        mu2_tr (np.ndarray): Frozen VAE-Y train-fold codes (targets).
        vae_y: The frozen VAE-Y (its ``decode``).
        c_va (np.ndarray): Val-fold context for decoder-Y.
        y_va (np.ndarray): Val-fold true SP(k).

    Returns:
        dict: ``per_seed`` val y-space RMSEs and their ``mean``.
    """
    per_seed = []
    for seed in PROBE_SEEDS:
        probe = LatentMapMLP(seed=seed, **PROBE_PARAMS)
        probe.fit(feats_tr, mu2_tr)
        y_hat = vae_y.decode(probe.predict(feats_va), c_va)
        per_seed.append(_rmse(y_hat, y_va))
    return {"per_seed": per_seed, "mean": float(np.mean(per_seed))}


def _write_arm_figure(arms: dict, label: str):
    """Horizontal bar chart of arm probe metrics (mean +- seed spread)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    names = list(arms)
    means = [arms[n]["mean"] for n in names]
    spread = [np.ptp(arms[n]["per_seed"]) / 2 for n in names]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(names) + 1.5))
    ax.barh([names[i] for i in order], [means[i] for i in order],
            xerr=[spread[i] for i in order], color="C0", alpha=0.8)
    ax.axvline(PCA_LINEAR_VAL_RMSE, color="C3", linestyle="--", linewidth=1,
               label=f"G4.0a floor (pca_linear {PCA_LINEAR_VAL_RMSE})")
    ax.set_xlabel("probe val y-space RMSE (mean over 3 seeds)")
    ax.set_title(f"{label}\nStage 4.0 probe arms")
    ax.legend(fontsize=8)
    out = figure_dir() / f"{label}__probe_arms.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 4.0 probe protocol, gates, and run record.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: 0 on success (incl. a caveated G4.0b); 1 on a STOP (G4.0a fail
            or G4.0b ratio > 1.25).
    """
    parser = argparse.ArgumentParser(
        description="Stage 4.0: fixed-nonlinear-probe codec selection (H3)."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--vae-y-run-id", required=True)
    parser.add_argument("--candidate-run-ids", nargs="+",
                        default=list(CANDIDATE_RUN_IDS))
    parser.add_argument("--seed", type=int, default=0,
                        help="RunConfig bookkeeping seed (probe seeds are the "
                        "fixed protocol set).")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)
    models_root = Path(scratch_root) / "models"

    td = load_training_data(data_config)
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    x_tr = np.asarray(td.X[masks["train"]], dtype=float)
    x_va = np.asarray(td.X[masks["val"]], dtype=float)
    y_tr = np.asarray(td.y[masks["train"]], dtype=float)
    y_va = np.asarray(td.y[masks["val"]], dtype=float)
    c_tr, c_va = td.X_params[masks["train"]], td.X_params[masks["val"]]

    vae_y = joblib.load(models_root / args.vae_y_run_id / "model.joblib")
    mu2_tr = vae_y.encode(y_tr, c_tr)

    arms: dict[str, dict] = {}

    # (a) ceiling: the full cropped profile.
    arms["ceiling_full_profile"] = probe_arm(x_tr, x_va, mu2_tr, vae_y, c_va, y_va)
    print(f"[ceiling] {arms['ceiling_full_profile']['mean']:.5f} "
          f"{arms['ceiling_full_profile']['per_seed']}")

    # G4.0a checked as soon as the ceiling exists (STOP semantics).
    g40a_pass = arms["ceiling_full_profile"]["mean"] < PCA_LINEAR_VAL_RMSE

    # (b) references: PCA-x scores at fixed dims.
    from sklearn.decomposition import PCA

    for d in REFERENCE_DIMS:
        pca = PCA(n_components=d, svd_solver="full", random_state=0).fit(x_tr)
        arms[f"pca_scores_d{d}"] = probe_arm(
            pca.transform(x_tr), pca.transform(x_va), mu2_tr, vae_y, c_va, y_va
        )
        print(f"[pca d={d}] {arms[f'pca_scores_d{d}']['mean']:.5f}")

    # (c) candidates: mu1 from every Stage 3 candidate.
    candidates: dict[str, dict] = {}
    for run_id in args.candidate_run_ids:
        summary = json.loads(
            (repo_root() / "experiments" / "runs" / run_id / "summary.json")
            .read_text()
        )
        vae_x = joblib.load(models_root / run_id / "model.joblib")
        name = f"vae_ld{summary['latent_dim']}_b{summary['beta']}"
        arm = probe_arm(
            vae_x.encode(x_tr, c_tr), vae_x.encode(x_va, c_va),
            mu2_tr, vae_y, c_va, y_va,
        )
        arm.update({
            "run_id": run_id,
            "latent_dim": summary["latent_dim"],
            "beta": summary["beta"],
            "x_recon_rmse": summary["val_recon_rmse"],
        })
        arms[name] = arm
        candidates[name] = arm
        print(f"[{name}] {arm['mean']:.5f}")

    # Selection (H3.4): best mean; tiebreak smaller ld, then better X-recon.
    sel_name = min(
        candidates,
        key=lambda n: (candidates[n]["mean"], candidates[n]["latent_dim"],
                       candidates[n]["x_recon_rmse"]),
    )
    selected = candidates[sel_name]
    best_ref_name = min(
        (n for n in arms if n.startswith("pca_scores_")),
        key=lambda n: arms[n]["mean"],
    )
    ratio = selected["mean"] / arms[best_ref_name]["mean"]
    if ratio <= G40B_CLEAN:
        g40b = "clean_pass"
    elif ratio <= G40B_CAVEAT:
        g40b = "proceed_with_caveat"
    else:
        g40b = "stop"

    findings = {
        "ceiling_vs_selected_gap": {
            "ceiling": arms["ceiling_full_profile"]["mean"],
            "selected": selected["mean"],
            "ratio": selected["mean"] / arms["ceiling_full_profile"]["mean"],
            "meaning": "information lost to x-compression, pre-mapping",
        },
        "ceiling_vs_mlp_regressor_gap": {
            "ceiling": arms["ceiling_full_profile"]["mean"],
            "mlp_regressor": MLP_REGRESSOR_VAL_RMSE,
            "mlp_regressor_run_id": MLP_REGRESSOR_RUN_ID,
            "ratio": arms["ceiling_full_profile"]["mean"] / MLP_REGRESSOR_VAL_RMSE,
            "meaning": "cost of routing through the two-stage architecture",
        },
        "f1_remeasurement_nonlinear_probe": {
            f"d{d}": {
                "best_vae_at_d": min(
                    (c["mean"] for c in candidates.values()
                     if c["latent_dim"] == d), default=None,
                ),
                "pca_ref": arms[f"pca_scores_d{d}"]["mean"],
            }
            for d in REFERENCE_DIMS
        },
    }

    summary_out = {
        "stage": "dual_vae_stage4_0",
        "probe_protocol": {
            "recipe": PROBE_PARAMS, "seeds": list(PROBE_SEEDS),
            "metric": "mean over seeds of val y-space RMSE (decoded via the "
                      "frozen VAE-Y decoder); targets mu2, fit in z-space, "
                      "judged in y-space -- the objective mismatch is inherent "
                      "to F0.5, identical across arms (Stage 6 addresses it).",
        },
        "frozen_vae_y_run_id": args.vae_y_run_id,
        "arms": arms,
        "G4.0a_pass": bool(g40a_pass),
        "G4.0a_floor": {"pca_linear": PCA_LINEAR_VAL_RMSE,
                        "run_id": PCA_LINEAR_RUN_ID},
        "selected": {k: selected[k] for k in
                     ("run_id", "latent_dim", "beta", "mean", "x_recon_rmse")},
        "G4.0b": {"ratio": float(ratio), "grade": g40b,
                  "best_reference_arm": best_ref_name,
                  "constants": {"clean": G40B_CLEAN, "caveat": G40B_CAVEAT}},
        "findings": findings,
    }
    run_config = RunConfig(
        model="probe_codec_selection",
        model_params={"probe": PROBE_PARAMS, "seeds": list(PROBE_SEEDS)},
        seed=args.seed,
        split=PINNED_SPLIT,
    )
    record = write_run_record(
        data_config, run_config,
        dataset_path=td.source_path,
        data_root=data_config.project_root,
        scratch_root=scratch_root,
        summary=summary_out,
        ledger_metrics={
            "stage": "dual_vae_stage4_0",
            "G4.0a_pass": bool(g40a_pass),
            "G4.0b_grade": g40b,
            "selected_run_id": selected["run_id"],
            "selected_probe_rmse": selected["mean"],
        },
        device=pick_device(),
        experiments_root=args.experiments_root,
    )
    if not args.no_figures:
        _write_arm_figure(arms, _figure_label(record.run_id, "probe_selection"))

    print(json.dumps({
        "record": record.run_id,
        "G4.0a_pass": bool(g40a_pass),
        "ceiling": arms["ceiling_full_profile"]["mean"],
        "selected": summary_out["selected"],
        "G4.0b": summary_out["G4.0b"],
        "findings_ceiling_vs_selected": findings["ceiling_vs_selected_gap"]["ratio"],
    }, indent=2))
    if not g40a_pass:
        print("G4.0a STOP: the unrestricted two-stage pathway cannot beat "
              "pca_linear; the architecture is unviable on this data.")
        return 1
    if g40b == "stop":
        print("G4.0b STOP: selected/reference ratio > 1.25 (structural "
              "tripwire); maintainer decision required.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
