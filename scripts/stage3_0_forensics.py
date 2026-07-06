"""Stage 3.0 of the dual-VAE programme: X-matrix forensics.

Maintainer amendment 2 (2026-07-05; `docs/dual_vae_staged_spec.md`) suspends
Stage 3 until the legitimacy of the f_gas matrix itself is established. This
script runs the five diagnostics (3.0.1-3.0.5) and writes one recorded
diagnostics run (record-only label ``"x_forensics"``); the human-readable
verdict lives in ``experiments/notes/dual_vae_stage3_forensics.md``.

* **3.0.1 Value census, both tags** -- per-radial-bin quantiles, non-finite
  and exact-zero counts, pooled distributions, on tags 20260702 and 20260617.
* **3.0.2 Offender census** -- hard-implausibility tier (outside the
  PROVISIONAL range [-0.5, 2.0]; note that the stamped f_gas definition,
  ``Delta Sigma_ionized / Delta Sigma_total / f_b``, is a ratio of excess
  surface densities and is NOT mathematically bounded when the denominator
  crosses zero -- the range is flagged for maintainer confirmation) and a
  robust per-bin z tier (median/MAD, |z| > 10). Spatial pattern and fold
  membership reported.
* **3.0.3 PCA poisoning check** -- explained-variance spectra and leading-
  component score domination across tags, cross-referenced with offenders.
* **3.0.4 Source-denominator check** -- for the worst offender simulations,
  reload the upstream per-halo ``DSigma`` profiles (lindajin scratch), rebuild
  the stack exactly as the builder does (rank by halo mass, slice
  ``int(nd * box^3)`` halos, ratio ``ionized / total / fb`` per halo and
  projection, nanmean over projections then halos), and report the minimum
  ``|Delta Sigma_total|`` cells behind each offending value. This tests the
  division-by-near-zero mechanism in `builder.py` (``fgas = ionized / total
  / fb``, no guard) directly against the data.
* **3.0.5 mlp attribution (diagnostic only, NOT a recorded baseline)** --
  the plain mlp retrained with offender simulations excluded (same folds
  otherwise), to measure how much of the 0.0617 -> 0.0429 improvement
  survives cleaning.

Usage::

    python scripts/stage3_0_forensics.py [--no-figures] [--no-mlp]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from fgas_spk.experiment import RunConfig
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.paths import figure_dir, fill_data_root, resolve_scratch_root
from fgas_spk.run_record import write_run_record
from fgas_spk.train import grouped_split, pick_device

from stage1_pca_dimensionality import PINNED_SPLIT, _figure_label
from stage2_vae_y import DEFAULT_DATA_CONFIG, _rmse

OLD_TAG = "20260617"

# 3.0.2 tiers. The hard range is PROVISIONAL (see module docstring).
HARD_RANGE = (-0.5, 2.0)
ROBUST_Z_THRESHOLD = 10.0
_MAD_TO_SIGMA = 1.4826

# 3.0.4 source-profile layout (the demo_builder / dataset provenance path).
PROFILES_BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROFILE_FILENAME = "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap74.npz"
BOX_SIZE_MPCH = 50.0
_PROJECTIONS = ("xy", "xz", "yz")

QUANTILE_LABELS = ("min", "q0.001", "q0.01", "q0.5", "q0.99", "q0.999", "max")
QUANTILE_POINTS = (0.001, 0.01, 0.5, 0.99, 0.999)


def value_census(x: np.ndarray, radii: np.ndarray) -> dict:
    """3.0.1: per-bin and pooled quantiles plus non-finite / zero counts.

    Args:
        x (np.ndarray): Raw served X, shape (n_rows, n_bins).
        radii (np.ndarray): Bin radii, shape (n_bins,).

    Returns:
        dict: ``per_bin`` rows of (radius, min, q0.001, q0.01, q0.5, q0.99,
            q0.999, max), ``pooled`` quantiles, ``n_nonfinite``, ``n_zero``.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    per_bin = []
    for j, r in enumerate(radii):
        col = x[:, j][finite[:, j]]
        qs = np.quantile(col, QUANTILE_POINTS)
        per_bin.append([float(r), float(col.min()), *[float(q) for q in qs],
                        float(col.max())])
    pooled_vals = x[finite]
    pooled = [float(pooled_vals.min()),
              *[float(q) for q in np.quantile(pooled_vals, QUANTILE_POINTS)],
              float(pooled_vals.max())]
    return {
        "columns": ["radius_mpch", *QUANTILE_LABELS],
        "per_bin": per_bin,
        "pooled": dict(zip(QUANTILE_LABELS, pooled)),
        "n_nonfinite": int((~finite).sum()),
        "n_zero": int((x == 0.0).sum()),
    }


def offender_census(
    x: np.ndarray, sim_index: np.ndarray, masks: dict
) -> dict:
    """3.0.2: two-tier offender flags with spatial pattern and fold membership.

    Args:
        x (np.ndarray): Raw served X, shape (n_rows, n_bins).
        sim_index (np.ndarray): Simulation id per row.
        masks (dict): Pinned-split fold masks over the rows.

    Returns:
        dict: Per-tier cell/sim counts, per-bin cell counts, fold membership
            of offender sims, and the worst sims by max |robust z|.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * _MAD_TO_SIGMA
    mad = np.where(mad < 1e-12, 1.0, mad)
    z = (x - med) / mad

    hard = (x < HARD_RANGE[0]) | (x > HARD_RANGE[1]) | ~np.isfinite(x)
    dist = np.abs(z) > ROBUST_Z_THRESHOLD

    def _tier(cells: np.ndarray) -> dict:
        sims = np.unique(sim_index[cells.any(axis=1)])
        folds = {
            fold: int(np.isin(sims, np.unique(sim_index[masks[fold]])).sum())
            for fold in ("train", "val", "test")
        }
        return {
            "n_cells": int(cells.sum()),
            "n_sims_any": int(sims.size),
            "cells_per_bin": [int(c) for c in cells.sum(axis=0)],
            "offender_sim_folds": folds,
        }

    zmax_per_sim = np.abs(z).max(axis=1)
    worst = np.argsort(zmax_per_sim)[::-1][:15]
    return {
        "hard_range": list(HARD_RANGE),
        "robust_z_threshold": ROBUST_Z_THRESHOLD,
        "hard": _tier(hard),
        "robust_z": _tier(dist),
        "worst_sims_by_z": [
            {"sim_index": int(sim_index[i]), "max_abs_z": float(zmax_per_sim[i]),
             "max_value": float(x[i].max()), "min_value": float(x[i].min())}
            for i in worst
        ],
        "_hard_rows": hard.any(axis=1),   # np arrays for downstream use
        "_dist_rows": dist.any(axis=1),
    }


def pca_poisoning(x_tr: np.ndarray, offender_rows_tr: np.ndarray) -> dict:
    """3.0.3: EVR spectrum and leading-component score domination (train fold).

    Args:
        x_tr (np.ndarray): Train-fold X, shape (n_tr, n_bins).
        offender_rows_tr (np.ndarray): Boolean offender mask over train rows.

    Returns:
        dict: EVR spectrum, and per leading component: the top-5 |score| rows'
            share of that component's score variance and how many of them are
            offender rows.
    """
    from sklearn.decomposition import PCA

    n_comp = min(10, x_tr.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full").fit(x_tr)
    scores = pca.transform(x_tr)
    comps = []
    for j in range(min(4, n_comp)):
        s = scores[:, j]
        order = np.argsort(np.abs(s))[::-1][:5]
        share = float(np.sum(s[order] ** 2) / np.sum(s ** 2))
        comps.append({
            "component": j,
            "evr": float(pca.explained_variance_ratio_[j]),
            "top5_score_variance_share": share,
            "top5_offender_overlap": int(offender_rows_tr[order].sum()),
        })
    return {
        "evr": [float(v) for v in pca.explained_variance_ratio_],
        "leading_components": comps,
    }


def denominator_check(sim_ids: list[int], nd_index: int = 2) -> list[dict]:
    """3.0.4: rebuild offender stacks from source and expose the denominators.

    Replicates the builder's stacking (rank by ``halo_masses`` descending,
    slice ``int(nd * box^3)`` halos, ``ionized / total / fb`` per halo and
    projection, nanmean over projections then halos) and reports, per
    simulation, the stacked profile's extreme value and the smallest
    ``|Delta Sigma_total|`` cells at that radial bin.

    Args:
        sim_ids (list[int]): Simulations to audit.
        nd_index (int): Number-density index (pinned config uses 2).

    Returns:
        list[dict]: One audit entry per readable simulation.
    """
    nd_values = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3)
    n_use_target = int(nd_values[nd_index] * BOX_SIZE_MPCH ** 3)
    out = []
    for sim in sim_ids:
        path = Path(PROFILES_BASE.format(sim)) / PROFILE_FILENAME
        if not path.exists():
            out.append({"sim_index": int(sim), "error": f"missing {path}"})
            continue
        d = np.load(path)
        masses = np.asarray(d["halo_masses"]).ravel()
        order = np.argsort(masses)[::-1]
        ionized = np.stack(
            [d[f"prof2_ionized_gas_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
        )[:, :, order]
        total = np.stack(
            [d[f"prof1_total_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
        )[:, :, order]
        fb = float(np.asarray(d["fb"]).ravel()[0])
        n_use = min(n_use_target, ionized.shape[2])
        ionized, total = ionized[:, :, :n_use], total[:, :, :n_use]
        ratio = ionized / total / fb                      # (n_r, n_proj, n_use)
        stacked = np.nanmean(np.nanmean(ratio, axis=1), axis=1)  # (n_r,)
        j = int(np.argmax(np.abs(stacked)))
        tot_j = total[j]                                  # (n_proj, n_use)
        flat = np.argsort(np.abs(tot_j).ravel())[:3]
        small = [
            {
                "proj": _PROJECTIONS[int(f // n_use)],
                "halo_rank": int(f % n_use),
                "total_dsigma": float(tot_j.ravel()[f]),
                "ratio_cell": float(ratio[j].ravel()[f]),
            }
            for f in flat
        ]
        med_abs_total = float(np.median(np.abs(tot_j)))
        out.append({
            "sim_index": int(sim),
            "n_use": int(n_use),
            "worst_bin_index": j,
            "stacked_value_rebuilt": float(stacked[j]),
            "median_abs_total_at_bin": med_abs_total,
            "smallest_abs_total_cells": small,
        })
    return out


def _write_census_figure(census_new: dict, census_old: dict, label: str):
    """Per-bin quantile fans for both tags (log-|value|, symmetric)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, census, name in ((axes[0], census_new, "tag 20260702"),
                             (axes[1], census_old, f"tag {OLD_TAG}")):
        tbl = np.asarray(census["per_bin"])
        r = tbl[:, 0]
        for idx, lab, style in ((1, "min", ":"), (2, "q0.001", "--"),
                                (4, "median", "-"), (6, "q0.999", "--"),
                                (7, "max", ":")):
            ax.plot(r, tbl[:, idx], linestyle=style, marker=".", label=lab)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("R [Mpc/h]")
        ax.set_title(name)
        ax.axhline(0, color="k", linewidth=0.5)
    axes[0].set_ylabel("f_gas value (symlog)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label}\n3.0.1 value census")
    out = figure_dir() / f"{label}__census.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 3.0 forensics and write the recorded diagnostics run.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(description="Stage 3.0 X-matrix forensics.")
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-denominator-sims", type=int, default=8,
                        help="Worst offender sims audited against source files.")
    parser.add_argument("--no-mlp", action="store_true",
                        help="Skip the 3.0.5 mlp attribution fits.")
    parser.add_argument("--experiments-root", default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    data_config = fill_data_root(DataConfig.from_yaml(args.config_data))
    scratch_root = resolve_scratch_root(explicit=args.scratch_root)

    td = load_training_data(data_config)
    td_old = load_training_data(dataclasses.replace(data_config, tag=OLD_TAG))
    if not np.array_equal(td.sim_index, td_old.sim_index):
        raise RuntimeError("tags serve different simulation sets")
    masks = grouped_split(td.sim_index, PINNED_SPLIT)

    x_new = np.asarray(td.X, dtype=float)
    x_old = np.asarray(td_old.X, dtype=float)

    # 3.0.1 census, both tags.
    census_new = value_census(x_new, td.radii_mpch)
    census_old = value_census(x_old, td_old.radii_mpch)

    # 3.0.2 offenders (primary: the pinned, served X on the new tag; the old
    # tag census is repeated for the cross-tag character comparison).
    off_new = offender_census(x_new, td.sim_index, masks)
    off_old = offender_census(x_old, td_old.sim_index, masks)
    hard_rows = off_new.pop("_hard_rows")
    dist_rows = off_new.pop("_dist_rows")
    off_old.pop("_hard_rows"), off_old.pop("_dist_rows")

    # 3.0.3 PCA poisoning (train fold, both tags; offender mask from 3.0.2).
    pca_new = pca_poisoning(x_new[masks["train"]], hard_rows[masks["train"]])
    pca_old = pca_poisoning(x_old[masks["train"]], hard_rows[masks["train"]])

    # 3.0.4 denominator audit on the worst offenders by |z|.
    worst_ids = [w["sim_index"] for w in off_new["worst_sims_by_z"][: args.n_denominator_sims]]
    denominators = denominator_check(worst_ids)

    # 3.0.5 mlp attribution (diagnostic only).
    mlp_attribution = None
    if not args.no_mlp:
        from fgas_spk.experiment import RunConfig as _RC
        from fgas_spk.models import REGISTRY
        from fgas_spk.train import _subset

        mlp_cfg = _RC.from_yaml("scripts/configs/run/mlp.yaml")
        mlp_attribution = {}
        for tier_name, rows in (("hard", hard_rows), ("robust_z", dist_rows)):
            keep = ~rows
            model = REGISTRY["mlp"](seed=mlp_cfg.seed, **mlp_cfg.model_params)
            model.fit(_subset(td, masks["train"] & keep))
            sub = _subset(td, masks["val"] & keep)
            rmse = _rmse(model.predict(sub.X, sub.X_cond, sub.X_params), sub.y)
            mlp_attribution[tier_name] = {
                "val_rmse_excluding_offenders": rmse,
                "n_train_kept": int((masks["train"] & keep).sum()),
                "n_val_kept": int((masks["val"] & keep).sum()),
            }
            print(f"[3.0.5 mlp, {tier_name} excluded] val RMSE {rmse:.5g} "
                  f"(all-rows reference 0.042921)")

    summary = {
        "stage": "dual_vae_stage3_0",
        "fgas_definition_stamped": (
            "Delta Sigma_ionized / Delta Sigma_total / f_b (dataset sidecar "
            "units block) -- a ratio of excess surface densities, "
            "mathematically unbounded where Delta Sigma_total ~ 0; the "
            "[-0.5, 2.0] hard range is therefore NOT a physical bound and "
            "awaits maintainer confirmation (3.0.2)."
        ),
        "census_tag20260702": census_new,
        f"census_tag{OLD_TAG}": census_old,
        "offenders_tag20260702": off_new,
        f"offenders_tag{OLD_TAG}": off_old,
        "pca_tag20260702": pca_new,
        f"pca_tag{OLD_TAG}": pca_old,
        "denominator_audit": denominators,
        "mlp_attribution_diagnostic_only": mlp_attribution,
    }
    run_config = RunConfig(model="x_forensics", model_params={}, seed=args.seed,
                           split=PINNED_SPLIT)
    record = write_run_record(
        data_config, run_config,
        dataset_path=td.source_path,
        data_root=data_config.project_root,
        scratch_root=scratch_root,
        summary=summary,
        ledger_metrics={
            "stage": "dual_vae_stage3_0",
            "hard_cells": off_new["hard"]["n_cells"],
            "hard_sims": off_new["hard"]["n_sims_any"],
            "robust_z_cells": off_new["robust_z"]["n_cells"],
        },
        device=pick_device(),
        experiments_root=args.experiments_root,
    )
    if not args.no_figures:
        _write_census_figure(
            census_new, census_old, _figure_label(record.run_id, "x_forensics")
        )
    print(f"run_id: {record.run_id}")
    print(json.dumps({
        "hard_cells": off_new["hard"]["n_cells"],
        "hard_sims": off_new["hard"]["n_sims_any"],
        "robust_z_cells": off_new["robust_z"]["n_cells"],
        "robust_z_sims": off_new["robust_z"]["n_sims_any"],
        "old_tag_hard_cells": off_old["hard"]["n_cells"],
        "old_tag_hard_sims": off_old["hard"]["n_sims_any"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
