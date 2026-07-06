"""Amendment 6, I1: suppressed-regime breakdown of the Stage 4.0 probe arms.

The Stage 4.0 record stores each arm's RMSEs but not its predictions, so this
script regenerates them by **deterministic refit** — the identical fixed
probe protocol (same recipe, same seeds, same folds) — and verifies each
arm's recomputed global mean against the recorded value before reporting
(re-gating in effect; no new training choices). It then reports, for the
ceiling, ``pca_scores_d4``, and selected-VAE arms, the val y-space RMSE
restricted to bins with TRUE SP(k) < t for t in {0.95, 0.9, 0.8}
(ground-truth-masked), with per-threshold bin counts.

Usage::

    python scripts/stage4_0_i1_breakdown.py \\
        --vae-y-run-id 20260706T190601Z__8a715039__8522dd4 \\
        --vae-x-run-id 20260706T200039Z__3080064e__8cbe5b2 \\
        --record 20260706T205815Z__e90f52b8__a6dab80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.models.dual_vae_components import LatentMapMLP
from fgas_spk.paths import fill_data_root, repo_root, resolve_scratch_root
from fgas_spk.train import grouped_split

from stage1_pca_dimensionality import PINNED_SPLIT
from stage2_vae_y import DEFAULT_DATA_CONFIG, _rmse, _suppressed_recon_rmse
from stage4_0_probe_selection import PROBE_PARAMS, PROBE_SEEDS


def main(argv: list[str] | None = None) -> int:
    """Refit the three arms, verify against the record, print the breakdown.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: 0 on success; 1 if a recomputed mean fails verification.
    """
    parser = argparse.ArgumentParser(description="I1 probe-arm breakdown.")
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--scratch-root", default=None)
    parser.add_argument("--vae-y-run-id", required=True)
    parser.add_argument("--vae-x-run-id", required=True,
                        help="The H3.4-selected VAE candidate (Arm V codec).")
    parser.add_argument("--record", required=True,
                        help="Stage 4.0 run id (verification reference).")
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

    from sklearn.decomposition import PCA

    pca4 = PCA(n_components=4, svd_solver="full", random_state=0).fit(x_tr)
    vae_x = joblib.load(models_root / args.vae_x_run_id / "model.joblib")
    arms = {
        "ceiling_full_profile": (x_tr, x_va),
        "pca_scores_d4": (pca4.transform(x_tr), pca4.transform(x_va)),
        "selected_vae_ld3_b0.01": (
            vae_x.encode(x_tr, c_tr), vae_x.encode(x_va, c_va)
        ),
    }
    recorded = json.loads(
        (repo_root() / "experiments" / "runs" / args.record / "summary.json")
        .read_text()
    )["arms"]
    ref_key = {"selected_vae_ld3_b0.01": "vae_ld3_b0.01"}

    results = {}
    ok = True
    for name, (f_tr, f_va) in arms.items():
        # Mean prediction over the protocol seeds; suppressed stats on it.
        preds = []
        for seed in PROBE_SEEDS:
            probe = LatentMapMLP(seed=seed, **PROBE_PARAMS)
            probe.fit(f_tr, mu2_tr)
            preds.append(vae_y.decode(probe.predict(f_va), c_va))
        per_seed = [_rmse(p, y_va) for p in preds]
        mean = float(np.mean(per_seed))
        rec = recorded[ref_key.get(name, name)]["mean"]
        match = np.isclose(mean, rec, rtol=1e-4)
        ok &= bool(match)
        # Per-seed suppressed RMSE, averaged over seeds (same convention as
        # the global metric: mean over seeds of the per-seed statistic).
        sup_per_seed = [_suppressed_recon_rmse(y_va, p) for p in preds]
        sup = {
            t: {
                "rmse": float(np.mean([s[t]["rmse"] for s in sup_per_seed])),
                "n_bins": sup_per_seed[0][t]["n_bins"],
            }
            for t in sup_per_seed[0]
        }
        results[name] = {"global": mean, "recorded": rec,
                         "verified": bool(match), "suppressed": sup}
        print(f"[{name}] global {mean:.5f} (recorded {rec:.5f}, "
              f"match={match}) " + " ".join(
                  f"SP<{t}: {sup[t]['rmse']:.5f} (n={sup[t]['n_bins']})"
                  for t in sup))

    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
