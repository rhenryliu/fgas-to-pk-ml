"""Amendment 4, F3.3: census of the pinned, cropped X slice before training.

Loads the pinned DataConfig (tag 20260706, nd index 2, R < 10 crop) and
reports, per retained radial bin, the min / max / quantiles of the served
f_gas values against the provisional sanity range [-1.0, 3.0]. Any retained
cell outside the range is a **STOP** (exit 1): the maintainer sets the final
crop boundary from where the violations sit; neither the range nor the crop
moves autonomously.

Usage::

    python scripts/f33_census.py [--config-data scripts/configs/data/config_dual_vae.yaml]
"""

from __future__ import annotations

import argparse

import numpy as np

from fgas_spk.builder import DEFAULT_FGAS_SANITY_RANGE
from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.paths import fill_data_root

DEFAULT_DATA_CONFIG = "scripts/configs/data/config_dual_vae.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run the census; exit 1 on any retained-cell violation (F3.3 STOP).

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: 0 when every retained cell lies inside the sanity range; 1
            otherwise.
    """
    parser = argparse.ArgumentParser(description="F3.3 cropped-slice census.")
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG)
    args = parser.parse_args(argv)

    config = fill_data_root(DataConfig.from_yaml(args.config_data))
    td = load_training_data(config)
    x = np.asarray(td.X, dtype=float)
    lo, hi = DEFAULT_FGAS_SANITY_RANGE

    print(f"tag {config.tag}, nd indices {config.number_density_indices}, "
          f"crop {config.radial_range_mpch} -> {x.shape[1]} retained bins")
    print(f"{'R [Mpc/h]':>10} | {'min':>9} | {'q0.001':>9} | {'q0.5':>7} | "
          f"{'q0.999':>9} | {'max':>9} | viol")
    n_viol = 0
    for j, r in enumerate(td.radii_mpch):
        col = x[:, j]
        qs = np.quantile(col, (0.001, 0.5, 0.999))
        v = int(((col < lo) | (col > hi)).sum())
        n_viol += v
        print(f"{r:10.3f} | {col.min():9.4f} | {qs[0]:9.4f} | {qs[1]:7.4f} | "
              f"{qs[2]:9.4f} | {col.max():9.4f} | {v}")
    print(f"retained-slice span: [{x.min():.4f}, {x.max():.4f}] vs sanity "
          f"[{lo}, {hi}]; violations: {n_viol}")
    if n_viol:
        print("F3.3 STOP: retained cells violate the sanity range. Do not "
              "widen the range or move the crop; report to the maintainer.")
        return 1
    print("F3.3 census: PASS (all retained cells inside the sanity range)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
