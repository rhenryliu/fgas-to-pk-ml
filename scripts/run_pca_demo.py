"""Worked example: run the PCA reference model end to end.

Runs :mod:`fgas_spk.models.pca_linear` through the runner against the bundled
configs (``scripts/configs/data/config.yaml`` + ``scripts/configs/run/
pca_reference.yaml``) and prints the run id, the recorded run-dir path, and the
held-out RMSE. This is the "how do I run a model" reference.

The data config resolves its read root from ``project_root`` (null defaults to
the repo, where the datasets are git-tracked; override with ``--data-root``). The
write root comes from ``--scratch-root`` or ``$FGAS_SCRATCH_ROOT``.

Usage::

    python scripts/run_pca_demo.py [--data-root PATH] [--scratch-root PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fgas_spk.experiment import load_configs
from fgas_spk.train import run_training

_HERE = Path(__file__).resolve().parent
_DATA_CONFIG = _HERE / "configs" / "data" / "config.yaml"
_RUN_CONFIG = _HERE / "configs" / "run" / "pca_reference.yaml"


def main(argv: list[str] | None = None) -> int:
    """Run the PCA reference and print run id, run dir, and RMSE.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Run the PCA reference model end to end (worked example)."
    )
    parser.add_argument("--data-root", default=None,
                        help="Override the read root (fills a null project_root).")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root for run outputs.")
    args = parser.parse_args(argv)

    data_config, run_config = load_configs(_DATA_CONFIG, _RUN_CONFIG)
    result = run_training(
        data_config, run_config,
        data_root_override=args.data_root,
        scratch_root_override=args.scratch_root,
    )

    held = result.summary["held_out_split"]
    n_held = result.summary[f"n_{held}"]
    print(f"run_id    : {result.run_id}")
    print(f"run_dir   : {result.run_dir}")
    print(f"held-out  : {held}  (n={n_held})")
    print(f"RMSE      : {result.summary['rmse']:.6g}")
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
