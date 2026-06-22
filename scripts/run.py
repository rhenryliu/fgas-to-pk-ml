"""Thin CLI: run a model from a (data config, run config) pair.

Loads the two YAML configs independently (no merge), runs the training runner,
and prints the run id and the headline held-out metric. Large artifacts go to
scratch; the small text record goes under ``experiments/runs/<run_id>/``.

Usage::

    python scripts/run.py --config-data scripts/configs/data/config.yaml \\
                          --config-run  scripts/configs/run/pca_reference.yaml \\
                          [--data-root PATH] [--scratch-root PATH]

``--data-root`` fills the read root only when the data config leaves
``project_root`` null; ``--scratch-root`` overrides the write root (else
``RunConfig.write_root`` / ``$FGAS_SCRATCH_ROOT``).
"""

from __future__ import annotations

import argparse

from fgas_spk.experiment import load_configs
from fgas_spk.train import run_training


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run training, and print the run id + headline metric.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Run a model from a (data config, run config) pair and "
        "record the run."
    )
    parser.add_argument("--config-data", required=True,
                        help="Path to the DataConfig YAML.")
    parser.add_argument("--config-run", required=True,
                        help="Path to the RunConfig YAML.")
    parser.add_argument("--data-root", default=None,
                        help="Override the read root (only fills a null "
                        "project_root in store-field mode).")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the write root for run outputs.")
    args = parser.parse_args(argv)

    data_config, run_config = load_configs(args.config_data, args.config_run)
    result = run_training(
        data_config, run_config,
        data_root_override=args.data_root,
        scratch_root_override=args.scratch_root,
    )

    print(f"run_id    : {result.run_id}")
    print(f"run_dir   : {result.run_dir}")
    print(f"held-out  : {result.summary['held_out_split']}")
    print(f"RMSE      : {result.summary['rmse']:.6g}")
    print(f"checkpoint: {result.checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
