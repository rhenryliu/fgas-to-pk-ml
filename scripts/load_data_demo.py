"""Demo: load f_gas(R) to SP(k) training data via :mod:`fgas_spk.loader`.

This is a runnable usage example for the configurable training-data loader. It
resolves a saved :class:`~fgas_spk.schema.FgasSpkDataset`, builds model-ready
numpy arrays through :func:`fgas_spk.loader.load_training_data`, prints the
loader's dry-run summary, and then shows how to access the resulting
:class:`~fgas_spk.loader.TrainingData` modalities. It is read-only: no
preprocessing, no training, no side effects.

By default it reads the bundled config at ``scripts/configs/data/config.yaml``;
pass ``--config`` to point at another YAML.

Run from the repository root::

    python scripts/load_data_demo.py
    python scripts/load_data_demo.py --config path/to/other_config.yaml
    python scripts/load_data_demo.py --data-root /path/to/store

Note on the read root: the bundled config leaves ``project_root`` null, which
defers to :func:`fgas_spk.paths.resolve_data_root` (``$FGAS_DATA_ROOT``, else the
repo, where the datasets are git-tracked). This demo fills it at load time, the
same way the runner and the loader CLI do; pass ``--data-root`` to read datasets
from elsewhere, or set ``project_root`` explicitly in the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fgas_spk import loader as L
from fgas_spk.paths import fill_data_root

# Default config shipped alongside this demo.
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "data" / "config.yaml"


def _describe_array(name: str, arr: np.ndarray | None) -> str:
    """Return a one-line ``name: shape dtype`` description, or ``None``.

    Args:
        name (str): Field label to print.
        arr (np.ndarray | None): The array, or None for an absent modality.

    Returns:
        str: A single formatted description line.
    """
    if arr is None:
        return f"  {name:<11}: None  (modality not requested / not present)"
    return f"  {name:<11}: shape {str(arr.shape):<16} dtype {arr.dtype}"


def _show_access_demo(td: L.TrainingData) -> None:
    """Print how to access the :class:`TrainingData` fields after loading.

    Args:
        td (L.TrainingData): The loaded training data.
    """
    print("\n--- TrainingData field access ---")
    print(_describe_array("X", td.X))
    print(_describe_array("X_cond", td.X_cond))
    print(_describe_array("X_params", td.X_params))
    print(_describe_array("y", td.y))
    print(_describe_array("nd", td.nd))
    print(_describe_array("sim_index", td.sim_index))
    print(_describe_array("k", td.k))
    print(_describe_array("radii_mpch", td.radii_mpch))

    if td.X.shape[0] == 0:
        print("\nNo examples in the loaded dataset; nothing to inspect per-row.")
        return

    # Inspect the first (simulation, number-density) row as a concrete example.
    print("\n--- First example (row 0) ---")
    print(f"  originating sim_index : {int(td.sim_index[0])}")
    print(f"  number density nd     : {float(td.nd[0]):.4g} (Mpc/h)^-3")
    print(f"  profile X[0]          : {td.X[0].shape[0]} radial bins")
    if td.X_cond is not None:
        print(f"  conditioning X_cond[0]: {td.X_cond[0].tolist()}")
    if td.X_params is not None:
        print(f"  CAMELS params X_params[0]: {td.X_params[0].shape[0]} values")
    # y is per-curve (2-D) for curve/k_range, scalar (1-D) for single_k.
    if td.y.ndim == 1:
        print(f"  target y[0]           : scalar {float(td.y[0]):.4g}")
    else:
        print(f"  target y[0]           : {td.y[0].shape[0]} k bins")

    n_sims = np.unique(td.sim_index).size
    print(
        f"\nTo split by simulation downstream, group rows by `sim_index` "
        f"({n_sims} unique sims here) -- never by (sim, nd) row, which would "
        f"leak a sim's shared SP(k) target across folds."
    )


def main(argv: list[str] | None = None) -> int:
    """Load a config, build the training arrays, and print a usage demo.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Demo loader for f_gas->SP(k) training data. Loads via "
        "fgas_spk.loader, prints a summary and a field-access example. "
        "Read-only: no training, no side effects."
    )
    parser.add_argument(
        "--config", type=str, default=str(_DEFAULT_CONFIG),
        help=f"Path to a DataConfig YAML file (default: {_DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override the read root; only fills a null project_root in "
        "store-field mode (else $FGAS_DATA_ROOT, else the repo).",
    )
    args = parser.parse_args(argv)

    # Fill a null project_root at this entry point (same helper the runner and
    # the loader CLI use), so the loader receives a fully-specified config.
    config = L.DataConfig.from_yaml(args.config)
    config = fill_data_root(config, explicit=args.data_root)
    td = L.load_training_data(config)

    print(f"config        : {args.config}")
    print(L._summarize(td))
    _show_access_demo(td)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
