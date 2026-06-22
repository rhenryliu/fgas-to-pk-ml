"""Assemble f_gas(R) to SP(k) training data from CAMELS stacking products.

This package learns the mapping from projected gas-fraction profiles
``f_gas(R)`` to matter power-spectrum suppression ``SP(k) = P_total(k) / P_DM(k)``
for the CAMELS IllustrisTNG SB35 suite. It is organized as three submodules:

- :mod:`fgas_spk.schema` -- the on-disk data contract and single source of
  truth (the :class:`FgasSpkDataset` container, the filename/directory
  convention, the save/load round-trip, and ``SCHEMA_VERSION``);
- :mod:`fgas_spk.builder` -- assemble a dataset from CAMELS profile products,
  correcting the upstream halo-ordering bug;
- :mod:`fgas_spk.loader` -- serve model-ready numpy arrays from a saved dataset
  per a :class:`DataConfig`;
- :mod:`fgas_spk.experiment` -- the trainer-side :class:`RunConfig` (model and
  training concerns only) and :func:`load_configs`.

The most commonly used names are re-exported here as a flat API, so callers can
write ``import fgas_spk as F; F.save_dataset(...)`` without reaching into the
submodules. The submodules remain importable directly for the full surface.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from fgas_spk.builder import build_fgas_spk_dataset, build_fgas_spk_from_compiled
from fgas_spk.experiment import RunConfig, SplitSpec, load_configs
from fgas_spk.loader import DataConfig, TrainingData, load_training_data
from fgas_spk.schema import (
    SCHEMA_VERSION,
    FgasSpkDataset,
    dataset_dir,
    ensure_store_dirs,
    load_dataset,
    resolve_dataset_path,
    save_dataset,
)

try:
    __version__ = _pkg_version("fgas-to-pk-ml")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # schema -- the on-disk contract
    "SCHEMA_VERSION",
    "FgasSpkDataset",
    "dataset_dir",
    "ensure_store_dirs",
    "load_dataset",
    "resolve_dataset_path",
    "save_dataset",
    # builder
    "build_fgas_spk_dataset",
    "build_fgas_spk_from_compiled",
    # loader
    "DataConfig",
    "TrainingData",
    "load_training_data",
    # experiment -- trainer-side config
    "RunConfig",
    "SplitSpec",
    "load_configs",
]
