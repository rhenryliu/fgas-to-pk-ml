"""Configurable training-data loader for f_gas(R) to SP(k) models.

This module turns a saved :class:`~fgas_spk_schema.FgasSpkDataset` (written by
:mod:`fgas_spk_builder` via :mod:`fgas_spk_schema`) into model-ready numpy
arrays, according to a :class:`DataConfig`. It is importable by future training
scripts and runnable as a CLI dry-run.

It does exactly one thing: read a saved dataset and serve arrays. It performs
**no preprocessing, no normalisation, and no splitting** -- those belong to the
training script. It is numpy-only and never imports a deep-learning framework.

Each training example is one ``(simulation, number-density)`` pair: the input is
that stack's gas-fraction profile (optionally cropped in radius and augmented
with conditioning features), and the target is the simulation's SP(k) -- the
full curve, a single nearest-k value, or an inclusive k-window, per the config.

Splitting is deliberately out of scope, but :attr:`TrainingData.sim_index`
exposes the originating simulation id for every row so a downstream script can
split *by simulation*. Splitting by ``(sim, nd)`` row would leak a simulation's
shared SP(k) target across train/test folds; grouping by ``sim_index`` avoids
that.

Example config (YAML)::

    # Option A -- resolve from store fields (never type the path beyond root):
    project_root: /pscratch/sd/r/rhliu/projects/fgas-to-pk-ml
    suite: CAMELS-IllustrisTNG-L50n512-SB35
    snapshot: 74
    redshift: 0.47
    source: lindajin
    rank: m500            # on-disk rank token (m500 / mstar / frozen)
    tag: latest           # or an explicit tag like 20260616 / v1 (preferred)

    # Option B -- instead of the seven fields above, give an explicit file:
    # path: /pscratch/.../fgas_spk__...__v1.npz

    # Simulation subset (by simulation id; null = all):
    sim_ids: null

    # Inputs / features:
    number_density_indices: [0, 1, 2, 3, 4]   # null = all five
    radial_range_mpch: [0.1, 2.0]             # null = full radial range
    include_nd_feature: true                  # append the number density
    include_mean_halo_mass: false             # append mean M_500c
    include_camels_params: true              # append CAMELS params (on by default)

    # Target:
    target_mode: curve        # one of: curve | single_k | k_range
    k_target: null            # required for single_k (nearest-k bin)
    k_range: null             # required for k_range, e.g. [0.5, 5.0] (inclusive)

CLI::

    python src/fgas_spk_loader.py --config config.yaml

prints a summary (resolved path, ``X``/``y`` shapes, number-density values,
target mode, n_sims) and exits. It loads but does not train and has no side
effects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from fgas_spk_schema import FgasSpkDataset, load_dataset, resolve_dataset_path

_TARGET_MODES = ("curve", "single_k", "k_range")


@dataclass
class DataConfig:
    """Declarative recipe for building a training array set from a saved dataset.

    Exactly one selection mode must be given: either an explicit ``path`` to a
    saved ``.npz``, or the seven store fields (``project_root``, ``suite``,
    ``snapshot``, ``redshift``, ``source``, ``rank``, ``tag``) that are resolved
    via :func:`~fgas_spk_schema.resolve_dataset_path`. The store-field route
    means a caller never types the path beyond ``project_root``; ``tag`` may be
    ``"latest"`` to auto-select the newest (see ``resolve_dataset_path``).

    Attributes:
        path (str, optional): Explicit ``.npz`` path. Mutually exclusive with the
            store fields.
        project_root (str, optional): Store root for field-based resolution.
        suite (str, optional): Simulation suite tag.
        snapshot (int, optional): Snapshot number (e.g. 74 -> z=0.47).
        redshift (float | str, optional): Redshift.
        source (str, optional): Provenance tag for the underlying profiles.
        rank (str, optional): On-disk rank token (``m500`` / ``mstar`` /
            ``frozen``).
        tag (str, optional): Version tag, or ``"latest"`` to auto-select.
        sim_ids (list[int], optional): Subset of simulation ids to keep, matched
            against the saved ``sim_ids``. None keeps all, in stored order.
        number_density_indices (list[int], optional): Subset of number-density
            indices to keep (into the saved ``number_densities``). None keeps all.
        radial_range_mpch (tuple[float, float], optional): Inclusive
            ``(r_min, r_max)`` crop of the profile radii in comoving Mpc/h. None
            keeps the full radial range.
        include_nd_feature (bool): Append the number density as a conditioning
            feature column. Defaults to True.
        include_mean_halo_mass (bool): Append the stack's mean M_500c [M_sun/h]
            as a feature column. Defaults to False.
        include_camels_params (bool): Append the simulation's CAMELS parameters
            as feature columns. Defaults to False, to preserve the
            model-independent framing; requires the dataset to carry params.
        target_mode (str): One of ``"curve"`` (full SP(k) curve), ``"single_k"``
            (scalar at the nearest k bin), or ``"k_range"`` (SP(k) over an
            inclusive k-window). Defaults to ``"curve"``.
        k_target (float, optional): Required for ``single_k``; the nearest k bin
            to this value (h/Mpc) is used.
        k_range (tuple[float, float], optional): Required for ``k_range``; the
            inclusive ``(k_min, k_max)`` window in h/Mpc.
    """

    # --- selection: exactly one of `path` or the store fields below ---
    path: str | None = None
    project_root: str | None = None
    suite: str | None = None
    snapshot: int | None = None
    redshift: float | str | None = None
    source: str | None = None
    rank: str | None = None
    tag: str | None = None

    # --- simulation subset ---
    sim_ids: list[int] | None = None

    # --- inputs / features ---
    number_density_indices: list[int] | None = None
    radial_range_mpch: tuple[float, float] | None = None
    include_nd_feature: bool = True
    include_mean_halo_mass: bool = False
    include_camels_params: bool = True

    # --- target ---
    target_mode: str = "curve"
    k_target: float | None = None
    k_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        # YAML round-trips tuples as lists; restore them so downstream code can
        # rely on the declared types. Validation of presence/consistency is done
        # at load time (see `load_training_data`) so deserialization never fails
        # spuriously on a partially-filled config.
        if self.radial_range_mpch is not None:
            self.radial_range_mpch = tuple(self.radial_range_mpch)  # type: ignore[assignment]
        if self.k_range is not None:
            self.k_range = tuple(self.k_range)  # type: ignore[assignment]
        if self.target_mode not in _TARGET_MODES:
            raise ValueError(
                f"target_mode must be one of {_TARGET_MODES}, got "
                f"{self.target_mode!r}."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataConfig":
        """Construct a config from a YAML file.

        Args:
            path (str | Path): Path to a YAML mapping of config fields.

        Returns:
            DataConfig: The parsed configuration.

        Raises:
            TypeError: If the YAML top level is not a mapping.
        """
        data = yaml.safe_load(Path(path).read_text())
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Config YAML must be a mapping of fields, got {type(data).__name__}."
            )
        return cls(**data)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the config to a YAML file.

        Args:
            path (str | Path): Destination path.

        Returns:
            Path: The written path.
        """
        out = Path(path)
        out.write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        return out

    def resolve_path(self) -> Path:
        """Resolve the saved-dataset ``.npz`` path this config points at.

        Returns:
            Path: The dataset file path (existence is only guaranteed when
                ``tag == "latest"``, which globs for an extant file).

        Raises:
            ValueError: If neither or both selection modes are provided, or if a
                store field is missing in field-resolution mode.
        """
        if self.path is not None:
            store_fields = (self.project_root, self.suite, self.snapshot,
                            self.redshift, self.source, self.rank, self.tag)
            if any(f is not None for f in store_fields):
                raise ValueError(
                    "Provide either `path` or the store fields "
                    "(project_root, suite, snapshot, redshift, source, rank, "
                    "tag), not both."
                )
            return Path(self.path)

        required = ["project_root", "suite", "snapshot", "redshift",
                    "source", "rank", "tag"]
        missing = [name for name in required if getattr(self, name) is None]
        if missing:
            raise ValueError(
                "Provide `path`, or all store fields "
                f"{required}; missing: {missing}."
            )
        return resolve_dataset_path(
            self.project_root, self.suite, self.snapshot, self.redshift, # type: ignore
            self.source, self.rank, self.tag, # type: ignore
        )


@dataclass
class TrainingData:
    """Model-ready numpy arrays produced from a :class:`DataConfig`.

    Attributes:
        X (np.ndarray): Inputs, shape (n_examples, n_features). The leading
            columns are the (optionally cropped) gas-fraction profile, followed
            by any appended features in this order: number density (if
            ``include_nd_feature``), mean halo mass (if
            ``include_mean_halo_mass``), CAMELS params (if
            ``include_camels_params``).
        y (np.ndarray): Target. Shape (n_examples, n_k_sel) for ``curve`` and
            ``k_range``; shape (n_examples,) for ``single_k``.
        nd (np.ndarray): Number-density value for each row, shape (n_examples,),
            in (Mpc/h)^-3.
        sim_index (np.ndarray): Originating simulation id for each row, shape
            (n_examples,). Use this to split by simulation downstream.
        k (np.ndarray): Selected wavenumbers, shape (n_k_sel,), in h/Mpc. For
            ``single_k`` this holds the single nearest bin (length 1).
        radii_mpch (np.ndarray): Selected profile radii, shape (n_radii_sel,),
            in comoving Mpc/h. These correspond to the leading profile columns
            of ``X``.
        source_path (str): The resolved ``.npz`` the data was read from.
        meta (dict): The dataset's embedded ``__meta__`` provenance.
        config (DataConfig): The config used to build this object.
    """

    X: np.ndarray
    y: np.ndarray
    nd: np.ndarray
    sim_index: np.ndarray
    k: np.ndarray
    radii_mpch: np.ndarray
    source_path: str
    meta: dict = field(default_factory=dict)
    config: DataConfig | None = None


def _read_meta(npz_path: Path) -> dict:
    """Return the embedded ``__meta__`` provenance dict from a saved ``.npz``."""
    data = np.load(npz_path, allow_pickle=False)
    if "__meta__" in data:
        return json.loads(str(data["__meta__"]))
    return {}


def _select_sim_rows(dataset: FgasSpkDataset, sim_ids: list[int] | None) -> np.ndarray:
    """Map requested simulation ids to row positions in the dataset arrays."""
    stored = [int(s) for s in dataset.sim_ids]
    if sim_ids is None:
        return np.arange(len(stored), dtype=int)
    position = {sid: i for i, sid in enumerate(stored)}
    missing = [s for s in sim_ids if int(s) not in position]
    if missing:
        raise ValueError(
            f"Requested sim_ids {missing} are not in the dataset "
            f"(available: {stored})."
        )
    return np.array([position[int(s)] for s in sim_ids], dtype=int)


def _select_nd_indices(dataset: FgasSpkDataset, nd_indices: list[int] | None) -> np.ndarray:
    """Validate and return the number-density column indices to keep."""
    n_nd = dataset.fgas.shape[1]
    if nd_indices is None:
        return np.arange(n_nd, dtype=int)
    bad = [j for j in nd_indices if not (0 <= int(j) < n_nd)]
    if bad:
        raise ValueError(
            f"number_density_indices {bad} out of range for n_nd={n_nd}."
        )
    if not nd_indices:
        raise ValueError("number_density_indices is empty; select at least one.")
    return np.array([int(j) for j in nd_indices], dtype=int)


def _radial_mask(dataset: FgasSpkDataset, rng: tuple[float, float] | None) -> np.ndarray:
    """Return a boolean mask selecting profile radii in the inclusive range."""
    radii = np.asarray(dataset.radii_mpch)
    if rng is None:
        return np.ones(radii.shape[0], dtype=bool)
    r_min, r_max = float(rng[0]), float(rng[1])
    mask = (radii >= r_min) & (radii <= r_max)
    if not mask.any():
        raise ValueError(
            f"radial_range_mpch {rng} selects no radii "
            f"(available {radii.min():.4g}..{radii.max():.4g} Mpc/h)."
        )
    return mask


def load_training_data(config: DataConfig) -> TrainingData:
    """Build model-ready arrays from a saved dataset per ``config``.

    Resolves the dataset path, loads it, applies the simulation subset, the
    number-density-index subset, and the radial crop, then flattens the grid
    into one row per ``(simulation, number-density)`` pair (simulation-major,
    number-density-minor). Appends the requested conditioning features and shapes
    the target according to ``target_mode``.

    Args:
        config (DataConfig): The selection / feature / target recipe.

    Returns:
        TrainingData: The assembled arrays, with per-row ``nd`` and
            ``sim_index`` aligned to ``X`` and ``y``.

    Raises:
        ValueError: On an inconsistent config (selection, missing target
            parameters, out-of-range subsets) or a missing requested feature.
    """
    npz_path = config.resolve_path()

    # Validate target parameters before doing any heavy work.
    if config.target_mode == "single_k" and config.k_target is None:
        raise ValueError("target_mode='single_k' requires `k_target`.")
    if config.target_mode == "k_range" and config.k_range is None:
        raise ValueError("target_mode='k_range' requires `k_range`.")

    dataset = load_dataset(npz_path)
    meta = _read_meta(npz_path)

    rows = _select_sim_rows(dataset, config.sim_ids)          # (S,)
    nd_idx = _select_nd_indices(dataset, config.number_density_indices)  # (M,)
    rmask = _radial_mask(dataset, config.radial_range_mpch)   # (n_radii,)
    r_pos = np.where(rmask)[0]

    n_sims_sel, n_nd_sel = rows.shape[0], nd_idx.shape[0]
    n_rows = n_sims_sel * n_nd_sel

    # Profile block: (S, M, R_sel) -> (S*M, R_sel), simulation-major.
    fgas_block = dataset.fgas[np.ix_(rows, nd_idx, r_pos)]
    x_profiles = fgas_block.reshape(n_rows, r_pos.shape[0])

    # Per-row bookkeeping (aligned with the sim-major / nd-minor flattening).
    nd_values = np.asarray(dataset.number_densities)[nd_idx]      # (M,)
    nd_per_row = np.tile(nd_values, n_sims_sel)                   # (S*M,)
    sim_ids_sel = np.asarray(dataset.sim_ids)[rows]              # (S,)
    sim_index = np.repeat(sim_ids_sel, n_nd_sel)                 # (S*M,)

    # Appended feature columns, in a fixed, documented order.
    feature_blocks = [x_profiles]
    if config.include_nd_feature:
        feature_blocks.append(nd_per_row[:, None])
    if config.include_mean_halo_mass:
        mhm = dataset.mean_halo_mass[np.ix_(rows, nd_idx)]       # (S, M)
        feature_blocks.append(mhm.reshape(n_rows, 1))
    if config.include_camels_params:
        if dataset.camels_params is None:
            raise ValueError(
                "include_camels_params=True but the dataset has no "
                "camels_params (it was saved without a params_path)."
            )
        params_sel = np.asarray(dataset.camels_params)[rows]     # (S, n_params)
        feature_blocks.append(np.repeat(params_sel, n_nd_sel, axis=0))
    X = np.hstack(feature_blocks)

    # Target shaping. The suppression curve is per-simulation, so each sim's row
    # is repeated across its number densities.
    k_all = np.asarray(dataset.k)
    sup_sel = np.asarray(dataset.suppression)[rows]             # (S, n_k)
    if config.target_mode == "curve":
        y = np.repeat(sup_sel, n_nd_sel, axis=0)               # (S*M, n_k)
        k_sel = k_all
    elif config.target_mode == "k_range":
        k_min, k_max = float(config.k_range[0]), float(config.k_range[1])  # type: ignore[index]
        kmask = (k_all >= k_min) & (k_all <= k_max)
        if not kmask.any():
            raise ValueError(
                f"k_range {config.k_range} selects no k bins "
                f"(available {k_all.min():.4g}..{k_all.max():.4g} h/Mpc)."
            )
        y = np.repeat(sup_sel[:, kmask], n_nd_sel, axis=0)     # (S*M, n_k_sel)
        k_sel = k_all[kmask]
    else:  # single_k
        if k_all.size == 0:
            raise ValueError("dataset has no k bins; cannot select a single_k target.")
        k_i = int(np.argmin(np.abs(k_all - float(config.k_target))))  # type: ignore[arg-type]
        y = np.repeat(sup_sel[:, k_i], n_nd_sel)               # (S*M,)
        k_sel = k_all[k_i:k_i + 1]

    return TrainingData(
        X=X,
        y=y,
        nd=nd_per_row,
        sim_index=sim_index,
        k=k_sel,
        radii_mpch=np.asarray(dataset.radii_mpch)[r_pos],
        source_path=str(npz_path),
        meta=meta,
        config=config,
    )


def _summarize(td: TrainingData) -> str:
    """Build the CLI dry-run summary string for a loaded TrainingData."""
    nd_values = np.unique(td.nd)
    target_mode = td.config.target_mode if td.config is not None else "?"
    lines = [
        f"resolved path : {td.source_path}",
        f"X shape       : {td.X.shape}  (n_examples, n_features)",
        f"y shape       : {td.y.shape}  (target_mode={target_mode})",
        f"n_examples    : {td.X.shape[0]}",
        f"n_sims        : {np.unique(td.sim_index).size}",
        f"nd values     : {nd_values.tolist()}",
        f"radii_mpch    : {td.radii_mpch.shape[0]} bins "
        f"[{td.radii_mpch.min():.4g}, {td.radii_mpch.max():.4g}] Mpc/h",
        f"k             : {td.k.shape[0]} bins "
        f"[{td.k.min():.4g}, {td.k.max():.4g}] h/Mpc",
        f"schema        : {td.meta.get('schema_version', 'unknown')}, "
        f"suite={td.meta.get('suite', '?')}, snap={td.meta.get('snapshot', '?')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI dry-run: resolve + load a config and print a summary (no training)."""
    parser = argparse.ArgumentParser(
        description="Resolve and load an f_gas->SP(k) training set; print a "
        "summary. Dry-run only: no training, no side effects."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to a DataConfig YAML file.",
    )
    args = parser.parse_args(argv)

    config = DataConfig.from_yaml(args.config)
    td = load_training_data(config)
    print(_summarize(td))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
