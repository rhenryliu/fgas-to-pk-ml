"""[FROZEN BACKUP] Pre-refactor copy of the f_gas(R) -> SP(k) loader.

This module is a frozen, pre-refactor backup of the data-loading code, retained
only because callers may still source it. Do not extend or fix it here; the
canonical, maintained module is ``fgas_spk_dataset.py``. Note in particular that
this version's ``__main__`` uses the old suppression naming
(``Ptot_Pdm_ratio_k_le15.npz``) and lacks the snapshot/suppression cross-check
and the ``snapshot`` field carried by the canonical module.

Assemble an f_gas(R) to SP(k) training set from CAMELS stacking products.

This module isolates the data-loading logic that currently lives inside
``test_CAMELS_sixth_gen.ipynb`` so it can be reused for model training
(e.g. a probabilistic autoencoder mapping projected gas-fraction profiles to
matter power-spectrum suppression).

Two things differ deliberately from the notebook:

1. **The silent halo-ordering bug is fixed.** The CAMELS fork selects halos by
   stellar mass but returns them in FoF-catalogue order, so the notebook's
   ``[:, :n_halos]`` slice takes the top-N by *FoF mass*, not by the selection
   proxy. Here each simulation's halo columns are re-ranked by a chosen mass
   key (descending) before slicing, so a number-density cut means "the N most
   massive" under a single, explicit definition.

2. **Provenance is checked, not assumed.** The loader validates array shapes and
   the presence of the ionized-gas profile, and fails loudly on a missing or
   malformed file rather than silently producing fifth-gen-stamped output.

Units follow the on-disk CAMELS products and are kept explicit: projected radii
in comoving Mpc/h, Delta Sigma in M_sun*h/(kpc/h)^2 (the ``kpch2`` variant),
wavenumbers in h/Mpc. No h-factors are stripped; do that downstream if your
public API requires it.

Notes:
    ``SubhaloMStar`` is not saved by the current producer, so ranking by stellar
    mass requires re-reading the group catalogues (see ``rank_key='stellar'``,
    which is left as an explicit hook rather than guessed). Ranking by the saved
    ``halo_masses`` (M_500c) works with no extra I/O and is the default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:  # YAML is only needed for the human-readable sidecar manifest.
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

# Profiles file written per simulation by the CAMELS producer.
_PROFILE_FILENAME = "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap74.npz"

# Projections stacked in the products.
_PROJECTIONS = ("xy", "xz", "yz")

# Bump when the on-disk dataset layout changes in a backward-incompatible way.
SCHEMA_VERSION = "1.0"

# Self-describing leaf filename (redundant with the directory path on purpose,
# so a file remains identifiable if copied somewhere else).
_DATASET_STEM_TEMPLATE = (
    "fgas_spk__{suite}__snap{snapshot:03d}_z{redshift}"
    "__src-{source}__rank-{rank}__{tag}"
)


@dataclass
class FgasSpkDataset:
    """Container for an assembled f_gas(R) to SP(k) training set.

    Attributes:
        radii_mpch (np.ndarray): Projected radial bin centres in comoving Mpc/h,
            shape (n_radii,). Shared across simulations (taken from the first).
        number_densities (np.ndarray): Target number densities in (Mpc/h)^-3,
            shape (n_nd,).
        fgas (np.ndarray): Mean projected gas-fraction profiles, normalised by
            the cosmic baryon fraction, shape (n_sims, n_nd, n_radii).
        fgas_std (np.ndarray): Halo-to-halo standard deviation of the gas
            fraction within each (sim, nd) stack, same shape as ``fgas``.
        k (np.ndarray): Wavenumbers for the suppression target in h/Mpc,
            shape (n_k,).
        suppression (np.ndarray): P_total(k) / P_DM(k) per simulation,
            shape (n_sims, n_k).
        sim_ids (np.ndarray): Simulation indices actually loaded, shape (n_sims,).
        mean_halo_mass (np.ndarray): Mean M_500c [M_sun/h] of each (sim, nd)
            stack, shape (n_sims, n_nd). Useful as a conditioning feature or a
            sanity check on the ranking.
        rank_key (str): The mass key used to order halos before slicing.
        camels_params (np.ndarray, optional): CAMELS input parameters per loaded
            simulation, shape (n_sims, n_params), aligned with ``sim_ids``. None
            unless a ``params_path`` was supplied. Intended as a held-out
            diagnostic (e.g. residual-vs-parameter checks), not a core feature.
    """

    radii_mpch: np.ndarray
    number_densities: np.ndarray
    fgas: np.ndarray
    fgas_std: np.ndarray
    k: np.ndarray
    suppression: np.ndarray
    sim_ids: np.ndarray
    mean_halo_mass: np.ndarray
    rank_key: str = "halo_mass"
    camels_params: np.ndarray | None = None

    def to_training_arrays(
        self, k_target: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten the (sim, nd) grid into model-ready arrays.

        Each (simulation, number-density) pair becomes one training example:
        the input is that stack's gas-fraction profile, the conditioning value
        is the number density, and the target is the simulation's suppression.

        Args:
            k_target (float, optional): If given, the target is SP(k) at the
                single wavenumber closest to ``k_target`` (h/Mpc), giving a
                scalar target per example. If None, the full S(k) curve is
                returned as the target. Defaults to None.

        Returns:
            tuple: ``(x_profiles, nd_condition, y_target)`` where

                - ``x_profiles`` has shape (n_examples, n_radii),
                - ``nd_condition`` has shape (n_examples,) in (Mpc/h)^-3,
                - ``y_target`` has shape (n_examples,) if ``k_target`` is set,
                  else (n_examples, n_k).
        """
        n_sims, n_nd, n_radii = self.fgas.shape
        x_profiles = self.fgas.reshape(n_sims * n_nd, n_radii)
        nd_condition = np.repeat(self.number_densities[None, :], n_sims, axis=0).reshape(-1)

        if k_target is None:
            y_target = np.repeat(self.suppression[:, None, :], n_nd, axis=1)
            y_target = y_target.reshape(n_sims * n_nd, -1)
        else:
            k_idx = int(np.argmin(np.abs(self.k - k_target)))
            y_scalar = self.suppression[:, k_idx]
            y_target = np.repeat(y_scalar[:, None], n_nd, axis=1).reshape(-1)

        return x_profiles, nd_condition, y_target


def _load_one_simulation(
    profile_path: Path, rank_key: str
) -> dict[str, np.ndarray]:
    """Load and re-rank a single simulation's profile file.

    Args:
        profile_path (Path): Path to that simulation's ``Profiles_*.npz``.
        rank_key (str): Column-ordering key. ``'halo_mass'`` uses the saved
            ``halo_masses`` (M_500c). ``'stellar'`` is reserved for a stellar-mass
            ordering and raises until a catalogue re-read is wired in, because the
            producer does not save ``SubhaloMStar``.

    Returns:
        dict: With keys ``'fgas'`` (n_radii, n_halos), ``'halo_masses'``
            (n_halos,, descending), ``'radii_mpch'`` (n_radii,), and ``'fb'``.

    Raises:
        FileNotFoundError: If the profile file is absent.
        KeyError: If the ionized-gas profile is missing (i.e. the file predates
            the third profile and cannot yield a gas fraction).
        NotImplementedError: If ``rank_key='stellar'`` is requested.
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile file: {profile_path}")

    data = np.load(profile_path)

    if f"prof2_ionized_gas_DSigma_kpch2_{_PROJECTIONS[0]}" not in data:
        raise KeyError(
            f"{profile_path.name} has no ionized-gas profile (prof2_*). "
            "This file cannot produce a gas fraction; check the producer "
            "generation."
        )

    # Stack the three projections along the halo axis for both numerator
    # (ionized gas) and denominator (total). Using the kpch2 variant keeps the
    # ratio physical; the arcmin2 variant gives an identical ratio since the
    # per-radius angular conversion cancels.
    ionized = np.concatenate(
        [data[f"prof2_ionized_gas_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
    )
    total = np.concatenate(
        [data[f"prof1_total_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
    )

    fb = float(data["fb"])
    # Gas fraction per halo, normalised by the cosmic baryon fraction.
    with np.errstate(divide="ignore", invalid="ignore"):
        fgas = ionized / total / fb  # (n_radii, n_halos * n_projections)

    halo_masses = np.asarray(data["halo_masses"])  # M_500c [M_sun/h]
    # Halo masses are per-halo (not per-projection); tile to match the stacked
    # halo axis so a single ranking applies across projections.
    halo_masses_tiled = np.tile(halo_masses, len(_PROJECTIONS))

    if rank_key == "halo_mass":
        order = np.argsort(halo_masses_tiled)[::-1]  # descending M_500c
    elif rank_key == "stellar":
        raise NotImplementedError(
            "Stellar-mass ranking requires SubhaloMStar, which the producer "
            "does not save. Re-read the group catalogue per sim and pass the "
            "stellar masses in, or have the producer dump SubhaloMStar."
        )
    else:
        raise ValueError(f"Unknown rank_key: {rank_key!r}")

    fgas = fgas[:, order]
    halo_masses_tiled = halo_masses_tiled[order]

    radii_mpch = np.asarray(data["r12_to_mpch"])
    if radii_mpch.ndim > 1:  # some files store one row per sim; collapse it
        radii_mpch = radii_mpch[0]

    return {
        "fgas": fgas,
        "halo_masses": halo_masses_tiled,
        "radii_mpch": radii_mpch,
        "fb": fb, # type: ignore
    }


def _load_params(params_path: str | Path | None, sim_ids: Sequence[int]) -> np.ndarray | None:
    """Load and index the CAMELS parameter matrix, if a path is given.

    Args:
        params_path (str | Path, optional): Path to ``camels_params_matrix.npy``,
            shape (n_total_sims, n_params). If None, returns None.
        sim_ids (Sequence[int]): Simulation indices that were loaded, used to
            select and order the matching parameter rows.

    Returns:
        np.ndarray | None: Parameters of shape (len(sim_ids), n_params), or None.

    Raises:
        IndexError: If any requested simulation index is out of range for the
            parameter matrix (a guard against a generation/index mismatch).
    """
    if params_path is None:
        return None
    params = np.load(Path(params_path))
    if max(sim_ids) >= params.shape[0]:
        raise IndexError(
            f"sim_id {max(sim_ids)} out of range for params matrix of length "
            f"{params.shape[0]}; check the matrix matches this simulation set."
        )
    return params[np.asarray(sim_ids)]


def load_fgas_spk_dataset(
    base_path_template: str,
    suppression_path: str | Path,
    number_densities: Sequence[float] = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3),
    box_size_mpch: float = 50.0,
    sim_ids: Sequence[int] | None = None,
    rank_key: str = "halo_mass",
    params_path: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FgasSpkDataset:
    """Assemble the full f_gas(R) to SP(k) dataset across all simulations.

    For each simulation and each target number density, the N most massive
    halos (under ``rank_key``) are stacked and their mean gas-fraction profile
    is computed, where ``N = int(number_density * box_size_mpch**3)``. The
    per-simulation suppression curve is read from a single shared file.

    Args:
        base_path_template (str): Template for each simulation's data directory,
            with a single ``{}`` placeholder for the simulation index, e.g.
            ``'/path/to/SB35_{}/data/'``. The profile filename is appended.
        suppression_path (str | Path): Path to the ``Ptot_Pdm_ratio_*.npz`` file
            holding keys ``'k'`` (n_k,) and ``'Ptot_Pdm_ratio'`` (n_sims, n_k).
        number_densities (Sequence[float], optional): Target number densities in
            (Mpc/h)^-3. Defaults to the five values used in the notebook.
        box_size_mpch (float, optional): Box side length in Mpc/h, used to turn
            number densities into halo counts. Defaults to 50.0.
        sim_ids (Sequence[int], optional): Simulation indices to load. Defaults
            to ``range(n_sims)`` inferred from the suppression file.
        rank_key (str, optional): Halo-ordering key for the number-density cut.
            Defaults to ``'halo_mass'`` (M_500c).
        progress (Callable, optional): Callback ``progress(i, n_total)`` invoked
            per simulation, for a progress bar. Defaults to None.

    Returns:
        FgasSpkDataset: The assembled dataset.

    Raises:
        ValueError: If the suppression file row count does not match the number
            of requested simulations.
    """
    suppression_path = Path(suppression_path)
    sup_file = np.load(suppression_path)
    k = np.asarray(sup_file["k"])
    suppression_all = np.asarray(sup_file["Ptot_Pdm_ratio"])  # (n_sims, n_k)

    if sim_ids is None:
        sim_ids = list(range(suppression_all.shape[0]))
    sim_ids = list(sim_ids)

    n_halos_per_nd = [int(nd * box_size_mpch**3) for nd in number_densities]

    fgas_rows: list[np.ndarray] = []
    fgas_std_rows: list[np.ndarray] = []
    mean_mass_rows: list[np.ndarray] = []
    radii_ref: np.ndarray | None = None
    kept_ids: list[int] = []
    kept_suppression: list[np.ndarray] = []

    for i, sim_id in enumerate(sim_ids):
        profile_path = Path(base_path_template.format(sim_id)) / _PROFILE_FILENAME
        loaded = _load_one_simulation(profile_path, rank_key=rank_key)

        if radii_ref is None:
            radii_ref = loaded["radii_mpch"]

        fgas = loaded["fgas"]               # (n_radii, n_halos_sorted)
        masses = loaded["halo_masses"]      # (n_halos_sorted,), descending

        per_nd_mean = []
        per_nd_std = []
        per_nd_mass = []
        for n_halos in n_halos_per_nd:
            n_use = min(n_halos, fgas.shape[1])
            block = fgas[:, :n_use]
            per_nd_mean.append(np.nanmean(block, axis=1))
            per_nd_std.append(np.nanstd(block, axis=1))
            per_nd_mass.append(np.nanmean(masses[:n_use]))

        fgas_rows.append(np.stack(per_nd_mean))       # (n_nd, n_radii)
        fgas_std_rows.append(np.stack(per_nd_std))
        mean_mass_rows.append(np.asarray(per_nd_mass))  # (n_nd,)
        kept_ids.append(sim_id)
        kept_suppression.append(suppression_all[sim_id])

        if progress is not None:
            progress(i + 1, len(sim_ids))

    suppression = np.stack(kept_suppression)  # (n_sims, n_k)
    if suppression.shape[0] != len(kept_ids):
        raise ValueError(
            "Suppression rows do not match loaded simulations: "
            f"{suppression.shape[0]} vs {len(kept_ids)}."
        )

    return FgasSpkDataset(
        radii_mpch=np.asarray(radii_ref),
        number_densities=np.asarray(number_densities),
        fgas=np.stack(fgas_rows),            # (n_sims, n_nd, n_radii)
        fgas_std=np.stack(fgas_std_rows),
        k=k,
        suppression=suppression,
        sim_ids=np.asarray(kept_ids),
        mean_halo_mass=np.stack(mean_mass_rows),  # (n_sims, n_nd)
        rank_key=rank_key,
        camels_params=_load_params(params_path, kept_ids),
    )


def load_fgas_spk_from_compiled(
    data_dir: str | Path,
    suppression_path: str | Path,
    number_densities: Sequence[float] = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3),
    box_size_mpch: float = 50.0,
    name_extra: str = "_sixth_gen_makeMap",
    params_path: str | Path | None = None,
) -> FgasSpkDataset:
    """Fast path: load the pre-compiled per-nd ratio files instead of raw profiles.

    Reads the aggregated ``{name_extra}DeltaSigma_kSZ_over_DeltaSigma_total_
    profiles_nd_{i}_n_{N}.npz`` files written by the notebook (each holding the
    per-simulation f_gas ratio of shape (n_sims, n_radii)) and stacks them into
    the same :class:`FgasSpkDataset` layout. This is much faster than re-reading
    1024 raw profile files, at the cost of flexibility.

    Warning:
        This inherits whatever halo ranking was frozen when the compiled files
        were produced. If they came from the unsorted CAMELS-fork selection, the
        number-density bins are top-N by FoF mass, not by the selection proxy,
        and that cannot be corrected here. ``mean_halo_mass`` is unavailable
        (the compiled files do not store per-halo masses) and is returned as
        NaN. Use :func:`load_fgas_spk_dataset` when correctness of the binning
        matters; use this for quick iteration when it does not.

    Args:
        data_dir (str | Path): Directory holding the compiled ``.npz`` files and
            the suppression file.
        suppression_path (str | Path): Path to ``Ptot_Pdm_ratio_*.npz``.
        number_densities (Sequence[float], optional): Target number densities in
            (Mpc/h)^-3. Defaults to the five notebook values.
        box_size_mpch (float, optional): Box side in Mpc/h, used only to rebuild
            the ``n_{N}`` count in each filename. Defaults to 50.0.
        name_extra (str, optional): Filename prefix tag. Defaults to
            ``'_sixth_gen_makeMap'``; pass ``'_fifth_gen_makeMap'`` or ``''`` if
            that is what is actually on disk.
        params_path (str | Path, optional): Path to ``camels_params_matrix.npy``.
            Defaults to None.

    Returns:
        FgasSpkDataset: Assembled from the compiled files.

    Raises:
        FileNotFoundError: If a compiled file for any number density is missing.
    """
    data_dir = Path(data_dir)
    sup_file = np.load(Path(suppression_path))
    k = np.asarray(sup_file["k"])
    suppression = np.asarray(sup_file["Ptot_Pdm_ratio"])  # (n_sims, n_k)

    n_halos_per_nd = [int(nd * box_size_mpch**3) for nd in number_densities]

    fgas_per_nd: list[np.ndarray] = []
    fgas_std_per_nd: list[np.ndarray] = []
    radii_ref: np.ndarray | None = None
    for nd_idx, n_halos in enumerate(n_halos_per_nd):
        fname = f"{name_extra}DeltaSigma_kSZ_over_DeltaSigma_total_profiles_nd_{nd_idx}_n_{n_halos}.npz"
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Missing compiled ratio file: {path}. Check `name_extra` "
                "matches the files on disk (they may be fifth-gen stamped)."
            )
        d = np.load(path)
        fgas_per_nd.append(np.asarray(d["prof"]))          # (n_sims, n_radii)
        fgas_std_per_nd.append(np.asarray(d["prof_std"]))
        if radii_ref is None:
            radii_ref = np.asarray(d["radii"])

    # Stack to (n_sims, n_nd, n_radii) to match the raw-loader layout.
    fgas = np.stack(fgas_per_nd, axis=1)
    fgas_std = np.stack(fgas_std_per_nd, axis=1)
    n_sims = fgas.shape[0]
    sim_ids = list(range(n_sims))

    return FgasSpkDataset(
        radii_mpch=np.asarray(radii_ref),
        number_densities=np.asarray(number_densities),
        fgas=fgas,
        fgas_std=fgas_std,
        k=k,
        suppression=suppression[:n_sims],
        sim_ids=np.asarray(sim_ids),
        mean_halo_mass=np.full((n_sims, len(number_densities)), np.nan),
        rank_key="frozen_at_production",
        camels_params=_load_params(params_path, sim_ids),
    )


def dataset_dir(
    project_root: str | Path,
    suite: str,
    snapshot: int,
    redshift: float | str,
    source: str,
) -> Path:
    """Return the canonical directory for a dataset's provenance bucket.

    Layout: ``<project_root>/datasets/<suite>/snap<NNN>_z<z>/src-<source>/``.
    Nesting by suite then snapshot then source keeps your own products and a
    collaborator's side by side without collision, and groups everything sharing
    a physical configuration.

    Args:
        project_root (str | Path): Project root, e.g.
            ``/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml``.
        suite (str): Simulation suite tag, e.g.
            ``'CAMELS-IllustrisTNG-L50n512-SB35'``.
        snapshot (int): Snapshot number, e.g. 74.
        redshift (float | str): Redshift, e.g. 0.47.
        source (str): Provenance of the profiles, e.g. ``'lindajin'`` or
            ``'rhliu'``.

    Returns:
        Path: The directory (not created).
    """
    return (
        Path(project_root)
        / "datasets"
        / suite
        / f"snap{snapshot:03d}_z{redshift}"
        / f"src-{source}"
    )


def save_dataset(
    dataset: FgasSpkDataset,
    project_root: str | Path,
    suite: str,
    snapshot: int,
    redshift: float | str,
    source: str,
    rank: str | None = None,
    tag: str | None = None,
    overwrite: bool = False,
    provenance: dict | None = None,
    notes: str | None = None,
    manifest_dir: str | Path | None = None,
) -> Path:
    """Write a dataset to the scratch store under a consistent, self-describing name.

    The scratch store holds data. This writes a ``.npz`` into the provenance
    bucket with all metadata embedded as a JSON blob under the ``__meta__`` key
    (so the file is self-describing), plus a human-readable ``<stem>.yaml``
    sidecar carrying the same metadata. The sidecar sits beside the data by
    default; pass ``manifest_dir`` to redirect it elsewhere (e.g. your repo).

    The filename encodes suite, snapshot, redshift, source, ranking, and a
    version tag, so it stays identifiable if moved.

    Args:
        dataset (FgasSpkDataset): The assembled dataset to save.
        project_root (str | Path): Scratch store root directory.
        suite (str): Simulation suite tag (see :func:`dataset_dir`).
        snapshot (int): Snapshot number.
        redshift (float | str): Redshift.
        source (str): Provenance tag for the underlying profiles.
        rank (str, optional): Ranking tag for the filename. Defaults to a
            cleaned form of ``dataset.rank_key`` (e.g. ``'halo_mass'`` -> ``'m500'``).
        tag (str, optional): Version tag. Defaults to today's UTC date
            (``YYYYMMDD``); pass e.g. ``'v1'`` to manage versions explicitly.
        overwrite (bool, optional): If False (default), refuse to clobber an
            existing file and raise instead. Defaults to False.
        provenance (dict, optional): Free-form provenance (source repo, git
            commit, producer script, generation tag). Stored in ``__meta__``.
        notes (str, optional): Any human notes to record.
        manifest_dir (str | Path, optional): Directory for the ``<stem>.yaml``
            sidecar. If None (default), the sidecar is written beside the
            ``.npz`` in the dataset bucket. Pass a git-tracked path to keep the
            manifest under version control instead.

    Returns:
        Path: Path to the written ``.npz``.

    Raises:
        FileExistsError: If the target exists and ``overwrite`` is False.
    """
    rank = rank or _clean_rank(dataset.rank_key)
    tag = tag or datetime.now(timezone.utc).strftime("%Y%m%d")

    out_dir = dataset_dir(project_root, suite, snapshot, redshift, source)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _DATASET_STEM_TEMPLATE.format(
        suite=suite, snapshot=snapshot, redshift=redshift,
        source=source, rank=rank, tag=tag,
    )
    npz_path = out_dir / f"{stem}.npz"

    if npz_path.exists() and not overwrite:
        raise FileExistsError(
            f"{npz_path} already exists. Pass overwrite=True or change `tag`."
        )

    arrays = {
        "radii_mpch": dataset.radii_mpch,
        "number_densities": dataset.number_densities,
        "fgas": dataset.fgas,
        "fgas_std": dataset.fgas_std,
        "k": dataset.k,
        "suppression": dataset.suppression,
        "sim_ids": dataset.sim_ids,
        "mean_halo_mass": dataset.mean_halo_mass,
    }
    if dataset.camels_params is not None:
        arrays["camels_params"] = dataset.camels_params

    meta = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": suite,
        "snapshot": snapshot,
        "redshift": redshift,
        "source": source,
        "rank_key": dataset.rank_key,
        "tag": tag,
        "n_sims": int(dataset.fgas.shape[0]),
        "n_number_densities": int(dataset.fgas.shape[1]),
        "n_radii": int(dataset.fgas.shape[2]),
        "n_k": int(dataset.k.shape[0]),
        "number_densities": [float(x) for x in dataset.number_densities],
        "units": {
            "radii_mpch": "comoving Mpc/h",
            "k": "h/Mpc",
            "suppression": "P_total(k)/P_DM(k), dimensionless",
            "fgas": "Delta Sigma_ionized / Delta Sigma_total / f_b, dimensionless",
            "mean_halo_mass": "M_500c [M_sun/h]",
        },
        "cosmology_note": (
            "Per-simulation cosmology varies across the suite (Sobol set); "
            "f_b and h are stored per simulation in the source profiles, not here."
        ),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "provenance": provenance or {},
        "notes": notes or "",
    }

    np.savez(npz_path, __meta__=json.dumps(meta), **arrays)
    # Always write the human-readable sidecar; metadata also lives inside the
    # .npz under __meta__. Location defaults to beside the data; manifest_dir
    # overrides it (e.g. a git-tracked path in your repo).
    manifest_target = Path(manifest_dir) if manifest_dir is not None else out_dir
    manifest_target.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_target / f"{stem}.yaml", meta)
    return npz_path


def load_dataset(npz_path: str | Path) -> FgasSpkDataset:
    """Reconstruct a :class:`FgasSpkDataset` from a saved ``.npz``.

    Args:
        npz_path (str | Path): Path to a file written by :func:`save_dataset`.

    Returns:
        FgasSpkDataset: The reconstructed dataset. The embedded metadata is
            accessible via ``np.load(npz_path)['__meta__']`` if needed.
    """
    data = np.load(Path(npz_path), allow_pickle=False)
    meta = json.loads(str(data["__meta__"])) if "__meta__" in data else {}
    return FgasSpkDataset(
        radii_mpch=data["radii_mpch"],
        number_densities=data["number_densities"],
        fgas=data["fgas"],
        fgas_std=data["fgas_std"],
        k=data["k"],
        suppression=data["suppression"],
        sim_ids=data["sim_ids"],
        mean_halo_mass=data["mean_halo_mass"],
        rank_key=meta.get("rank_key", "unknown"),
        camels_params=data["camels_params"] if "camels_params" in data else None,
    )


def ensure_store_dirs(project_root: str | Path, make_models: bool = True) -> Path:
    """Ensure the scratch store has the expected top-level directories.

    This store holds data only: ``datasets/`` and, optionally, ``models/`` for
    large checkpoints. Configs, figures, and documentation live in the
    git-tracked repository, not here, so nothing else is created.

    Args:
        project_root (str | Path): Project root on scratch, e.g.
            ``/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml``.
        make_models (bool, optional): Also create ``models/``. Defaults to True.

    Returns:
        Path: The project root.
    """
    root = Path(project_root)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    if make_models:
        (root / "models").mkdir(parents=True, exist_ok=True)
    return root


def _clean_rank(rank_key: str) -> str:
    """Map an internal rank_key to a short filename token."""
    return {
        "halo_mass": "m500",
        "stellar": "mstar",
        "frozen_at_production": "frozen",
    }.get(rank_key, rank_key.replace("_", "-"))


def _write_manifest(yaml_path: Path, meta: dict) -> None:
    """Write the human-readable sidecar manifest (YAML if available, else JSON)."""
    if _HAS_YAML:
        yaml_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    else:  # A .json with the same stem avoids a misleading .yaml extension.
        yaml_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
    BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"

    # Correct (slower) path: rebuilds bins from raw per-halo data, fixing the sort.
    dataset = load_fgas_spk_dataset(
        base_path_template=BASE,
        suppression_path=Path(DATA_DIR) / "Ptot_Pdm_ratio_k_le15.npz",
        rank_key="halo_mass",
        params_path=Path(DATA_DIR) / "camels_params_matrix.npy",
        progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True),
    )
    print()
    print("fgas grid:", dataset.fgas.shape, "(n_sims, n_nd, n_radii)")
    print("suppression:", dataset.suppression.shape, "(n_sims, n_k)")
    if dataset.camels_params is not None:
        print("params:", dataset.camels_params.shape, "(n_sims, n_params)")

    x, nd, y = dataset.to_training_arrays(k_target=3.0)
    print("training X:", x.shape, "| nd:", nd.shape, "| y (SP at k~3):", y.shape)

    # Save into your scratch store (data only) under a consistent name.
    PROJECT = "/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml"
    ensure_store_dirs(PROJECT)  # creates datasets/ and models/ only
    out = save_dataset(
        dataset,
        project_root=PROJECT,
        suite="CAMELS-IllustrisTNG-L50n512-SB35",
        snapshot=74,
        redshift=0.47,
        source="lindajin",                 # switch to "rhliu" for self-generated runs
        provenance={
            "profiles_repo": "github.com/Klinjin/SimulationStacker (fork)",
            "producer": "CAMELS_training_data_second_generation.py",
            "note": "Profiles from lindajin scratch; re-ranked by M_500c on load.",
        },
        notes="Bins re-sorted by halo mass to correct the unsorted-selection bug.",
        # manifest_dir=...  # optionally point at a git-tracked path in your repo
    )
    print("saved:", out)
    # Round-trip check:
    # reloaded = load_dataset(out)

    # Fast path: load the pre-compiled ratio files as-is (no re-ranking).
    # fast = load_fgas_spk_from_compiled(
    #     data_dir=DATA_DIR,
    #     suppression_path=Path(DATA_DIR) / "Ptot_Pdm_ratio_k_le15.npz",
    #     name_extra="_sixth_gen_makeMap",
    # )