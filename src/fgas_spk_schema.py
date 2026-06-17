"""On-disk data contract for the f_gas(R) to SP(k) training sets.

This module is the **single source of truth** for the dataset schema: the
:class:`FgasSpkDataset` container, the filename/directory convention, the
save/load round-trip, and the schema version. The builder
(:mod:`fgas_spk_builder`) and the training-data loader (:mod:`fgas_spk_loader`)
both import from here; neither redefines the contract.

Keeping the contract in one place means the on-disk format has exactly one
definition. The saved ``.npz`` layout is frozen at :data:`SCHEMA_VERSION`: each
file stores the dataset arrays plus a JSON ``__meta__`` blob, with a
human-readable ``.yaml`` sidecar written beside it.

Units follow the on-disk CAMELS products and are kept explicit: projected radii
in comoving Mpc/h, wavenumbers in h/Mpc, masses in M_sun/h. CAMELS cosmology
varies per simulation (Sobol set); per-sim f_b and h live in the source
profiles, not in the assembled dataset.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:  # YAML is only needed for the human-readable sidecar manifest.
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False

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
        snapshot (int, optional): Snapshot the dataset was built for (e.g. 74 ->
            z=0.47, 82 -> z=0.21). Populated by the loaders so the redshift is
            carried with the data; None for datasets assembled without it.
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
    snapshot: int | None = None

    def to_training_arrays(
        self, k_target: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten the (sim, nd) grid into model-ready arrays.

        Each (simulation, number-density) pair becomes one training example:
        the input is that stack's gas-fraction profile, the conditioning value
        is the number density, and the target is the simulation's suppression.

        Note:
            This is a minimal in-memory convenience for a dataset you already
            hold (e.g. a quick check right after building). It is **not** the
            training entry point: it offers no simulation/number-density subset,
            no radial crop, no ``k`` targeting beyond a single bin, and no
            ``X_cond``/``X_params`` modality split. For real training work, save
            the dataset with :func:`save_dataset` and load it through
            :mod:`fgas_spk_loader` (``load_training_data``), which is the
            configurable, reproducible path. Calling this method emits a
            ``UserWarning`` to that effect; silence it with
            :func:`warnings.filterwarnings` if the shortcut is what you want.

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
        warnings.warn(
            "FgasSpkDataset.to_training_arrays is an in-memory convenience, not "
            "the training entry point: it has no subsetting, radial crop, or "
            "X_cond/X_params modality split. For training, save the dataset and "
            "load it via fgas_spk_loader.load_training_data. Silence this with "
            "warnings.filterwarnings if the shortcut is intended.",
            UserWarning,
            stacklevel=2,
        )

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


def resolve_dataset_path(
    project_root: str | Path,
    suite: str,
    snapshot: int,
    redshift: float | str,
    source: str,
    rank: str,
    tag: str,
) -> Path:
    """Build the ``.npz`` path for a dataset from its identifying fields.

    This reconstructs the canonical path a :func:`save_dataset` call would have
    written, so a caller never has to type the path beyond ``project_root``.
    ``rank`` is the on-disk rank *token* that appears in the filename (e.g.
    ``'m500'``, ``'mstar'``, ``'frozen'`` -- the output of :func:`_clean_rank`),
    not the internal ``rank_key``.

    If ``tag == "latest"`` the dataset bucket is globbed and the newest tag is
    chosen automatically:

    - if every on-disk tag is an 8-digit ``YYYYMMDD`` date, the maximum
      (most recent) date is used;
    - if every tag is ``vN``, the maximum ``N`` is used;
    - if the tags are mixed (some dates, some versions, or anything else),
      resolution is refused and an explicit ``tag`` is required.

    Note on reproducibility:
        ``tag="latest"`` is a convenience for interactive iteration. For
        anything you want to reproduce (a training run, a saved config), prefer
        an explicit tag so the resolved file cannot drift when a newer dataset
        is written into the same bucket.

    Args:
        project_root (str | Path): Project / store root directory.
        suite (str): Simulation suite tag (see :func:`dataset_dir`).
        snapshot (int): Snapshot number.
        redshift (float | str): Redshift.
        source (str): Provenance tag for the underlying profiles.
        rank (str): On-disk rank token used in the filename.
        tag (str): Version tag, or the literal ``"latest"`` to auto-select.

    Returns:
        Path: Path to the resolved ``.npz`` (not checked for existence unless
            ``tag == "latest"``, in which case it is the newest match found).

    Raises:
        FileNotFoundError: If ``tag == "latest"`` and no matching files exist.
        ValueError: If ``tag == "latest"`` and the on-disk tags are mixed
            (cannot be ordered unambiguously); pass an explicit tag instead.
    """
    out_dir = dataset_dir(project_root, suite, snapshot, redshift, source)

    if tag != "latest":
        stem = _DATASET_STEM_TEMPLATE.format(
            suite=suite, snapshot=snapshot, redshift=redshift,
            source=source, rank=rank, tag=tag,
        )
        return out_dir / f"{stem}.npz"

    # Glob the bucket for every tag under this (suite, snap, source, rank).
    # The prefix is fully fixed, so the remaining tail is exactly the tag --
    # robust even if the suite name itself contains the "__" separator.
    prefix = _DATASET_STEM_TEMPLATE.format(
        suite=suite, snapshot=snapshot, redshift=redshift,
        source=source, rank=rank, tag="",
    )  # ends with the trailing "__"
    matches = sorted(out_dir.glob(f"{prefix}*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No datasets matching '{prefix}*.npz' in {out_dir}; cannot resolve "
            "tag='latest'. Check the fields or pass an explicit tag."
        )

    tags = [p.name[len(prefix):-len(".npz")] for p in matches]
    date_re = re.compile(r"^\d{8}$")
    version_re = re.compile(r"^v(\d+)$")

    if all(date_re.match(t) for t in tags):
        chosen = max(tags)  # YYYYMMDD sorts chronologically as a string
    elif all(version_re.match(t) for t in tags):
        chosen = max(tags, key=lambda t: int(version_re.match(t).group(1)))  # type: ignore[union-attr]
    else:
        raise ValueError(
            f"tag='latest' cannot order mixed tags {sorted(tags)} in {out_dir}. "
            "Use all-date (YYYYMMDD) or all-version (vN) tags in a bucket, or "
            "pass an explicit tag."
        )

    return out_dir / f"{prefix}{chosen}.npz"


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
        ValueError: If ``snapshot`` contradicts a non-None ``dataset.snapshot``.
    """
    if dataset.snapshot is not None and dataset.snapshot != snapshot:
        raise ValueError(
            f"save_dataset(snapshot={snapshot}) contradicts dataset.snapshot="
            f"{dataset.snapshot}. Pass the snapshot the dataset was built for."
        )

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
        snapshot=meta.get("snapshot"),
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
