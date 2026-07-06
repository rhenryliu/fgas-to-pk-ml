"""Assemble an f_gas(R) to SP(k) training set from CAMELS stacking products.

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

import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# The on-disk contract -- the dataset container, the filename/directory
# convention, and the save/load round-trip -- lives in ``fgas_spk.schema``, the
# single source of truth. The names below are imported for use by the builders
# and re-exported so existing callers can keep importing them from this module
# (now ``fgas_spk.builder``).
from fgas_spk.schema import (  # noqa: F401  (re-exported for backward compatibility)
    SCHEMA_VERSION,
    FgasSpkDataset,
    dataset_dir,
    ensure_store_dirs,
    load_dataset,
    resolve_dataset_path,
    save_dataset,
)

# Per-simulation profiles file written by the CAMELS producer. The snapshot
# number is part of the filename (e.g. snap74 -> z=0.47, snap82 -> z=0.21).
_PROFILE_FILENAME_TEMPLATE = (
    "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap{snapshot}.npz"
)

# Projections stacked in the products.
_PROJECTIONS = ("xy", "xz", "yz")

# The gas-fraction statistic this builder computes, stamped verbatim into the
# dataset ``__meta__`` (``fgas_definition``). Datasets lacking this stamp, or
# carrying a different one, were built with the superseded per-halo-ratio
# statistic and must not be compared against stamped ones (Branch A, E2.2).
FGAS_DEFINITION = (
    "ratio of stacked profiles: nanmean of Delta Sigma_ionized over the "
    "selected halos x 3 projections per radial bin, divided by the same "
    "nanmean of Delta Sigma_total, divided by the per-simulation f_b. "
    "Matches the measurements-paper convention and the reference notebook's "
    "ratio-of-means; immune to per-halo Delta Sigma_total zero-crossings at "
    "R >~ 0.7 Mpc/h (see experiments/notes/dual_vae_stage3_forensics.md)."
)

# Build-time sanity range for the stacked f_gas (E2.1). PROVISIONAL, maintainer
# to confirm; a build producing any value outside this range hard-fails.
DEFAULT_FGAS_SANITY_RANGE = (-1.0, 3.0)


def fgas_meta_stamp(
    fgas: np.ndarray, sanity_range: tuple[float, float]
) -> dict:
    """Return the additive ``__meta__`` stamp for a built f_gas array (E2.1/E2.2).

    Args:
        fgas (np.ndarray): The built array, shape (n_sims, n_nd, n_radii).
        sanity_range (tuple[float, float]): The build's validated range.

    Returns:
        dict: ``fgas_definition`` (the statistic, verbatim),
            ``fgas_sanity_range``, and ``fgas_realized_range_per_bin`` with
            per-radial-bin ``min`` / ``max`` lists, so every build documents
            its own realized range.
    """
    fgas = np.asarray(fgas)
    return {
        "fgas_definition": FGAS_DEFINITION,
        "fgas_sanity_range": [float(sanity_range[0]), float(sanity_range[1])],
        "fgas_realized_range_per_bin": {
            "min": [float(v) for v in fgas.min(axis=(0, 1))],
            "max": [float(v) for v in fgas.max(axis=(0, 1))],
        },
    }


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
        dict: With keys ``'ionized'`` and ``'total'`` -- the per-halo
            ``Delta Sigma`` profiles, each (n_radii, n_proj, n_halos) with the
            three projections on a separate axis and the halo axis re-ranked
            descending -- ``'halo_masses'`` (n_halos,, descending, per distinct
            halo), ``'radii_mpch'`` (n_radii,), and ``'fb'``. The gas-fraction
            ratio is deliberately NOT formed here: it is a ratio of *stacked*
            profiles, taken in :func:`build_fgas_spk_dataset` after the
            number-density cut (see the Stage 3.0 forensics note -- a per-halo
            ratio is singular wherever a halo's ``Delta Sigma_total`` crosses
            zero, which is generic at R >~ 0.7 Mpc/h).

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

    # Keep the three projections on a separate axis (not concatenated into the
    # halo axis), so a number-density cut selects N *distinct* halos with all of
    # their projections, rather than N halo-projection columns (which would be
    # ~N/3 distinct halos). Using the kpch2 variant keeps the ratio physical; the
    # arcmin2 variant gives an identical ratio since the per-radius angular
    # conversion cancels.
    ionized = np.stack(
        [data[f"prof2_ionized_gas_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
    )
    total = np.stack(
        [data[f"prof1_total_DSigma_kpch2_{p}"] for p in _PROJECTIONS], axis=1
    )

    fb = float(data["fb"])
    halo_masses = np.asarray(data["halo_masses"])  # M_500c [M_sun/h], per halo

    if rank_key == "halo_mass":
        order = np.argsort(halo_masses)[::-1]  # descending M_500c, per distinct halo
    elif rank_key == "stellar":
        raise NotImplementedError(
            "Stellar-mass ranking requires SubhaloMStar, which the producer "
            "does not save. Re-read the group catalogue per sim and pass the "
            "stellar masses in, or have the producer dump SubhaloMStar."
        )
    else:
        raise ValueError(f"Unknown rank_key: {rank_key!r}")

    # Reorder the halo axis (shared across projections) by descending mass.
    ionized = ionized[:, :, order]
    total = total[:, :, order]
    halo_masses = halo_masses[order]

    # radii_mpch = np.asarray(data["r12_to_mpch"])
    radii_mpch = np.asarray(data["r12_mpch"])
    if radii_mpch.ndim > 1:  # some files store one row per sim; collapse it
        radii_mpch = radii_mpch[0]

    return {
        "ionized": ionized,
        "total": total,
        "halo_masses": halo_masses,
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


def _check_suppression_snapshot(suppression_path: Path, snapshot: int) -> None:
    """Guard against pairing profiles with a mismatched-snapshot suppression file.

    The suppression filename encodes its snapshot (e.g.
    ``Ptot_Pdm_ratio_snap74.npz``). If that token disagrees with the ``snapshot``
    argument, the loaders would silently pair one snapshot's profiles with
    another's suppressions -- a wrong-redshift bug the row-count shape guard
    cannot catch. Filenames without a ``snap<NN>`` token (e.g. older naming
    conventions) are left unchecked rather than rejected.

    Args:
        suppression_path (Path): Path to the suppression ``.npz``.
        snapshot (int): Snapshot the profiles are being loaded for.

    Raises:
        ValueError: If the filename's snapshot token contradicts ``snapshot``.
    """
    match = re.search(r"snap(\d+)", suppression_path.name)
    if match and int(match.group(1)) != snapshot:
        raise ValueError(
            f"suppression_path '{suppression_path.name}' is snap{match.group(1)} "
            f"but snapshot={snapshot} was passed. These must match -- a mismatch "
            "would silently pair profiles and suppressions from different redshifts."
        )


def build_fgas_spk_dataset(
    base_path_template: str,
    suppression_path: str | Path,
    snapshot: int,
    number_densities: Sequence[float] = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3),
    box_size_mpch: float = 50.0,
    sim_ids: Sequence[int] | None = None,
    rank_key: str = "halo_mass",
    params_path: str | Path | None = None,
    fgas_sanity_range: tuple[float, float] = DEFAULT_FGAS_SANITY_RANGE,
    progress: Callable[[int, int], None] | None = None,
) -> FgasSpkDataset:
    """Assemble the full f_gas(R) to SP(k) dataset across all simulations.

    For each simulation and each target number density, the N most massive
    *distinct* halos (under ``rank_key``) are selected, where
    ``N = int(number_density * box_size_mpch**3)``, and the gas fraction is the
    **ratio of stacked profiles** (:data:`FGAS_DEFINITION`): ``Delta
    Sigma_ionized`` is nanmean-pooled over the selected halos and all three
    projections per radial bin, ``Delta Sigma_total`` likewise, the two stacks
    are divided once, and the result is normalised by the per-simulation
    ``f_b``. This matches the observational estimator (stacked lensing over a
    stacked sample) and the reference notebook's explicit ratio-of-means; it
    replaced a per-halo ratio-then-average statistic that was singular
    wherever one halo's ``Delta Sigma_total`` crossed zero (generic at
    R >~ 0.7 Mpc/h) -- see ``experiments/notes/dual_vae_stage3_forensics.md``.
    (Ratio of per-halo means equals ratio of per-halo sums over the same halo
    set, so "mean then ratio" and "sum then ratio" are the same estimator.)
    ``fgas_std`` propagates the two stacks' pooled standard deviations in
    quadrature, mirroring the reference notebook:
    ``|fgas| * sqrt((std_ion/mean_ion)^2 + (std_tot/mean_tot)^2)``.

    The build **hard-fails** (E2.1) if any stacked f_gas value is non-finite
    or falls outside ``fgas_sanity_range``. The per-simulation suppression
    curve is read from a single shared file.

    Args:
        base_path_template (str): Template for each simulation's data directory,
            with a single ``{}`` placeholder for the simulation index, e.g.
            ``'/path/to/SB35_{}/data/'``. The profile filename is appended.
        suppression_path (str | Path): Path to the ``Ptot_Pdm_ratio_snap{NN}.npz``
            file holding keys ``'k'`` (n_k,, in h/Mpc) and ``'Ptot_Pdm_ratio'``
            (n_sims, n_k). Must correspond to ``snapshot``.
        snapshot (int): Snapshot number, threaded into the per-simulation profile
            filename (e.g. 74 -> z=0.47, 82 -> z=0.21).
        number_densities (Sequence[float], optional): Target number densities in
            (Mpc/h)^-3. Defaults to the five values used in the notebook.
        box_size_mpch (float, optional): Box side length in Mpc/h, used to turn
            number densities into halo counts. Defaults to 50.0.
        sim_ids (Sequence[int], optional): Simulation indices to load. Defaults
            to ``range(n_sims)`` inferred from the suppression file.
        rank_key (str, optional): Halo-ordering key for the number-density cut.
            Defaults to ``'halo_mass'`` (M_500c).
        fgas_sanity_range (tuple[float, float], optional): Inclusive build-time
            validation range for every stacked f_gas value; any violation (or
            any non-finite value) hard-fails the build. Defaults to
            :data:`DEFAULT_FGAS_SANITY_RANGE` ([-1.0, 3.0], provisional).
        progress (Callable, optional): Callback ``progress(i, n_total)`` invoked
            per simulation, for a progress bar. Defaults to None.

    Returns:
        FgasSpkDataset: The assembled dataset.

    Raises:
        ValueError: If the suppression file row count does not match the number
            of requested simulations.
    """
    suppression_path = Path(suppression_path)
    _check_suppression_snapshot(suppression_path, snapshot)
    sup_file = np.load(suppression_path)
    k = np.asarray(sup_file["k"])
    suppression_all = np.asarray(sup_file["Ptot_Pdm_ratio"])  # (n_sims, n_k)

    if sim_ids is None:
        sim_ids = list(range(suppression_all.shape[0]))
    sim_ids = list(sim_ids)

    n_halos_per_nd = [int(nd * box_size_mpch**3) for nd in number_densities]
    profile_filename = _PROFILE_FILENAME_TEMPLATE.format(snapshot=snapshot)

    fgas_rows: list[np.ndarray] = []
    fgas_std_rows: list[np.ndarray] = []
    mean_mass_rows: list[np.ndarray] = []
    radii_ref: np.ndarray | None = None
    kept_ids: list[int] = []
    kept_suppression: list[np.ndarray] = []

    for i, sim_id in enumerate(sim_ids):
        profile_path = Path(base_path_template.format(sim_id)) / profile_filename
        loaded = _load_one_simulation(profile_path, rank_key=rank_key)

        if radii_ref is None:
            radii_ref = loaded["radii_mpch"]

        ionized = loaded["ionized"]         # (n_radii, n_proj, n_halos), desc mass
        total = loaded["total"]             # same shape and ordering
        fb = loaded["fb"]
        masses = loaded["halo_masses"]      # (n_halos,), descending
        n_halos_total = ionized.shape[2]

        per_nd_mean = []
        per_nd_std = []
        per_nd_mass = []
        for n_halos in n_halos_per_nd:
            n_use = min(n_halos, n_halos_total)
            ion_block = ionized[:, :, :n_use]        # (n_radii, n_proj, n_use)
            tot_block = total[:, :, :n_use]
            # Ratio of stacked profiles (FGAS_DEFINITION): pool halos and
            # projections in one unweighted nanmean per bin, divide once, then
            # normalise by f_b. The stacked denominator is far from zero, so no
            # per-halo zero-crossing can poison the value.
            mean_ion = np.nanmean(ion_block, axis=(1, 2))   # (n_radii,)
            mean_tot = np.nanmean(tot_block, axis=(1, 2))
            per_nd_mean.append(mean_ion / mean_tot / fb)
            # Reference-notebook error propagation: the two stacks' pooled
            # standard deviations combined in quadrature on the ratio.
            std_ion = np.nanstd(ion_block, axis=(1, 2))
            std_tot = np.nanstd(tot_block, axis=(1, 2))
            per_nd_std.append(
                np.abs(per_nd_mean[-1])
                * np.sqrt((std_ion / mean_ion) ** 2 + (std_tot / mean_tot) ** 2)
            )
            per_nd_mass.append(np.nanmean(masses[:n_use]))   # over distinct halos

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

    # E2.1 build-time validation: a single non-finite or out-of-range stacked
    # f_gas value fails the whole build, loudly and with coordinates -- the
    # Stage 3.0 corruption must never ship silently again.
    fgas_all = np.stack(fgas_rows)                    # (n_sims, n_nd, n_radii)
    bad = ~np.isfinite(fgas_all)
    lo, hi = float(fgas_sanity_range[0]), float(fgas_sanity_range[1])
    bad |= (fgas_all < lo) | (fgas_all > hi)
    if bad.any():
        sim_i, nd_i, bin_i = np.nonzero(bad)
        listing = ", ".join(
            f"(sim {kept_ids[s]}, nd {n}, bin {b}: {fgas_all[s, n, b]:.6g})"
            for s, n, b in list(zip(sim_i, nd_i, bin_i))[:10]
        )
        raise ValueError(
            f"Built f_gas violates the sanity range [{lo}, {hi}] or is "
            f"non-finite in {int(bad.sum())} cell(s): {listing}"
            + (" ..." if bad.sum() > 10 else "")
            + ". The build is rejected (E2.1); inspect the source profiles or "
            "adjust fgas_sanity_range deliberately."
        )

    return FgasSpkDataset(
        radii_mpch=np.asarray(radii_ref),
        number_densities=np.asarray(number_densities),
        fgas=fgas_all,                       # (n_sims, n_nd, n_radii)
        fgas_std=np.stack(fgas_std_rows),
        k=k,
        suppression=suppression,
        sim_ids=np.asarray(kept_ids),
        mean_halo_mass=np.stack(mean_mass_rows),  # (n_sims, n_nd)
        rank_key=rank_key,
        camels_params=_load_params(params_path, kept_ids),
        snapshot=snapshot,
    )


def build_fgas_spk_from_compiled(
    data_dir: str | Path,
    suppression_path: str | Path,
    snapshot: int,
    number_densities: Sequence[float] = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3),
    box_size_mpch: float = 50.0,
    name_extra: str = "_sixth_gen_makeMap",
    params_path: str | Path | None = None,
) -> FgasSpkDataset:
    """Fast path: load the pre-compiled per-nd ratio files instead of raw profiles.

    Reads the aggregated ``{name_extra}DeltaSigma_kSZ_over_DeltaSigma_total_
    profiles_snap{NN}_nd_{i}_n_{N}.npz`` files written by the notebook (each
    holding the per-simulation f_gas ratio of shape (n_sims, n_radii)) and stacks
    them into the same :class:`FgasSpkDataset` layout. This is much faster than
    re-reading 1024 raw profile files, at the cost of flexibility.

    Warning:
        This inherits whatever halo ranking was frozen when the compiled files
        were produced. If they came from the unsorted CAMELS-fork selection, the
        number-density bins are top-N by FoF mass, not by the selection proxy,
        and that cannot be corrected here. ``mean_halo_mass`` is unavailable
        (the compiled files do not store per-halo masses) and is returned as
        NaN. Use :func:`build_fgas_spk_dataset` when correctness of the binning
        matters; use this for quick iteration when it does not.

    Args:
        data_dir (str | Path): Directory holding the compiled ``.npz`` files and
            the suppression file.
        suppression_path (str | Path): Path to ``Ptot_Pdm_ratio_snap{NN}.npz``.
        snapshot (int): Snapshot number, threaded into the compiled filenames
            (e.g. 74 -> z=0.47, 82 -> z=0.21). Must correspond to the data.
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
    suppression_path = Path(suppression_path)
    _check_suppression_snapshot(suppression_path, snapshot)
    sup_file = np.load(suppression_path)
    k = np.asarray(sup_file["k"])
    suppression = np.asarray(sup_file["Ptot_Pdm_ratio"])  # (n_sims, n_k)

    n_halos_per_nd = [int(nd * box_size_mpch**3) for nd in number_densities]

    fgas_per_nd: list[np.ndarray] = []
    fgas_std_per_nd: list[np.ndarray] = []
    radii_ref: np.ndarray | None = None
    for nd_idx, n_halos in enumerate(n_halos_per_nd):
        fname = (
            f"{name_extra}DeltaSigma_kSZ_over_DeltaSigma_total_profiles"
            f"_snap{snapshot}_nd_{nd_idx}_n_{n_halos}.npz"
        )
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
    if n_sims > suppression.shape[0]:
        raise ValueError(
            f"Compiled fgas files contain {n_sims} sims but the suppression file "
            f"has only {suppression.shape[0]} rows; cannot pair them."
        )
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
        snapshot=snapshot,
    )


if __name__ == "__main__":
    DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
    BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
    SNAPSHOT, REDSHIFT = 74, 0.47  # change to (82, 0.21) for the snap82 dataset

    # Correct (slower) path: rebuilds bins from raw per-halo data, fixing the sort.
    dataset = build_fgas_spk_dataset(
        base_path_template=BASE,
        suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
        snapshot=SNAPSHOT,
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
        snapshot=SNAPSHOT,
        redshift=REDSHIFT,
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
    # fast = build_fgas_spk_from_compiled(
    #     data_dir=DATA_DIR,
    #     suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
    #     snapshot=SNAPSHOT,
    #     name_extra="_sixth_gen_makeMap",
    # )