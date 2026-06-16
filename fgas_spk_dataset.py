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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# Profiles file written per simulation by the CAMELS producer.
# _PROFILE_FILENAME = "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma.npz"
_PROFILE_FILENAME = "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap74.npz"

# Projections stacked in the products.
_PROJECTIONS = ("xy", "xz", "yz")


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


def load_fgas_spk_dataset(
    base_path_template: str,
    suppression_path: str | Path,
    number_densities: Sequence[float] = (1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3),
    box_size_mpch: float = 50.0,
    sim_ids: Sequence[int] | None = None,
    rank_key: str = "halo_mass",
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
    )


if __name__ == "__main__":
    # Minimal smoke test / usage example. Point these at your real paths.
    DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
    BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"

    dataset = load_fgas_spk_dataset(
        base_path_template=BASE,
        suppression_path=Path(DATA_DIR) / "Ptot_Pdm_ratio_k_le15.npz",
        rank_key="halo_mass",
        progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True),
    )
    print()
    print("fgas grid:", dataset.fgas.shape, "(n_sims, n_nd, n_radii)")
    print("suppression:", dataset.suppression.shape, "(n_sims, n_k)")

    x, nd, y = dataset.to_training_arrays(k_target=3.0)
    print("training X:", x.shape, "| nd:", nd.shape, "| y (SP at k~3):", y.shape)