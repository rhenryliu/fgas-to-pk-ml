"""Worked example: build and save the f_gas(R) -> SP(k) dataset from raw profiles.

This mirrors the ``__main__`` block of :mod:`fgas_spk.builder` as a runnable
script. It rebuilds the number-density bins from the raw per-simulation profile
files via :func:`fgas_spk.builder.build_fgas_spk_dataset` (the correct, slower
path that re-ranks halos by mass before the cut), prints a few shape checks, and
saves the assembled dataset into the scratch store under a self-describing name.
This is the path that produced the on-disk ``src-lindajin`` datasets.

The input and output locations are the same hardcoded NERSC paths as the original
``builder.py`` ``__main__``: the ``lindajin`` scratch for the raw profiles, and
the ``rhliu`` project scratch for the output. Edit the constants in :func:`main`
to target the other snapshot (82, z=0.21) or a different store.

Run from the repository root (in the ``fgas-ml`` env)::

    python scripts/demo_builder.py

Note:
    This reads ~1024 raw profile files and writes a dataset ``.npz`` to the
    scratch store; it is the slow, correct path. ``save_dataset`` refuses to
    overwrite an existing file -- change ``tag`` or pass ``overwrite=True`` to
    re-save. Use the compiled fast path (``build_fgas_spk_from_compiled``) only
    for quick iteration when binning correctness does not matter.
"""

from __future__ import annotations

from pathlib import Path

from fgas_spk.builder import (
    build_fgas_spk_dataset,
    ensure_store_dirs,
    save_dataset,
)


def main() -> int:
    """Build the dataset from raw profiles, save it, and print shape checks.

    Returns:
        int: Process exit status (0 on success).
    """
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
