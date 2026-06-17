"""Synthetic tests for the loader fixes in ``fgas_spk_builder.py``.

No real CAMELS data is required; every fixture is built on the fly. Coverage
maps to the code-review fixes:

- the snapshot/suppression filename cross-check in both loaders,
- the ``snapshot`` field round-trip through save/load and its save-time
  cross-check,
- the compiled-loader lower-bound guard on suppression rows.
"""

from pathlib import Path

import numpy as np
import pytest

import fgas_spk_builder as m


# --- builders --------------------------------------------------------------

def _write_suppression(path: Path, n_sims: int = 4, n_k: int = 8) -> None:
    """Write a minimal ``Ptot_Pdm_ratio`` file with ``n_sims`` rows."""
    np.savez(path, k=np.linspace(0.1, 10, n_k),
             Ptot_Pdm_ratio=np.ones((n_sims, n_k)))


def _write_compiled(
    data_dir: Path,
    number_densities,
    n_sims: int,
    snapshot: int = 74,
    box_size_mpch: float = 50.0,
    n_radii: int = 10,
    name_extra: str = "_sixth_gen_makeMap",
) -> None:
    """Write the per-nd compiled ratio files the fast loader expects."""
    radii = np.linspace(0.1, 5.0, n_radii)
    for i, nd in enumerate(number_densities):
        n = int(nd * box_size_mpch ** 3)
        fname = (
            f"{name_extra}DeltaSigma_kSZ_over_DeltaSigma_total_profiles"
            f"_snap{snapshot}_nd_{i}_n_{n}.npz"
        )
        np.savez(
            data_dir / fname,
            prof=np.ones((n_sims, n_radii)),
            prof_std=np.zeros((n_sims, n_radii)),
            radii=radii,
        )


def _make_dataset(snapshot: int | None = 74) -> m.FgasSpkDataset:
    """Build a small in-memory dataset for save/load round-trips."""
    n_sims, n_nd, n_radii, n_k = 3, 2, 5, 8
    rng = np.random.default_rng(0)
    return m.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4]),
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.random((n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        snapshot=snapshot,
    )


# --- #1 snapshot/suppression cross-check -----------------------------------

def test_check_suppression_snapshot_matching_ok():
    # Matching snapshot token must not raise.
    m._check_suppression_snapshot(Path("Ptot_Pdm_ratio_snap74.npz"), 74)


def test_check_suppression_snapshot_mismatch_raises():
    with pytest.raises(ValueError, match="snap74"):
        m._check_suppression_snapshot(Path("Ptot_Pdm_ratio_snap74.npz"), 82)


def test_check_suppression_snapshot_no_token_unchecked():
    # A filename without a ``snap<NN>`` token (e.g. the v0 naming) is left alone.
    m._check_suppression_snapshot(Path("Ptot_Pdm_ratio_k_le15.npz"), 74)


def test_compiled_loader_rejects_snapshot_mismatch(tmp_path):
    # The guard fires before the per-nd file-read loop, so no compiled files needed.
    sup = tmp_path / "Ptot_Pdm_ratio_snap74.npz"
    _write_suppression(sup)
    with pytest.raises(ValueError, match="snap74"):
        m.build_fgas_spk_from_compiled(tmp_path, sup, snapshot=82)


# --- #3 compiled-loader lower-bound guard ----------------------------------

def test_compiled_loader_too_few_suppression_rows_raises(tmp_path):
    nds = (1.0e-4, 2.8e-4)
    _write_compiled(tmp_path, nds, n_sims=6)
    sup = tmp_path / "Ptot_Pdm_ratio_snap74.npz"
    _write_suppression(sup, n_sims=4)  # fewer rows than the 6 compiled sims
    with pytest.raises(ValueError, match="only 4"):
        m.build_fgas_spk_from_compiled(
            tmp_path, sup, snapshot=74, number_densities=nds
        )


def test_compiled_loader_truncates_longer_suppression(tmp_path):
    nds = (1.0e-4, 2.8e-4)
    _write_compiled(tmp_path, nds, n_sims=6)
    sup = tmp_path / "Ptot_Pdm_ratio_snap74.npz"
    _write_suppression(sup, n_sims=10)  # more rows than fgas -> truncate to 6
    ds = m.build_fgas_spk_from_compiled(
        tmp_path, sup, snapshot=74, number_densities=nds
    )
    assert ds.fgas.shape[0] == 6
    assert ds.suppression.shape[0] == 6
    assert ds.snapshot == 74


# --- #2 snapshot field round-trip + save-time cross-check ------------------

def test_snapshot_round_trip(tmp_path):
    ds = _make_dataset(snapshot=74)
    out = m.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="vtest",
    )
    assert m.load_dataset(out).snapshot == 74


def test_save_dataset_snapshot_cross_check_raises(tmp_path):
    ds = _make_dataset(snapshot=74)
    with pytest.raises(ValueError, match="contradicts dataset.snapshot"):
        m.save_dataset(
            ds, project_root=tmp_path, suite="TEST",
            snapshot=82, redshift=0.21, source="rhliu", tag="vbad",
        )


def test_save_dataset_none_snapshot_ok_and_surfaced_from_meta(tmp_path):
    ds = _make_dataset(snapshot=None)  # dataset built without a snapshot
    out = m.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=82, redshift=0.21, source="rhliu", tag="vnone",
    )
    # No cross-check error, and the snapshot is still surfaced from the metadata.
    assert m.load_dataset(out).snapshot == 82
