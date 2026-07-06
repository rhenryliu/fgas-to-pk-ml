"""Tests for the Branch-A gas-fraction statistic (ratio of stacked profiles).

Coverage (spec amendment 3, E2.3): (i) a regression test on synthetic per-halo
profiles containing one near-zero ``Delta Sigma_total`` cell among healthy
halos -- the sim-429 pathology in miniature -- asserting the built f_gas stays
in the sanity range and matches the hand-computed ratio-of-stacks value, while
the superseded per-halo-ratio statistic on the same input explodes; (ii) the
E2.1 build-time guard hard-fails a genuinely poisoned stack; (iii) the
``fgas_definition`` stamp and realized-range metadata are written through
``save_dataset(extra_meta=...)`` and protected against reserved-key
collisions.

The two pre-existing ``r12_mpch`` failures in ``test_fgas_spk_builder.py``
remain out of scope (prior amendment, D1).
"""

from pathlib import Path

import json
import numpy as np
import pytest

from fgas_spk.builder import (
    DEFAULT_FGAS_SANITY_RANGE,
    FGAS_DEFINITION,
    build_fgas_spk_dataset,
    fgas_meta_stamp,
)
from fgas_spk.schema import save_dataset

_N_RADII = 3
_N_HALOS = 8
_FB = 0.16


def _write_sim(
    base: Path, sim_id: int, total: np.ndarray, ionized: np.ndarray
) -> None:
    """Write one synthetic per-sim profile archive in the producer layout."""
    data_dir = base / f"SB35_{sim_id}" / "data"
    data_dir.mkdir(parents=True)
    arrays = {
        "fb": np.array(_FB),
        "halo_masses": np.linspace(5e13, 1e13, _N_HALOS),  # already descending
        "r12_mpch": np.array([0.5, 2.0, 8.0]),
    }
    for i, proj in enumerate(("xy", "xz", "yz")):
        arrays[f"prof1_total_DSigma_kpch2_{proj}"] = total[:, i, :]
        arrays[f"prof2_ionized_gas_DSigma_kpch2_{proj}"] = ionized[:, i, :]
    np.savez(
        data_dir / "Profiles_tau-CAP_total-DSigma_ionized_gas-DSigma_snap74.npz",
        **arrays,
    )


def _write_suppression(path: Path, n_sims: int) -> Path:
    """Write a minimal suppression file for ``n_sims`` simulations."""
    out = path / "Ptot_Pdm_ratio_snap74.npz"
    np.savez(out, k=np.array([0.5, 1.0]),
             Ptot_Pdm_ratio=np.ones((n_sims, 2)) * 0.9)
    return out


def _healthy_profiles(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-halo (n_radii, n_proj, n_halos) stacks with healthy denominators."""
    rng = np.random.default_rng(seed)
    total = 1e5 * (1.0 + 0.2 * rng.random((_N_RADII, 3, _N_HALOS)))
    ionized = 0.12 * total * (1.0 + 0.1 * rng.random(total.shape))
    return total, ionized


def test_ratio_of_stacks_survives_near_zero_denominator_cell(tmp_path):
    total, ionized = _healthy_profiles()
    # The sim-429 pathology in miniature: one cell's Delta Sigma_total sits at
    # a near-zero crossing while its ionized value stays ordinary.
    total[1, 1, 3] = -0.3
    ionized[1, 1, 3] = 4.0e4
    _write_sim(tmp_path, 0, total, ionized)
    sup = _write_suppression(tmp_path, 1)

    dataset = build_fgas_spk_dataset(
        base_path_template=str(tmp_path / "SB35_{}" / "data") + "/",
        suppression_path=sup,
        snapshot=74,
        sim_ids=[0],
    )
    # Hand-computed ratio of stacks (all nd clamp to the 8 available halos, so
    # every nd row is identical): pooled mean over (proj, halo), divide, / fb.
    expected = (
        ionized.mean(axis=(1, 2)) / total.mean(axis=(1, 2)) / _FB
    )
    assert dataset.fgas.shape == (1, 5, _N_RADII)
    for nd in range(5):
        np.testing.assert_allclose(dataset.fgas[0, nd], expected, rtol=1e-12)
    lo, hi = DEFAULT_FGAS_SANITY_RANGE
    assert (dataset.fgas >= lo).all() and (dataset.fgas <= hi).all()
    assert np.isfinite(dataset.fgas_std).all()

    # The superseded statistic (mean of per-halo ratios) explodes on the same
    # input -- the regression this fix exists to prevent.
    old_stat = np.nanmean(ionized / total / _FB, axis=(1, 2))
    assert np.abs(old_stat[1]) > hi


def test_build_guard_hard_fails_poisoned_stack(tmp_path):
    total, ionized = _healthy_profiles(seed=1)
    # Poison the *stacked* denominator: the whole bin-1 total sums to ~0.
    total[1] = np.linspace(-1.0, 1.0, 3 * _N_HALOS).reshape(3, _N_HALOS)
    _write_sim(tmp_path, 0, total, ionized)
    sup = _write_suppression(tmp_path, 1)

    with pytest.raises(ValueError, match="sanity range"):
        build_fgas_spk_dataset(
            base_path_template=str(tmp_path / "SB35_{}" / "data") + "/",
            suppression_path=sup,
            snapshot=74,
            sim_ids=[0],
        )


def test_definition_stamp_and_realized_range_written(tmp_path):
    total, ionized = _healthy_profiles(seed=2)
    _write_sim(tmp_path, 0, total, ionized)
    sup = _write_suppression(tmp_path, 1)
    dataset = build_fgas_spk_dataset(
        base_path_template=str(tmp_path / "SB35_{}" / "data") + "/",
        suppression_path=sup,
        snapshot=74,
        sim_ids=[0],
    )
    stamp = fgas_meta_stamp(dataset.fgas, DEFAULT_FGAS_SANITY_RANGE)
    out = save_dataset(
        dataset,
        project_root=tmp_path / "store",
        suite="TEST-SUITE",
        snapshot=74,
        redshift=0.47,
        source="synthetic",
        tag="t1",
        stamp_param_names=False,
        extra_meta=stamp,
    )
    meta = json.loads(str(np.load(out, allow_pickle=False)["__meta__"]))
    assert meta["fgas_definition"] == FGAS_DEFINITION
    assert meta["fgas_sanity_range"] == list(DEFAULT_FGAS_SANITY_RANGE)
    realized = meta["fgas_realized_range_per_bin"]
    np.testing.assert_allclose(realized["min"], dataset.fgas.min(axis=(0, 1)))
    np.testing.assert_allclose(realized["max"], dataset.fgas.max(axis=(0, 1)))
    # The human-readable sidecar carries the stamp too.
    sidecar = out.with_suffix(".yaml").read_text()
    assert "fgas_definition" in sidecar

    # Reserved-key collision is rejected.
    with pytest.raises(ValueError, match="collide"):
        save_dataset(
            dataset, project_root=tmp_path / "store2", suite="TEST-SUITE",
            snapshot=74, redshift=0.47, source="synthetic", tag="t2",
            stamp_param_names=False, extra_meta={"suite": "sneaky"},
        )
