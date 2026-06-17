"""Tests for the extracted data contract in ``fgas_spk_schema``.

The hard requirement is that the on-disk saved format is *frozen*: the same
array keys, the same ``__meta__`` keys, a ``.yaml`` sidecar beside the ``.npz``,
the same filename stem, and ``SCHEMA_VERSION == "1.0"``. These tests pin all of
that, plus the new :func:`resolve_dataset_path` behaviour.
"""

from pathlib import Path

import numpy as np
import pytest

import fgas_spk_schema as S


# --- builders --------------------------------------------------------------

def _make_dataset(with_params: bool = False, snapshot: int | None = 74) -> S.FgasSpkDataset:
    """Build a small in-memory dataset for save/load round-trips."""
    n_sims, n_nd, n_radii, n_k, n_params = 3, 2, 5, 8, 4
    rng = np.random.default_rng(0)
    return S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4]),
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.random((n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        camels_params=rng.random((n_sims, n_params)) if with_params else None,
        snapshot=snapshot,
    )


# --- frozen on-disk format -------------------------------------------------

_EXPECTED_ARRAY_KEYS = {
    "radii_mpch", "number_densities", "fgas", "fgas_std", "k", "suppression",
    "sim_ids", "mean_halo_mass", "__meta__",
}

_EXPECTED_META_KEYS = {
    "schema_version", "created_utc", "suite", "snapshot", "redshift", "source",
    "rank_key", "tag", "n_sims", "n_number_densities", "n_radii", "n_k",
    "number_densities", "units", "cosmology_note", "arrays", "provenance", "notes",
}


def test_schema_version_unchanged():
    assert S.SCHEMA_VERSION == "1.0"


def test_saved_npz_array_keys_with_params(tmp_path):
    out = S.save_dataset(
        _make_dataset(with_params=True), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    keys = set(np.load(out, allow_pickle=False).files)
    assert keys == _EXPECTED_ARRAY_KEYS | {"camels_params"}


def test_saved_npz_array_keys_without_params(tmp_path):
    out = S.save_dataset(
        _make_dataset(with_params=False), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    keys = set(np.load(out, allow_pickle=False).files)
    assert keys == _EXPECTED_ARRAY_KEYS
    assert "camels_params" not in keys


def test_saved_meta_keys_frozen(tmp_path):
    import json
    out = S.save_dataset(
        _make_dataset(with_params=True), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    meta = json.loads(str(np.load(out, allow_pickle=False)["__meta__"]))
    assert set(meta) == _EXPECTED_META_KEYS
    assert meta["schema_version"] == "1.0"


def test_filename_stem_unchanged(tmp_path):
    out = S.save_dataset(
        _make_dataset(), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    assert out.name == "fgas_spk__TEST__snap074_z0.47__src-rhliu__rank-m500__v1.npz"


def test_yaml_sidecar_written_beside_npz(tmp_path):
    out = S.save_dataset(
        _make_dataset(), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    sidecar = out.with_suffix(".yaml" if S._HAS_YAML else ".json")
    assert sidecar.exists()
    assert sidecar.parent == out.parent


def test_save_load_round_trip_values(tmp_path):
    ds = _make_dataset(with_params=True)
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    back = S.load_dataset(out)
    np.testing.assert_allclose(back.fgas, ds.fgas)
    np.testing.assert_allclose(back.suppression, ds.suppression)
    np.testing.assert_allclose(back.camels_params, ds.camels_params)
    assert back.rank_key == ds.rank_key
    assert back.snapshot == ds.snapshot


# --- resolve_dataset_path --------------------------------------------------

def test_resolve_explicit_tag_matches_save(tmp_path):
    ds = _make_dataset()
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    resolved = S.resolve_dataset_path(
        tmp_path, "TEST", 74, 0.47, "rhliu", "m500", "v1",
    )
    assert resolved == out


def _touch_tags(tmp_path, tags):
    """Create empty .npz files for the given tags in a TEST bucket."""
    bucket = S.dataset_dir(tmp_path, "TEST", 74, 0.47, "rhliu")
    bucket.mkdir(parents=True, exist_ok=True)
    for t in tags:
        stem = S._DATASET_STEM_TEMPLATE.format(
            suite="TEST", snapshot=74, redshift=0.47,
            source="rhliu", rank="m500", tag=t,
        )
        (bucket / f"{stem}.npz").write_bytes(b"")
    return bucket


def test_resolve_latest_picks_newest_date(tmp_path):
    _touch_tags(tmp_path, ["20260101", "20260615", "20251231"])
    resolved = S.resolve_dataset_path(
        tmp_path, "TEST", 74, 0.47, "rhliu", "m500", "latest",
    )
    assert resolved.name.endswith("__20260615.npz")


def test_resolve_latest_picks_max_version_numerically(tmp_path):
    # v10 must beat v2 (numeric, not lexicographic).
    _touch_tags(tmp_path, ["v1", "v2", "v10"])
    resolved = S.resolve_dataset_path(
        tmp_path, "TEST", 74, 0.47, "rhliu", "m500", "latest",
    )
    assert resolved.name.endswith("__v10.npz")


def test_resolve_latest_mixed_tags_raises(tmp_path):
    _touch_tags(tmp_path, ["20260101", "v2"])
    with pytest.raises(ValueError, match="mixed tags"):
        S.resolve_dataset_path(tmp_path, "TEST", 74, 0.47, "rhliu", "m500", "latest")


def test_resolve_latest_no_matches_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.resolve_dataset_path(tmp_path, "TEST", 74, 0.47, "rhliu", "m500", "latest")
