"""Tests for the extracted data contract in ``fgas_spk_schema``.

The hard requirement is that the on-disk saved format is *frozen*: the same
array keys, the same ``__meta__`` keys, a ``.yaml`` sidecar beside the ``.npz``,
the same filename stem, and ``SCHEMA_VERSION == "1.0"``. These tests pin all of
that, plus the new :func:`resolve_dataset_path` behaviour.
"""

from pathlib import Path

import numpy as np
import pytest

import fgas_spk.schema as S


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
    # stamp_param_names=False: pin the base format for a param matrix without
    # stamped names ("TEST" is not a registered suite). The stamped-format keys
    # are pinned separately in test_stamped_param_names_add_array_and_meta_keys.
    out = S.save_dataset(
        _make_dataset(with_params=True), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
        stamp_param_names=False,
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
        stamp_param_names=False,
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
        stamp_param_names=False,
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


# --- parameter-name stamping (optional, additive) --------------------------

import json  # noqa: E402  (kept local to the stamping tests below)

import fgas_spk.camels_params as CP  # noqa: E402

SB35 = "CAMELS-IllustrisTNG-L50n512-SB35"


def _make_dataset_with_names(names):
    """A with-params dataset carrying its own column names (len == n_params)."""
    ds = _make_dataset(with_params=True)
    # The fixture has n_params = 4; size the names to match.
    ds.camels_params = ds.camels_params[:, : len(names)]
    ds.camels_param_names = tuple(names)
    return ds


def test_dataset_carried_names_are_stamped_for_any_suite(tmp_path):
    # A dataset carrying its own names stamps without needing the registry, so
    # an unregistered suite works when the names travel on the object.
    ds = _make_dataset_with_names(["p0", "p1", "p2", "p3"])
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    keys = set(np.load(out, allow_pickle=False).files)
    assert "camels_param_names" in keys
    meta = json.loads(str(np.load(out, allow_pickle=False)["__meta__"]))
    assert meta["camels_param_names"] == ["p0", "p1", "p2", "p3"]
    back = S.load_dataset(out)
    assert back.camels_param_names == ("p0", "p1", "p2", "p3")


def test_stamp_false_omits_names_and_keeps_base_format(tmp_path):
    ds = _make_dataset_with_names(["p0", "p1", "p2", "p3"])
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
        stamp_param_names=False,
    )
    keys = set(np.load(out, allow_pickle=False).files)
    assert "camels_param_names" not in keys
    meta = json.loads(str(np.load(out, allow_pickle=False)["__meta__"]))
    assert "camels_param_names" not in meta
    assert S.load_dataset(out).camels_param_names is None


def test_stamp_from_registry_for_registered_suite(tmp_path):
    # A real SB35-sized matrix stamps the registry names by suite alone.
    expected = CP.param_names_for(SB35)
    ds = _make_dataset(with_params=True)
    rng = np.random.default_rng(0)
    ds.camels_params = rng.random((ds.camels_params.shape[0], len(expected)))
    out = S.save_dataset(
        ds, project_root=tmp_path, suite=SB35,
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    assert S.load_dataset(out).camels_param_names == expected


def test_stamp_unknown_suite_raises(tmp_path):
    with pytest.raises(KeyError, match="No CAMELS parameter-name list"):
        S.save_dataset(
            _make_dataset(with_params=True), project_root=tmp_path, suite="TEST",
            snapshot=74, redshift=0.47, source="rhliu", tag="v1",
        )


def test_stamp_length_mismatch_raises(tmp_path):
    # Registered suite, but the matrix has the wrong number of columns.
    with pytest.raises(ValueError, match="stamp_param_names"):
        S.save_dataset(
            _make_dataset(with_params=True), project_root=tmp_path, suite=SB35,
            snapshot=74, redshift=0.47, source="rhliu", tag="v1",
        )  # fixture has 4 columns, SB35 registers 35


def test_param_less_dataset_ignores_stamp_flag(tmp_path):
    # No camels_params -> nothing to stamp, even with the default flag on.
    out = S.save_dataset(
        _make_dataset(with_params=False), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )
    keys = set(np.load(out, allow_pickle=False).files)
    assert "camels_param_names" not in keys
    assert S.load_dataset(out).camels_param_names is None
