"""Tests for the configurable training-data loader ``fgas_spk_loader``.

Every fixture is synthetic. Coverage maps to the Part B verification gate:
shapes for each ``target_mode``; honouring ``number_density_indices``,
``radial_range_mpch``, and the feature flags; row-for-row alignment of
``sim_index`` / ``nd`` with ``X`` / ``y``; the selection/validation rules; the
YAML config round-trip; and the CLI dry-run summary.
"""

from pathlib import Path

import numpy as np
import pytest

import fgas_spk_loader as L
import fgas_spk_schema as S

# The five fixed number densities for the suite (n = int(nd * 50**3)).
NUMBER_DENSITIES = [1.0e-4, 2.8e-4, 5.0e-4, 1.0e-3, 2.4e-3]


# --- builders --------------------------------------------------------------

def _make_dataset(with_params: bool = True) -> S.FgasSpkDataset:
    """Dataset whose values encode (sim, nd, radius) so alignment is checkable.

    ``fgas[s, j, r] = 100*s + 10*j + r`` and ``suppression[s, q] = 1000*s + q``,
    so any flattened row can be traced back to its originating simulation and
    number density.
    """
    n_sims, n_nd, n_radii, n_k, n_params = 3, 5, 6, 8, 4
    s = np.arange(n_sims)[:, None, None]
    j = np.arange(n_nd)[None, :, None]
    r = np.arange(n_radii)[None, None, :]
    fgas = (100 * s + 10 * j + r).astype(float)
    sup = (1000 * np.arange(n_sims)[:, None] + np.arange(n_k)[None, :]).astype(float)
    return S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 3.0, n_radii),
        number_densities=np.array(NUMBER_DENSITIES),
        fgas=fgas,
        fgas_std=np.zeros_like(fgas),
        k=np.linspace(0.1, 10.0, n_k),
        suppression=sup,
        sim_ids=np.array([10, 20, 30]),  # non-trivial ids
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        camels_params=(np.arange(n_sims * n_params).reshape(n_sims, n_params)
                       .astype(float) if with_params else None),
        snapshot=74,
    )


def _save(tmp_path, with_params: bool = True, tag: str = "v1") -> Path:
    return S.save_dataset(
        _make_dataset(with_params), project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag=tag,
    )


# --- target_mode shapes ----------------------------------------------------

def test_curve_mode_shapes(tmp_path):
    td = L.load_training_data(L.DataConfig(path=str(_save(tmp_path))))
    # 3 sims x 5 nds = 15 rows; 6 radii + 1 nd feature = 7 features; full 8 k.
    assert td.X.shape == (15, 7)
    assert td.y.shape == (15, 8)
    assert td.nd.shape == (15,)
    assert td.sim_index.shape == (15,)
    assert td.k.shape == (8,)
    assert td.radii_mpch.shape == (6,)


def test_single_k_mode_shapes_and_nearest_bin(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), target_mode="single_k", k_target=3.0)
    td = L.load_training_data(cfg)
    assert td.y.shape == (15,)            # scalar target per row
    assert td.k.shape == (1,)
    k_all = np.linspace(0.1, 10.0, 8)
    assert td.k[0] == k_all[int(np.argmin(np.abs(k_all - 3.0)))]


def test_k_range_mode_shapes(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), target_mode="k_range", k_range=(0.5, 5.0))
    td = L.load_training_data(cfg)
    k_all = np.linspace(0.1, 10.0, 8)
    n_sel = int(((k_all >= 0.5) & (k_all <= 5.0)).sum())
    assert td.y.shape == (15, n_sel)
    assert td.k.shape == (n_sel,)
    assert td.k.min() >= 0.5 and td.k.max() <= 5.0


# --- subsets ---------------------------------------------------------------

def test_number_density_index_subset(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), number_density_indices=[0, 2])
    td = L.load_training_data(cfg)
    assert td.X.shape[0] == 3 * 2  # 3 sims x 2 selected nds
    assert sorted(np.unique(td.nd).tolist()) == sorted(
        [NUMBER_DENSITIES[0], NUMBER_DENSITIES[2]]
    )


def test_radial_range_crop(tmp_path):
    radii = np.linspace(0.1, 3.0, 6)
    lo, hi = 0.5, 2.0
    expected = int(((radii >= lo) & (radii <= hi)).sum())
    cfg = L.DataConfig(path=str(_save(tmp_path)), radial_range_mpch=(lo, hi))
    td = L.load_training_data(cfg)
    assert td.radii_mpch.shape == (expected,)
    # X = cropped profile (expected cols) + 1 nd feature.
    assert td.X.shape[1] == expected + 1
    assert td.radii_mpch.min() >= lo and td.radii_mpch.max() <= hi


def test_sim_id_subset(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), sim_ids=[10, 30])
    td = L.load_training_data(cfg)
    assert set(np.unique(td.sim_index).tolist()) == {10, 30}
    assert td.X.shape[0] == 2 * 5


# --- feature flags ---------------------------------------------------------

def test_feature_flags_change_width(tmp_path):
    npz = str(_save(tmp_path, with_params=True))
    n_radii, n_params = 6, 4
    # No nd feature -> just the profile.
    td0 = L.load_training_data(L.DataConfig(path=npz, include_nd_feature=False))
    assert td0.X.shape[1] == n_radii
    # nd only (default).
    td1 = L.load_training_data(L.DataConfig(path=npz))
    assert td1.X.shape[1] == n_radii + 1
    # nd + mean halo mass.
    td2 = L.load_training_data(L.DataConfig(path=npz, include_mean_halo_mass=True))
    assert td2.X.shape[1] == n_radii + 2
    # nd + mhm + camels params.
    td3 = L.load_training_data(L.DataConfig(
        path=npz, include_mean_halo_mass=True, include_camels_params=True))
    assert td3.X.shape[1] == n_radii + 2 + n_params


def test_camels_params_requested_but_absent_raises(tmp_path):
    npz = str(_save(tmp_path, with_params=False))
    with pytest.raises(ValueError, match="camels_params"):
        L.load_training_data(L.DataConfig(path=npz, include_camels_params=True))


def test_mean_halo_mass_feature_value(tmp_path):
    npz = str(_save(tmp_path))
    td = L.load_training_data(L.DataConfig(path=npz, include_mean_halo_mass=True))
    # Last column is the mean halo mass (constant 1e13 in the fixture).
    np.testing.assert_allclose(td.X[:, -1], 1e13)


# --- row-for-row alignment -------------------------------------------------

def test_sim_index_and_nd_align_with_X_and_y(tmp_path):
    """For every row, the profile and target must match its (sim, nd) labels."""
    ds = _make_dataset(with_params=False)
    td = L.load_training_data(L.DataConfig(path=str(_save(tmp_path, with_params=False))))
    sim_ids = list(ds.sim_ids)
    nds = list(ds.number_densities)
    for row in range(td.X.shape[0]):
        p = sim_ids.index(int(td.sim_index[row]))      # sim position
        j = int(np.argmin(np.abs(np.array(nds) - td.nd[row])))  # nd position
        # Profile columns (before the appended nd feature) match fgas[p, j].
        np.testing.assert_allclose(td.X[row, :ds.fgas.shape[2]], ds.fgas[p, j])
        # Curve target matches that simulation's suppression row.
        np.testing.assert_allclose(td.y[row], ds.suppression[p])
        # The appended nd feature equals the row's nd value.
        np.testing.assert_allclose(td.X[row, ds.fgas.shape[2]], td.nd[row])


def test_row_order_is_sim_major(tmp_path):
    td = L.load_training_data(L.DataConfig(path=str(_save(tmp_path))))
    # First 5 rows are sim 10 across its 5 nds, then sim 20, then sim 30.
    assert td.sim_index[:5].tolist() == [10] * 5
    assert td.sim_index[5:10].tolist() == [20] * 5


# --- selection / validation ------------------------------------------------

def test_resolve_via_store_fields_and_latest(tmp_path):
    _save(tmp_path, tag="v1")
    cfg = L.DataConfig(
        project_root=str(tmp_path), suite="TEST", snapshot=74, redshift=0.47,
        source="rhliu", rank="m500", tag="latest",
    )
    td = L.load_training_data(cfg)
    assert td.source_path.endswith("__v1.npz")
    assert td.X.shape == (15, 7)


def test_both_selection_modes_raises(tmp_path):
    cfg = L.DataConfig(path="x.npz", project_root="/root")
    with pytest.raises(ValueError, match="not both"):
        cfg.resolve_path()


def test_no_selection_mode_raises():
    with pytest.raises(ValueError, match="missing"):
        L.DataConfig().resolve_path()


def test_missing_store_field_raises():
    cfg = L.DataConfig(project_root="/root", suite="TEST")  # incomplete
    with pytest.raises(ValueError, match="missing"):
        cfg.resolve_path()


def test_single_k_without_k_target_raises(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), target_mode="single_k")
    with pytest.raises(ValueError, match="k_target"):
        L.load_training_data(cfg)


def test_k_range_without_k_range_raises(tmp_path):
    cfg = L.DataConfig(path=str(_save(tmp_path)), target_mode="k_range")
    with pytest.raises(ValueError, match="k_range"):
        L.load_training_data(cfg)


def test_bad_target_mode_raises():
    with pytest.raises(ValueError, match="target_mode"):
        L.DataConfig(target_mode="nonsense")


def test_single_k_empty_k_raises(tmp_path):
    ds = _make_dataset(with_params=False)
    ds.k = np.array([])                                  # no k bins
    ds.suppression = np.zeros((ds.suppression.shape[0], 0))
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="vk0",
    )
    cfg = L.DataConfig(path=str(out), target_mode="single_k", k_target=1.0)
    with pytest.raises(ValueError, match="no k bins"):
        L.load_training_data(cfg)


# --- YAML config round-trip ------------------------------------------------

def test_config_yaml_round_trip(tmp_path):
    cfg = L.DataConfig(
        path="some.npz", number_density_indices=[0, 2],
        radial_range_mpch=(0.5, 2.0), include_mean_halo_mass=True,
        target_mode="k_range", k_range=(0.5, 5.0),
    )
    out = cfg.to_yaml(tmp_path / "cfg.yaml")
    back = L.DataConfig.from_yaml(out)
    assert back == cfg
    # Tuples survive the YAML list round-trip via __post_init__ coercion.
    assert isinstance(back.radial_range_mpch, tuple)
    assert isinstance(back.k_range, tuple)


# --- CLI dry-run -----------------------------------------------------------

def test_summarize_handles_missing_config():
    # A hand-built TrainingData with config=None must not crash _summarize.
    td = L.TrainingData(
        X=np.zeros((2, 3)), y=np.zeros((2, 4)), nd=np.array([1.0e-4, 1.0e-4]),
        sim_index=np.array([1, 1]), k=np.linspace(0.1, 1.0, 4),
        radii_mpch=np.linspace(0.1, 1.0, 3), source_path="x.npz",
    )
    assert "target_mode=?" in L._summarize(td)


def test_cli_dry_run_prints_summary(tmp_path, capsys):
    _save(tmp_path, tag="v1")
    cfg = L.DataConfig(
        project_root=str(tmp_path), suite="TEST", snapshot=74, redshift=0.47,
        source="rhliu", rank="m500", tag="v1",
    )
    cfg_path = cfg.to_yaml(tmp_path / "cfg.yaml")
    rc = L.main(["--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved path" in out
    assert "X shape" in out
    assert "target_mode=curve" in out
