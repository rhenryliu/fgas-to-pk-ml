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

import fgas_spk.loader as L
import fgas_spk.schema as S

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
    # 3 sims x 5 nds = 15 rows. Inputs are split by modality: X is the profile
    # alone (6 radii), nd goes to X_cond (1 col), CAMELS params to X_params (4);
    # full 8 k in the curve target.
    assert td.X.shape == (15, 6)
    assert td.X_cond.shape == (15, 1)
    assert td.X_params.shape == (15, 4)
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
    # X is the cropped profile alone; the nd conditioning feature lives in X_cond.
    assert td.X.shape[1] == expected
    assert td.X_cond.shape[1] == 1
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
    # X is always the profile alone; the feature flags resize X_cond / X_params,
    # never X. No conditioning, no params.
    td0 = L.load_training_data(L.DataConfig(
        path=npz, include_nd_feature=False, include_camels_params=False))
    assert td0.X.shape[1] == n_radii
    assert td0.X_cond is None
    assert td0.X_params is None
    # nd only in X_cond (default); CAMELS params on by default in X_params.
    td1 = L.load_training_data(L.DataConfig(path=npz))
    assert td1.X.shape[1] == n_radii
    assert td1.X_cond.shape[1] == 1                    # nd
    assert td1.X_params.shape[1] == n_params
    # nd + mean halo mass in X_cond.
    td2 = L.load_training_data(L.DataConfig(path=npz, include_mean_halo_mass=True))
    assert td2.X.shape[1] == n_radii
    assert td2.X_cond.shape[1] == 2                    # nd, mean halo mass
    # CAMELS params can be turned off -> X_params is None.
    td3 = L.load_training_data(L.DataConfig(path=npz, include_camels_params=False))
    assert td3.X.shape[1] == n_radii
    assert td3.X_params is None


def test_camels_params_requested_but_absent_raises(tmp_path):
    npz = str(_save(tmp_path, with_params=False))
    with pytest.raises(ValueError, match="camels_params"):
        L.load_training_data(L.DataConfig(path=npz, include_camels_params=True))


def test_mean_halo_mass_feature_value(tmp_path):
    npz = str(_save(tmp_path))
    td = L.load_training_data(L.DataConfig(path=npz, include_mean_halo_mass=True))
    # X_cond columns are (nd, mean halo mass); the last is the mean halo mass
    # (constant 1e13 in the fixture).
    np.testing.assert_allclose(td.X_cond[:, -1], 1e13)


# --- row-for-row alignment -------------------------------------------------

def test_sim_index_and_nd_align_with_X_and_y(tmp_path):
    """For every row, the profile and target must match its (sim, nd) labels."""
    ds = _make_dataset(with_params=False)
    td = L.load_training_data(L.DataConfig(
        path=str(_save(tmp_path, with_params=False)), include_camels_params=False))
    sim_ids = list(ds.sim_ids)
    nds = list(ds.number_densities)
    for row in range(td.X.shape[0]):
        p = sim_ids.index(int(td.sim_index[row]))      # sim position
        j = int(np.argmin(np.abs(np.array(nds) - td.nd[row])))  # nd position
        # X is the profile alone and matches fgas[p, j].
        np.testing.assert_allclose(td.X[row], ds.fgas[p, j])
        # Curve target matches that simulation's suppression row.
        np.testing.assert_allclose(td.y[row], ds.suppression[p])
        # The nd conditioning scalar (X_cond column 0) equals the row's nd value.
        np.testing.assert_allclose(td.X_cond[row, 0], td.nd[row])


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
    assert td.X.shape == (15, 6)


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


def test_k_range_with_non_k_range_mode_raises():
    # k_range set under target_mode='curve' is contradictory: the k-field would
    # be silently ignored. Fail at construction.
    with pytest.raises(ValueError, match="k_range"):
        L.DataConfig(target_mode="curve", k_range=(0.5, 5.0))


def test_k_range_with_single_k_mode_raises():
    # k_range contradicts single_k even when k_target is also supplied.
    with pytest.raises(ValueError, match="k_range"):
        L.DataConfig(target_mode="single_k", k_target=3.0, k_range=(0.5, 5.0))


def test_k_target_with_non_single_k_mode_raises():
    with pytest.raises(ValueError, match="k_target"):
        L.DataConfig(target_mode="curve", k_target=3.0)


def test_valid_target_mode_k_field_combinations_construct():
    # The three consistent combinations must build without error.
    assert L.DataConfig(target_mode="curve").k_range is None
    assert L.DataConfig(target_mode="k_range", k_range=(0.5, 5.0)).k_target is None
    assert L.DataConfig(target_mode="single_k", k_target=3.0).k_range is None


def test_single_k_empty_k_raises(tmp_path):
    ds = _make_dataset(with_params=False)
    ds.k = np.array([])                                  # no k bins
    ds.suppression = np.zeros((ds.suppression.shape[0], 0))
    out = S.save_dataset(
        ds, project_root=tmp_path, suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="vk0",
    )
    cfg = L.DataConfig(path=str(out), target_mode="single_k", k_target=1.0,
                       include_camels_params=False)
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
    # A hand-built TrainingData with config=None (and no X_cond / X_params)
    # must not crash _summarize.
    td = L.TrainingData(
        X=np.zeros((2, 3)), X_cond=None, X_params=None, y=np.zeros((2, 4)),
        nd=np.array([1.0e-4, 1.0e-4]), sim_index=np.array([1, 1]),
        k=np.linspace(0.1, 1.0, 4), radii_mpch=np.linspace(0.1, 1.0, 3),
        source_path="x.npz",
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


# --- entry-point read-root fill (null project_root through the loader CLI) --

def _null_root_config(tag: str = "v1") -> "L.DataConfig":
    """A store-field DataConfig with project_root EXPLICITLY null."""
    return L.DataConfig(
        project_root=None, suite="TEST", snapshot=74, redshift=0.47,
        source="rhliu", rank="m500", tag=tag,
    )


def test_cli_fills_null_project_root_from_data_root_env(tmp_path, monkeypatch, capsys):
    # A null project_root must resolve via $FGAS_DATA_ROOT at the CLI entry point
    # -- not raise "missing project_root".
    _save(tmp_path, tag="v1")
    monkeypatch.setenv("FGAS_DATA_ROOT", str(tmp_path))
    cfg_path = _null_root_config().to_yaml(tmp_path / "config_data.yaml")
    rc = L.main(["--config", str(cfg_path)])
    assert rc == 0
    assert "resolved path" in capsys.readouterr().out


def test_cli_honours_data_root_override(tmp_path, monkeypatch, capsys):
    # The --data-root override fills the null project_root even with no env var.
    _save(tmp_path, tag="v1")
    monkeypatch.delenv("FGAS_DATA_ROOT", raising=False)
    cfg_path = _null_root_config().to_yaml(tmp_path / "config_data.yaml")
    rc = L.main(["--config", str(cfg_path), "--data-root", str(tmp_path)])
    assert rc == 0
    assert "resolved path" in capsys.readouterr().out
