"""End-to-end tests for the training runner ``fgas_spk.train``.

Coverage: a tiny synthetic run drives the runner all the way to a written record
(run dir + checkpoint on a tmp scratch root + a ledger line), and a dedicated
test asserts the grouped-by-``sim_index`` split never puts a simulation in both
the train and the held-out fold. No real CAMELS data is needed.
"""

import json
from pathlib import Path

import numpy as np

import fgas_spk.schema as S
from fgas_spk.experiment import RunConfig, SplitSpec
from fgas_spk.loader import DataConfig
from fgas_spk.train import grouped_split, run_training

_RECORD_FILES = {
    "config_data.yaml", "config_run.yaml", "env.json",
    "metrics.jsonl", "summary.json",
}


# --- builders --------------------------------------------------------------

def _save_dataset(tmp_path: Path, n_sims: int = 10, n_nd: int = 2,
                  n_radii: int = 5, n_k: int = 8) -> Path:
    """Save a small real dataset .npz (no camels_params) and return its path."""
    rng = np.random.default_rng(0)
    ds = S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4])[:n_nd],
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.random((n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        snapshot=74,
    )
    return S.save_dataset(
        ds, project_root=tmp_path / "data", suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )


def _data_config(npz: Path) -> DataConfig:
    # Profile + nd conditioning; the synthetic set carries no CAMELS params.
    return DataConfig(
        path=str(npz), target_mode="curve",
        include_nd_feature=True, include_camels_params=False,
    )


def _run_config() -> RunConfig:
    return RunConfig(
        model="pca_linear",
        model_params={"n_components": 3},
        seed=0,
        split=SplitSpec(train_frac=0.7, val_frac=0.0, test_frac=0.3, seed=0),
    )


# --- end-to-end smoke test -------------------------------------------------

def test_runner_writes_record_checkpoint_and_ledger(tmp_path):
    npz = _save_dataset(tmp_path)
    result = run_training(
        _data_config(npz), _run_config(),
        scratch_root_override=tmp_path / "scratch",
        experiments_root=tmp_path / "experiments",
        write_figures=False,
        device="cpu",
    )

    # Record directory with all five text files.
    assert result.run_dir.is_dir()
    assert {p.name for p in result.run_dir.iterdir()} == _RECORD_FILES
    assert result.run_dir == tmp_path / "experiments" / "runs" / result.run_id

    # Checkpoint on the (tmp) scratch root under models/<run_id>/.
    assert result.checkpoint_path.exists()
    assert result.checkpoint_path == (
        tmp_path / "scratch" / "models" / result.run_id / "model.joblib"
    )

    # Exactly one ledger line, well formed, naming this run.
    ledger = tmp_path / "experiments" / "runs.jsonl"
    lines = ledger.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == result.run_id

    # Held-out evaluation happened.
    assert result.summary["held_out_split"] == "test"
    assert "test_rmse" in result.summary
    assert result.summary["rmse"] == result.summary["test_rmse"]
    assert np.isfinite(result.summary["rmse"])


def test_runner_split_is_disjoint(tmp_path):
    npz = _save_dataset(tmp_path)
    result = run_training(
        _data_config(npz), _run_config(),
        scratch_root_override=tmp_path / "scratch",
        experiments_root=tmp_path / "experiments",
        write_figures=False,
        device="cpu",
    )
    masks = result.split_masks
    # The two non-empty folds for this config are train and test.
    assert masks["train"].any() and masks["test"].any()
    assert not (masks["train"] & masks["test"]).any()  # no shared rows


# --- no-leakage: grouped split keeps simulations whole ---------------------

def test_grouped_split_no_sim_in_two_folds():
    # 12 simulations, each contributing several rows (non-contiguous ids).
    sim_index = np.repeat(np.array([10, 20, 30, 40, 50, 60,
                                    70, 80, 90, 100, 110, 120]), 3)
    split = SplitSpec(train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=7)
    masks = grouped_split(sim_index, split)

    train = set(sim_index[masks["train"]].tolist())
    val = set(sim_index[masks["val"]].tolist())
    test = set(sim_index[masks["test"]].tolist())

    # No simulation appears in more than one fold.
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    # Every simulation is placed somewhere.
    assert train | val | test == set(sim_index.tolist())
    # The masks partition the rows exactly.
    assert int(masks["train"].sum() + masks["val"].sum() + masks["test"].sum()) == len(sim_index)
    assert not (masks["train"] & masks["val"]).any()
    assert not (masks["train"] & masks["test"]).any()
    assert not (masks["val"] & masks["test"]).any()


def test_grouped_split_two_way_when_val_zero():
    sim_index = np.repeat(np.arange(10), 2)
    split = SplitSpec(train_frac=0.8, val_frac=0.0, test_frac=0.2, seed=0)
    masks = grouped_split(sim_index, split)
    assert not masks["val"].any()  # no validation fold
    assert masks["train"].any() and masks["test"].any()


def test_grouped_split_val_empty_for_rounding_prone_n():
    # 0.5/0.0/0.5 at odd n used to leak a simulation into val via independent
    # rounding; the cumulative-boundary split keeps val empty by construction.
    for n in (5, 9, 13, 17, 25):
        sim_index = np.arange(n)
        masks = grouped_split(sim_index, SplitSpec(0.5, 0.0, 0.5, seed=0))
        assert not masks["val"].any(), f"val fold not empty at n={n}"
        total = int(masks["train"].sum() + masks["val"].sum() + masks["test"].sum())
        assert total == n  # still a partition
        assert masks["train"].any() and masks["test"].any()


# --- X_params is threaded through to predict --------------------------------

def _save_dataset_with_params(tmp_path: Path, n_sims: int = 10, n_nd: int = 2,
                              n_radii: int = 5, n_k: int = 8,
                              n_params: int = 4) -> Path:
    """Save a small dataset that carries CAMELS params, and return its path."""
    rng = np.random.default_rng(1)
    ds = S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4])[:n_nd],
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.random((n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        camels_params=rng.random((n_sims, n_params)),
        snapshot=74,
    )
    return S.save_dataset(
        ds, project_root=tmp_path / "data", suite="TESTP",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1",
    )


class _ParamProbe:
    """Module-level (so it is picklable for the checkpoint) probe model.

    Records the ``X_params`` the runner passes to ``predict`` on a class
    attribute, so the test can assert params are threaded through.
    """

    last_X_params = "unset"

    def __init__(self, seed: int = 0, **kwargs):
        self._n_k = 1

    def fit(self, training_data):
        self._n_k = training_data.y.shape[1]

    def predict(self, X, X_cond=None, X_params=None):
        type(self).last_X_params = X_params  # record what the runner passed
        return np.zeros((np.asarray(X).shape[0], self._n_k))


def test_runner_threads_x_params_to_predict(tmp_path):
    from fgas_spk.models.base import REGISTRY

    _ParamProbe.last_X_params = "unset"
    REGISTRY["_param_probe"] = _ParamProbe
    npz = _save_dataset_with_params(tmp_path)
    data_cfg = DataConfig(
        path=str(npz), target_mode="curve",
        include_nd_feature=True, include_camels_params=True,
    )
    run_cfg = RunConfig(
        model="_param_probe", seed=0,
        split=SplitSpec(train_frac=0.7, val_frac=0.0, test_frac=0.3, seed=0),
    )
    try:
        run_training(
            data_cfg, run_cfg,
            scratch_root_override=tmp_path / "scratch",
            experiments_root=tmp_path / "experiments",
            write_figures=False, device="cpu",
        )
    finally:
        REGISTRY.pop("_param_probe", None)

    assert _ParamProbe.last_X_params is not None    # params reached predict
    assert _ParamProbe.last_X_params.shape[1] == 4  # the four CAMELS params
