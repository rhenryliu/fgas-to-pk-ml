"""Tests for the git-tracked run record and ledger in ``fgas_spk.run_record``.

Coverage: writing a synthetic run produces the full per-run directory with all
five files; ``runs.jsonl`` gains exactly one well-formed line; the record
round-trips (``env.json`` plus the two config YAMLs read back, fields match); and
a second run appends a second distinct line without clobbering the first. Every
fixture is synthetic -- no real CAMELS data, but a real saved ``.npz`` so the
embedded ``__meta__`` capture is exercised.
"""

import json
from pathlib import Path

import numpy as np

import fgas_spk.schema as S
from fgas_spk.experiment import RunConfig, SplitSpec
from fgas_spk.loader import DataConfig
from fgas_spk.run_record import write_run_record

_RUN_FILES = {
    "config_data.yaml", "config_run.yaml", "env.json",
    "metrics.jsonl", "summary.json",
}


# --- builders --------------------------------------------------------------

def _save_dataset(tmp_path: Path) -> Path:
    """Save a small real dataset .npz (with __meta__) and return its path."""
    n_sims, n_nd, n_radii, n_k = 3, 2, 5, 8
    rng = np.random.default_rng(0)
    ds = S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4]),
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.random((n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        snapshot=74,
    )
    # overwrite=True so repeated calls in one tmp_path (e.g. two runs sharing a
    # dataset) re-save the identical .npz instead of colliding.
    return S.save_dataset(
        ds, project_root=tmp_path / "data", suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1", overwrite=True,
    )


def _data_config(npz_path: Path) -> DataConfig:
    return DataConfig(path=str(npz_path), target_mode="single_k", k_target=3.0)


def _run_config(seed: int = 0) -> RunConfig:
    return RunConfig(
        model="pca_linear",
        model_params={"n_components": 4},
        seed=seed,
        split=SplitSpec(train_frac=0.8, val_frac=0.0, test_frac=0.2),
    )


def _write(tmp_path: Path, seed: int = 0):
    npz = _save_dataset(tmp_path)
    return write_run_record(
        _data_config(npz),
        _run_config(seed),
        dataset_path=npz,
        data_root=tmp_path / "data",
        scratch_root=tmp_path / "scratch",
        summary={"test_rmse": 0.123, "final_rmse": 0.1},
        metrics=[{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.4}],
        device="cpu",
        experiments_root=tmp_path / "experiments",
    )


# --- full directory written ------------------------------------------------

def test_run_dir_has_all_files(tmp_path):
    rec = _write(tmp_path)
    assert rec.run_dir.is_dir()
    assert {p.name for p in rec.run_dir.iterdir()} == _RUN_FILES


def test_run_id_format(tmp_path):
    rec = _write(tmp_path)
    ts, config_hash, sha7 = rec.run_id.split("__")
    assert ts.endswith("Z") and len(ts) == len("YYYYMMDDTHHMMSSZ")
    assert len(config_hash) == 8
    assert len(sha7) == 7


def test_metrics_jsonl_one_object_per_line(tmp_path):
    rec = _write(tmp_path)
    lines = (rec.run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]


# --- env.json content + round-trip -----------------------------------------

def test_env_json_captures_provenance(tmp_path):
    rec = _write(tmp_path)
    env = json.loads((rec.run_dir / "env.json").read_text())

    assert isinstance(env["git"]["sha"], str) and env["git"]["sha"]
    assert isinstance(env["git"]["dirty"], bool)
    assert env["seed"] == 0
    assert env["device"] == "cpu"
    # Both roots recorded.
    assert env["roots"]["data_root"] == str(tmp_path / "data")
    assert env["roots"]["scratch_root"] == str(tmp_path / "scratch")
    # Dataset path + its embedded __meta__.
    assert env["dataset"]["path"].endswith("__v1.npz")
    assert env["dataset"]["meta"]["suite"] == "TEST"
    assert env["dataset"]["meta"]["snapshot"] == 74
    # Suppression definition recorded (the within-hydro proxy).
    assert env["suppression_target"]["definition"].startswith("P_total")
    # Target mode + k selection from the DataConfig.
    assert env["target_selection"]["target_mode"] == "single_k"
    assert env["target_selection"]["k_target"] == 3.0
    assert env["target_selection"]["k_range"] is None
    # Package versions present.
    assert "python" in env["packages"]
    assert env["packages"]["numpy"]


def test_configs_round_trip(tmp_path):
    npz = _save_dataset(tmp_path)
    data_cfg = _data_config(npz)
    run_cfg = _run_config(seed=5)
    rec = write_run_record(
        data_cfg, run_cfg, dataset_path=npz,
        data_root=tmp_path / "data", scratch_root=tmp_path / "scratch",
        summary={"test_rmse": 0.2},
        experiments_root=tmp_path / "experiments",
    )
    assert DataConfig.from_yaml(rec.run_dir / "config_data.yaml") == data_cfg
    assert RunConfig.from_yaml(rec.run_dir / "config_run.yaml") == run_cfg


def test_summary_round_trips(tmp_path):
    rec = _write(tmp_path)
    summary = json.loads((rec.run_dir / "summary.json").read_text())
    assert summary == {"test_rmse": 0.123, "final_rmse": 0.1}


def test_metrics_empty_when_none(tmp_path):
    npz = _save_dataset(tmp_path)
    rec = write_run_record(
        _data_config(npz), _run_config(), dataset_path=npz,
        data_root=tmp_path / "data", scratch_root=tmp_path / "scratch",
        summary={"test_rmse": 0.2}, metrics=None,
        experiments_root=tmp_path / "experiments",
    )
    assert (rec.run_dir / "metrics.jsonl").read_text() == ""


# --- ledger ----------------------------------------------------------------

def test_ledger_gains_exactly_one_wellformed_line(tmp_path):
    rec = _write(tmp_path)
    lines = rec.ledger_path.read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {
        "run_id", "config_hash", "git_sha", "dirty", "hparams",
        "data", "metrics", "run_dir",
    }
    assert entry["run_id"] == rec.run_id
    assert entry["hparams"]["model"] == "pca_linear"
    assert entry["hparams"]["model_params"] == {"n_components": 4}
    # The data block records the target selection so a single_k run's target
    # wavenumber stays recoverable from the grep-friendly ledger alone.
    assert entry["data"]["target_mode"] == "single_k"
    assert entry["data"]["k_target"] == 3.0
    assert entry["data"]["k_range"] is None
    assert entry["metrics"] == {"test_rmse": 0.123, "final_rmse": 0.1}
    assert entry["run_dir"].startswith("experiments/runs/")


def test_second_run_appends_distinct_line_without_clobbering(tmp_path):
    rec1 = _write(tmp_path, seed=0)
    env1_before = (rec1.run_dir / "env.json").read_text()

    rec2 = _write(tmp_path, seed=1)  # different config -> different run_id

    assert rec2.run_id != rec1.run_id
    assert rec2.run_dir != rec1.run_dir

    # Ledger now has two distinct, well-formed lines.
    lines = rec1.ledger_path.read_text().splitlines()
    assert len(lines) == 2
    run_ids = [json.loads(line)["run_id"] for line in lines]
    assert run_ids == [rec1.run_id, rec2.run_id]

    # The first run's directory and files are untouched.
    assert rec1.run_dir.is_dir()
    assert {p.name for p in rec1.run_dir.iterdir()} == _RUN_FILES
    assert (rec1.run_dir / "env.json").read_text() == env1_before
