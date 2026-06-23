"""Tests for the trainer-side config in ``fgas_spk.experiment``.

Coverage: the ``RunConfig`` YAML round-trip (with nested ``SplitSpec``
coercion), split-spec validation (out-of-range and non-summing fractions, plus
the ``val_frac = 0`` two-way case which must be accepted), that ``load_configs``
returns two independent objects with no field bleed, and that an existing
``DataConfig`` YAML still loads unchanged.
"""

from pathlib import Path

import pytest

import fgas_spk.experiment as E
from fgas_spk.experiment import RunConfig, SplitSpec, load_configs
from fgas_spk.loader import DataConfig

# tests/ lives at the repo root, so its parent is the repository.
_REPO = Path(__file__).resolve().parents[1]


# --- RunConfig YAML round-trip ---------------------------------------------

def test_runconfig_yaml_round_trip(tmp_path):
    cfg = RunConfig(
        model="pca",
        model_params={"n_components": 8, "whiten": True},
        seed=42,
        split=SplitSpec(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=1),
        write_root="/scratch/run",
    )
    out = cfg.to_yaml(tmp_path / "config_run.yaml")
    back = RunConfig.from_yaml(out)
    assert back == cfg
    # The nested split survives the YAML mapping round-trip as a SplitSpec.
    assert isinstance(back.split, SplitSpec)


def test_runconfig_defaults_round_trip(tmp_path):
    cfg = RunConfig(model="mlp")
    back = RunConfig.from_yaml(cfg.to_yaml(tmp_path / "config_run.yaml"))
    assert back == cfg
    assert isinstance(back.split, SplitSpec)


def test_runconfig_coerces_split_mapping():
    cfg = RunConfig(
        model="pca",
        split={"train_frac": 0.6, "val_frac": 0.2, "test_frac": 0.2, "seed": 3},
    )
    assert isinstance(cfg.split, SplitSpec)
    assert cfg.split.seed == 3


# --- split-spec validation -------------------------------------------------

def test_split_rejects_non_summing_fractions():
    with pytest.raises(ValueError, match="sum to 1"):
        SplitSpec(train_frac=0.5, val_frac=0.5, test_frac=0.5)


def test_split_rejects_out_of_range_fractions():
    with pytest.raises(ValueError, match="out of range"):
        SplitSpec(train_frac=1.2, val_frac=0.0, test_frac=-0.2)


def test_split_rejects_zero_train():
    with pytest.raises(ValueError, match="train_frac must be > 0"):
        SplitSpec(train_frac=0.0, val_frac=0.5, test_frac=0.5)


def test_split_validation_fires_through_runconfig():
    # An incoherent split passed as a mapping is rejected during coercion.
    with pytest.raises(ValueError, match="sum to 1"):
        RunConfig(model="pca",
                  split={"train_frac": 0.4, "val_frac": 0.4, "test_frac": 0.4})


def test_split_allows_two_way_via_zero_val(tmp_path):
    # val_frac == 0 is a valid two-way train/test split, and round-trips.
    cfg = RunConfig(model="pca",
                    split=SplitSpec(train_frac=0.8, val_frac=0.0, test_frac=0.2))
    assert cfg.split.val_frac == 0.0
    back = RunConfig.from_yaml(cfg.to_yaml(tmp_path / "config_run.yaml"))
    assert back == cfg


# --- load_configs: independent, not merged ---------------------------------

def test_load_configs_returns_independent_objects(tmp_path):
    data_cfg = DataConfig(path="some.npz", number_density_indices=[0, 2])
    data_yaml = data_cfg.to_yaml(tmp_path / "config_data.yaml")
    run_cfg = RunConfig(model="pca", seed=7)
    run_yaml = run_cfg.to_yaml(tmp_path / "config_run.yaml")

    d, r = load_configs(data_yaml, run_yaml)

    assert isinstance(d, DataConfig)
    assert isinstance(r, RunConfig)
    assert d == data_cfg
    assert r == run_cfg
    # No merge / no field bleed between the two configs.
    assert not hasattr(r, "path")
    assert not hasattr(d, "model")


# --- existing DataConfig YAML still loads ----------------------------------

def test_existing_data_config_yaml_loads_unchanged():
    cfg_path = _REPO / "scripts" / "configs" / "data" / "config.yaml"
    assert cfg_path.exists()
    cfg = DataConfig.from_yaml(cfg_path)
    assert isinstance(cfg, DataConfig)
    # Spot-check fields from the shipped file.
    assert cfg.suite == "CAMELS-IllustrisTNG-L50n512-SB35"
    assert cfg.target_mode == "k_range"
    assert cfg.k_range == (0.5, 5.0)


# --- flat-API exposure -----------------------------------------------------

def test_flat_api_exposes_run_config():
    import fgas_spk as F
    assert F.RunConfig is E.RunConfig
    assert F.SplitSpec is E.SplitSpec
    assert F.load_configs is E.load_configs
