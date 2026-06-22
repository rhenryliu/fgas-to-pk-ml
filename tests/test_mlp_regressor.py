"""Tests for the ``mlp_regressor`` model plugin.

Coverage: importing ``fgas_spk.models`` registers ``"mlp_regressor"`` *without*
eagerly importing torch (the lazy-dependency contract); the model satisfies the
:class:`~fgas_spk.models.base.ProfileToSpk` protocol; fit/predict/latents return
the right shapes for the with-conditioning and profile-only paths; the modality
and before-fit guards raise; the training loop actually drives the loss down
(overfit-one-batch gate); and training is deterministic on a fixed CPU backend.
Every fixture is synthetic -- no CAMELS data.

The lazy-torch check runs in a fresh subprocess. Asserting ``"torch" not in
sys.modules`` *inside* this pytest process would be order-dependent: the fit
tests import torch and pytest shares ``sys.modules`` across a session, so the
assertion would flip depending on test order. A clean subprocess tests the real
property -- ``import fgas_spk.models`` does not pull torch -- without that
fragility.
"""

import subprocess
import sys

import numpy as np
import pytest

import fgas_spk.models as models
from fgas_spk.loader import TrainingData
from fgas_spk.models.base import ProfileToSpk
from fgas_spk.models.mlp_regressor import MlpRegressor


# --- builders --------------------------------------------------------------

def _training_data(
    with_cond: bool = True,
    n: int = 30,
    n_radii: int = 6,
    n_cond: int = 2,
    n_k: int = 5,
    seed: int = 0,
) -> TrainingData:
    """Build a synthetic TrainingData bundle in the loader's shape."""
    rng = np.random.default_rng(seed)
    return TrainingData(
        X=rng.random((n, n_radii)),
        X_cond=rng.random((n, n_cond)) if with_cond else None,
        X_params=None,
        y=rng.random((n, n_k)),
        nd=rng.random(n),
        sim_index=np.arange(n),
        k=np.linspace(0.1, 1.0, n_k),
        radii_mpch=np.linspace(0.1, 3.0, n_radii),
        source_path="synthetic",
    )


# --- registry + lazy torch -------------------------------------------------

def test_registry_populated_with_mlp_regressor():
    # Importing fgas_spk.models must have run the @register side effect.
    assert "mlp_regressor" in models.REGISTRY
    assert models.REGISTRY["mlp_regressor"] is MlpRegressor


def test_importing_models_does_not_import_torch():
    # Robust check: a fresh interpreter imports the package and asserts torch was
    # not pulled in transitively. Run out-of-process so prior in-session fits
    # (which do import torch) cannot taint the result.
    code = (
        "import sys; import fgas_spk.models; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_construction_is_torch_free_and_cheap():
    # __init__ stores config only; it must not need torch or build modules.
    model = MlpRegressor(latent_dim=3, hidden=16, n_layers=1, seed=0)
    assert model.latent_dim == 3
    assert model._encoder is None and model._decoder is None
    assert model.history == []


def test_satisfies_protocol():
    model = MlpRegressor(seed=0)
    assert isinstance(model, ProfileToSpk)  # runtime_checkable structural check


# --- fit / predict / latents shapes ---------------------------------------

def test_fit_predict_latents_with_cond_shapes():
    td = _training_data(with_cond=True, n=24, n_radii=6, n_cond=2, n_k=5)
    model = MlpRegressor(
        latent_dim=4, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == td.y.shape  # (n, n_k)
    z = model.latents(td.X, td.X_cond)
    assert z.shape == (td.X.shape[0], 4)  # (n, latent_dim)
    # One history entry per epoch.
    assert len(model.history) == 3
    assert set(model.history[0]) == {"epoch", "train_mse"}


def test_fit_predict_profile_only_shapes():
    td = _training_data(with_cond=False, n=24, n_radii=6, n_k=5)
    model = MlpRegressor(
        latent_dim=3, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X)  # profile-only (X_cond is None)
    assert pred.shape == td.y.shape
    z = model.latents(td.X)
    assert z.shape == (td.X.shape[0], 3)


# --- guards ----------------------------------------------------------------

def test_predict_before_fit_raises():
    model = MlpRegressor(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((2, 6)))


def test_latents_before_fit_raises():
    model = MlpRegressor(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.latents(np.zeros((2, 6)))


def test_predict_modality_must_match_fit():
    # Fit with conditioning, then predict without it -> the input widths would
    # not line up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True, n=16)
    model = MlpRegressor(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X)


def test_predict_profile_only_rejects_cond():
    td = _training_data(with_cond=False, n=16)
    model = MlpRegressor(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X, X_cond=np.zeros((td.X.shape[0], 2)))


# --- learning + determinism ------------------------------------------------

def test_overfit_one_batch_drives_loss_down():
    # On a tiny fixed set, enough epochs must clearly reduce the training MSE --
    # proving the loop learns and gradients flow end to end.
    td = _training_data(with_cond=True, n=16, n_radii=5, n_cond=2, n_k=3, seed=7)
    model = MlpRegressor(
        latent_dim=4, hidden=32, n_layers=2, epochs=300, batch_size=16,
        lr=1e-2, seed=0, device="cpu",
    )
    model.fit(td)
    initial = model.history[0]["train_mse"]
    final = model.history[-1]["train_mse"]
    assert final < 0.5 * initial or final < 1e-3


def test_determinism_on_fixed_cpu_backend():
    td = _training_data(with_cond=True, n=20, seed=3)
    a = MlpRegressor(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=123, device="cpu",
    )
    b = MlpRegressor(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=123, device="cpu",
    )
    a.fit(td)
    b.fit(td)
    np.testing.assert_array_equal(
        a.predict(td.X, td.X_cond), b.predict(td.X, td.X_cond)
    )
