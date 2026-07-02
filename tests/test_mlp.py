"""Tests for the ``mlp`` model plugin (the plain, un-bottlenecked MLP).

Coverage: importing ``fgas_spk.models`` registers ``"mlp"`` *without* eagerly
importing torch (the lazy-dependency contract); the model satisfies the
:class:`~fgas_spk.models.base.ProfileToSpk` protocol; fit/predict return the right
shapes for the with-conditioning and profile-only paths; the modality and
before-fit guards raise; the training loop actually drives the loss down
(overfit-one-batch gate); and training is deterministic on a fixed CPU backend.
The model deliberately exposes no ``latents`` (there is no bottleneck) and no
``predict_samples`` (it is a deterministic point predictor) -- both are asserted
absent. Every fixture is synthetic -- no CAMELS data.

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
from fgas_spk.models.mlp import Mlp


# --- builders --------------------------------------------------------------

def _training_data(
    with_cond: bool = True,
    with_params: bool = False,
    n: int = 30,
    n_radii: int = 6,
    n_cond: int = 2,
    n_params: int = 3,
    n_k: int = 5,
    seed: int = 0,
) -> TrainingData:
    """Build a synthetic TrainingData bundle in the loader's shape."""
    rng = np.random.default_rng(seed)
    return TrainingData(
        X=rng.random((n, n_radii)),
        X_cond=rng.random((n, n_cond)) if with_cond else None,
        X_params=rng.random((n, n_params)) if with_params else None,
        y=rng.random((n, n_k)),
        nd=rng.random(n),
        sim_index=np.arange(n),
        k=np.linspace(0.1, 1.0, n_k),
        radii_mpch=np.linspace(0.1, 3.0, n_radii),
        source_path="synthetic",
    )


# --- registry + lazy torch -------------------------------------------------

def test_registry_populated_with_mlp():
    # Importing fgas_spk.models must have run the @register side effect.
    assert "mlp" in models.REGISTRY
    assert models.REGISTRY["mlp"] is Mlp


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
    # __init__ stores config only; it must not need torch or build the network.
    model = Mlp(hidden=16, n_layers=1, seed=0)
    assert model.hidden == 16
    assert model._net is None
    assert model.history == []


def test_satisfies_protocol():
    model = Mlp(seed=0)
    assert isinstance(model, ProfileToSpk)  # runtime_checkable structural check


def test_no_bottleneck_diagnostics_exposed():
    # No bottleneck -> no latents; deterministic point predictor -> no
    # predict_samples. The runner duck-types on both, so their absence is what
    # keeps the latent-norm and coverage diagnostics from firing for this model.
    model = Mlp(seed=0)
    assert not hasattr(model, "latents")
    assert not hasattr(model, "predict_samples")


# --- fit / predict shapes --------------------------------------------------

def test_fit_predict_with_cond_shapes():
    td = _training_data(with_cond=True, n=24, n_radii=6, n_cond=2, n_k=5)
    model = Mlp(
        hidden=16, n_layers=1, epochs=3, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == td.y.shape  # (n, n_k)
    # One history entry per epoch.
    assert len(model.history) == 3
    assert set(model.history[0]) == {"epoch", "train_mse"}


def test_fit_predict_profile_only_shapes():
    td = _training_data(with_cond=False, n=24, n_radii=6, n_k=5)
    model = Mlp(
        hidden=16, n_layers=1, epochs=3, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X)  # profile-only (X_cond is None)
    assert pred.shape == td.y.shape


def test_fit_predict_with_cond_and_params_shapes():
    # Both conditioning modalities present: X_params used just like X_cond.
    td = _training_data(
        with_cond=True, with_params=True, n=24, n_radii=6, n_cond=2,
        n_params=3, n_k=5,
    )
    model = Mlp(
        hidden=16, n_layers=1, epochs=3, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond, td.X_params)
    assert pred.shape == td.y.shape


def test_params_only_path_shapes():
    # X_params present, X_cond absent: the param modality alone conditions the net.
    td = _training_data(with_cond=False, with_params=True, n=24, n_k=5)
    model = Mlp(
        hidden=16, n_layers=1, epochs=3, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, X_params=td.X_params)
    assert pred.shape == td.y.shape


def test_single_k_target_predict_is_1d():
    # A 1-D single_k target must round-trip back to 1-D at predict time.
    td = _training_data(with_cond=True, n=24, n_k=5)
    td.y = td.y[:, 0]  # collapse to a single k bin: shape (n,)
    model = Mlp(
        hidden=16, n_layers=1, epochs=3, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == (td.X.shape[0],)


def test_params_actually_influence_prediction():
    # Changing X_params must change the prediction -- proof the parameter modality
    # is wired into the forward pass, not silently dropped.
    td = _training_data(with_cond=False, with_params=True, n=24, n_k=5, seed=5)
    model = Mlp(
        hidden=16, n_layers=2, epochs=20, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    p0 = model.predict(td.X, X_params=td.X_params)
    p1 = model.predict(td.X, X_params=td.X_params + 1.0)
    assert not np.allclose(p0, p1)


# --- guards ----------------------------------------------------------------

def test_predict_before_fit_raises():
    model = Mlp(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((2, 6)))


def test_predict_modality_must_match_fit():
    # Fit with conditioning, then predict without it -> the input widths would
    # not line up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True, n=16)
    model = Mlp(
        hidden=16, n_layers=1, epochs=2, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X)


def test_predict_profile_only_rejects_cond():
    td = _training_data(with_cond=False, n=16)
    model = Mlp(
        hidden=16, n_layers=1, epochs=2, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X, X_cond=np.zeros((td.X.shape[0], 2)))


def test_predict_params_modality_must_match_fit():
    # Fit with X_params, then predict without it -> input widths would not line
    # up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True, with_params=True, n=16)
    model = Mlp(
        hidden=16, n_layers=1, epochs=2, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond)  # X_params omitted


def test_predict_rejects_unfit_params():
    # Fit without X_params, then predict with it -> must raise.
    td = _training_data(with_cond=True, with_params=False, n=16)
    model = Mlp(
        hidden=16, n_layers=1, epochs=2, batch_size=8, seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond, X_params=np.zeros((td.X.shape[0], 3)))


# --- learning + determinism ------------------------------------------------

def test_overfit_one_batch_drives_loss_down():
    # On a tiny fixed set, enough epochs must clearly reduce the training MSE --
    # proving the loop learns and gradients flow end to end.
    td = _training_data(with_cond=True, n=16, n_radii=5, n_cond=2, n_k=3, seed=7)
    model = Mlp(
        hidden=32, n_layers=2, epochs=300, batch_size=16, lr=1e-2,
        seed=0, device="cpu",
    )
    model.fit(td)
    initial = model.history[0]["train_mse"]
    final = model.history[-1]["train_mse"]
    assert final < 0.5 * initial or final < 1e-3


def test_determinism_on_fixed_cpu_backend():
    td = _training_data(with_cond=True, n=20, seed=3)
    a = Mlp(
        hidden=16, n_layers=2, epochs=10, batch_size=8, seed=123, device="cpu",
    )
    b = Mlp(
        hidden=16, n_layers=2, epochs=10, batch_size=8, seed=123, device="cpu",
    )
    a.fit(td)
    b.fit(td)
    np.testing.assert_array_equal(
        a.predict(td.X, td.X_cond), b.predict(td.X, td.X_cond)
    )
