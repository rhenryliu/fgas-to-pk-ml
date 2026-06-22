"""Tests for the model-plugin layer in ``fgas_spk.models``.

Coverage: importing the package populates ``REGISTRY`` with ``"pca_linear"``
(the registration side effect actually runs); the reference model fits on small
synthetic ``(X, X_cond, y)`` arrays and ``predict`` returns the right shapes for
both the with-conditioning and profile-only paths; and the model is
deterministic under a fixed seed. Every fixture is synthetic -- no CAMELS data.
"""

import numpy as np
import pytest

import fgas_spk.models as models
from fgas_spk.loader import TrainingData
from fgas_spk.models.base import ProfileToSpk
from fgas_spk.models.pca_linear import PcaLinear


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


# --- registry --------------------------------------------------------------

def test_registry_populated_with_pca_linear():
    # Importing fgas_spk.models must have run the @register side effect.
    assert "pca_linear" in models.REGISTRY
    assert models.REGISTRY["pca_linear"] is PcaLinear


def test_pca_linear_satisfies_protocol():
    model = models.REGISTRY["pca_linear"](n_components=3, seed=0)
    assert isinstance(model, ProfileToSpk)  # runtime_checkable structural check


# --- fit / predict shapes --------------------------------------------------

def test_fit_predict_with_cond_shapes():
    td = _training_data(with_cond=True)
    model = models.REGISTRY["pca_linear"](n_components=3, seed=0)
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == td.y.shape  # (n, n_k)


def test_fit_predict_profile_only_shapes():
    td = _training_data(with_cond=False)
    model = PcaLinear(n_components=3, seed=0)
    model.fit(td)
    pred = model.predict(td.X)  # profile-only (X_cond is None)
    assert pred.shape == td.y.shape


def test_predict_modality_must_match_fit():
    # Fit with conditioning, then predict without it -> the feature layout would
    # not line up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True)
    model = PcaLinear(n_components=3, seed=0)
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X)


def test_predict_before_fit_raises():
    model = PcaLinear(n_components=3, seed=0)
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((2, 6)))


# --- determinism -----------------------------------------------------------

def test_determinism_under_fixed_seed():
    td = _training_data(with_cond=True)
    a = PcaLinear(n_components=3, seed=123)
    b = PcaLinear(n_components=3, seed=123)
    a.fit(td)
    b.fit(td)
    np.testing.assert_array_equal(
        a.predict(td.X, td.X_cond), b.predict(td.X, td.X_cond)
    )
