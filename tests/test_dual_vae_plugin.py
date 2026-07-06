"""Tests for the codec-agnostic ``dual_vae`` registry plugin.

Coverage: importing ``fgas_spk.models`` registers ``"dual_vae"`` without
importing torch; the plugin satisfies :class:`~fgas_spk.models.base.ProfileToSpk`
and the construction contract (``seed`` keyword, tolerant of unexpected keys);
fit/predict/predict_samples/latents/latents_y/obs_sigma shapes for the
(vae, vae), (pca, pca), and one cross codec pair (amendment B7); the F0.5
sampling path is seeded-reproducible; non-MDN configurations refuse
``predict_samples``; conditioning-modality mismatches raise; and ``history``
carries the three-phase trace. Every fixture is synthetic.
"""

import subprocess
import sys

import numpy as np
import pytest

import fgas_spk.models as models
from fgas_spk.loader import TrainingData
from fgas_spk.models.base import ProfileToSpk
from fgas_spk.models.dual_vae import DualVae


def _training_data(n: int = 80, n_radii: int = 6, n_k: int = 5,
                   n_params: int = 3, seed: int = 0) -> TrainingData:
    """Synthetic TrainingData with a smooth X -> y dependence."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_radii))
    y = 0.9 + 0.1 * np.tanh(X[:, :1] - X[:, 1:2]) + rng.random((n, n_k)) * 0.01
    return TrainingData(
        X=X, X_cond=None, X_params=rng.random((n, n_params)),
        y=y, nd=rng.random(n), sim_index=np.arange(n),
        k=np.linspace(0.5, 5.0, n_k), radii_mpch=np.linspace(0.1, 9.0, n_radii),
        source_path="synthetic",
    )


def _fast(**kw) -> DualVae:
    """A small, quick CPU configuration for tests."""
    defaults = dict(
        latent_dim_x=2, latent_dim_y=2, hidden=32, n_layers=2, epochs=8,
        map_epochs=15, lr=1e-2, batch_size=64, seed=0, device="cpu",
    )
    defaults.update(kw)
    return DualVae(**defaults)


def test_registered_and_torch_free():
    assert "dual_vae" in models.REGISTRY
    assert models.REGISTRY["dual_vae"] is DualVae
    code = ("import sys; import fgas_spk.models; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_construction_contract_and_protocol():
    model = DualVae(seed=3, not_a_real_key=1)
    assert isinstance(model, ProfileToSpk)
    assert model.seed == 3 and model.codec_x == "vae" and model.codec_y == "vae"
    with pytest.raises(ValueError):
        DualVae(codec_x="nope")
    with pytest.raises(ValueError):
        DualVae(mapping="flow")


@pytest.mark.parametrize("codec_x,codec_y", [
    ("vae", "vae"), ("pca", "pca"), ("pca", "vae"),
])
def test_fit_predict_all_codec_pairs(codec_x, codec_y):
    td = _training_data()
    model = _fast(codec_x=codec_x, codec_y=codec_y, mapping="ridge")
    model.fit(td)
    pred = model.predict(td.X, None, td.X_params)
    assert pred.shape == td.y.shape
    assert model.latents(td.X, None, td.X_params).shape == (td.X.shape[0], 2)
    assert model.latents_y(td.y, None, td.X_params).shape == (td.X.shape[0], 2)
    assert model.obs_sigma().shape == (td.y.shape[1],)


def test_mdn_samples_seeded_and_f05_floor():
    td = _training_data()
    model = _fast(mapping="mdn", mdn_components=2)
    model.fit(td)
    s1 = model.predict_samples(td.X[:7], None, td.X_params[:7], n_samples=30, seed=5)
    s2 = model.predict_samples(td.X[:7], None, td.X_params[:7], n_samples=30, seed=5)
    assert s1.shape == (30, 7, td.y.shape[1])
    assert np.array_equal(s1, s2)
    # F0.5: decoder-Y obs noise is part of the spread.
    assert (s1.std(axis=0) > 0).all()
    # Three-phase history for the all-torch configuration.
    assert {row["phase"] for row in model.history} == {"codec_y", "codec_x", "mapping"}


def test_non_mdn_refuses_samples_and_guards():
    td = _training_data()
    model = _fast(mapping="ridge")
    with pytest.raises(RuntimeError):
        model.predict(td.X)  # before fit
    model.fit(td)
    with pytest.raises(RuntimeError):
        model.predict_samples(td.X, None, td.X_params)
    with pytest.raises(ValueError):
        model.predict(td.X)  # fitted with X_params, called without
    with pytest.raises(ValueError):
        model.predict(td.X, np.zeros((td.X.shape[0], 1)), td.X_params)
