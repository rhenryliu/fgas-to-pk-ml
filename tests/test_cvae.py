"""Tests for the ``cvae`` model plugin.

Coverage: importing ``fgas_spk.models`` registers ``"cvae"`` *without* eagerly
importing torch (the lazy-dependency contract); the model satisfies the
:class:`~fgas_spk.models.base.ProfileToSpk` protocol; fit/predict/latents return
the right shapes for the profile-only, with-conditioning, and both-modalities
paths; :meth:`predict` is deterministic (posterior mean); :meth:`predict_samples`
returns the right shape, has non-zero across-sample spread, and is reproducible
under a fixed seed; the ELBO loop drives the reconstruction term down under KL
annealing (overfit-one-batch gate) while the per-epoch history exposes the
recon/kl/beta_eff collapse diagnostics; the modality and before-fit guards raise;
and training is deterministic on a fixed CPU backend. Every fixture is synthetic
-- no CAMELS data.

The lazy-torch check runs in a fresh subprocess for the same reason as the MLP
test: asserting ``"torch" not in sys.modules`` inside this pytest process would be
order-dependent, since the fit tests import torch and pytest shares
``sys.modules`` across a session.
"""

import subprocess
import sys

import numpy as np
import pytest

import fgas_spk.models as models
from fgas_spk.loader import TrainingData
from fgas_spk.models.base import ProfileToSpk
from fgas_spk.models.cvae import Cvae


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

def test_registry_populated_with_cvae():
    # Importing fgas_spk.models must have run the @register side effect.
    assert "cvae" in models.REGISTRY
    assert models.REGISTRY["cvae"] is Cvae


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
    model = Cvae(latent_dim=3, hidden=16, n_layers=1, seed=0)
    assert model.latent_dim == 3
    assert model._encoder is None and model._decoder is None
    assert model.history == []


def test_anneal_epochs_default_is_quarter_schedule():
    # None resolves to epochs // 4 (pure arithmetic, torch-free), explicit passes.
    assert Cvae(epochs=200).anneal_epochs == 50
    assert Cvae(epochs=200, anneal_epochs=10).anneal_epochs == 10


def test_satisfies_protocol():
    model = Cvae(seed=0)
    assert isinstance(model, ProfileToSpk)  # runtime_checkable structural check


# --- fit / predict / latents shapes ---------------------------------------

def test_fit_predict_latents_with_cond_shapes():
    td = _training_data(with_cond=True, n=24, n_radii=6, n_cond=2, n_k=5)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == td.y.shape  # (n, n_k)
    z = model.latents(td.X, td.X_cond)
    assert z.shape == (td.X.shape[0], 4)  # (n, latent_dim)
    # One history entry per epoch, carrying the collapse diagnostics.
    assert len(model.history) == 3
    assert {"epoch", "recon", "kl", "beta_eff", "train_loss"} <= set(model.history[0])


def test_fit_predict_profile_only_shapes():
    td = _training_data(with_cond=False, n=24, n_radii=6, n_k=5)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X)  # profile-only (X_cond is None)
    assert pred.shape == td.y.shape
    z = model.latents(td.X)
    assert z.shape == (td.X.shape[0], 3)


def test_fit_predict_latents_with_cond_and_params_shapes():
    # Both conditioning modalities present: X_params used just like X_cond.
    td = _training_data(
        with_cond=True, with_params=True, n=24, n_radii=6, n_cond=2,
        n_params=3, n_k=5,
    )
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond, td.X_params)
    assert pred.shape == td.y.shape
    z = model.latents(td.X, td.X_cond, td.X_params)
    assert z.shape == (td.X.shape[0], 4)


def test_params_only_path_shapes():
    # X_params present, X_cond absent: the param modality alone conditions the net.
    td = _training_data(with_cond=False, with_params=True, n=24, n_k=5)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, X_params=td.X_params)
    assert pred.shape == td.y.shape
    z = model.latents(td.X, X_params=td.X_params)
    assert z.shape == (td.X.shape[0], 3)


def test_single_k_target_shapes():
    # A 1-D (single_k) target must round-trip to 1-D predictions and samples.
    td = _training_data(with_cond=True, n=20, n_k=5)
    td.y = td.y[:, 0]  # 1-D single_k target
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=3, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == (td.X.shape[0],)
    samples = model.predict_samples(td.X, td.X_cond, n_samples=7)
    assert samples.shape == (7, td.X.shape[0])


# --- guards ----------------------------------------------------------------

def test_predict_before_fit_raises():
    model = Cvae(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict(np.zeros((2, 6)))


def test_latents_before_fit_raises():
    model = Cvae(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.latents(np.zeros((2, 6)))


def test_predict_samples_before_fit_raises():
    model = Cvae(seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.predict_samples(np.zeros((2, 6)))


def test_predict_modality_must_match_fit():
    # Fit with conditioning, then predict without it -> the input widths would
    # not line up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X)


def test_predict_profile_only_rejects_cond():
    td = _training_data(with_cond=False, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_cond"):
        model.predict(td.X, X_cond=np.zeros((td.X.shape[0], 2)))


def test_predict_params_modality_must_match_fit():
    # Fit with X_params, then predict without it -> input widths would not line
    # up, so this must raise rather than silently mispredict.
    td = _training_data(with_cond=True, with_params=True, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond)  # X_params omitted


def test_predict_rejects_unfit_params():
    # Fit without X_params, then predict with it -> must raise.
    td = _training_data(with_cond=True, with_params=False, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond, X_params=np.zeros((td.X.shape[0], 3)))


# --- point prediction: determinism -----------------------------------------

def test_predict_is_deterministic():
    # predict uses the posterior mean (no sampling): repeated calls on the same
    # fitted model must return identical arrays.
    td = _training_data(with_cond=True, n=20, seed=2)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    p0 = model.predict(td.X, td.X_cond)
    p1 = model.predict(td.X, td.X_cond)
    np.testing.assert_array_equal(p0, p1)


def test_determinism_on_fixed_cpu_backend():
    td = _training_data(with_cond=True, n=20, seed=3)
    a = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=123, device="cpu",
    )
    b = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=123, device="cpu",
    )
    a.fit(td)
    b.fit(td)
    np.testing.assert_array_equal(
        a.predict(td.X, td.X_cond), b.predict(td.X, td.X_cond)
    )


# --- predictive spread: the science payload --------------------------------

def test_predict_samples_shape_and_spread():
    # predict_samples returns (n_samples, n_examples, n_k) and -- for a model fit
    # with a non-trivial KL term (so sigma > 0) -- has non-zero across-sample
    # spread. A collapsed latent would give ~0 spread; annealing prevents that.
    td = _training_data(with_cond=True, n=24, n_k=5, seed=4)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=40, batch_size=8,
        beta=1.0, seed=0, device="cpu",
    )
    model.fit(td)
    samples = model.predict_samples(td.X, td.X_cond, n_samples=32)
    assert samples.shape == (32, td.X.shape[0], td.y.shape[1])
    # Spread across the sample axis is strictly positive somewhere.
    assert samples.std(axis=0).max() > 1e-6


def test_predict_samples_is_reproducible_under_fixed_seed():
    # Sampling is seeded: same seed -> identical draws; a different seed differs.
    td = _training_data(with_cond=True, n=20, n_k=4, seed=6)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=30, batch_size=8,
        beta=1.0, seed=0, device="cpu",
    )
    model.fit(td)
    s_a = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=11)
    s_b = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=11)
    np.testing.assert_array_equal(s_a, s_b)
    s_c = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=12)
    assert not np.allclose(s_a, s_c)


# --- learning + collapse diagnostics ---------------------------------------

def test_overfit_one_batch_drives_recon_down():
    # On a tiny fixed set, enough epochs under KL annealing must clearly reduce the
    # reconstruction term -- proving the loop learns and gradients flow end to end.
    # We assert on RECON, not total loss: annealing makes the total non-monotonic
    # by design (beta_eff ramps up), so total-loss monotonicity is not expected.
    td = _training_data(with_cond=True, n=16, n_radii=5, n_cond=2, n_k=3, seed=7)
    model = Cvae(
        latent_dim=4, hidden=32, n_layers=2, epochs=400, batch_size=16,
        lr=1e-2, beta=1.0, seed=0, device="cpu",
    )
    model.fit(td)
    initial = model.history[0]["recon"]
    final = model.history[-1]["recon"]
    assert final < 0.5 * initial or final < 1e-3


def test_history_carries_collapse_diagnostics():
    # The per-epoch trace must expose recon/kl/beta_eff so KL-vanishing is visible.
    td = _training_data(with_cond=True, n=16, n_k=4, seed=8)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=5, batch_size=8,
        anneal_epochs=4, beta=1.0, seed=0, device="cpu",
    )
    model.fit(td)
    assert len(model.history) == 5
    for row in model.history:
        assert {"epoch", "recon", "kl", "beta_eff", "train_loss"} <= set(row)
    # beta_eff anneals: it starts at 0 and reaches beta by anneal_epochs.
    assert model.history[0]["beta_eff"] == 0.0
    assert model.history[-1]["beta_eff"] == pytest.approx(1.0)
