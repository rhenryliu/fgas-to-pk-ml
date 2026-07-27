"""Tests for the Sohn-style conditional VAE (``cvae``) model plugin.

Coverage: importing ``fgas_spk.models`` registers ``"cvae"`` *without* eagerly
importing torch (the lazy-dependency contract); the model satisfies the
:class:`~fgas_spk.models.base.ProfileToSpk` protocol; fit/predict/predict_samples/
latents return the right shapes for the profile-only, with-conditioning, and
both-modalities paths; :meth:`predict` (prior mean) is deterministic;
:meth:`predict_samples` (conditional-prior draws) has non-zero spread and is
reproducible under a fixed seed; :meth:`latents` returns the prior mean by default
and the recognition mean when a target is passed; the prior's logvar head is
zero-initialised; the modality and before-fit guards raise; and training is
deterministic on a fixed CPU backend.

It then runs the four-part **verification gate** for the conditional VAE:

1. the two-Gaussian KL reduces to the standard ``KL(q || N(0, I))`` form when
   ``mu_p = 0`` and ``logvar_p = 0`` (to 1e-6 on random tensors);
2. the reconstruction term equals ``n_k`` times the mean-over-bins MSE;
3. every logvar head respects the ``[-8, 8]`` clamp for inputs scaled by 1e6;
4. **heteroscedastic recovery** (the decisive test): on synthetic data with a
   known input-dependent noise scale, the per-example predictive std rank-tracks
   the true ``sigma(x)`` (Spearman > 0.8) and the central-68% intervals are
   calibrated (empirical coverage in [0.58, 0.78]) for both a heteroscedastic and
   a bimodal conditional.

A final section covers the **learned observation-noise head**
(``obs_noise_head=True``): toggled off, a default-constructed model and an
explicit ``obs_noise_head=False`` model give bit-identical seeded predictions
and samples; toggled on, training runs, :meth:`obs_sigma` has shape ``(n_k,)``
and is strictly positive, :meth:`predict_samples` is seed-reproducible and
strictly wider per example than the latent-only spread from the *same* fitted
weights, and the 1-D ``single_k`` path works; and, the decisive calibration
gate, on synthetic ``y = f(x) + eps`` with known homoscedastic per-k sigma the
fitted ``obs_sigma()`` recovers the true sigma and the central-68% interval
covers ~68% of held-out truth.

Every fixture is synthetic -- no CAMELS data. The lazy-torch check runs in a fresh
subprocess (the fit tests import torch and pytest shares ``sys.modules`` across a
session, so an in-process assertion would be order-dependent).
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
    assert model._recognition is None and model._prior is None
    assert model._decoder is None
    assert model.history == []
    assert model.best_epoch is None


def test_defaults_are_the_task_b_values():
    m = Cvae()
    assert (m.latent_dim, m.hidden, m.n_layers, m.epochs) == (4, 128, 2, 200)
    assert m.lr == pytest.approx(5e-4)
    assert m.weight_decay == pytest.approx(1e-4)
    assert m.dropout == pytest.approx(0.0)
    assert m.batch_size == 128
    assert m.beta == pytest.approx(1.0)
    assert m.val_frac == pytest.approx(0.1)


def test_anneal_epochs_default_is_quarter_schedule():
    assert Cvae(epochs=200).anneal_epochs == 50
    assert Cvae(epochs=200, anneal_epochs=10).anneal_epochs == 10


def test_satisfies_protocol():
    model = Cvae(seed=0)
    assert isinstance(model, ProfileToSpk)  # runtime_checkable structural check


# --- fit / predict / latents / predict_samples shapes ----------------------

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
    # One history entry per epoch, carrying the CVAE trace incl. latent_gap.
    assert len(model.history) == 3
    assert {
        "epoch", "recon", "kl", "beta_eff", "train_loss",
        "val_loss", "recon_prior", "latent_gap",
    } <= set(model.history[0])


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


# --- latents: prior mean vs recognition mean -------------------------------

def test_latents_prior_vs_recognition():
    # latents() returns mu_p (prior); latents(y=...) returns mu_q (recognition).
    # After training the two codes differ -- the recognition net sees the target.
    td = _training_data(with_cond=True, n=24, n_k=4, seed=3)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=30, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    mu_p = model.latents(td.X, td.X_cond)
    mu_q = model.latents(td.X, td.X_cond, y=td.y)
    assert mu_p.shape == mu_q.shape == (td.X.shape[0], 4)
    assert not np.allclose(mu_p, mu_q)


def test_latent_dists_mu_agrees_with_latents_and_std_is_positive():
    # latent_dists is the primitive latents() delegates to: the means must match
    # exactly on BOTH branches, so the two APIs cannot drift apart.
    td = _training_data(with_cond=True, n=24, n_k=4, seed=3)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=30, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)

    mu_p, std_p = model.latent_dists(td.X, td.X_cond)
    mu_q, std_q = model.latent_dists(td.X, td.X_cond, y=td.y)
    assert mu_p.shape == std_p.shape == (td.X.shape[0], 4)
    assert mu_q.shape == std_q.shape == (td.X.shape[0], 4)
    assert np.array_equal(mu_p, model.latents(td.X, td.X_cond))
    assert np.array_equal(mu_q, model.latents(td.X, td.X_cond, y=td.y))

    # std = exp(0.5 * logvar) with logvar clamped to [-8, 8]: strictly positive
    # and inside the clamp's implied bounds.
    for std in (std_p, std_q):
        assert np.all(std > 0.0)
        assert np.all(std >= np.exp(-4.0)) and np.all(std <= np.exp(4.0))
    # The prior and the recognition posterior are genuinely different laws.
    assert not np.allclose(mu_p, mu_q)


def test_latent_dists_before_fit_raises_and_checks_modality():
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=1, seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.latent_dists(np.zeros((2, 6)))

    td = _training_data(with_cond=True, n=16, n_k=3, seed=1)
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=2, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    # Fitted WITH conditioning, so omitting it must raise, as latents() does.
    with pytest.raises(ValueError, match="X_cond"):
        model.latent_dists(td.X)


def test_decode_at_the_prior_mean_reproduces_predict():
    # predict() IS the prior-mean decode, so decode(latents(X), X) must equal it
    # exactly. This pins the new public method to existing behaviour.
    td = _training_data(with_cond=True, n=24, n_k=4, seed=3)
    model = Cvae(latent_dim=3, hidden=16, n_layers=2, epochs=20, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    mu_p = model.latents(td.X, td.X_cond)
    decoded = model.decode(mu_p, td.X, td.X_cond)
    assert decoded.shape == td.y.shape
    assert np.allclose(decoded, model.predict(td.X, td.X_cond), atol=0.0, rtol=0.0)


def test_decode_responds_to_the_latent():
    # Perturbing z must change the decode -- otherwise the traversal figure is
    # measuring nothing.
    td = _training_data(with_cond=False, n=24, n_k=5, seed=4)
    model = Cvae(latent_dim=3, hidden=16, n_layers=2, epochs=25, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    base = model.latents(td.X)
    shifted = base.copy()
    shifted[:, 0] += 3.0
    assert not np.allclose(model.decode(base, td.X), model.decode(shifted, td.X))


def test_decode_single_k_target_returns_1d():
    td = _training_data(with_cond=False, n=20, n_k=1, seed=6)
    td = TrainingData(
        X=td.X, X_cond=None, X_params=None, y=td.y[:, 0], nd=td.nd,
        sim_index=td.sim_index, k=td.k[:1], radii_mpch=td.radii_mpch,
        source_path=td.source_path, param_names=None, meta={}, config=None,
    )
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=5, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    out = model.decode(model.latents(td.X), td.X)
    assert out.shape == (td.X.shape[0],)
    assert np.allclose(out, model.predict(td.X))


def test_decode_validates_shapes_and_modality():
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=1, seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        model.decode(np.zeros((2, 2)), np.zeros((2, 6)))

    td = _training_data(with_cond=True, n=16, n_k=3, seed=1)
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=2, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    n = td.X.shape[0]
    with pytest.raises(ValueError, match="z must have shape"):
        model.decode(np.zeros((n, 5)), td.X, td.X_cond)      # wrong latent width
    with pytest.raises(ValueError, match="same number of rows"):
        model.decode(np.zeros((n - 1, 2)), td.X, td.X_cond)  # row mismatch
    with pytest.raises(ValueError, match="X_cond"):
        model.decode(np.zeros((n, 2)), td.X)                 # modality mismatch


def test_latent_dists_is_deterministic():
    # Means and widths are read off the networks, never sampled.
    td = _training_data(with_cond=False, n=20, n_k=3, seed=5)
    model = Cvae(latent_dim=3, hidden=16, n_layers=1, epochs=5, batch_size=8,
                 seed=0, device="cpu")
    model.fit(td)
    first = model.latent_dists(td.X)
    second = model.latent_dists(td.X)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_prior_logvar_head_is_zero_initialised():
    # The prior's logvar head is zero-init so p(z|x) starts at unit variance: an
    # untrained (epochs=0) prior returns logvar_p == 0 for every input.
    import torch

    td = _training_data(with_cond=False, n=20, n_radii=5, n_k=3)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=1, epochs=0, val_frac=0.0,
        seed=0, device="cpu",
    )
    model.fit(td)  # builds + zero-inits the modules; no training steps run
    x = model._to_tensor(np.random.default_rng(1).random((6, td.X.shape[1])))
    with torch.no_grad():
        _mu_p, logvar_p = model._prior_dist(x, None)
    assert torch.allclose(logvar_p, torch.zeros_like(logvar_p), atol=1e-6)


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
    td = _training_data(with_cond=True, with_params=True, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond)  # X_params omitted


def test_predict_rejects_unfit_params():
    td = _training_data(with_cond=True, with_params=False, n=16)
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    with pytest.raises(ValueError, match="X_params"):
        model.predict(td.X, td.X_cond, X_params=np.zeros((td.X.shape[0], 3)))


# --- point prediction + sampling: determinism ------------------------------

def test_predict_is_deterministic():
    td = _training_data(with_cond=True, n=20, seed=2)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    np.testing.assert_array_equal(
        model.predict(td.X, td.X_cond), model.predict(td.X, td.X_cond)
    )


def test_determinism_on_fixed_cpu_backend():
    td = _training_data(with_cond=True, n=20, seed=3)
    a = Cvae(latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
             seed=123, device="cpu")
    b = Cvae(latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
             seed=123, device="cpu")
    a.fit(td)
    b.fit(td)
    np.testing.assert_array_equal(
        a.predict(td.X, td.X_cond), b.predict(td.X, td.X_cond)
    )


def test_predict_samples_shape_and_spread():
    td = _training_data(with_cond=True, n=24, n_k=5, seed=4)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=40, batch_size=8,
        beta=1.0, seed=0, device="cpu",
    )
    model.fit(td)
    samples = model.predict_samples(td.X, td.X_cond, n_samples=32)
    assert samples.shape == (32, td.X.shape[0], td.y.shape[1])
    assert samples.std(axis=0).max() > 1e-6  # non-zero spread somewhere


def test_predict_samples_is_reproducible_under_fixed_seed():
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


# ===========================================================================
# VERIFICATION GATE
# ===========================================================================

# --- gate 1: two-Gaussian KL reduces to KL(. || N(0, I)) -------------------

def test_kl_reduces_to_standard_against_unit_normal():
    import torch

    torch.manual_seed(0)
    b, d = 8, 4
    mu_q = torch.randn(b, d)
    logvar_q = torch.randn(b, d)
    mu_p = torch.zeros(b, d)
    logvar_p = torch.zeros(b, d)
    two_gaussian = Cvae._kl(mu_q, logvar_q, mu_p, logvar_p)
    standard = (-0.5 * (1 + logvar_q - mu_q.pow(2) - logvar_q.exp()).sum(dim=1)).mean()
    assert torch.allclose(two_gaussian, standard, atol=1e-6)


def test_kl_is_zero_for_identical_distributions():
    import torch

    torch.manual_seed(1)
    mu, logvar = torch.randn(5, 3), torch.randn(5, 3)
    assert torch.allclose(
        Cvae._kl(mu, logvar, mu, logvar), torch.zeros(()), atol=1e-6
    )


# --- gate 2: recon == n_k * mean-over-bins MSE -----------------------------

def test_recon_equals_n_k_times_mse():
    import torch

    torch.manual_seed(0)
    b, n_k = 7, 6
    pred = torch.randn(b, n_k)
    y = torch.randn(b, n_k)
    recon = Cvae._recon(pred, y)
    mse = torch.mean((pred - y) ** 2)
    assert torch.allclose(recon, n_k * mse, atol=1e-6)


# --- gate 3: every logvar head respects the clamp under huge inputs --------

def test_all_logvar_heads_respect_clamp_under_1e6_inputs():
    import torch

    td = _training_data(with_cond=False, n=30, n_radii=5, n_k=3)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=1, epochs=2, batch_size=8,
        seed=0, device="cpu",
    )
    model.fit(td)
    x_big = model._to_tensor(np.full((4, td.X.shape[1]), 1e6))
    y_big = model._to_tensor(np.full((4, td.y.shape[1]), 1e6))
    with torch.no_grad():
        _mu_p, logvar_p = model._prior_dist(x_big, None)
        _mu_q, logvar_q = model._recognition_dist(x_big, y_big, None)
    for logvar in (logvar_p, logvar_q):
        assert float(logvar.min()) >= -8.0 - 1e-6
        assert float(logvar.max()) <= 8.0 + 1e-6


# --- gate 4: synthetic heteroscedastic / bimodal recovery ------------------

def _hetero_bimodal_data(kind: str, n: int, seed: int):
    """Synthetic (x, y, sigma) with a known input-dependent conditional spread.

    Profile-only ``x ~ U(-1, 1)`` (shape (n, 1)); a smooth 2-D mean ``f(x)``; and
    either heteroscedastic Gaussian noise ``eps ~ N(0, sigma(x)^2)`` with
    ``sigma(x) = 0.05 + 0.3 x^2``, or a bimodal conditional ``y = f(x) +/- g(x)``
    with equal-probability sign and ``g(x)`` an O(0.3) smooth function. ``sigma``
    is the true per-example conditional-spread scale used by the gate.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    f = np.hstack([0.5 * np.sin(2.0 * x) + 0.3 * x, 0.4 * np.cos(1.5 * x)])
    sigma = 0.05 + 0.3 * x[:, 0] ** 2
    if kind == "hetero":
        y = f + rng.normal(size=(n, 2)) * sigma[:, None]
    else:  # bimodal
        g = 0.3 * (1.0 + 0.3 * np.sin(2.0 * x[:, 0]))
        s = rng.choice([-1.0, 1.0], size=n)
        y = f + (s * g)[:, None]
        sigma = g
    return x, y, sigma


def _fit_and_measure(kind: str, seed: int, n: int = 2000, n_samples: int = 300):
    """Fit a profile-only Cvae and return (held-out Spearman, held-out coverage).

    The predictive std per held-out example (mean across target bins of the
    across-sample std) is rank-correlated with the true ``sigma(x)``; the empirical
    coverage is the fraction of held-out targets inside the central-68% interval
    formed from the conditional-prior samples.
    """
    from scipy.stats import spearmanr

    x, y, sigma = _hetero_bimodal_data(kind, n=n, seed=seed)
    order = np.random.default_rng(seed + 1).permutation(n)
    n_tr = int(0.8 * n)
    tr, ho = order[:n_tr], order[n_tr:]
    td = TrainingData(
        X=x[tr], X_cond=None, X_params=None, y=y[tr], nd=np.zeros(n_tr),
        sim_index=np.arange(n_tr), k=np.linspace(0.5, 5.0, y.shape[1]),
        radii_mpch=np.array([0.1]), source_path="synthetic",
    )
    # beta < 1 is deliberate: with no explicit observation-noise head, ALL
    # predictive spread comes from the latent, so the KL must not over-compress it
    # for the samples to be calibrated. This is the CVAE's calibration knob.
    model = Cvae(
        latent_dim=2, hidden=64, n_layers=2, epochs=300, lr=1e-3,
        beta=0.01, batch_size=128, val_frac=0.1, seed=0, device="cpu",
    )
    model.fit(td)
    samples = model.predict_samples(x[ho], n_samples=n_samples, seed=123)
    per_ex_std = samples.std(axis=0).mean(axis=1)  # (n_ho,)
    rho = float(spearmanr(per_ex_std, sigma[ho]).statistic)
    lo = np.quantile(samples, 0.16, axis=0)
    hi = np.quantile(samples, 0.84, axis=0)
    coverage = float(np.mean((y[ho] >= lo) & (y[ho] <= hi)))
    return rho, coverage


@pytest.fixture(scope="module")
def _gate4_recovery():
    """Fit the gate-4 recovery models once and cache (Spearman, coverage) results.

    Heteroscedastic coverage is tight across seeds, so one fit (seed 0) supplies
    both its Spearman and its coverage. The bimodal conditional is a discrete
    mixture whose per-fit coverage is genuinely seed-noisy, so it is averaged over
    five independent fits -- the coverage of a *calibrated* conditional model is
    68% by the probability-integral-transform whatever the modality, and the mean
    over fits is the stable empirical estimate. ~6 CPU fits; this fixture is the
    slow part of the suite by design (it is the decisive verification gate).
    """
    hetero = _fit_and_measure("hetero", seed=0)  # (rho, coverage)
    bimodal = [_fit_and_measure("bimodal", seed=s) for s in range(5)]
    return {"hetero": hetero, "bimodal": bimodal}


def test_gate4a_hetero_spread_tracks_true_sigma(_gate4_recovery):
    # (a) The per-example predictive std rank-tracks the true sigma(x): Spearman
    # well above 0.8 on held-out data. This is the property the Task-A VIB
    # architecture CANNOT have -- its N(0, I)-prior sample spread is a
    # regularisation floor independent of the structure of p(y|x), so it cannot
    # rank-order examples by their true conditional width (that is why
    # vib_regressor exposes no predict_samples at all).
    rho, _cov = _gate4_recovery["hetero"]
    assert rho > 0.8, f"held-out Spearman(pred_std, sigma) = {rho:.3f}"


def test_gate4b_coverage_is_calibrated_both_variants(_gate4_recovery):
    # (b) Empirical central-68% coverage on held-out data lies in [0.58, 0.78] for
    # both the heteroscedastic and the bimodal conditional (the latter averaged
    # over independent fits; see the fixture).
    _rho, hetero_cov = _gate4_recovery["hetero"]
    assert 0.58 <= hetero_cov <= 0.78, f"hetero coverage {hetero_cov:.3f}"
    bimodal_covs = [cov for _r, cov in _gate4_recovery["bimodal"]]
    mean_cov = float(np.mean(bimodal_covs))
    assert 0.58 <= mean_cov <= 0.78, f"bimodal coverage {mean_cov:.3f} from {bimodal_covs}"


# ===========================================================================
# OBSERVATION-NOISE HEAD (obs_noise_head=True)
# ===========================================================================

def test_obs_noise_head_defaults_off_and_off_toggle_is_equivalent():
    # obs_noise_head defaults to False, and an explicit False is bit-identical to
    # a default construction: same seeded fit, predictions, and samples. (The
    # before/after-edit regression on the pre-head code is run out-of-suite; this
    # pins the in-suite equivalence the default relies on.)
    td = _training_data(with_cond=True, n=24, n_k=5, seed=8)
    kwargs = dict(latent_dim=4, hidden=16, n_layers=2, epochs=10, batch_size=8,
                  seed=0, device="cpu")
    default = Cvae(**kwargs)
    explicit = Cvae(obs_noise_head=False, **kwargs)
    assert default.obs_noise_head is False
    default.fit(td)
    explicit.fit(td)
    np.testing.assert_array_equal(
        default.predict(td.X, td.X_cond), explicit.predict(td.X, td.X_cond)
    )
    np.testing.assert_array_equal(
        default.predict_samples(td.X, td.X_cond, n_samples=16, seed=3),
        explicit.predict_samples(td.X, td.X_cond, n_samples=16, seed=3),
    )
    assert default._obs_logvar is None and explicit._obs_logvar is None


def test_obs_noise_head_trains_and_obs_sigma_is_positive_per_k():
    td = _training_data(with_cond=True, n=24, n_k=5, seed=9)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=20, batch_size=8,
        obs_noise_head=True, seed=0, device="cpu",
    )
    model.fit(td)
    sigma = model.obs_sigma()
    assert sigma.shape == (td.y.shape[1],)
    assert (sigma > 0.0).all()
    # The head is in the optimiser: it must have moved off its init value
    # (exp(0.5 * 0) * y_std = y_std exactly).
    assert not np.allclose(sigma, model._y_std)


def test_obs_sigma_raises_when_head_disabled_or_before_fit():
    td = _training_data(with_cond=True, n=16, n_k=4)
    off = Cvae(latent_dim=3, hidden=16, n_layers=1, epochs=2, batch_size=8,
               seed=0, device="cpu")
    off.fit(td)
    with pytest.raises(RuntimeError, match="obs_noise_head"):
        off.obs_sigma()
    unfit = Cvae(obs_noise_head=True, seed=0, device="cpu")
    with pytest.raises(RuntimeError, match="before fit"):
        unfit.obs_sigma()


def test_obs_noise_head_widens_samples_over_latent_only():
    # From the SAME fitted weights, samples with the observation noise must be
    # strictly wider per example than the latent-only spread (the head-off
    # sampling path, obtained by toggling the flag on the fitted model).
    td = _training_data(with_cond=True, n=20, n_k=5, seed=10)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=30, batch_size=8,
        obs_noise_head=True, seed=0, device="cpu",
    )
    model.fit(td)
    with_noise = model.predict_samples(td.X, td.X_cond, n_samples=200, seed=5)
    model.obs_noise_head = False  # sampling dispatch only; weights untouched
    latent_only = model.predict_samples(td.X, td.X_cond, n_samples=200, seed=5)
    model.obs_noise_head = True
    spread_on = with_noise.std(axis=0).mean(axis=1)    # (n_examples,)
    spread_off = latent_only.std(axis=0).mean(axis=1)  # (n_examples,)
    assert (spread_on > spread_off).all()


def test_obs_noise_head_samples_reproducible_under_fixed_seed():
    td = _training_data(with_cond=True, n=20, n_k=4, seed=11)
    model = Cvae(
        latent_dim=4, hidden=16, n_layers=2, epochs=15, batch_size=8,
        obs_noise_head=True, seed=0, device="cpu",
    )
    model.fit(td)
    s_a = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=21)
    s_b = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=21)
    np.testing.assert_array_equal(s_a, s_b)
    s_c = model.predict_samples(td.X, td.X_cond, n_samples=16, seed=22)
    assert not np.allclose(s_a, s_c)


def test_obs_noise_head_single_k_path():
    # A 1-D (single_k) target: obs_logvar has shape (1,), obs_sigma() returns
    # (1,), and sampling keeps the (n_samples, n_examples) shape.
    td = _training_data(with_cond=True, n=20, n_k=5, seed=12)
    td.y = td.y[:, 0]
    model = Cvae(
        latent_dim=3, hidden=16, n_layers=1, epochs=5, batch_size=8,
        obs_noise_head=True, seed=0, device="cpu",
    )
    model.fit(td)
    sigma = model.obs_sigma()
    assert sigma.shape == (1,)
    assert (sigma > 0.0).all()
    samples = model.predict_samples(td.X, td.X_cond, n_samples=9)
    assert samples.shape == (9, td.X.shape[0])
    pred = model.predict(td.X, td.X_cond)
    assert pred.shape == (td.X.shape[0],)


def test_nll_matches_closed_form_and_clamps():
    # The NLL reconstruction equals 0.5 * sum_k(lv + err^2 / e^lv), batch mean,
    # with obs_logvar clamped to [-8, 8] (GaussianVAE convention, log 2pi
    # dropped).
    import torch

    td = _training_data(with_cond=False, n=16, n_k=3)
    model = Cvae(latent_dim=2, hidden=8, n_layers=1, epochs=0, val_frac=0.0,
                 obs_noise_head=True, seed=0, device="cpu")
    model.fit(td)  # builds modules; no training steps
    torch.manual_seed(0)
    pred, y = torch.randn(6, 3), torch.randn(6, 3)
    with torch.no_grad():
        model._obs_logvar.copy_(torch.tensor([-20.0, 0.5, 20.0]))
        got = model._nll(pred, y)
        lv = torch.tensor([-8.0, 0.5, 8.0])  # the clamped values
        want = (0.5 * ((pred - y).pow(2) / lv.exp() + lv).sum(dim=1)).mean()
    assert torch.allclose(got, want, atol=1e-6)


# --- the decisive noise-head gate: known-sigma recovery + calibration -------

@pytest.fixture(scope="module")
def _noise_head_recovery():
    """Fit the noise-head model once on y = f(x) + eps with known per-k sigma.

    Homoscedastic Gaussian noise with distinct per-k scales (0.10, 0.20) on a
    smooth 2-D mean: the conditional spread is pure observation noise, so a
    correct head absorbs it into ``obs_logvar`` and the latent contributes
    little. One CPU fit at the frozen operating point (beta = 1.0 -- unlike the
    head-off gate-4 fits, no lowered beta is needed: the aleatoric term lives in
    the head, not the latent).
    """
    true_sigma = np.array([0.10, 0.20])
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    f = np.hstack([0.5 * np.sin(2.0 * x) + 0.3 * x, 0.4 * np.cos(1.5 * x)])
    y = f + rng.normal(size=(n, 2)) * true_sigma
    order = rng.permutation(n)
    tr, ho = order[: int(0.8 * n)], order[int(0.8 * n):]
    td = TrainingData(
        X=x[tr], X_cond=None, X_params=None, y=y[tr], nd=np.zeros(tr.size),
        sim_index=np.arange(tr.size), k=np.linspace(0.5, 5.0, 2),
        radii_mpch=np.array([0.1]), source_path="synthetic",
    )
    model = Cvae(
        latent_dim=2, hidden=64, n_layers=2, epochs=300, lr=1e-3, beta=1.0,
        batch_size=128, val_frac=0.1, obs_noise_head=True, seed=0, device="cpu",
    )
    model.fit(td)
    samples = model.predict_samples(x[ho], n_samples=300, seed=123)
    lo = np.quantile(samples, 0.16, axis=0)
    hi = np.quantile(samples, 0.84, axis=0)
    coverage = float(np.mean((y[ho] >= lo) & (y[ho] <= hi)))
    return {"model": model, "true_sigma": true_sigma, "coverage": coverage}


def test_noise_head_recovers_known_sigma(_noise_head_recovery):
    # obs_sigma() recovers the true homoscedastic per-k sigma. The learned scale
    # also absorbs the (small) mean-fit residual, so the tolerance is one-sided
    # -friendly: within [0.8, 1.3] of truth per k bin.
    sigma = _noise_head_recovery["model"].obs_sigma()
    true_sigma = _noise_head_recovery["true_sigma"]
    ratio = sigma / true_sigma
    assert ratio.shape == (2,)
    assert (0.8 <= ratio).all() and (ratio <= 1.3).all(), f"sigma ratio {ratio}"


def test_noise_head_central_interval_is_calibrated(_noise_head_recovery):
    # Held-out central-68% coverage ~ 0.68 (same band as the gate-4 checks).
    coverage = _noise_head_recovery["coverage"]
    assert 0.58 <= coverage <= 0.78, f"held-out 68% coverage {coverage:.3f}"


# --- observation likelihood: gaussian | student_t --------------------------

def test_obs_likelihood_defaults_to_gaussian():
    m = Cvae()
    assert m.obs_likelihood == "gaussian"
    assert m.obs_df == pytest.approx(5.0)


def test_invalid_obs_likelihood_raises():
    with pytest.raises(ValueError, match="obs_likelihood must be in"):
        Cvae(obs_likelihood="cauchy")


@pytest.mark.parametrize("df", [2.0, 1.5, 0.0, -1.0])
def test_obs_df_at_or_below_two_raises(df):
    # nu <= 2 has infinite variance, so the calibration and obs_predictive_sd
    # would be undefined.
    with pytest.raises(ValueError, match="obs_df must exceed"):
        Cvae(obs_likelihood="student_t", obs_df=df)


def test_gaussian_likelihood_is_bit_identical_to_the_pre_change_default():
    td = _training_data(with_cond=False)
    common = dict(epochs=8, hidden=16, n_layers=1, latent_dim=2, seed=0,
                  device="cpu", obs_noise_head=True)
    implicit = Cvae(**common)
    explicit = Cvae(obs_likelihood="gaussian", obs_df=5.0, **common)
    implicit.fit(td)
    explicit.fit(td)
    assert np.array_equal(implicit.predict(td.X), explicit.predict(td.X))
    assert np.array_equal(implicit.obs_sigma(), explicit.obs_sigma())


def test_student_t_nll_tends_to_gaussian_nll_as_df_grows():
    torch = pytest.importorskip("torch")
    import math

    torch.manual_seed(0)
    n_k = 4
    pred = torch.randn(20, n_k)
    y = pred + 0.3 * torch.randn(20, n_k)
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", device="cpu")
    m._device = "cpu"
    m._obs_logvar = torch.nn.Parameter(torch.randn(n_k) * 0.3)
    lv = m._obs_logvar.clamp(-8.0, 8.0)
    # The Gaussian NLL as implemented drops log(2*pi); the t keeps every
    # constant, so compare against the Gaussian WITH the constant restored.
    gauss = (0.5 * (lv + ((y - pred) ** 2) / torch.exp(lv)
                    + math.log(2 * math.pi))).sum(dim=1).mean()
    m.obs_df = 5.0
    near = abs(m._nll_student_t(pred, y).item() - gauss.item())
    m.obs_df = 1e7
    far = abs(m._nll_student_t(pred, y).item() - gauss.item())
    assert far < 1e-3, f"t-NLL should approach the Gaussian NLL; got {far:.2e}"
    assert far < near, "convergence must be monotone in nu, not divergent"


def test_learned_df_stays_above_two():
    torch = pytest.importorskip("torch")
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=None,
             device="cpu")
    m._device = "cpu"
    m._obs_raw_df = torch.nn.Parameter(
        torch.tensor([-100.0, -10.0, 0.0, 10.0, 100.0])
    )
    nu = m._current_df().detach().numpy()
    assert (nu > 2.0).all(), f"nu must stay > 2 for finite variance; got {nu}"


def test_student_t_samples_match_exact_quantiles():
    # Moment-based checks are useless for heavy tails (the 4th moment barely
    # converges), so verify the sampler against exact t quantiles instead.
    stats = pytest.importorskip("scipy.stats")
    td = _training_data(with_cond=False, n=60, n_k=3)
    nu = 5.0
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=nu,
             epochs=8, hidden=16, n_layers=1, latent_dim=2, seed=0, device="cpu")
    m.fit(td)
    samples = m.predict_samples(td.X[:1], None, None, n_samples=40000, seed=0)
    z = (samples[:, 0, :] - m.predict(td.X[:1], None, None)[0]) / m.obs_sigma()
    for q in (0.05, 0.25, 0.75, 0.95):
        emp = np.quantile(z, q, axis=0).mean()
        assert emp == pytest.approx(stats.t.ppf(q, nu), abs=0.06), (
            f"quantile {q}: empirical {emp:.3f} vs exact {stats.t.ppf(q, nu):.3f}"
        )


def test_student_t_sampling_is_seeded_and_reproducible():
    td = _training_data(with_cond=False, n=40)
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=4.0,
             epochs=8, hidden=16, n_layers=1, latent_dim=2, seed=0, device="cpu")
    m.fit(td)
    a = m.predict_samples(td.X[:4], None, None, n_samples=25, seed=3)
    b = m.predict_samples(td.X[:4], None, None, n_samples=25, seed=3)
    c = m.predict_samples(td.X[:4], None, None, n_samples=25, seed=4)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


@pytest.mark.parametrize("nu,sigma", [(3.0, 0.2), (5.0, 0.05), (10.0, 0.5)])
def test_calibration_recovers_known_student_t_scale(nu, sigma):
    stats = pytest.importorskip("scipy.stats")
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=nu,
             device="cpu")
    resid = stats.t.rvs(nu, size=(4000, 3), random_state=np.random.default_rng(0))
    var, fitted_nu = m._fit_student_t(resid * sigma)
    assert np.sqrt(var).mean() == pytest.approx(sigma, rel=0.06)
    assert fitted_nu == pytest.approx(nu)


def test_calibration_recovers_learned_df():
    stats = pytest.importorskip("scipy.stats")
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=None,
             device="cpu")
    resid = stats.t.rvs(4.0, size=(8000, 3), random_state=np.random.default_rng(1))
    _var, nu = m._fit_student_t(resid * 0.1)
    assert nu.mean() == pytest.approx(4.0, rel=0.25), f"fitted nu {nu}"


def test_calibration_picks_light_tails_for_gaussian_residuals():
    # A learned nu must not invent heavy tails that are not there.
    m = Cvae(obs_noise_head=True, obs_likelihood="student_t", obs_df=None,
             device="cpu")
    resid = np.random.default_rng(2).normal(scale=0.1, size=(8000, 3))
    _var, nu = m._fit_student_t(resid)
    assert nu.mean() > 30.0, f"expected a near-Gaussian nu; got {nu}"


def test_obs_accessors_are_consistent_across_likelihoods():
    td = _training_data(with_cond=False, n=60)
    common = dict(epochs=8, hidden=16, n_layers=1, latent_dim=2, seed=0,
                  device="cpu", obs_noise_head=True)
    g = Cvae(**common)
    g.fit(td)
    assert np.isinf(g.obs_fitted_df()).all()
    assert np.array_equal(g.obs_predictive_sd(), g.obs_sigma())

    nu = 5.0
    t = Cvae(obs_likelihood="student_t", obs_df=nu, **common)
    t.fit(td)
    assert t.obs_fitted_df() == pytest.approx(nu)
    # sd = scale * sqrt(nu / (nu - 2)); scale alone would understate the spread
    assert t.obs_predictive_sd() == pytest.approx(
        t.obs_sigma() * np.sqrt(nu / (nu - 2.0))
    )


def test_student_t_beats_gaussian_coverage_on_heavy_tailed_data():
    """The end-to-end claim: when the residuals are heavy-tailed, a
    variance-matched Gaussian over-covers a central interval (its sd is inflated
    by rare large errors, so the bulk sits well inside +-1 sd) while the
    Student-t lands near nominal.

    The target is deliberately LINEAR and the injected noise large: the effect
    only exists when the residual is dominated by the heavy-tailed noise rather
    than by the model's own approximation error. The kurtosis assertion guards
    that precondition -- without it the test can pass vacuously on residuals
    that are not heavy-tailed at all.
    """
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    n, n_k, n_tr = 800, 4, 600
    X = rng.normal(size=(n, 6))
    W = rng.normal(size=(6, n_k)) / np.sqrt(6)
    y = X @ W + 0.30 * stats.t.rvs(3.0, size=(n, n_k), random_state=rng)
    tr = TrainingData(
        X=X[:n_tr], X_cond=None, X_params=None, y=y[:n_tr], nd=np.zeros(n_tr),
        sim_index=np.arange(n_tr), k=np.linspace(0.1, 1.0, n_k),
        radii_mpch=np.linspace(0.1, 3.0, 6), source_path="synthetic",
    )
    common = dict(epochs=300, hidden=64, n_layers=2, latent_dim=2, seed=0,
                  device="cpu", obs_noise_head=True, obs_calibrate_on_val=True)

    def cov68(model):
        s = model.predict_samples(X[n_tr:], None, None, n_samples=400, seed=0)
        lo, hi = np.quantile(s, 0.16, axis=0), np.quantile(s, 0.84, axis=0)
        return float(np.mean((y[n_tr:] >= lo) & (y[n_tr:] <= hi)))

    g = Cvae(obs_likelihood="gaussian", **common)
    g.fit(tr)
    resid = (y[n_tr:] - g.predict(X[n_tr:], None, None)).ravel()
    assert stats.kurtosis(resid) > 3.0, (
        f"precondition failed: residual kurtosis {stats.kurtosis(resid):.1f} is "
        "not heavy-tailed, so this test would pass vacuously"
    )
    t = Cvae(obs_likelihood="student_t", obs_df=3.0, **common)
    t.fit(tr)
    cov_g, cov_t = cov68(g), cov68(t)
    assert cov_g > 0.75, f"Gaussian should over-cover here; got {cov_g:.3f}"
    assert abs(cov_t - 0.68) < abs(cov_g - 0.68), (
        f"student_t should be closer to nominal 0.68: gaussian {cov_g:.3f} "
        f"vs t {cov_t:.3f}"
    )
