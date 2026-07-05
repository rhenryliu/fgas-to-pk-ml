"""Tests for the dual-VAE :class:`~fgas_spk.models.dual_vae_components.GaussianVAE`.

Coverage: the module imports without torch (lazy-dependency contract, checked
in a fresh subprocess); fit/encode/decode/reconstruct/obs_sigma/latent_stats
return the right shapes on the with-context and no-context paths; the context
and before-fit guards raise; training is deterministic on a fixed CPU backend;
the history carries the spec-required per-epoch keys (total loss, NLL, total
KL, per-dimension KL, activity A_j); the loss terms match their documented
formulas (GR7 one-convention reduction); and -- the decisive check for the
F0.2 noise head -- on synthetic data with a known per-bin observation noise,
the fitted ``obs_sigma()`` recovers that noise per bin.

Every fixture is synthetic -- no CAMELS data.
"""

import math
import subprocess
import sys

import numpy as np
import pytest

from fgas_spk.models.dual_vae_components import GaussianVAE


def _linear_latent_data(
    n: int = 512,
    n_bins: int = 8,
    n_latent: int = 2,
    noise: float | np.ndarray = 0.02,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic modality: a linear map of a low-dim latent plus Gaussian noise.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(v, z_true)`` with ``v`` of shape
            (n, n_bins) and the generating latents of shape (n, n_latent).
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, n_latent))
    basis = rng.normal(size=(n_latent, n_bins))
    v = 1.0 + z @ basis + rng.normal(size=(n, n_bins)) * noise
    return v, z


def _fast_vae(**kw) -> GaussianVAE:
    """A small, quick CPU configuration for tests."""
    defaults = dict(
        latent_dim=2, hidden=32, n_layers=2, epochs=60, lr=1e-2,
        weight_decay=0.0, batch_size=128, val_frac=0.1, seed=0, device="cpu",
    )
    defaults.update(kw)
    return GaussianVAE(**defaults)


# --- lazy torch / construction ----------------------------------------------

def test_importing_component_does_not_import_torch():
    code = (
        "import sys; import fgas_spk.models.dual_vae_components; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code])
    assert result.returncode == 0


def test_construction_tolerates_unexpected_keys():
    model = GaussianVAE(seed=0, device="cpu", not_a_real_key=123)
    assert model.latent_dim == 2
    assert model.anneal_epochs == 400 // 4  # None -> epochs // 4


# --- shapes and guards --------------------------------------------------------

def test_fit_encode_decode_shapes_with_context():
    v, _ = _linear_latent_data()
    ctx = np.random.default_rng(1).normal(size=(v.shape[0], 5))
    model = _fast_vae(epochs=10)
    model.fit(v, ctx)
    mu = model.encode(v, ctx)
    assert mu.shape == (v.shape[0], 2)
    mu2, std = model.encode(v, ctx, return_std=True)
    assert np.array_equal(mu, mu2) and std.shape == mu.shape and (std > 0).all()
    recon = model.reconstruct(v, ctx)
    assert recon.shape == v.shape
    dec = model.decode(mu, ctx)
    assert np.allclose(dec, recon)
    assert model.obs_sigma().shape == (v.shape[1],)
    stats = model.latent_stats(v, ctx)
    assert len(stats["kl_per_dim"]) == 2 and len(stats["activity"]) == 2
    assert stats["mu"].shape == mu.shape
    assert model.best_epoch is not None


def test_fit_without_context_and_guards():
    v, _ = _linear_latent_data(n=128)
    ctx = np.zeros((v.shape[0], 3))
    model = _fast_vae(epochs=5)
    model.fit(v)
    assert model.encode(v).shape == (v.shape[0], 2)
    with pytest.raises(ValueError):
        model.encode(v, ctx)  # fit without context, called with one
    model_ctx = _fast_vae(epochs=5)
    model_ctx.fit(v, ctx)
    with pytest.raises(ValueError):
        model_ctx.reconstruct(v)  # fit with context, called without


def test_before_fit_raises():
    model = _fast_vae()
    v = np.zeros((4, 8))
    with pytest.raises(RuntimeError):
        model.encode(v)
    with pytest.raises(RuntimeError):
        model.decode(np.zeros((4, 2)))
    with pytest.raises(RuntimeError):
        model.obs_sigma()


# --- history / determinism ----------------------------------------------------

def test_history_carries_spec_required_keys():
    v, _ = _linear_latent_data(n=128)
    model = _fast_vae(epochs=6)
    model.fit(v)
    assert len(model.history) == 6
    row = model.history[-1]
    for key in ("epoch", "train_loss", "nll", "kl", "kl_per_dim", "activity",
                "beta_eff", "val_loss"):
        assert key in row
    assert len(row["kl_per_dim"]) == 2 and len(row["activity"]) == 2
    assert row["beta_eff"] == pytest.approx(1.0)  # past the anneal ramp


def test_determinism_on_fixed_cpu_backend():
    v, _ = _linear_latent_data(n=128)
    a = _fast_vae(epochs=8, seed=123)
    b = _fast_vae(epochs=8, seed=123)
    a.fit(v)
    b.fit(v)
    assert np.array_equal(a.reconstruct(v), b.reconstruct(v))
    assert np.array_equal(a.obs_sigma(), b.obs_sigma())


# --- loss conventions (GR7) ---------------------------------------------------

def test_nll_matches_documented_formula():
    import torch

    v, _ = _linear_latent_data(n=64)
    model = _fast_vae(epochs=2)
    model.fit(v)
    v_hat = torch.zeros(3, v.shape[1])
    vt = torch.ones(3, v.shape[1])
    with torch.no_grad():
        model._obs_logvar.zero_()
        got = float(model._nll(v_hat, vt).item())
    # obs_logvar == 0: nll = 0.5 * sum_bins(err^2 + log 2pi), batch mean.
    expected = 0.5 * v.shape[1] * (1.0 + math.log(2.0 * math.pi))
    assert got == pytest.approx(expected, rel=1e-6)


def test_kl_per_dim_zero_at_standard_normal_and_positive_otherwise():
    import torch

    kl0 = GaussianVAE._kl_per_dim(torch.zeros(5, 3), torch.zeros(5, 3))
    assert torch.allclose(kl0, torch.zeros(3), atol=1e-7)
    kl1 = GaussianVAE._kl_per_dim(torch.ones(5, 3), torch.zeros(5, 3))
    assert torch.allclose(kl1, 0.5 * torch.ones(3), atol=1e-6)


# --- the decisive check: noise-head recovery ----------------------------------

def test_obs_sigma_recovers_per_bin_noise():
    # Rank-2 signal plus per-bin noise spanning a factor of 8. A learned noise
    # head absorbs residual decoder-approximation error on top of the true
    # observation noise, so the guarantees are one-sided where noise is small:
    # (i) bins where the noise dominates the residual (true sigma >= 0.08) are
    # recovered within a factor of 2; (ii) no bin is *under*-estimated below
    # half its true noise (the under-coverage failure mode the F0.2 head
    # exists to prevent); (iii) the noisiest bin is identified.
    n_bins = 6
    true_sigma = np.array([0.02, 0.04, 0.08, 0.16, 0.08, 0.04])
    v, _ = _linear_latent_data(n=1024, n_bins=n_bins, noise=true_sigma, seed=3)
    model = _fast_vae(epochs=200, hidden=64, seed=0)
    model.fit(v)
    fitted = model.obs_sigma()
    ratio = fitted / true_sigma
    loud = true_sigma >= 0.08
    assert (ratio[loud] > 0.5).all() and (ratio[loud] < 2.0).all(), (
        fitted, true_sigma,
    )
    assert (ratio > 0.5).all(), (fitted, true_sigma)
    assert int(np.argmax(fitted)) == int(np.argmax(true_sigma))


# --- latent-map rungs (Stage 4) ------------------------------------------------

def test_latent_map_mlp_recovers_nonlinear_map():
    from fgas_spk.models.dual_vae_components import LatentMapMLP

    rng = np.random.default_rng(0)
    z1 = rng.normal(size=(600, 2))
    z2_clean = np.tanh(z1 @ np.array([[1.5, -0.7], [0.3, 2.0]]))
    z2 = z2_clean + rng.normal(size=z2_clean.shape) * 0.05
    mlp = LatentMapMLP(epochs=250, seed=0, device="cpu")
    mlp.fit(z1, z2)
    pred = mlp.predict(z1)
    assert pred.shape == z2.shape
    assert float(np.sqrt(np.mean((pred - z2_clean) ** 2))) < 0.06
    assert mlp.best_epoch is not None and len(mlp.history) == 250


def test_latent_map_mdn_bimodal_and_seeded():
    from fgas_spk.models.dual_vae_components import LatentMapMDN

    rng = np.random.default_rng(1)
    z1 = rng.normal(size=(600, 2))
    sign = rng.choice([-1.0, 1.0], size=(600, 1))
    z2 = np.hstack([z1[:, :1] * 0.5 + sign, -z1[:, 1:] * 0.3]) \
        + rng.normal(size=(600, 2)) * 0.05
    mdn = LatentMapMDN(n_components=3, epochs=300, seed=0, device="cpu")
    mdn.fit(z1, z2)
    s = mdn.sample(z1[:4], n_samples=2000, seed=1)
    assert s.shape == (2000, 4, 2)
    # Both branches of the bimodal conditional must be populated.
    frac_hi = float((s[:, 0, 0] > z1[0, 0] * 0.5).mean())
    assert 0.2 < frac_hi < 0.8
    # Mixture-mean point prediction has the target shape.
    assert mdn.predict(z1[:4]).shape == (4, 2)
    # Sampling is reproducible under a fixed seed.
    assert np.array_equal(mdn.sample(z1[:3], 10, seed=5),
                          mdn.sample(z1[:3], 10, seed=5))


def test_composite_helpers_shapes_and_f05_semantics():
    from fgas_spk.models.dual_vae_components import (
        GaussianVAE, LatentMapMDN, composite_predict, composite_samples,
    )

    rng = np.random.default_rng(2)
    n, n_r, n_k, n_ctx = 200, 6, 5, 3
    ctx = rng.normal(size=(n, n_ctx))
    x = rng.normal(size=(n, n_r))
    y = rng.normal(size=(n, n_k))
    vx = _fast_vae(epochs=10)
    vx.fit(x, ctx)
    vy = _fast_vae(epochs=10)
    vy.fit(y, ctx)
    mdn = LatentMapMDN(n_components=2, epochs=30, seed=0, device="cpu")
    mdn.fit(vx.encode(x, ctx), vy.encode(y, ctx))

    point = composite_predict(vx, mdn, vy, x, ctx)
    assert point.shape == (n, n_k)
    s = composite_samples(vx, mdn, vy, x, ctx, n_samples=20, seed=0)
    assert s.shape == (20, n, n_k)
    # F0.5: the spread includes decoder-Y obs noise, so the across-sample std
    # is at least on the order of obs_sigma for every bin.
    spread = s.std(axis=0).mean(axis=0)
    assert (spread > 0.3 * vy.obs_sigma()).all()
