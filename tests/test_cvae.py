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
