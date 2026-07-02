"""Conditional variational autoencoder for f_gas(R) to SP(k) (PyTorch).

The probabilistic successor to :mod:`~fgas_spk.models.mlp_regressor`. It reuses
that model's encoder/decoder skeleton, its internal standardisation, its
conditioning-context wiring, and its device/guard machinery verbatim; the one
structural change is the bottleneck. Where the MLP has a *plain deterministic*
latent, the CVAE has a *sampled* latent with a KL-regularised evidence lower
bound (ELBO). This buys two things at once:

1. **A point prediction** -- the posterior mean -- that is directly comparable to
   the deterministic MLP on the same split. Point accuracy is a first-class goal:
   the KL term, dropout, and weight decay are all regularisers that can close (or
   reverse) a deterministic net's generalisation gap on this small suite, so
   beating the MLP's held-out RMSE is a legitimate target, not merely a tie.
2. **A predictive spread** -- the spread of decoded samples for a fixed profile --
   which is the scientific payload: how tightly a given f_gas(R) profile
   constrains SP(k). Read it off via :meth:`predict_samples`.

**Architecture (conditional VAE).**

The conditioning *context* is whichever of two optional modalities are present:
``X_cond`` (observable scalars -- number density, optional mean halo mass) and
``X_params`` (the simulation's CAMELS parameters, e.g. the five cosmological
parameters). Each is standardised on its own scale and the two are concatenated
into a single context vector, exactly as :class:`~fgas_spk.models.mlp_regressor.MlpRegressor`
does. The model does not assume which modality is present -- it handles whichever
are supplied, and the profile-only (no conditioning) path is supported too.

1. **encoder** maps ``concat(X, context)`` to *two* heads of width ``latent_dim``:
   the posterior mean ``mu`` and log-variance ``logvar`` of an approximate
   posterior ``q(z|x) = N(mu, exp(logvar))``. At train time the latent is drawn by
   the reparameterisation trick ``z = mu + exp(0.5 * logvar) * eps`` with
   ``eps ~ N(0, I)``, so gradients flow through ``mu`` / ``logvar``. At predict
   time the latent is the posterior mean (``z = mu``, no sampling), which makes
   :meth:`predict` deterministic.
2. **decoder** maps ``concat(z, context)`` to the SP(k) target of width ``n_k``.
   The decoder is conditioned on the context too -- mirroring the Lin et al. trick
   of conditioning the decoder on cosmology (the CAMELS parameters make that
   literal) -- so the latent is pushed to carry the residual feedback information
   that the profile underdetermines, rather than re-encoding the number density or
   the known parameters. With no conditioning the decoder reads the latent alone.

Input and output widths are inferred from the first :meth:`fit` batch; nothing
about the data dimensions is hardcoded. The default ``hidden`` is a deliberately
modest 128: with a small training set, regularisation is favoured over raw
capacity.

**Loss -- ELBO with KL annealing.** Per minibatch the loss is

    total = recon + beta_eff * KL

where ``recon`` is the mean-squared error between the decoder output and the
standardised target (mean over the batch *and* over ``n_k``, matching the MLP's
standardised-space objective), and ``KL`` is the analytic Kullback-Leibler
divergence between ``q(z|x) = N(mu, sigma^2)`` and the ``N(0, I)`` prior, summed
over the latent dimensions and averaged over the batch:

    KL = mean_batch( -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)) ).

``beta_eff`` is *annealed*: it ramps linearly from ~0 at epoch 0 up to the
configured ``beta`` (default 1.0) over the first ``anneal_epochs`` epochs
(default ``epochs // 4``), then holds at ``beta``. A plain ``beta = 1`` from the
first step tends to collapse the posterior at small ``N`` (see below); the
warm-up lets the decoder first learn to reconstruct through the latent before the
KL term pulls the posterior towards the prior. This is a monotonic (linear) ramp;
Lin et al. adopt a *cyclical* annealing schedule for the same purpose (avoiding
KL-vanishing at small ``N``), and the linear ramp is the simpler special case
used here.

**Posterior collapse (KL-vanishing).** With an expressive decoder and few
examples, a CVAE can reconstruct while ignoring the latent: the KL term decays
toward 0, ``sigma`` shrinks, and the predictive spread collapses to zero. That
failure is dangerous *scientifically*, not just numerically -- a collapsed latent
yields an artificially narrow posterior that is indistinguishable from a genuine
"f_gas tightly constrains SP(k)" finding. Annealing is the standard mitigation,
but it is not a guarantee, so the KL trace must stay monitorable: :attr:`history`
records ``kl`` (and ``beta_eff``) per epoch, and a ``kl`` decaying toward 0 in the
trace is the signal that any narrow posterior must **not** be trusted until
collapse is ruled out.

**Target scoping (documentation only).** This model is intended to be run against
the ``k_range`` target (k in [0.5, 5.0] h/Mpc), not the full SP(k) curve. The
full-curve high-k bins carry a known feedback-correlated bias in the within-hydro
suppression proxy (see CLAUDE.md, "Suppression target is a proxy"); on the full
curve, part of any predictive spread would reflect that target bias rather than
genuine f_gas underdetermination. The CVAE imposes no target choice -- it consumes
whatever ``td.y`` width it is given -- but the predictive spread from
:meth:`predict_samples` is only cleanly interpretable on the proxy-sound
k-window. No target-selection logic lives in the model; this is a documentation
requirement.

**Normalisation.** As in the MLP: raw f_gas, SP(k), and each conditioning modality
live on very different scales, so the model standardises internally. At
:meth:`fit` a per-column mean and standard deviation are fitted on the training
batch for the profile ``X``, each conditioning modality present (``X_cond`` and
``X_params``), and the target ``y``; the network sees and predicts standardised
quantities, and :meth:`predict` / :meth:`predict_samples` invert the target
standardisation on the way out. Zero-variance columns are given unit scale.

**Determinism.** :meth:`fit` seeds torch via :func:`torch.manual_seed`, which
gives run-to-run stability on a *fixed* backend (same machine, same device); it is
**not** bit-reproducible across machines or backends (CUDA/MPS kernels are not
guaranteed identical to CPU or to each other). Pass ``device="cpu"`` for a
backend-stable comparison. :meth:`predict` is deterministic (posterior mean, no
sampling). :meth:`predict_samples` is stochastic *by design* but seeded, so it is
reproducible on a fixed backend for a fixed seed.

In a run, the model is constructed from the
:class:`~fgas_spk.experiment.RunConfig` as
``Cvae(seed=run_config.seed, **run_config.model_params)`` and looked up by name
via ``fgas_spk.models.REGISTRY["cvae"]``.

Dependencies:
    PyTorch (pinned in the project environment, an opt-in training dependency,
    not part of the core numpy+pyyaml import surface). torch is imported **lazily
    inside the methods** -- never at module import or at registration -- so
    ``import fgas_spk.models`` stays torch-free and the ``@register`` side effect
    runs without torch present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register

if TYPE_CHECKING:  # import only for type hints; never pulls torch/loader at runtime
    from fgas_spk.loader import TrainingData


@register("cvae")
class Cvae:
    """Conditional VAE mapping f_gas(R) (+ conditioning) to a posterior over SP(k).

    Trains an ``encoder -> sampled latent -> decoder`` network by maximising a
    KL-annealed ELBO (see the module docstring), standardising its inputs and
    target internally. :meth:`predict` returns the deterministic posterior-mean
    point prediction; :meth:`predict_samples` returns decoded latent samples whose
    across-sample spread is the predictive uncertainty. Widths are inferred from
    the first :meth:`fit` batch; nothing about the data dimensions is hardcoded.

    Args:
        latent_dim (int): Width of the stochastic bottleneck. Defaults to 4.
        hidden (int): Hidden-layer width on each side of the bottleneck. A
            deliberately modest 128 by default: with a small training set, favour
            regularisation over raw capacity. Defaults to 128.
        n_layers (int): Number of hidden layers in the encoder and in the decoder
            (each). Defaults to 2.
        epochs (int): Number of training epochs. Defaults to 200.
        lr (float): Adam/AdamW learning rate. Defaults to 1e-3.
        weight_decay (float): Weight decay (regularisation). When > 0 the optimiser
            is AdamW (decoupled weight decay); when 0 it is plain Adam. Defaults to
            0.0.
        dropout (float): Dropout probability applied after each hidden activation
            in the encoder and decoder (regularisation). 0.0 disables it. Defaults
            to 0.0.
        batch_size (int): Minibatch size; capped at the training-set size at fit
            time. Defaults to 128.
        beta (float): Terminal KL weight in the ELBO after annealing. Defaults to
            1.0.
        anneal_epochs (int | None): Number of epochs over which ``beta_eff`` ramps
            linearly from ~0 to ``beta``. None selects ``epochs // 4``; a value
            <= 0 disables annealing (``beta`` applies from epoch 0). Defaults to
            None.
        seed (int): Reproducibility seed; from ``RunConfig.seed`` in a real run.
            Threaded into :func:`torch.manual_seed` at fit and used as the default
            sampling seed in :meth:`predict_samples`. Defaults to 0.
        device (str | None): Torch device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"mps"``). None auto-selects cuda -> mps -> cpu, mirroring
            :func:`fgas_spk.train.pick_device`. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored, so a stray
            ``model_params`` key never breaks construction.

    Attributes:
        history (list[dict]): One ``{"epoch", "recon", "kl", "beta_eff",
            "train_loss"}`` dict per epoch, populated by :meth:`fit`. ``recon`` and
            ``kl`` are the full-batch standardised-space reconstruction MSE (of the
            posterior-mean prediction) and the analytic KL; ``beta_eff`` is that
            epoch's annealed KL weight; ``train_loss`` is ``recon + beta_eff * kl``.
            The ``kl`` column is the collapse diagnostic (see the module docstring).
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        batch_size: int = 128,
        beta: float = 1.0,
        anneal_epochs: int | None = None,
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        # Construction is intentionally cheap and torch-free: store config only.
        # The modules are built (and torch imported) in fit, once dims are known.
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.beta = beta
        # Resolve the anneal window now (pure arithmetic, no torch): None -> a
        # quarter of the schedule; a non-positive value disables annealing.
        self.anneal_epochs = epochs // 4 if anneal_epochs is None else anneal_epochs
        self.seed = seed
        self.device = device

        self.history: list[dict] = []

        self._encoder = None
        self._decoder = None
        self._device = None
        self._uses_cond: bool = False
        self._uses_params: bool = False
        self._y_was_1d: bool = False
        # Standardisation statistics, fitted on the train batch in fit.
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._cond_mean: np.ndarray | None = None
        self._cond_std: np.ndarray | None = None
        self._params_mean: np.ndarray | None = None
        self._params_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    # --- public API --------------------------------------------------------

    def fit(self, training_data: "TrainingData") -> None:
        """Fit the CVAE on a TrainingData bundle by maximising the annealed ELBO.

        Consumes ``training_data.X`` (profiles), ``training_data.y`` (SP(k) target,
        the curve of shape ``(n_examples, n_k)``; a 1-D ``single_k`` target is
        supported and inverted back to 1-D at predict time), and -- when present --
        the two conditioning modalities ``training_data.X_cond`` (observable
        scalars) and ``training_data.X_params`` (CAMELS parameters). Both
        conditioning modalities are standardised on their own scale and
        concatenated into one context vector fed to the encoder *and* the decoder;
        whichever modalities are present at fit are then required at predict.
        Standardisation statistics are fitted here and stored for predict (see the
        module docstring). torch is imported lazily inside this method.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.
        """
        import torch
        from torch import nn

        X = np.asarray(training_data.X, dtype=np.float64)
        y = np.asarray(training_data.y, dtype=np.float64)
        X_cond = training_data.X_cond
        X_params = training_data.X_params

        self._uses_cond = X_cond is not None
        if self._uses_cond:
            print("Using conditioning scalars (X_cond) in Cvae.fit.")
            X_cond = np.asarray(X_cond, dtype=np.float64)

        self._uses_params = X_params is not None
        if self._uses_params:
            print("Using CAMELS parameters (X_params) in Cvae.fit.")
            X_params = np.asarray(X_params, dtype=np.float64)

        # Keep the target 2-D internally; remember a 1-D (single_k) input so the
        # prediction can be squeezed back to the caller's shape.
        self._y_was_1d = y.ndim == 1
        y2d = y[:, None] if self._y_was_1d else y

        # Fit normalisation on the training batch (each modality on its own scale).
        self._x_mean, self._x_std = self._fit_norm(X)
        self._y_mean, self._y_std = self._fit_norm(y2d)
        if X_cond is not None:
            self._cond_mean, self._cond_std = self._fit_norm(X_cond)
        if X_params is not None:
            self._params_mean, self._params_std = self._fit_norm(X_params)

        # Infer widths from this first batch -- no hardcoded dimensions. X_cond and
        # X_params are concatenated into one conditioning context of width n_context.
        n_profile = X.shape[1]
        n_cond = X_cond.shape[1] if X_cond is not None else 0
        n_params = X_params.shape[1] if X_params is not None else 0
        n_context = n_cond + n_params
        n_k = y2d.shape[1]

        # Seed, resolve the device, and build the modules now that dims are known.
        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._build_modules(n_profile, n_context, n_k)
        assert self._encoder is not None and self._decoder is not None
        self._encoder.to(self._device)
        self._decoder.to(self._device)

        # Standardise and move to the device as float32 tensors. The conditioning
        # tensor is the standardised [X_cond | X_params] context (None if neither).
        Xs = self._to_tensor(self._apply_norm(X, self._x_mean, self._x_std))
        ys = self._to_tensor(self._apply_norm(y2d, self._y_mean, self._y_std))
        context = self._context(X_cond, X_params)
        conds = self._to_tensor(context) if context is not None else None

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()  # mean over batch and n_k -> standardised-space recon

        n = Xs.shape[0]
        batch_size = min(self.batch_size, n)
        self.history = []
        for epoch in range(self.epochs):
            beta_eff = self._beta_eff(epoch)
            self._encoder.train()
            self._decoder.train()
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = Xs[idx]
                yb = ys[idx]
                cb = conds[idx] if conds is not None else None
                optimizer.zero_grad()
                mu, logvar = self._encode_dist(xb, cb)
                # Reparameterised sample at train time (global-seeded eps).
                z = self._reparameterise(mu, logvar)
                pred = self._decode(z, cb)
                recon = loss_fn(pred, yb)
                kl = self._kl(mu, logvar)
                loss = recon + beta_eff * kl
                loss.backward()
                optimizer.step()

            # Per-epoch trace: full-batch standardised-space recon of the POINT
            # prediction (posterior mean, matching predict), the analytic KL, the
            # annealed weight, and the total. The kl column is the collapse
            # diagnostic -- a value decaying toward 0 flags KL-vanishing.
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                mu_all, logvar_all = self._encode_dist(Xs, conds)
                pred_all = self._decode(mu_all, conds)
                recon_all = float(loss_fn(pred_all, ys).item())
                kl_all = float(self._kl(mu_all, logvar_all).item())
            self.history.append(
                {
                    "epoch": epoch,
                    "recon": recon_all,
                    "kl": kl_all,
                    "beta_eff": beta_eff,
                    "train_loss": recon_all + beta_eff * kl_all,
                }
            )

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict SP(k) for profiles ``X`` (the deterministic point prediction).

        Uses the posterior **mean** as the latent (``z = mu``, no sampling), so the
        result is deterministic and directly comparable to
        :class:`~fgas_spk.models.mlp_regressor.MlpRegressor` /
        :mod:`~fgas_spk.models.pca_linear` on the same split. numpy in, torch on the
        device, forward pass, numpy out, with the target standardisation inverted.
        The conditioning modalities must match :meth:`fit`: whichever of
        ``X_cond`` / ``X_params`` were used at fit are required here, and ones
        unused at fit must stay None.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Observable conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                ``X_cond``. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params). Required iff the model was fit with
                ``X_params``. Defaults to None.

        Returns:
            np.ndarray: Predicted SP(k), shape (n_examples, n_k) for a curve
                target or (n_examples,) for a 1-D ``single_k`` target.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit (the input widths would not line up).
        """
        import torch

        if self._encoder is None or self._decoder is None:
            raise RuntimeError("Cvae.predict called before fit.")
        self._check_modality(X_cond, X_params)
        assert self._x_mean is not None and self._x_std is not None

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        context = self._context(X_cond, X_params)
        cond_t = self._to_tensor(context) if context is not None else None

        self._encoder.eval()
        self._decoder.eval()
        with torch.no_grad():
            mu, _ = self._encode_dist(x_t, cond_t)
            out = self._decode(mu, cond_t)
        out = out.detach().cpu().numpy()

        assert self._y_mean is not None and self._y_std is not None
        y = self._invert_norm(out, self._y_mean, self._y_std)
        if self._y_was_1d:
            y = y[:, 0]
        return y

    def latents(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the posterior-mean codes ``mu`` for profiles ``X``.

        Not part of the :class:`~fgas_spk.models.base.ProfileToSpk` protocol -- the
        runner never calls it -- but it lets the latent structure be inspected after
        training. Returns the posterior *mean* ``mu`` (no sampling), so it is
        deterministic. The conditioning modalities must match :meth:`fit`, exactly
        as in :meth:`predict`.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Observable conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                ``X_cond``. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params). Required iff the model was fit with
                ``X_params``. Defaults to None.

        Returns:
            np.ndarray: Posterior-mean latent codes ``mu``, shape
                (n_examples, latent_dim).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit.
        """
        import torch

        if self._encoder is None:
            raise RuntimeError("Cvae.latents called before fit.")
        self._check_modality(X_cond, X_params)
        assert self._x_mean is not None and self._x_std is not None

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        context = self._context(X_cond, X_params)
        cond_t = self._to_tensor(context) if context is not None else None

        self._encoder.eval()
        with torch.no_grad():
            mu, _ = self._encode_dist(x_t, cond_t)
        return mu.detach().cpu().numpy()

    def predict_samples(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
        n_samples: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Draw decoded latent samples for profiles ``X`` (the science payload).

        For each example this draws ``n_samples`` latents ``z ~ q(z|x) = N(mu,
        sigma^2)`` via the reparameterisation trick, decodes each, and inverts the
        target standardisation. The spread **across the sample axis** (axis 0) is
        the predictive uncertainty: how much a given f_gas(R) profile leaves SP(k)
        underdetermined. This method is *not* part of the
        :class:`~fgas_spk.models.base.ProfileToSpk` protocol -- the runner never
        calls it.

        The sampling is seeded (default: the model ``seed``) so it is reproducible
        on a fixed backend; it is stochastic by design (unlike :meth:`predict`,
        which returns the posterior mean). Dropout is disabled here, so all
        stochasticity comes from the latent, not the network.

        The spread is only cleanly interpretable on the proxy-sound k-window (the
        ``k_range`` target, k in [0.5, 5.0] h/Mpc): on the full SP(k) curve the
        high-k bins carry a feedback-correlated target bias (see the module
        docstring, "Target scoping"), so part of the spread there would reflect the
        target proxy rather than genuine f_gas underdetermination.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Observable conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                ``X_cond``. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params). Required iff the model was fit with
                ``X_params``. Defaults to None.
            n_samples (int): Number of latent draws per example. Defaults to 100.
            seed (int | None): Seed for the latent sampling. None uses the model's
                ``seed``. Defaults to None.

        Returns:
            np.ndarray: Decoded samples, shape (n_samples, n_examples, n_k) for a
                curve target, or (n_samples, n_examples) for a 1-D ``single_k``
                target.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit.
        """
        import torch

        if self._encoder is None or self._decoder is None:
            raise RuntimeError("Cvae.predict_samples called before fit.")
        self._check_modality(X_cond, X_params)
        assert self._x_mean is not None and self._x_std is not None

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        context = self._context(X_cond, X_params)
        cond_t = self._to_tensor(context) if context is not None else None

        n_examples = x_t.shape[0]
        # Seed a CPU generator explicitly: this is reproducible on every backend
        # (torch.Generator(device="mps") is unsupported), and eps is moved to the
        # working device after sampling.
        seed = self.seed if seed is None else seed
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        self._encoder.eval()
        self._decoder.eval()
        with torch.no_grad():
            mu, logvar = self._encode_dist(x_t, cond_t)  # (n_examples, latent_dim)
            std = torch.exp(0.5 * logvar)
            latent_dim = mu.shape[1]
            # Sample all draws at once, then decode as one (n_samples * n_examples)
            # batch. eps is drawn on CPU (seeded) and moved to the device.
            eps = torch.randn(
                (n_samples, n_examples, latent_dim), generator=gen
            ).to(self._device)
            z = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, N, latent_dim)
            z_flat = z.reshape(n_samples * n_examples, latent_dim)
            if cond_t is not None:
                cond_flat = cond_t.unsqueeze(0).expand(n_samples, -1, -1).reshape(
                    n_samples * n_examples, cond_t.shape[1]
                )
            else:
                cond_flat = None
            out = self._decode(z_flat, cond_flat)  # (S * N, n_k)
        out = out.detach().cpu().numpy().reshape(n_samples, n_examples, -1)

        assert self._y_mean is not None and self._y_std is not None
        samples = self._invert_norm(out, self._y_mean, self._y_std)
        if self._y_was_1d:
            samples = samples[..., 0]  # (n_samples, n_examples)
        return samples

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_profile: int, n_context: int, n_k: int) -> None:
        """Build the encoder and decoder (torch imported lazily here).

        The encoder emits ``2 * latent_dim`` outputs -- the ``mu`` and ``logvar``
        heads, split in :meth:`_encode_dist`. ``n_context`` is the combined width of
        the conditioning modalities (``X_cond`` plus ``X_params``); it widens both
        the encoder input and the decoder input, so the decoder is conditioned on
        the context exactly as the encoder is.
        """
        self._encoder = self._mlp(n_profile + n_context, 2 * self.latent_dim)
        self._decoder = self._mlp(self.latent_dim + n_context, n_k)

    def _mlp(self, in_dim: int, out_dim: int):
        """Return an ``n_layers``-deep GELU MLP ``in_dim -> hidden... -> out_dim``.

        A dropout layer (probability ``self.dropout``) follows each hidden
        activation; ``dropout == 0.0`` makes it an identity, so the module matches
        the deterministic MLP when dropout is disabled.
        """
        from torch import nn

        layers: list = []
        d = in_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(d, self.hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            d = self.hidden
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def _encode_dist(self, x, cond):
        """Encode ``concat(x, cond)`` (or ``x`` alone) to ``(mu, logvar)``."""
        import torch

        enc_in = x if cond is None else torch.cat([x, cond], dim=1)
        assert self._encoder is not None
        h = self._encoder(enc_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def _reparameterise(self, mu, logvar):
        """Sample ``z = mu + exp(0.5 * logvar) * eps`` with ``eps ~ N(0, I)``.

        Uses :func:`torch.randn_like` (the global torch seed set in :meth:`fit`
        governs it), so gradients flow through ``mu`` and ``logvar`` at train time.
        """
        import torch

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _decode(self, z, cond):
        """Decode ``concat(z, cond)`` (or ``z`` alone) to standardised SP(k)."""
        import torch

        dec_in = z if cond is None else torch.cat([z, cond], dim=1)
        assert self._decoder is not None
        return self._decoder(dec_in)

    @staticmethod
    def _kl(mu, logvar):
        """Analytic KL(q(z|x) || N(0, I)): summed over latents, mean over batch.

        ``KL = mean_batch( -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j)) )``,
        the standard closed form for a diagonal-Gaussian posterior against a
        standard-normal prior. Returns a scalar tensor.
        """
        kl_per_example = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return kl_per_example.mean()

    def _beta_eff(self, epoch: int) -> float:
        """Return the annealed KL weight at ``epoch``.

        Ramps linearly from 0 at epoch 0 to ``beta`` at epoch ``anneal_epochs``,
        then holds at ``beta``. A non-positive ``anneal_epochs`` disables annealing
        (``beta`` from epoch 0). See the module docstring for the rationale.
        """
        if self.anneal_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / self.anneal_epochs)

    def _resolve_device(self):
        """Resolve the torch device (cuda -> mps -> cpu), honouring ``device``."""
        import torch

        if self.device is not None:
            print(f"Using user-specified device '{self.device}' for Cvae.")
            return torch.device(self.device)
        if torch.cuda.is_available():
            print("Using CUDA device for Cvae.")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using MPS device for Cvae.")
            return torch.device("mps")
        print("Using CPU device for Cvae.")
        return torch.device("cpu")

    def _to_tensor(self, a: np.ndarray):
        """Convert a numpy array to a float32 tensor on the resolved device."""
        import torch

        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
            self._device
        )

    def _context(
        self, X_cond: np.ndarray | None, X_params: np.ndarray | None
    ) -> np.ndarray | None:
        """Standardise and concatenate the conditioning modalities into one array.

        Returns the ``[X_cond | X_params]`` context (each modality standardised
        with its stored statistics) fed to both the encoder and the decoder, or
        None if neither modality is in use. Presence is assumed already validated
        by :meth:`_check_modality`.

        Args:
            X_cond (np.ndarray | None): Observable conditioning scalars, or None.
            X_params (np.ndarray | None): CAMELS parameters, or None.

        Returns:
            np.ndarray | None: The standardised context, shape
                (n_examples, n_context), or None.
        """
        blocks = []
        if self._uses_cond:
            assert self._cond_mean is not None and self._cond_std is not None
            blocks.append(
                self._apply_norm(
                    np.asarray(X_cond, dtype=np.float64),
                    self._cond_mean,
                    self._cond_std,
                )
            )
        if self._uses_params:
            assert self._params_mean is not None and self._params_std is not None
            blocks.append(
                self._apply_norm(
                    np.asarray(X_params, dtype=np.float64),
                    self._params_mean,
                    self._params_std,
                )
            )
        if not blocks:
            return None
        return np.hstack(blocks)

    def _check_modality(
        self, X_cond: np.ndarray | None, X_params: np.ndarray | None
    ) -> None:
        """Raise if X_cond / X_params presence does not match how the model was fit."""
        if self._uses_cond and X_cond is None:
            raise ValueError(
                "This model was fit with conditioning (X_cond); predict/latents "
                "requires X_cond as well."
            )
        if not self._uses_cond and X_cond is not None:
            raise ValueError(
                "This model was fit without X_cond but X_cond was given. Pass "
                "X_cond=None to match the fitted modality."
            )
        if self._uses_params and X_params is None:
            raise ValueError(
                "This model was fit with CAMELS parameters (X_params); "
                "predict/latents requires X_params as well."
            )
        if not self._uses_params and X_params is not None:
            raise ValueError(
                "This model was fit without X_params but X_params was given. Pass "
                "X_params=None to match the fitted modality."
            )

    @staticmethod
    def _fit_norm(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-column mean and std; zero-variance columns get unit scale."""
        a = np.asarray(a, dtype=np.float64)
        mean = a.mean(axis=0)
        std = a.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std

    @staticmethod
    def _apply_norm(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Standardise ``a`` with the stored ``mean`` / ``std``."""
        return (np.asarray(a, dtype=np.float64) - mean) / std

    @staticmethod
    def _invert_norm(a: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Invert the standardisation: map standardised ``a`` back to raw scale."""
        return np.asarray(a, dtype=np.float64) * std + mean
