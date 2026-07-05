"""Reusable Gaussian VAE component for the dual-VAE architecture (PyTorch).

This module holds :class:`GaussianVAE`, the single component class used for
**both** modalities of the dual-VAE programme (staged spec v2): VAE-Y
autoencodes the SP(k) suppression curve (Stage 2, the Lin et al. 2026
methodological replication) and VAE-X autoencodes the f_gas(R) profile
(Stage 3). It is deliberately **not** a registry plugin: a y-only autoencoder
cannot honestly implement ``ProfileToSpk``'s ``predict(X) -> SP(k)`` (decision
F0.9). Stage scripts under ``scripts/`` construct it directly; the Stage 5
``dual_vae`` plugin composes it.

**Model.** A plain VAE with an optional conditioning context (decision F0.4:
the cosmological CAMELS parameters), concatenated into **both** the encoder and
the decoder inputs, following the existing plugins' ``_context`` pattern:

    encoder q(z | v, ctx):  concat(v, ctx) -> (mu, logvar)   [logvar clamped +-8]
    decoder p(v | z, ctx):  concat(z, ctx) -> v_hat
    obs_logvar:             per-bin observation-noise parameter, shape (n_bins,)

where ``v`` is the modality being autoencoded (SP(k) or f_gas(R)), standardised
per column internally. Both networks are ``n_layers``-deep GELU MLPs with
dropout, as in :mod:`~fgas_spk.models.cvae`.

**Likelihood -- Gaussian NLL with a learned noise head (decision F0.2).** The
per-bin ``obs_logvar`` (initialised to 0.0, i.e. unit variance on the
standardised scale = one per-bin standard deviation on the raw scale) makes the
reconstruction a proper heteroscedastic Gaussian log-likelihood. ``beta`` is a
**tuned hyperparameter** (F0.2 as amended 2026-07-05, B1): for a compression
autoencoder it prices information rate, so the spec sweeps it over
{1, 0.1, 0.01, 0.001} per usage. :meth:`obs_sigma` exposes the fitted noise on
the **raw** target scale. Note: the repo's ``cvae``
plugin does *not* carry this head (its known under-coverage is the motivation);
this is the design the spec prescribes for the dual-VAE components.

**Loss and reduction convention (GR7).** Within the ELBO both terms are
**per-example sums over their own dimensions, averaged over the batch**:

    nll = mean_batch 0.5 * sum_bins ( (v - v_hat)^2 / e^{obs_logvar}
                                      + obs_logvar + log 2*pi )
    kl  = mean_batch sum_j 0.5 * ( mu_j^2 + e^{logvar_j} - logvar_j - 1 )
    total = nll + beta_eff * kl

(standardised scale throughout training). The effective per-dimension weighting
is therefore ~``0.5 / e^{obs_logvar_b}`` per target bin against ``beta`` per
latent dimension -- one convention, no mixed mean/sum reductions. ``beta_eff``
anneals linearly from ~0 to ``beta`` over ``anneal_epochs`` (default
``epochs // 4``), as in the cvae/vib plugins.

**Per-epoch history.** Each entry records ``epoch``, ``train_loss``, ``nll``,
``kl`` (total), ``kl_per_dim`` (list, length ``latent_dim``), ``activity``
(list: the Lin et al. Eq. F21-analogue statistic ``A_j = Var_batch(mu_j) /
mean_batch(sigma_j^2)``, on the fit's training rows), ``beta_eff`` and
``val_loss`` (internal holdout, terminal ``beta``). :meth:`latent_stats`
recomputes ``kl_per_dim`` / ``activity`` / ``mu`` for any fold after fitting.

**Internal validation, determinism, standardisation.** Exactly the cvae
pattern: a seeded ``val_frac`` holdout with best-epoch restore (encoder,
decoder, and ``obs_logvar`` together); per-column standardisation fitted on the
training rows only, zero-variance columns get unit scale; ``torch.manual_seed``
seeding, deterministic on a fixed CPU backend; gradients clipped to max norm
1.0; AdamW when ``weight_decay > 0`` else Adam.

Dependencies:
    PyTorch, imported **lazily inside methods** -- importing this module (or
    ``fgas_spk.models``) stays torch-free.
"""

from __future__ import annotations

import copy
import math

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


class GaussianVAE:
    """Conditional Gaussian VAE for one modality (SP(k) or f_gas(R)).

    Autoencodes a single array ``v`` of shape (n_examples, n_bins) with an
    optional conditioning ``context`` of shape (n_examples, n_context) fed to
    both encoder and decoder (see the module docstring for the loss and
    conventions). Widths are inferred at :meth:`fit`; nothing is hardcoded.

    Args:
        latent_dim (int): Width of the stochastic latent. Defaults to 2
            (decision F0.3's first configuration).
        hidden (int): Hidden-layer width in both networks. Defaults to 128.
        n_layers (int): Hidden layers in each network. Defaults to 2.
        epochs (int): Training epochs. Defaults to 400.
        lr (float): Adam/AdamW learning rate. Defaults to 1e-3.
        weight_decay (float): Weight decay; > 0 selects AdamW (decoupled),
            0 selects plain Adam. Defaults to 1e-4.
        dropout (float): Dropout probability after each hidden activation.
            Defaults to 0.0.
        batch_size (int): Minibatch size, capped at the training-set size.
            Defaults to 128.
        beta (float): Terminal KL weight after annealing. A tuned
            hyperparameter (F0.2 as amended, B1), swept over
            {1, 0.1, 0.01, 0.001} by the stage scripts. Defaults to 1.0.
        anneal_epochs (int | None): Linear KL warm-up length; None selects
            ``epochs // 4``; <= 0 disables annealing. Defaults to None.
        val_frac (float): Internal holdout fraction for best-epoch restore
            (seeded shuffle, ``ceil(val_frac * n)`` rows); 0 disables it.
            Defaults to 0.1.
        seed (int): Reproducibility seed (torch, the holdout shuffle).
            Defaults to 0.
        device (str | None): Torch device string; None auto-selects
            cuda -> mps -> cpu. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored.

    Attributes:
        history (list[dict]): Per-epoch trace (see the module docstring).
        best_epoch (int | None): Epoch whose weights were restored, or None
            when ``val_frac == 0``.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 400,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        batch_size: int = 128,
        beta: float = 1.0,
        anneal_epochs: int | None = None,
        val_frac: float = 0.1,
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        # Construction is cheap and torch-free: store config only. The modules
        # are built (and torch imported) in fit, once the widths are known.
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.beta = beta
        self.anneal_epochs = epochs // 4 if anneal_epochs is None else anneal_epochs
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

        self.history: list[dict] = []
        self.best_epoch: int | None = None

        self._encoder = None
        self._decoder = None
        self._obs_logvar = None  # torch.nn.Parameter, shape (n_bins,), std scale
        self._device = None
        self._uses_context: bool = False
        self._v_mean: np.ndarray | None = None
        self._v_std: np.ndarray | None = None
        self._ctx_mean: np.ndarray | None = None
        self._ctx_std: np.ndarray | None = None

    # --- public API --------------------------------------------------------

    def fit(self, data: np.ndarray, context: np.ndarray | None = None) -> None:
        """Fit the VAE on ``data`` by maximising the annealed Gaussian ELBO.

        Args:
            data (np.ndarray): The modality to autoencode, shape
                (n_examples, n_bins).
            context (np.ndarray | None): Conditioning context, shape
                (n_examples, n_context), fed to both encoder and decoder; a
                context passed here is then required by every later call.
                Defaults to None.
        """
        import torch

        v = np.asarray(data, dtype=np.float64)
        self._uses_context = context is not None
        ctx = np.asarray(context, dtype=np.float64) if self._uses_context else None

        # Seeded internal holdout for best-epoch restore (cvae pattern).
        n_total = v.shape[0]
        n_val = math.ceil(self.val_frac * n_total) if self.val_frac > 0 else 0
        use_val = 0 < n_val < n_total
        if use_val:
            shuffled = np.random.default_rng(self.seed).permutation(n_total)
            val_rows, train_rows = shuffled[:n_val], shuffled[n_val:]
        else:
            train_rows = np.arange(n_total)
            val_rows = np.empty(0, dtype=int)

        # Standardisation fitted on the training rows only.
        self._v_mean, self._v_std = self._fit_norm(v[train_rows])
        if ctx is not None:
            self._ctx_mean, self._ctx_std = self._fit_norm(ctx[train_rows])

        n_bins = v.shape[1]
        n_context = ctx.shape[1] if ctx is not None else 0

        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._build_modules(n_bins, n_context)
        assert self._encoder is not None and self._decoder is not None

        vs = self._to_tensor(self._apply_norm(v, self._v_mean, self._v_std))
        cs = (
            self._to_tensor(self._apply_norm(ctx, self._ctx_mean, self._ctx_std))
            if ctx is not None
            else None
        )
        train_t = torch.as_tensor(train_rows, dtype=torch.long, device=self._device)
        v_tr = vs[train_t]
        c_tr = cs[train_t] if cs is not None else None
        if use_val:
            val_t = torch.as_tensor(val_rows, dtype=torch.long, device=self._device)
            v_va = vs[val_t]
            c_va = cs[val_t] if cs is not None else None

        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + [self._obs_logvar]
        )
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)

        n_train = v_tr.shape[0]
        batch_size = min(self.batch_size, n_train)
        self.history = []
        self.best_epoch = None
        best_val = math.inf
        best_state = None
        for epoch in range(self.epochs):
            beta_eff = self._beta_eff(epoch)
            self._train_mode()
            perm = torch.randperm(n_train, device=self._device)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                vb = v_tr[idx]
                cb = c_tr[idx] if c_tr is not None else None
                optimizer.zero_grad()
                mu, logvar = self._encode_dist(vb, cb)
                z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                v_hat = self._decode_std(z, cb)
                loss = self._nll(v_hat, vb) + beta_eff * self._kl_per_dim(
                    mu, logvar
                ).sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            # Per-epoch trace on the full training rows (posterior mean, no
            # sampling). val_loss uses the TERMINAL beta so epochs are compared
            # on one fixed objective; it drives best-epoch selection.
            self._eval_mode()
            with torch.no_grad():
                mu_tr, logvar_tr = self._encode_dist(v_tr, c_tr)
                nll_tr = float(self._nll(self._decode_std(mu_tr, c_tr), v_tr).item())
                kl_dims = self._kl_per_dim(mu_tr, logvar_tr)
                kl_tr = float(kl_dims.sum().item())
                activity = self._activity(mu_tr, logvar_tr)
                if use_val:
                    mu_va, logvar_va = self._encode_dist(v_va, c_va)
                    nll_va = float(
                        self._nll(self._decode_std(mu_va, c_va), v_va).item()
                    )
                    kl_va = float(self._kl_per_dim(mu_va, logvar_va).sum().item())
                    val_loss = nll_va + self.beta * kl_va
                else:
                    val_loss = None
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": nll_tr + beta_eff * kl_tr,
                    "nll": nll_tr,
                    "kl": kl_tr,
                    "kl_per_dim": [float(k) for k in kl_dims.cpu().numpy()],
                    "activity": activity,
                    "beta_eff": beta_eff,
                    "val_loss": val_loss,
                }
            )
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {
                    "encoder": copy.deepcopy(self._encoder.state_dict()),
                    "decoder": copy.deepcopy(self._decoder.state_dict()),
                    "obs_logvar": self._obs_logvar.detach().clone(),
                }

        if best_state is not None:
            self._encoder.load_state_dict(best_state["encoder"])
            self._decoder.load_state_dict(best_state["decoder"])
            with torch.no_grad():
                self._obs_logvar.copy_(best_state["obs_logvar"])

    def encode(
        self,
        data: np.ndarray,
        context: np.ndarray | None = None,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return posterior means ``mu`` (and optionally stds) for ``data``.

        Args:
            data (np.ndarray): Modality values, shape (n_examples, n_bins).
            context (np.ndarray | None): Context, required iff fitted with one.
            return_std (bool): Also return the posterior standard deviations
                ``exp(0.5 * logvar)``. Defaults to False.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: ``mu`` of shape
                (n_examples, latent_dim), or ``(mu, std)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If context presence does not match :meth:`fit`.
        """
        import torch

        self._require_fitted("encode")
        self._check_context(context)
        v_t, c_t = self._prepare(data, context)
        self._eval_mode()
        with torch.no_grad():
            mu, logvar = self._encode_dist(v_t, c_t)
        if return_std:
            return (
                mu.detach().cpu().numpy(),
                torch.exp(0.5 * logvar).detach().cpu().numpy(),
            )
        return mu.detach().cpu().numpy()

    def decode(self, z: np.ndarray, context: np.ndarray | None = None) -> np.ndarray:
        """Decode latent codes to the modality on the **raw** scale.

        Args:
            z (np.ndarray): Latent codes, shape (n_examples, latent_dim).
            context (np.ndarray | None): Context, required iff fitted with one.

        Returns:
            np.ndarray: Decoded values, shape (n_examples, n_bins), raw scale
                (the internal standardisation is inverted).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If context presence does not match :meth:`fit`.
        """
        import torch

        self._require_fitted("decode")
        self._check_context(context)
        z_t = self._to_tensor(np.asarray(z, dtype=np.float64))
        c_t = self._context_tensor(context)
        self._eval_mode()
        with torch.no_grad():
            out = self._decode_std(z_t, c_t)
        assert self._v_mean is not None and self._v_std is not None
        return self._invert_norm(out.detach().cpu().numpy(), self._v_mean, self._v_std)

    def reconstruct(
        self, data: np.ndarray, context: np.ndarray | None = None
    ) -> np.ndarray:
        """Posterior-mean reconstruction of ``data`` on the raw scale.

        Equivalent to ``decode(encode(data, context), context)``: encode to the
        posterior mean ``mu`` (no sampling) and decode back.

        Args:
            data (np.ndarray): Modality values, shape (n_examples, n_bins).
            context (np.ndarray | None): Context, required iff fitted with one.

        Returns:
            np.ndarray: Reconstruction, shape (n_examples, n_bins), raw scale.
        """
        return self.decode(self.encode(data, context), context)

    def obs_sigma(self) -> np.ndarray:
        """Fitted per-bin observation noise on the **raw** target scale.

        ``obs_logvar`` lives on the standardised scale (init 0.0 = one per-bin
        standard deviation), so the raw-scale noise is
        ``exp(0.5 * obs_logvar) * v_std``.

        Returns:
            np.ndarray: Noise sigma per bin, shape (n_bins,).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted("obs_sigma")
        assert self._obs_logvar is not None and self._v_std is not None
        std_scale = np.exp(0.5 * self._obs_logvar.detach().cpu().numpy())
        return std_scale * self._v_std

    def latent_stats(
        self, data: np.ndarray, context: np.ndarray | None = None
    ) -> dict:
        """Per-dimension latent diagnostics on an arbitrary fold.

        Args:
            data (np.ndarray): Modality values, shape (n_examples, n_bins).
            context (np.ndarray | None): Context, required iff fitted with one.

        Returns:
            dict: ``kl_per_dim`` (list, batch-mean analytic KL to N(0, I) per
                dimension), ``activity`` (list, ``A_j = Var(mu_j) /
                mean(sigma_j^2)``), and ``mu`` (np.ndarray, the posterior
                means, shape (n_examples, latent_dim)).
        """
        import torch

        self._require_fitted("latent_stats")
        self._check_context(context)
        v_t, c_t = self._prepare(data, context)
        self._eval_mode()
        with torch.no_grad():
            mu, logvar = self._encode_dist(v_t, c_t)
            kl_dims = self._kl_per_dim(mu, logvar)
            activity = self._activity(mu, logvar)
        return {
            "kl_per_dim": [float(k) for k in kl_dims.cpu().numpy()],
            "activity": activity,
            "mu": mu.detach().cpu().numpy(),
        }

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_bins: int, n_context: int) -> None:
        """Build encoder, decoder, and the obs_logvar parameter (torch lazy)."""
        import torch
        from torch import nn

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list = []
            d = in_dim
            for _ in range(self.n_layers):
                layers.append(nn.Linear(d, self.hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
                d = self.hidden
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        self._encoder = mlp(n_bins + n_context, 2 * self.latent_dim).to(self._device)
        self._decoder = mlp(self.latent_dim + n_context, n_bins).to(self._device)
        # Per-bin observation-noise head, init 0.0 (decision F0.2): unit
        # variance on the standardised scale = one per-bin std on the raw scale.
        self._obs_logvar = torch.nn.Parameter(
            torch.zeros(n_bins, device=self._device)
        )

    def _encode_dist(self, v, ctx):
        """Encoder ``q(z|v,ctx)`` -> ``(mu, logvar)`` (logvar clamped +-8)."""
        import torch

        enc_in = v if ctx is None else torch.cat([v, ctx], dim=1)
        h = self._encoder(enc_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _decode_std(self, z, ctx):
        """Decode to the standardised modality scale (training-side path)."""
        import torch

        dec_in = z if ctx is None else torch.cat([z, ctx], dim=1)
        return self._decoder(dec_in)

    def _nll(self, v_hat, v):
        """Gaussian NLL: 0.5 * sum_bins((err^2)/e^lv + lv + log 2pi), batch mean."""
        lv = self._obs_logvar.clamp(-8.0, 8.0)
        per_example = 0.5 * (
            (v_hat - v).pow(2) / lv.exp() + lv + _LOG_2PI
        ).sum(dim=1)
        return per_example.mean()

    @staticmethod
    def _kl_per_dim(mu, logvar):
        """Analytic KL(q || N(0, I)) per latent dimension, batch mean.

        Returns a tensor of shape (latent_dim,); ``.sum()`` gives the total KL
        under the module's one-convention reduction (sum over dims, batch mean).
        """
        return (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)).mean(dim=0)

    @staticmethod
    def _activity(mu, logvar) -> list[float]:
        """Lin et al. Eq. F21-analogue activity ``A_j = Var(mu_j)/mean(sigma_j^2)``."""
        var_mu = mu.var(dim=0, unbiased=False)
        mean_sigma2 = logvar.exp().mean(dim=0)
        return [float(a) for a in (var_mu / mean_sigma2).cpu().numpy()]

    def _beta_eff(self, epoch: int) -> float:
        """Annealed KL weight: linear ramp to ``beta`` over ``anneal_epochs``."""
        if self.anneal_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / self.anneal_epochs)

    def _train_mode(self) -> None:
        """Put both networks in train mode."""
        self._encoder.train()  # type: ignore[union-attr]
        self._decoder.train()  # type: ignore[union-attr]

    def _eval_mode(self) -> None:
        """Put both networks in eval mode."""
        self._encoder.eval()  # type: ignore[union-attr]
        self._decoder.eval()  # type: ignore[union-attr]

    def _require_fitted(self, method: str) -> None:
        """Raise if the model has not been fitted yet."""
        if self._encoder is None or self._decoder is None:
            raise RuntimeError(f"GaussianVAE.{method} called before fit.")

    def _check_context(self, context: np.ndarray | None) -> None:
        """Raise if context presence does not match how the model was fit."""
        if self._uses_context and context is None:
            raise ValueError(
                "This GaussianVAE was fit with a context; pass the matching "
                "context array."
            )
        if not self._uses_context and context is not None:
            raise ValueError(
                "This GaussianVAE was fit without a context but one was given; "
                "pass context=None."
            )

    def _prepare(self, data: np.ndarray, context: np.ndarray | None):
        """Standardise ``data`` (and context) and return device tensors."""
        assert self._v_mean is not None and self._v_std is not None
        v_t = self._to_tensor(
            self._apply_norm(np.asarray(data, dtype=np.float64), self._v_mean, self._v_std)
        )
        return v_t, self._context_tensor(context)

    def _context_tensor(self, context: np.ndarray | None):
        """Standardise the context and return it as a device tensor (or None)."""
        if not self._uses_context:
            return None
        assert self._ctx_mean is not None and self._ctx_std is not None
        return self._to_tensor(
            self._apply_norm(
                np.asarray(context, dtype=np.float64), self._ctx_mean, self._ctx_std
            )
        )

    def _resolve_device(self):
        """Resolve the torch device (cuda -> mps -> cpu), honouring ``device``."""
        import torch

        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to_tensor(self, a: np.ndarray):
        """Convert a numpy array to a float32 tensor on the resolved device."""
        import torch

        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
            self._device
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


class _LatentMapBase:
    """Shared machinery for the torch latent-map rungs (MLP and MDN).

    Maps frozen VAE-X posterior means ``z1`` to frozen VAE-Y posterior means
    ``z2`` (Stage 4 of the dual-VAE staged spec; decision F0.5: trained on
    posterior means, not samples). Inputs and targets are standardised per
    column on the training rows (the frozen codes are not unit-scale at low
    beta); an internal seeded ``val_frac`` holdout drives best-epoch restore,
    as in :class:`GaussianVAE`. torch is imported lazily inside methods.
    """

    def __init__(
        self,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 2000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        val_frac: float = 0.1,
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self._net = None
        self._device = None
        self._in_mean = self._in_std = None
        self._out_mean = self._out_std = None

    # -- template hooks implemented by the rungs ----------------------------

    def _out_width(self, d_out: int) -> int:
        """Network output width for a ``d_out``-dimensional target."""
        raise NotImplementedError

    def _loss(self, raw, target):
        """Per-batch training loss from raw network output and target."""
        raise NotImplementedError

    # -- shared fit ----------------------------------------------------------

    def fit(self, z1: np.ndarray, z2: np.ndarray) -> None:
        """Fit the map on ``(z1, z2)`` posterior-mean pairs.

        Args:
            z1 (np.ndarray): Inputs, shape (n_examples, d1).
            z2 (np.ndarray): Targets, shape (n_examples, d2).
        """
        import torch

        z1 = np.asarray(z1, dtype=np.float64)
        z2 = np.asarray(z2, dtype=np.float64)

        n_total = z1.shape[0]
        n_val = math.ceil(self.val_frac * n_total) if self.val_frac > 0 else 0
        use_val = 0 < n_val < n_total
        if use_val:
            shuffled = np.random.default_rng(self.seed).permutation(n_total)
            val_rows, train_rows = shuffled[:n_val], shuffled[n_val:]
        else:
            train_rows = np.arange(n_total)

        self._in_mean, self._in_std = GaussianVAE._fit_norm(z1[train_rows])
        self._out_mean, self._out_std = GaussianVAE._fit_norm(z2[train_rows])

        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._d_out = z2.shape[1]
        self._build(z1.shape[1], self._d_out)

        z1s = self._to_tensor(GaussianVAE._apply_norm(z1, self._in_mean, self._in_std))
        z2s = self._to_tensor(GaussianVAE._apply_norm(z2, self._out_mean, self._out_std))
        tr = torch.as_tensor(train_rows, dtype=torch.long, device=self._device)
        x_tr, y_tr = z1s[tr], z2s[tr]
        if use_val:
            va = torch.as_tensor(val_rows, dtype=torch.long, device=self._device)
            x_va, y_va = z1s[va], z2s[va]

        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        n_train = x_tr.shape[0]
        batch_size = min(self.batch_size, n_train)
        self.history = []
        self.best_epoch = None
        best_val = math.inf
        best_state = None
        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n_train, device=self._device)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                loss = self._loss(self._net(x_tr[idx]), y_tr[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

            self._net.eval()
            with torch.no_grad():
                train_loss = float(self._loss(self._net(x_tr), y_tr).item())
                val_loss = (
                    float(self._loss(self._net(x_va), y_va).item())
                    if use_val else None
                )
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self._net.state_dict())

        if best_state is not None:
            self._net.load_state_dict(best_state)

    # -- shared internals ----------------------------------------------------

    def _build(self, d_in: int, d_out: int) -> None:
        """Build the GELU MLP trunk (torch imported lazily here)."""
        from torch import nn

        layers: list = []
        d = d_in
        for _ in range(self.n_layers):
            layers.append(nn.Linear(d, self.hidden))
            layers.append(nn.GELU())
            d = self.hidden
        layers.append(nn.Linear(d, self._out_width(d_out)))
        self._net = nn.Sequential(*layers).to(self._device)

    def _resolve_device(self):
        """Resolve the torch device (cuda -> mps -> cpu), honouring ``device``."""
        import torch

        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _to_tensor(self, a: np.ndarray):
        """Convert a numpy array to a float32 tensor on the resolved device."""
        import torch

        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(
            self._device
        )

    def _prepared_input(self, z1: np.ndarray):
        """Standardise ``z1`` and return it as a device tensor."""
        if self._net is None:
            raise RuntimeError(f"{type(self).__name__} used before fit.")
        return self._to_tensor(
            GaussianVAE._apply_norm(
                np.asarray(z1, dtype=np.float64), self._in_mean, self._in_std
            )
        )


class LatentMapMLP(_LatentMapBase):
    """Stage 4 rung 2: deterministic GELU-MLP map ``z1 -> z2``.

    Plain MSE regression (mean over all elements) on the standardised code
    scales; :meth:`predict` inverts the target standardisation.
    """

    def _out_width(self, d_out: int) -> int:
        return d_out

    def _loss(self, raw, target):
        return ((raw - target) ** 2).mean()

    def predict(self, z1: np.ndarray) -> np.ndarray:
        """Predict ``z2`` for codes ``z1`` (deterministic).

        Args:
            z1 (np.ndarray): Inputs, shape (n_examples, d1).

        Returns:
            np.ndarray: Predicted codes, shape (n_examples, d2).
        """
        import torch

        x = self._prepared_input(z1)
        self._net.eval()
        with torch.no_grad():
            out = self._net(x).cpu().numpy()
        return GaussianVAE._invert_norm(out, self._out_mean, self._out_std)


class LatentMapMDN(_LatentMapBase):
    """Stage 4 rung 3: mixture density network for ``p(z2 | z1)``.

    The trunk emits, per example, ``K`` mixture logits, ``K x d2`` component
    means, and ``K x d2`` diagonal log-variances (clamped to [-8, 8]); the
    loss is the exact mixture negative log-likelihood on the standardised
    code scale (log-sum-exp over components; sums over target dimensions,
    mean over the batch). :meth:`predict` returns the mixture mean;
    :meth:`sample` draws components then Gaussians, seeded via a CPU
    generator (MPS-safe, the cvae pattern).

    Args:
        n_components (int): Mixture size ``K``. Defaults to 3.
        **kwargs: Trunk and training options, as :class:`_LatentMapBase`.
    """

    def __init__(self, n_components: int = 3, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.n_components = n_components

    def _out_width(self, d_out: int) -> int:
        return self.n_components * (1 + 2 * d_out)

    def _split(self, raw):
        """Split trunk output into (logits, means, logvars) mixture blocks."""
        import torch

        k, d = self.n_components, self._d_out
        logits = raw[:, :k]
        means = raw[:, k:k + k * d].reshape(-1, k, d)
        logvars = torch.clamp(
            raw[:, k + k * d:].reshape(-1, k, d), -8.0, 8.0
        )
        return logits, means, logvars

    def _loss(self, raw, target):
        import torch

        logits, means, logvars = self._split(raw)
        log_w = torch.log_softmax(logits, dim=1)                # (N, K)
        diff2 = (target.unsqueeze(1) - means) ** 2              # (N, K, d)
        comp_ll = -0.5 * (diff2 / logvars.exp() + logvars + _LOG_2PI).sum(dim=2)
        return -torch.logsumexp(log_w + comp_ll, dim=1).mean()

    def _mixture(self, z1: np.ndarray):
        """Evaluate the mixture parameters for ``z1`` (standardised scale)."""
        import torch

        x = self._prepared_input(z1)
        self._net.eval()
        with torch.no_grad():
            logits, means, logvars = self._split(self._net(x))
            weights = torch.softmax(logits, dim=1)
        return weights, means, logvars

    def predict(self, z1: np.ndarray) -> np.ndarray:
        """Predict the mixture-mean ``z2`` for codes ``z1``.

        Args:
            z1 (np.ndarray): Inputs, shape (n_examples, d1).

        Returns:
            np.ndarray: Mixture means, shape (n_examples, d2).
        """
        weights, means, _ = self._mixture(z1)
        mix_mean = (weights.unsqueeze(2) * means).sum(dim=1).cpu().numpy()
        return GaussianVAE._invert_norm(mix_mean, self._out_mean, self._out_std)

    def sample(
        self, z1: np.ndarray, n_samples: int = 100, seed: int | None = None
    ) -> np.ndarray:
        """Draw ``z2 ~ p(z2 | z1)`` mixture samples.

        Args:
            z1 (np.ndarray): Inputs, shape (n_examples, d1).
            n_samples (int): Draws per example. Defaults to 100.
            seed (int | None): Sampling seed; None uses the model seed.

        Returns:
            np.ndarray: Samples, shape (n_samples, n_examples, d2).
        """
        import torch

        weights, means, logvars = self._mixture(z1)
        n, k, d = means.shape
        seed = self.seed if seed is None else seed
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        # Component choice and Gaussian draws on CPU (seeded, MPS/CUDA-safe),
        # then gathered against the (possibly device-resident) parameters.
        comp = torch.multinomial(
            weights.cpu(), n_samples, replacement=True, generator=gen
        ).T                                                     # (S, N)
        eps = torch.randn((n_samples, n, d), generator=gen)     # (S, N, d)
        means_c, logvars_c = means.cpu(), logvars.cpu()
        idx = comp.unsqueeze(2).expand(-1, -1, d)               # (S, N, d)
        mu_sel = torch.gather(
            means_c.unsqueeze(0).expand(n_samples, -1, -1, -1), 2,
            idx.unsqueeze(2),
        ).squeeze(2)
        lv_sel = torch.gather(
            logvars_c.unsqueeze(0).expand(n_samples, -1, -1, -1), 2,
            idx.unsqueeze(2),
        ).squeeze(2)
        z = (mu_sel + torch.exp(0.5 * lv_sel) * eps).numpy()
        return GaussianVAE._invert_norm(z, self._out_mean, self._out_std)


def composite_predict(
    vae_x, mapping, vae_y, X: np.ndarray, context: np.ndarray | None
) -> np.ndarray:
    """Dual-VAE composite point prediction ``x -> mu1 -> M(mu1) -> decoder-Y``.

    Stage 4 task 5's point-prediction path (and the Stage 5 plugin's
    ``predict``): encode the profile to the VAE-X posterior mean, map it with
    the rung's ``predict`` (rungs 1-2: the deterministic map; rung 3: the MDN
    mixture mean), and decode through decoder-Y.

    Args:
        vae_x: Fitted :class:`GaussianVAE` for f_gas(R) (its ``encode``).
        mapping: A fitted rung exposing ``predict(z1) -> z2``.
        vae_y: Fitted :class:`GaussianVAE` for SP(k) (its ``decode``).
        X (np.ndarray): Profiles, shape (n_examples, n_radii).
        context (np.ndarray | None): Conditioning context for both VAEs.

    Returns:
        np.ndarray: Predicted SP(k), shape (n_examples, n_k), raw scale.
    """
    mu1 = vae_x.encode(X, context)
    return vae_y.decode(mapping.predict(mu1), context)


def composite_samples(
    vae_x,
    mdn,
    vae_y,
    X: np.ndarray,
    context: np.ndarray | None,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Dual-VAE composite predictive samples (rung 3 only).

    Per decision F0.5 the spread here is the **mapping density plus decoder-Y
    observation noise** and deliberately EXCLUDES the encoder-X posterior
    spread and any input measurement noise (input-noise propagation is a
    later, separate addition): ``z2 ~ p(z2 | mu1)`` from the MDN, each draw
    decoded through decoder-Y, plus per-bin Gaussian noise from
    ``vae_y.obs_sigma()``.

    Args:
        vae_x: Fitted :class:`GaussianVAE` for f_gas(R).
        mdn: Fitted :class:`LatentMapMDN` (its ``sample``).
        vae_y: Fitted :class:`GaussianVAE` for SP(k).
        X (np.ndarray): Profiles, shape (n_examples, n_radii).
        context (np.ndarray | None): Conditioning context for both VAEs.
        n_samples (int): Draws per example. Defaults to 200.
        seed (int): Seed for both the MDN sampling and the observation noise.

    Returns:
        np.ndarray: Samples, shape (n_samples, n_examples, n_k), raw scale.
    """
    mu1 = vae_x.encode(X, context)
    z2 = mdn.sample(mu1, n_samples=n_samples, seed=seed)   # (S, N, d2)
    n_s, n_x, d2 = z2.shape
    decoded = vae_y.decode(
        z2.reshape(n_s * n_x, d2),
        None if context is None else np.repeat(
            np.asarray(context, dtype=float)[None, :, :], n_s, axis=0
        ).reshape(n_s * n_x, -1),
    ).reshape(n_s, n_x, -1)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=decoded.shape) * vae_y.obs_sigma()
    return decoded + noise
