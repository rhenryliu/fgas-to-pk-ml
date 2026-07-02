"""MLP regressor with a variational information bottleneck (deep VIB) (PyTorch).

This is a **point-prediction regressor**, not a predictive-uncertainty model. It
is the deterministic :mod:`~fgas_spk.models.mlp_regressor` skeleton with its plain
bottleneck replaced by a *stochastic* one and a KL term added -- the deep
variational information bottleneck (deep VIB) of Alemi et al. The stochastic
latent and its KL are used here purely as a **regulariser**: injecting noise into
the bottleneck and penalising its information content (KL to an ``N(0, I)`` prior)
discourages the encoder from memorising the small training set, which can close
(or reverse) a deterministic net's generalisation gap. The scientific payload is
the single number the model returns from :meth:`predict`: the point estimate of
SP(k), directly comparable to the MLP and the PCA reference on the same split.

**Not a conditional-uncertainty model.** The prior is a *fixed* ``N(0, I)`` and the
posterior ``q(z|x)`` is regularised toward it, so at small ``N`` the encoder's
per-example ``sigma`` tends toward a roughly input-independent value (it is pushed
to the prior, not shaped by the data's conditional structure). Decoding latent
draws would therefore produce a spread that reflects the *regularisation noise
floor*, **not** an estimate of ``p(y|x)`` -- it does not track how tightly a given
f_gas(R) profile constrains SP(k), and it does not widen where the target is
genuinely underdetermined. Presenting that spread as a predictive uncertainty
would be actively misleading, so **this model deliberately exposes no
``predict_samples``**. A model whose sample spread *is* an estimate of ``p(y|x)``
requires a conditional prior and a recognition network that sees the target during
training -- that is the Sohn-style conditional VAE in
:mod:`~fgas_spk.models.cvae`, a separate plugin. Use that one for predictive
spread; use this one for a regularised point prediction.

**Architecture.** An encoder/decoder pair around a narrow stochastic latent:

The conditioning *context* is whichever of two optional modalities are present:
``X_cond`` (observable scalars -- number density, optional mean halo mass) and
``X_params`` (the simulation's CAMELS parameters). Each is standardised on its own
scale and the two are concatenated into a single context vector, wired into both
the encoder and the decoder identically; the profile-only (no-conditioning) path
is supported too.

1. **encoder** maps ``concat(X, context)`` to *two* heads of width ``latent_dim``:
   the posterior mean ``mu`` and log-variance ``logvar`` of ``q(z|x) = N(mu,
   exp(logvar))``. ``logvar`` is clamped to ``[-8, 8]`` for numerical safety. At
   train time the latent is drawn by the reparameterisation trick ``z = mu +
   exp(0.5 * logvar) * eps`` with ``eps ~ N(0, I)``; at predict time the latent is
   the posterior mean (``z = mu``), which makes :meth:`predict` deterministic.
2. **decoder** maps ``concat(z, context)`` to the SP(k) target of width ``n_k``.

Input and output widths are inferred from the first :meth:`fit` batch; nothing
about the data dimensions is hardcoded. The default ``hidden`` is a deliberately
modest 128: with a small training set, regularisation is favoured over capacity.

**Loss -- ELBO used as a regulariser, with KL annealing.** Per minibatch the loss
is ``total = recon + beta_eff * KL``, where ``recon`` is the reconstruction term
-- the squared error **summed over the ``n_k`` target bins and averaged over the
batch**, ``((pred - y)**2).sum(dim=1).mean()`` -- and ``KL`` is the analytic
Kullback-Leibler divergence of ``q(z|x)`` from the ``N(0, I)`` prior, summed over
the latent dims and averaged over the batch. ``beta_eff`` is *annealed*: it ramps
linearly from ~0 to the configured ``beta`` (default 1.0) over the first
``anneal_epochs`` epochs (default ``epochs // 4``), then holds. The warm-up lets
the decoder learn to reconstruct through the latent before the KL term starts
pulling the posterior toward the prior. Gradients are clipped to a max norm of 1.0
between ``backward`` and the optimiser step.

**Internal validation and best-epoch restore.** ``val_frac`` (default 0.1) of the
training rows are held out via a seeded shuffle (``ceil(val_frac * n)`` rows). At
each epoch the validation loss is computed with the **terminal** ``beta``
(``self.beta``, not the annealed ``beta_eff``, so successive epochs are compared on
one fixed objective) and recorded in :attr:`history` as ``"val_loss"``. The
encoder/decoder state at the epoch of lowest validation loss is snapshotted and
restored at the end of :meth:`fit`, and :attr:`best_epoch` records that epoch.
``val_frac == 0`` disables all of this -- the model trains on every row, keeps the
final-epoch weights, records ``val_loss`` as ``None``, and leaves
:attr:`best_epoch` ``None`` -- i.e. the pre-validation behaviour.

**Normalisation.** Raw f_gas, SP(k), and each conditioning modality live on very
different scales, so the model standardises internally. At :meth:`fit` a per-column
mean and standard deviation are fitted on the training batch for the profile
``X``, each conditioning modality present, and the target ``y``; the network sees
and predicts standardised quantities, and :meth:`predict` inverts the target
standardisation on the way out. Zero-variance columns are given unit scale.

**Determinism.** :meth:`fit` seeds torch via :func:`torch.manual_seed` and draws
the validation holdout from a NumPy generator seeded with ``seed``, giving
run-to-run stability on a *fixed* backend (same machine, same device); it is
**not** bit-reproducible across machines or backends (CUDA/MPS kernels differ from
CPU and from each other). Pass ``device="cpu"`` for a backend-stable comparison.
:meth:`predict` is deterministic (posterior mean, no sampling).

In a run, the model is constructed from the
:class:`~fgas_spk.experiment.RunConfig` as
``VibRegressor(seed=run_config.seed, **run_config.model_params)`` and looked up by
name via ``fgas_spk.models.REGISTRY["vib_regressor"]``.

Dependencies:
    PyTorch (pinned in the project environment, an opt-in training dependency,
    not part of the core numpy+pyyaml import surface). torch is imported **lazily
    inside the methods** -- never at module import or at registration -- so
    ``import fgas_spk.models`` stays torch-free and the ``@register`` side effect
    runs without torch present.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register

if TYPE_CHECKING:  # import only for type hints; never pulls torch/loader at runtime
    from fgas_spk.loader import TrainingData


@register("vib_regressor")
class VibRegressor:
    """Encoder/decoder MLP with a variational information bottleneck (deep VIB).

    A regularised point-prediction regressor mapping f_gas(R) (+ optional
    conditioning) to SP(k). The stochastic latent and its KL term to an ``N(0, I)``
    prior act purely as a regulariser (see the module docstring); the model returns
    a single point prediction from :meth:`predict` and, by design, has **no**
    predictive-spread method -- its latent-sample spread is a regularisation noise
    floor, not an estimate of ``p(y|x)``. Widths are inferred from the first
    :meth:`fit` batch; nothing about the data dimensions is hardcoded.

    Args:
        latent_dim (int): Width of the stochastic bottleneck. Defaults to 4.
        hidden (int): Hidden-layer width on each side of the bottleneck. A
            deliberately modest 128 by default: with a small training set, favour
            regularisation over raw capacity. Defaults to 128.
        n_layers (int): Number of hidden layers in the encoder and in the decoder
            (each). Defaults to 2.
        epochs (int): Number of training epochs. Defaults to 200.
        lr (float): Adam/AdamW learning rate. Defaults to 5e-4.
        weight_decay (float): Weight decay (regularisation). When > 0 the optimiser
            is AdamW (decoupled weight decay); when 0 it is plain Adam. Defaults to
            1e-4.
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
        val_frac (float): Fraction of the training rows held out for internal
            validation and best-epoch selection (seeded shuffle, ``ceil(val_frac *
            n)`` rows). ``0`` disables validation entirely -- train on all rows,
            keep the final-epoch weights. Defaults to 0.1.
        seed (int): Reproducibility seed; from ``RunConfig.seed`` in a real run.
            Threaded into :func:`torch.manual_seed` and the validation-holdout
            shuffle. Defaults to 0.
        device (str | None): Torch device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"mps"``). None auto-selects cuda -> mps -> cpu, mirroring
            :func:`fgas_spk.train.pick_device`. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored, so a stray
            ``model_params`` key never breaks construction.

    Attributes:
        history (list[dict]): One ``{"epoch", "recon", "kl", "beta_eff",
            "train_loss", "val_loss"}`` dict per epoch, populated by :meth:`fit`.
            ``recon`` and ``kl`` are the full-batch training reconstruction term
            (of the posterior-mean prediction, summed over ``n_k`` and averaged
            over the batch) and the analytic KL; ``beta_eff`` is that epoch's
            annealed KL weight; ``train_loss`` is ``recon + beta_eff * kl``;
            ``val_loss`` is the held-out loss at the terminal ``beta`` (``None``
            when ``val_frac == 0``).
        best_epoch (int | None): Epoch of lowest ``val_loss`` whose weights were
            restored, or ``None`` when ``val_frac == 0`` (no validation performed).
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 200,
        lr: float = 5e-4,
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
        self.val_frac = val_frac
        self.seed = seed
        self.device = device

        self.history: list[dict] = []
        self.best_epoch: int | None = None

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
        """Fit the deep-VIB regressor on a TrainingData bundle.

        Consumes ``training_data.X`` (profiles), ``training_data.y`` (SP(k) target,
        the curve of shape ``(n_examples, n_k)``; a 1-D ``single_k`` target is
        supported and inverted back to 1-D at predict time), and -- when present --
        the two conditioning modalities ``training_data.X_cond`` (observable
        scalars) and ``training_data.X_params`` (CAMELS parameters). Both
        conditioning modalities are standardised on their own scale and
        concatenated into one context vector fed to the encoder *and* the decoder;
        whichever modalities are present at fit are then required at predict.

        A seeded ``val_frac`` slice of the rows is held out for validation and
        best-epoch selection (see the module docstring); the network trains on the
        remainder and its lowest-validation-loss weights are restored at the end.
        Standardisation statistics are fitted on the training rows only. torch is
        imported lazily inside this method.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.
        """
        import torch

        X = np.asarray(training_data.X, dtype=np.float64)
        y = np.asarray(training_data.y, dtype=np.float64)
        X_cond = training_data.X_cond
        X_params = training_data.X_params

        self._uses_cond = X_cond is not None
        if self._uses_cond:
            print("Using conditioning scalars (X_cond) in VibRegressor.fit.")
            X_cond = np.asarray(X_cond, dtype=np.float64)

        self._uses_params = X_params is not None
        if self._uses_params:
            print("Using CAMELS parameters (X_params) in VibRegressor.fit.")
            X_params = np.asarray(X_params, dtype=np.float64)

        # Keep the target 2-D internally; remember a 1-D (single_k) input so the
        # prediction can be squeezed back to the caller's shape.
        self._y_was_1d = y.ndim == 1
        y2d = y[:, None] if self._y_was_1d else y

        # Seeded validation holdout: shuffle the row indices with a NumPy generator
        # (backend-independent, unlike a torch shuffle) and hold out the first
        # ceil(val_frac * n). val_frac == 0 (or a degenerate split that would leave
        # no training rows) means "no validation": train on all rows, keep the
        # final weights -- the pre-validation behaviour.
        n_total = X.shape[0]
        n_val = math.ceil(self.val_frac * n_total) if self.val_frac > 0 else 0
        use_val = 0 < n_val < n_total
        if use_val:
            shuffled = np.random.default_rng(self.seed).permutation(n_total)
            val_rows = shuffled[:n_val]
            train_rows = shuffled[n_val:]
        else:
            train_rows = np.arange(n_total)
            val_rows = np.empty(0, dtype=int)

        # Fit normalisation on the TRAINING rows only (each modality on its own
        # scale), so the held-out rows never leak into the standardisation.
        self._x_mean, self._x_std = self._fit_norm(X[train_rows])
        self._y_mean, self._y_std = self._fit_norm(y2d[train_rows])
        if X_cond is not None:
            self._cond_mean, self._cond_std = self._fit_norm(X_cond[train_rows])
        if X_params is not None:
            self._params_mean, self._params_std = self._fit_norm(X_params[train_rows])

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

        # Standardise and move to the device as float32 tensors, then slice into the
        # train / validation subsets. The conditioning tensor is the standardised
        # [X_cond | X_params] context (None if neither modality is in use).
        Xs = self._to_tensor(self._apply_norm(X, self._x_mean, self._x_std))
        ys = self._to_tensor(self._apply_norm(y2d, self._y_mean, self._y_std))
        context = self._context(X_cond, X_params)
        conds = self._to_tensor(context) if context is not None else None

        train_t = torch.as_tensor(train_rows, dtype=torch.long, device=self._device)
        X_tr, y_tr = Xs[train_t], ys[train_t]
        c_tr = conds[train_t] if conds is not None else None
        if use_val:
            val_t = torch.as_tensor(val_rows, dtype=torch.long, device=self._device)
            X_va, y_va = Xs[val_t], ys[val_t]
            c_va = conds[val_t] if conds is not None else None

        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(params, lr=self.lr)

        n_train = X_tr.shape[0]
        batch_size = min(self.batch_size, n_train)
        self.history = []
        self.best_epoch = None
        best_val = math.inf
        best_state = None
        for epoch in range(self.epochs):
            beta_eff = self._beta_eff(epoch)
            self._encoder.train()
            self._decoder.train()
            perm = torch.randperm(n_train, device=self._device)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_tr[idx]
                yb = y_tr[idx]
                cb = c_tr[idx] if c_tr is not None else None
                optimizer.zero_grad()
                mu, logvar = self._encode_dist(xb, cb)
                # Reparameterised sample at train time (global-seeded eps).
                z = self._reparameterise(mu, logvar)
                pred = self._decode(z, cb)
                recon = self._recon(pred, yb)
                kl = self._kl(mu, logvar)
                loss = recon + beta_eff * kl
                loss.backward()
                # Clip the gradient norm across all parameters before the step.
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            # Per-epoch trace: full-batch (training-rows) reconstruction of the
            # POINT prediction (posterior mean, matching predict), the analytic KL,
            # the annealed weight, and the total. The held-out val_loss uses the
            # TERMINAL beta (self.beta) so successive epochs are compared on one
            # fixed objective; it is the best-epoch selection criterion.
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                mu_tr, logvar_tr = self._encode_dist(X_tr, c_tr)
                recon_tr = float(self._recon(self._decode(mu_tr, c_tr), y_tr).item())
                kl_tr = float(self._kl(mu_tr, logvar_tr).item())
                if use_val:
                    mu_va, logvar_va = self._encode_dist(X_va, c_va)
                    recon_va = float(self._recon(self._decode(mu_va, c_va), y_va).item())
                    kl_va = float(self._kl(mu_va, logvar_va).item())
                    val_loss = recon_va + self.beta * kl_va
                else:
                    val_loss = None
            self.history.append(
                {
                    "epoch": epoch,
                    "recon": recon_tr,
                    "kl": kl_tr,
                    "beta_eff": beta_eff,
                    "train_loss": recon_tr + beta_eff * kl_tr,
                    "val_loss": val_loss,
                }
            )

            # Snapshot the best-validation weights (deep-copied so later epochs do
            # not mutate the saved tensors).
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {
                    "encoder": copy.deepcopy(self._encoder.state_dict()),
                    "decoder": copy.deepcopy(self._decoder.state_dict()),
                }

        # Restore the best-validation weights (no-op when val_frac == 0).
        if best_state is not None:
            self._encoder.load_state_dict(best_state["encoder"])
            self._decoder.load_state_dict(best_state["decoder"])

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
            raise RuntimeError("VibRegressor.predict called before fit.")
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
            raise RuntimeError("VibRegressor.latents called before fit.")
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
        """Encode ``concat(x, cond)`` (or ``x`` alone) to ``(mu, logvar)``.

        ``logvar`` is clamped to ``[-8, 8]`` so ``exp(0.5 * logvar)`` and the KL
        term stay numerically bounded even on out-of-distribution inputs.
        """
        import torch

        enc_in = x if cond is None else torch.cat([x, cond], dim=1)
        assert self._encoder is not None
        h = self._encoder(enc_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -8.0, 8.0)
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
    def _recon(pred, y):
        """Reconstruction term: squared error summed over ``n_k``, mean over batch.

        ``((pred - y)**2).sum(dim=1).mean()`` -- a per-example *sum* across the
        target bins, averaged over the batch. This is ``n_k`` times the plain
        mean-over-all-elements MSE, matching the ELBO's per-example log-likelihood
        convention (one term per target bin). Returns a scalar tensor.
        """
        return ((pred - y) ** 2).sum(dim=1).mean()

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
            print(f"Using user-specified device '{self.device}' for VibRegressor.")
            return torch.device(self.device)
        if torch.cuda.is_available():
            print("Using CUDA device for VibRegressor.")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using MPS device for VibRegressor.")
            return torch.device("mps")
        print("Using CPU device for VibRegressor.")
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
