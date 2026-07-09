"""Sohn-style conditional variational autoencoder for f_gas(R) to SP(k) (PyTorch).

Unlike :mod:`~fgas_spk.models.vib_regressor` -- which uses a stochastic bottleneck
with a *fixed* ``N(0, I)`` prior purely as a regulariser for a point prediction --
this model is a genuine **conditional generative model** of ``p(SP(k) | f_gas(R),
context)``. It is the conditional VAE of Sohn, Lee & Yan (2015): a
*conditional* prior ``p(z | x)`` and a recognition network ``q(z | x, y)`` that
sees the target during training. That is exactly what lets the decoded sample
spread from :meth:`predict_samples` be an estimate of the conditional
distribution's width -- how tightly a given f_gas(R) profile constrains SP(k) --
rather than a regularisation noise floor. The scientific payload here is the
*spread*, not only the point prediction.

**Three networks** (each a shared ``n_layers``-deep GELU MLP with dropout):

1. **recognition** ``q(z | x, y, ctx)`` -- input ``concat(X, y, ctx)``, output the
   ``(mu_q, logvar_q)`` of the approximate posterior (``2 * latent_dim`` values).
   **Used only during training** (and by :meth:`latents` when a target is passed):
   it needs ``y``, which is unavailable at inference.
2. **prior** ``p(z | x, ctx)`` -- input ``concat(X, ctx)``, output ``(mu_p,
   logvar_p)``. This is the *conditional* prior; its ``logvar_p`` head is
   zero-initialised so the prior starts at unit variance and the KL is well-behaved
   from epoch 0. This network is what inference samples from.
3. **decoder** ``p(y | z, x, ctx)`` -- input ``concat(z, X, ctx)``, output SP(k) of
   width ``n_k``. The decoder consumes the **profile directly** (not only the
   context), so ``x`` informs the mean and the latent ``z`` carries the residual,
   underdetermined variation.

All ``logvar`` outputs are clamped to ``[-8, 8]``. The conditioning *context* is
handled exactly as in :mod:`~fgas_spk.models.vib_regressor`: whichever of
``X_cond`` (observable scalars) and ``X_params`` (CAMELS parameters) are present is
standardised on its own scale and concatenated; the profile-only path is
supported. Whichever modalities are present at :meth:`fit` are required at predict.

**Loss -- the conditional ELBO (in ELBO units), with KL annealing.** Per minibatch:

    z ~ q(z | x, y, ctx)   (reparameterised)
    recon = ((decode(z, x, ctx) - y)**2).sum(dim=1).mean()
    kl    = mean_batch sum_j KL( N(mu_q, e^{logvar_q}) || N(mu_p, e^{logvar_p}) )
          = 0.5 * sum_j ( logvar_p - logvar_q
                + (e^{logvar_q} + (mu_q - mu_p)^2) / e^{logvar_p} - 1 )
    total = recon + beta_eff * kl

The two-Gaussian KL reduces to the familiar ``KL(q || N(0, I))`` when ``mu_p = 0``
and ``logvar_p = 0``. ``recon`` is the squared error **summed over the ``n_k``
target bins and averaged over the batch** (= ``n_k`` times the plain MSE), matching
the ELBO's per-example log-likelihood convention. ``beta_eff`` anneals linearly
from ~0 to the configured ``beta`` (default 1.0) over the first ``anneal_epochs``
epochs (default ``epochs // 4``). Gradients are clipped to a max norm of 1.0
between ``backward`` and the step; the optimiser (AdamW when ``weight_decay > 0``,
else Adam) spans **all three** networks' parameters.

**Optional learned observation-noise head (``obs_noise_head=True``; default
off).** The squared-error ``recon`` above carries **no aleatoric term**: all
predictive spread comes from the latent, which is the diagnosed cause of this
model's under-coverage. With the head enabled the model mirrors
:class:`~fgas_spk.models.dual_vae_components.GaussianVAE`'s noise head: a
learnable per-k parameter ``obs_logvar`` of shape ``(n_k,)`` (``(1,)`` for a 1-D
``single_k`` target), initialised to 0.0 (unit variance on the standardised
scale) and clamped to ``[-8, 8]`` inside the loss, joins the optimiser and the
best-epoch snapshot/restore alongside the three networks, and the reconstruction
term becomes the Gaussian negative log-likelihood on the standardised scale::

    recon = mean_batch 0.5 * sum_k ( obs_logvar_k
                                     + (decode(z, x, ctx) - y)_k^2 / e^{obs_logvar_k} )

(the additive ``log 2*pi`` constant is dropped, consistently in both the training
loss and the held-out ``val_loss``, so it never affects best-epoch selection).
The held-out ``val_loss`` uses the same NLL with the terminal ``beta``.
:meth:`predict` is unchanged (the deterministic prior-mean decode);
:meth:`predict_samples` adds, per decoded draw, seeded Gaussian observation noise
``eps ~ N(0, e^{0.5 * obs_logvar})`` on the standardised scale before the target
standardisation is inverted, so its spread carries the aleatoric term the
toggled-off model lacks; :meth:`obs_sigma` exposes the fitted noise on the raw
SP(k) scale (``e^{0.5 * obs_logvar} * y_std``). With the head **off**, behaviour
is exactly the pre-head model (same loss, same sampling, same seeded results).

**Caveat -- ``beta`` semantics differ between the two modes.** Swapping the
squared-error sum for the NLL rescales the reconstruction term relative to the
KL (at the ``obs_logvar = 0`` init the NLL's error term is *half* the squared
-error sum, and it shrinks further as the noise scale grows), so a given
``beta`` weights the KL differently with the head on than off. Do not transfer
a tuned ``beta`` across modes without re-checking.

**Internal validation and best-epoch restore.** As in
:mod:`~fgas_spk.models.vib_regressor`: ``val_frac`` (default 0.1) of the rows are
held out by a seeded shuffle (``ceil(val_frac * n)`` rows), the per-epoch
validation loss is computed with the **terminal** ``beta`` (not the annealed
``beta_eff``), the lowest-validation-loss weights of all three networks are
restored at the end of :meth:`fit`, and :attr:`best_epoch` records that epoch.
``val_frac == 0`` disables it (train on all rows, keep the final weights,
``val_loss`` recorded as ``None``, :attr:`best_epoch` ``None``).

**Per-epoch history and the collapse diagnostic.** :attr:`history` records, per
epoch, ``recon`` (full training-batch reconstruction with ``z = mu_q``),
``recon_prior`` (the same but with ``z = mu_p``), ``latent_gap = recon_prior -
recon``, ``kl``, ``beta_eff``, ``train_loss = recon + beta_eff * kl``, and
``val_loss``. **``latent_gap ~ 0`` together with ``kl ~ 0`` means the latent
transmits no information about ``y`` beyond ``x``** -- the prior mean already
reconstructs as well as the posterior mean. That is either posterior collapse
(KL-vanishing) or a genuinely conditionally-deterministic target; the two are
**distinguished by the coverage diagnostics, not by this trace**. A healthy,
informative latent shows ``latent_gap > 0`` and ``kl > 0``.

**Inference.**

- :meth:`predict` decodes with ``z = mu_p`` (the conditional-prior mean). It is
  deterministic and is an approximation to ``E[y | x]`` -- *exact* only when the
  decoder is linear in ``z`` (otherwise ``E[decode(z)] != decode(E[z])`` by
  Jensen); for a mildly non-linear decoder it is a close, and by far the cheapest,
  point summary.
- :meth:`predict_samples` draws ``z ~ N(mu_p, e^{logvar_p})`` from the conditional
  prior and decodes each draw (plus, with the noise head on, per-draw Gaussian
  observation noise), so the across-sample spread estimates ``p(y | x)``.
- :meth:`latents` returns ``mu_p`` by default, or the recognition code ``mu_q``
  when a target ``y`` is supplied.

**Normalisation / determinism.** As in :mod:`~fgas_spk.models.vib_regressor`:
per-column standardisation fitted on the training rows (profile, each conditioning
modality, and the target), with the target standardisation inverted on the way
out; zero-variance columns get unit scale. :meth:`fit` seeds torch and the
validation shuffle for run-to-run stability on a *fixed* backend (not
bit-reproducible across backends -- pass ``device="cpu"`` to compare).
:meth:`predict` is deterministic; :meth:`predict_samples` is stochastic by design
but seeded (a CPU :class:`torch.Generator`, with ``eps`` moved to the device so it
is MPS-safe), hence reproducible on a fixed backend for a fixed seed.

In a run, the model is constructed from the
:class:`~fgas_spk.experiment.RunConfig` as
``Cvae(seed=run_config.seed, **run_config.model_params)`` and looked up by name via
``fgas_spk.models.REGISTRY["cvae"]``.

Dependencies:
    PyTorch (pinned in the project environment, an opt-in training dependency, not
    part of the core numpy+pyyaml import surface). torch is imported **lazily inside
    the methods** -- never at module import or at registration -- so
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


@register("cvae")
class Cvae:
    """Sohn-style conditional VAE mapping f_gas(R) (+ conditioning) to p(SP(k)).

    Trains a ``recognition / conditional-prior / decoder`` triple by maximising a
    KL-annealed conditional ELBO (see the module docstring), standardising its
    inputs and target internally. :meth:`predict` returns the deterministic
    prior-mean point prediction (an approximation to ``E[y|x]``);
    :meth:`predict_samples` returns decoded conditional-prior samples whose
    across-sample spread estimates ``p(y|x)``. Widths are inferred from the first
    :meth:`fit` batch; nothing about the data dimensions is hardcoded.

    Args:
        latent_dim (int): Width of the stochastic latent. Defaults to 4.
        hidden (int): Hidden-layer width in each network. A deliberately modest 128
            by default: with a small training set, favour regularisation over raw
            capacity. Defaults to 128.
        n_layers (int): Number of hidden layers in each of the three networks.
            Defaults to 2.
        epochs (int): Number of training epochs. Defaults to 200.
        lr (float): Adam/AdamW learning rate. Defaults to 5e-4.
        weight_decay (float): Weight decay. When > 0 the optimiser is AdamW
            (decoupled weight decay); when 0 it is plain Adam. Defaults to 1e-4.
        dropout (float): Dropout probability after each hidden activation. 0.0
            disables it. Defaults to 0.0.
        batch_size (int): Minibatch size; capped at the training-set size at fit
            time. Defaults to 128.
        beta (float): Terminal KL weight in the ELBO after annealing. Defaults to
            1.0.
        anneal_epochs (int | None): Number of epochs over which ``beta_eff`` ramps
            linearly from ~0 to ``beta``. None selects ``epochs // 4``; a value
            <= 0 disables annealing (``beta`` applies from epoch 0). Defaults to
            None.
        val_frac (float): Fraction of the rows held out for internal validation and
            best-epoch selection (seeded shuffle, ``ceil(val_frac * n)`` rows).
            ``0`` disables validation (train on all rows, keep the final weights).
            Defaults to 0.1.
        obs_noise_head (bool): Enable the learned per-k observation-noise head:
            a learnable ``obs_logvar`` of shape ``(n_k,)`` turns the
            reconstruction term into a Gaussian NLL and adds seeded observation
            noise to :meth:`predict_samples` (see the module docstring, including
            the ``beta``-semantics caveat). Off (the default) reproduces the
            pre-head model exactly. Defaults to False.
        seed (int): Reproducibility seed; from ``RunConfig.seed`` in a real run.
            Threaded into :func:`torch.manual_seed`, the validation-holdout shuffle,
            and the default :meth:`predict_samples` sampling seed. Defaults to 0.
        device (str | None): Torch device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"mps"``). None auto-selects cuda -> mps -> cpu, mirroring
            :func:`fgas_spk.train.pick_device`. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored, so a stray
            ``model_params`` key never breaks construction.

    Attributes:
        history (list[dict]): One dict per epoch with keys ``epoch``, ``recon``,
            ``kl``, ``beta_eff``, ``train_loss``, ``val_loss``, ``recon_prior`` and
            ``latent_gap`` (see the module docstring; ``latent_gap ~ 0`` with
            ``kl ~ 0`` flags a non-informative latent -- collapse or genuine
            conditional determinism, told apart by the coverage diagnostics).
            With ``obs_noise_head`` the ``recon`` / ``recon_prior`` entries are
            the Gaussian NLL, not the squared-error sum -- do not compare their
            values across the two modes.
        best_epoch (int | None): Epoch of lowest ``val_loss`` whose weights were
            restored, or ``None`` when ``val_frac == 0``.
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
        obs_noise_head: bool = False,
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
        self.obs_noise_head = obs_noise_head
        self.seed = seed
        self.device = device

        self.history: list[dict] = []
        self.best_epoch: int | None = None

        self._recognition = None
        self._prior = None
        self._decoder = None
        self._obs_logvar = None  # torch.nn.Parameter (n_k,), std scale; head only
        self._device = None
        self._uses_cond: bool = False
        self._uses_params: bool = False
        self._y_was_1d: bool = False
        # Standardisation statistics, fitted on the train rows in fit.
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
        """Fit the conditional VAE by maximising the annealed conditional ELBO.

        Consumes ``training_data.X`` (profiles), ``training_data.y`` (SP(k) target,
        curve of shape ``(n_examples, n_k)``; a 1-D ``single_k`` target is
        supported and inverted back to 1-D at predict time), and -- when present --
        the two conditioning modalities ``training_data.X_cond`` and
        ``training_data.X_params``. Both are standardised on their own scale and
        concatenated into one context fed to all three networks; whichever
        modalities are present at fit are then required at predict.

        A seeded ``val_frac`` slice of the rows is held out for validation and
        best-epoch selection; the network trains on the remainder and its
        lowest-validation-loss weights are restored at the end. Standardisation
        statistics are fitted on the training rows only. torch is imported lazily
        inside this method.

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

        # Seeded validation holdout (backend-independent NumPy shuffle); the first
        # ceil(val_frac * n) rows are held out. val_frac == 0 (or a degenerate
        # split leaving no training rows) means "no validation".
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

        # Infer widths from this first batch -- no hardcoded dimensions.
        n_profile = X.shape[1]
        n_cond = X_cond.shape[1] if X_cond is not None else 0
        n_params = X_params.shape[1] if X_params is not None else 0
        n_context = n_cond + n_params
        n_k = y2d.shape[1]

        # Seed, resolve the device, and build the modules now that dims are known.
        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._build_modules(n_profile, n_context, n_k)
        assert (
            self._recognition is not None
            and self._prior is not None
            and self._decoder is not None
        )
        self._recognition.to(self._device)
        self._prior.to(self._device)
        self._decoder.to(self._device)

        # Standardise, move to the device as float32 tensors, and slice into the
        # train / validation subsets.
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

        params = (
            list(self._recognition.parameters())
            + list(self._prior.parameters())
            + list(self._decoder.parameters())
        )
        if self.obs_noise_head:
            params.append(self._obs_logvar)
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
            self._train_mode()
            perm = torch.randperm(n_train, device=self._device)
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                xb = X_tr[idx]
                yb = y_tr[idx]
                cb = c_tr[idx] if c_tr is not None else None
                optimizer.zero_grad()
                mu_q, logvar_q = self._recognition_dist(xb, yb, cb)
                # Reparameterised sample from the posterior at train time.
                z = self._reparameterise(mu_q, logvar_q)
                pred = self._decode(z, xb, cb)
                recon = self._recon_loss(pred, yb)
                mu_p, logvar_p = self._prior_dist(xb, cb)
                kl = self._kl(mu_q, logvar_q, mu_p, logvar_p)
                loss = recon + beta_eff * kl
                loss.backward()
                # Clip the gradient norm across all parameters before the step.
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

            # Per-epoch trace (full training-batch). recon uses z = mu_q (the
            # posterior mean), recon_prior uses z = mu_p (the prior mean, matching
            # predict); latent_gap = recon_prior - recon. The held-out val_loss
            # uses the TERMINAL beta so successive epochs are compared on one fixed
            # objective; it is the best-epoch selection criterion.
            self._eval_mode()
            with torch.no_grad():
                mu_q_tr, logvar_q_tr = self._recognition_dist(X_tr, y_tr, c_tr)
                mu_p_tr, logvar_p_tr = self._prior_dist(X_tr, c_tr)
                recon_tr = float(
                    self._recon_loss(self._decode(mu_q_tr, X_tr, c_tr), y_tr).item()
                )
                recon_prior_tr = float(
                    self._recon_loss(self._decode(mu_p_tr, X_tr, c_tr), y_tr).item()
                )
                kl_tr = float(self._kl(mu_q_tr, logvar_q_tr, mu_p_tr, logvar_p_tr).item())
                if use_val:
                    mu_q_va, logvar_q_va = self._recognition_dist(X_va, y_va, c_va)
                    mu_p_va, logvar_p_va = self._prior_dist(X_va, c_va)
                    recon_va = float(
                        self._recon_loss(self._decode(mu_q_va, X_va, c_va), y_va).item()
                    )
                    kl_va = float(
                        self._kl(mu_q_va, logvar_q_va, mu_p_va, logvar_p_va).item()
                    )
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
                    "recon_prior": recon_prior_tr,
                    "latent_gap": recon_prior_tr - recon_tr,
                }
            )

            # Snapshot the best-validation weights (deep-copied so later epochs do
            # not mutate the saved tensors).
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                self.best_epoch = epoch
                best_state = {
                    "recognition": copy.deepcopy(self._recognition.state_dict()),
                    "prior": copy.deepcopy(self._prior.state_dict()),
                    "decoder": copy.deepcopy(self._decoder.state_dict()),
                }
                if self.obs_noise_head:
                    best_state["obs_logvar"] = self._obs_logvar.detach().clone()

        # Restore the best-validation weights (no-op when val_frac == 0).
        if best_state is not None:
            self._recognition.load_state_dict(best_state["recognition"])
            self._prior.load_state_dict(best_state["prior"])
            self._decoder.load_state_dict(best_state["decoder"])
            if self.obs_noise_head:
                with torch.no_grad():
                    self._obs_logvar.copy_(best_state["obs_logvar"])

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict SP(k) for profiles ``X`` (the deterministic point prediction).

        Decodes with the conditional-prior mean (``z = mu_p``, no sampling), so the
        result is deterministic and directly comparable to the other models on the
        same split. This is an approximation to ``E[y | x]`` -- *exact* only for a
        decoder linear in ``z`` (otherwise ``E[decode(z)] != decode(E[z])`` by
        Jensen's inequality). The conditioning modalities must match :meth:`fit`.

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

        if self._prior is None or self._decoder is None:
            raise RuntimeError("Cvae.predict called before fit.")
        self._check_modality(X_cond, X_params)

        x_t, cond_t = self._prepare_inputs(X, X_cond, X_params)
        self._eval_mode()
        with torch.no_grad():
            mu_p, _ = self._prior_dist(x_t, cond_t)
            out = self._decode(mu_p, x_t, cond_t)
        out = out.detach().cpu().numpy()

        assert self._y_mean is not None and self._y_std is not None
        y = self._invert_norm(out, self._y_mean, self._y_std)
        if self._y_was_1d:
            y = y[:, 0]
        return y

    def predict_samples(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
        n_samples: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Draw decoded conditional-prior samples for profiles ``X``.

        For each example this draws ``n_samples`` latents ``z ~ p(z | x, ctx) =
        N(mu_p, exp(logvar_p))`` from the conditional prior, decodes each, and
        inverts the target standardisation. With ``obs_noise_head`` enabled, each
        decoded draw additionally receives Gaussian observation noise ``eps ~
        N(0, exp(0.5 * obs_logvar))`` on the standardised scale (before the
        inversion), so the spread carries the learned aleatoric term as well as
        the latent one. The spread **across the sample axis** (axis 0) estimates
        ``p(y | x)``: how much a given f_gas(R) profile leaves SP(k)
        underdetermined. Not part of the
        :class:`~fgas_spk.models.base.ProfileToSpk` protocol.

        The sampling is seeded (default: the model ``seed``) via a CPU
        :class:`torch.Generator`, with ``eps`` (latent and, when enabled,
        observation) moved to the working device, so it is reproducible on a fixed
        backend and MPS-safe. Dropout is disabled here, so all stochasticity comes
        from the latent (and the observation noise, when enabled).

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

        if self._prior is None or self._decoder is None:
            raise RuntimeError("Cvae.predict_samples called before fit.")
        self._check_modality(X_cond, X_params)

        x_t, cond_t = self._prepare_inputs(X, X_cond, X_params)
        n_examples = x_t.shape[0]
        # Seed a CPU generator explicitly: this is reproducible on every backend
        # (torch.Generator(device="mps") is unsupported); eps is moved to the
        # working device after sampling.
        seed = self.seed if seed is None else seed
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        self._eval_mode()
        with torch.no_grad():
            mu_p, logvar_p = self._prior_dist(x_t, cond_t)  # (n_examples, latent_dim)
            std = torch.exp(0.5 * logvar_p)
            latent_dim = mu_p.shape[1]
            # Sample all draws at once, then decode as one (n_samples * n_examples)
            # batch. eps is drawn on CPU (seeded) and moved to the device.
            eps = torch.randn(
                (n_samples, n_examples, latent_dim), generator=gen
            ).to(self._device)
            z = mu_p.unsqueeze(0) + std.unsqueeze(0) * eps  # (S, N, latent_dim)
            z_flat = z.reshape(n_samples * n_examples, latent_dim)
            # The decoder consumes the profile (and context) too, so broadcast them
            # across the sample axis to match z_flat.
            x_flat = x_t.unsqueeze(0).expand(n_samples, -1, -1).reshape(
                n_samples * n_examples, x_t.shape[1]
            )
            if cond_t is not None:
                cond_flat = cond_t.unsqueeze(0).expand(n_samples, -1, -1).reshape(
                    n_samples * n_examples, cond_t.shape[1]
                )
            else:
                cond_flat = None
            out = self._decode(z_flat, x_flat, cond_flat)  # (S * N, n_k)
            out = out.reshape(n_samples, n_examples, -1)
            if self.obs_noise_head:
                # Learned observation noise, per decoded draw, on the
                # standardised scale -- drawn from the same seeded CPU generator
                # as the latent eps (after it, so toggled-off draws are
                # unchanged) and moved to the working device (MPS-safe).
                eps_obs = torch.randn(out.shape, generator=gen).to(self._device)
                out = out + eps_obs * torch.exp(0.5 * self._obs_logvar)
        out = out.detach().cpu().numpy()

        assert self._y_mean is not None and self._y_std is not None
        samples = self._invert_norm(out, self._y_mean, self._y_std)
        if self._y_was_1d:
            samples = samples[..., 0]  # (n_samples, n_examples)
        return samples

    def obs_sigma(self) -> np.ndarray:
        """Fitted per-k observation noise on the **raw** SP(k) scale.

        ``obs_logvar`` lives on the standardised target scale (init 0.0 = one
        per-k standard deviation), so the raw-scale noise is
        ``exp(0.5 * obs_logvar) * y_std``, mirroring
        :meth:`~fgas_spk.models.dual_vae_components.GaussianVAE.obs_sigma`.

        Returns:
            np.ndarray: Noise sigma per k bin, shape (n_k,) -- ``(1,)`` for a
                1-D ``single_k`` target.

        Raises:
            RuntimeError: If the model was constructed with
                ``obs_noise_head=False``, or if called before :meth:`fit`.
        """
        if not self.obs_noise_head:
            raise RuntimeError(
                "Cvae.obs_sigma requires obs_noise_head=True; this model was "
                "constructed without the observation-noise head."
            )
        if self._obs_logvar is None:
            raise RuntimeError("Cvae.obs_sigma called before fit.")
        assert self._y_std is not None
        std_scale = np.exp(0.5 * self._obs_logvar.detach().cpu().numpy())
        return std_scale * self._y_std

    def latents(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return latent codes for profiles ``X``: prior means, or recognition means.

        With ``y=None`` (the default, and what the runner calls) this returns the
        conditional-**prior** mean ``mu_p`` from ``p(z | x, ctx)`` -- available at
        inference, since it needs no target. When a target ``y`` is supplied it
        returns the **recognition** mean ``mu_q`` from ``q(z | x, y, ctx)`` -- the
        code the encoder assigns when it *sees* the target, useful for inspecting
        how the posterior differs from the prior. Both are deterministic (means, no
        sampling). Not part of the :class:`~fgas_spk.models.base.ProfileToSpk`
        protocol. The conditioning modalities must match :meth:`fit`.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Observable conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                ``X_cond``. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params). Required iff the model was fit with
                ``X_params``. Defaults to None.
            y (np.ndarray | None): Optional SP(k) target, shape (n_examples, n_k)
                (or 1-D for a ``single_k`` model). When given, the recognition mean
                ``mu_q`` is returned instead of the prior mean ``mu_p``. Defaults to
                None.

        Returns:
            np.ndarray: Latent means, shape (n_examples, latent_dim) -- ``mu_p``
                when ``y`` is None, else ``mu_q``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit.
        """
        import torch

        if self._prior is None or self._recognition is None:
            raise RuntimeError("Cvae.latents called before fit.")
        self._check_modality(X_cond, X_params)

        x_t, cond_t = self._prepare_inputs(X, X_cond, X_params)
        self._eval_mode()
        with torch.no_grad():
            if y is None:
                mu, _ = self._prior_dist(x_t, cond_t)
            else:
                assert self._y_mean is not None and self._y_std is not None
                y_arr = np.asarray(y, dtype=np.float64)
                y2d = y_arr[:, None] if y_arr.ndim == 1 else y_arr
                y_t = self._to_tensor(self._apply_norm(y2d, self._y_mean, self._y_std))
                mu, _ = self._recognition_dist(x_t, y_t, cond_t)
        return mu.detach().cpu().numpy()

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_profile: int, n_context: int, n_k: int) -> None:
        """Build the recognition, prior and decoder networks (torch lazy here).

        - recognition ``q(z|x,y,ctx)``: ``n_profile + n_k + n_context -> 2*latent``.
        - prior ``p(z|x,ctx)``: ``n_profile + n_context -> 2*latent``; its logvar
          head is zero-initialised so the prior starts at unit variance.
        - decoder ``p(y|z,x,ctx)``: ``latent + n_profile + n_context -> n_k`` -- the
          decoder consumes the profile directly, not only the context.
        - with ``obs_noise_head``: the learnable ``obs_logvar`` parameter of shape
          ``(n_k,)``, initialised to 0.0 (unit variance on the standardised
          scale), mirroring
          :class:`~fgas_spk.models.dual_vae_components.GaussianVAE`.

        ``n_context`` is the combined width of the conditioning modalities
        (``X_cond`` plus ``X_params``); 0 on the profile-only path.
        """
        import torch

        self._recognition = self._mlp(n_profile + n_k + n_context, 2 * self.latent_dim)
        self._prior = self._mlp(n_profile + n_context, 2 * self.latent_dim)
        self._decoder = self._mlp(self.latent_dim + n_profile + n_context, n_k)

        # Optional per-k observation-noise head (torch.zeros consumes no RNG, so
        # building it leaves the seeded initialisation of the networks -- and
        # therefore the toggled-off training trajectory -- untouched).
        if self.obs_noise_head:
            self._obs_logvar = torch.nn.Parameter(
                torch.zeros(n_k, device=self._device)
            )

        # Zero-init the prior's logvar head (the second latent_dim outputs of the
        # final layer) so logvar_p == 0 for every input at epoch 0 -- a unit-variance
        # prior, keeping the two-Gaussian KL well-behaved from the start.
        final = self._prior[-1]
        with torch.no_grad():
            final.weight[self.latent_dim:].zero_() # type: ignore[union-attr]
            final.bias[self.latent_dim:].zero_() # type: ignore[union-attr]

    def _mlp(self, in_dim: int, out_dim: int):
        """Return an ``n_layers``-deep GELU MLP ``in_dim -> hidden... -> out_dim``.

        A dropout layer (probability ``self.dropout``) follows each hidden
        activation; ``dropout == 0.0`` makes it an identity.
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

    def _recognition_dist(self, x, y, cond):
        """Recognition ``q(z|x,y,ctx)`` -> ``(mu_q, logvar_q)`` (logvar clamped)."""
        import torch

        parts = [x, y] if cond is None else [x, y, cond]
        assert self._recognition is not None
        h = self._recognition(torch.cat(parts, dim=1))
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _prior_dist(self, x, cond):
        """Conditional prior ``p(z|x,ctx)`` -> ``(mu_p, logvar_p)`` (logvar clamped)."""
        import torch

        enc_in = x if cond is None else torch.cat([x, cond], dim=1)
        assert self._prior is not None
        h = self._prior(enc_in)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, torch.clamp(logvar, -8.0, 8.0)

    def _decode(self, z, x, cond):
        """Decode ``concat(z, x, ctx)`` (or ``concat(z, x)``) to standardised SP(k)."""
        import torch

        parts = [z, x] if cond is None else [z, x, cond]
        assert self._decoder is not None
        return self._decoder(torch.cat(parts, dim=1))

    def _reparameterise(self, mu, logvar):
        """Sample ``z = mu + exp(0.5 * logvar) * eps`` with ``eps ~ N(0, I)``.

        Uses :func:`torch.randn_like` (the global torch seed set in :meth:`fit`
        governs it), so gradients flow through ``mu`` and ``logvar`` at train time.
        """
        import torch

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _recon_loss(self, pred, y):
        """Reconstruction term for the configured mode (see the module docstring).

        Dispatches to the Gaussian NLL :meth:`_nll` when ``obs_noise_head`` is
        enabled, else to the original squared-error :meth:`_recon` -- one call
        site for the training loss, the per-epoch trace, and the held-out
        ``val_loss``, so both modes select their best epoch on their own
        reconstruction convention. Returns a scalar tensor.
        """
        if self.obs_noise_head:
            return self._nll(pred, y)
        return self._recon(pred, y)

    @staticmethod
    def _recon(pred, y):
        """Reconstruction term: squared error summed over ``n_k``, mean over batch.

        ``((pred - y)**2).sum(dim=1).mean()`` -- ``n_k`` times the plain
        mean-over-all-elements MSE, matching the ELBO's per-example log-likelihood
        convention. Returns a scalar tensor.
        """
        return ((pred - y) ** 2).sum(dim=1).mean()

    def _nll(self, pred, y):
        """Gaussian NLL reconstruction with the learned observation noise.

        ``mean_batch 0.5 * sum_k( obs_logvar_k + (pred - y)_k^2 /
        e^{obs_logvar_k} )``, with ``obs_logvar`` clamped to ``[-8, 8]`` as in
        :class:`~fgas_spk.models.dual_vae_components.GaussianVAE`. The additive
        ``log 2*pi`` constant is dropped (consistently in the training loss and
        ``val_loss``; it cannot affect optimisation or best-epoch selection).
        Returns a scalar tensor.
        """
        lv = self._obs_logvar.clamp(-8.0, 8.0)
        per_example = 0.5 * ((pred - y).pow(2) / lv.exp() + lv).sum(dim=1)
        return per_example.mean()

    @staticmethod
    def _kl(mu_q, logvar_q, mu_p, logvar_p):
        """Analytic KL( N(mu_q, e^{logvar_q}) || N(mu_p, e^{logvar_p}) ).

        Summed over the latent dims and averaged over the batch::

            0.5 * sum_j ( logvar_p - logvar_q
                  + (e^{logvar_q} + (mu_q - mu_p)^2) / e^{logvar_p} - 1 )

        Reduces to ``KL(q || N(0, I))`` when ``mu_p = 0`` and ``logvar_p = 0``.
        Returns a scalar tensor.
        """
        kl_per_example = 0.5 * (
            logvar_p
            - logvar_q
            + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
            - 1.0
        ).sum(dim=1)
        return kl_per_example.mean()

    def _beta_eff(self, epoch: int) -> float:
        """Return the annealed KL weight at ``epoch``.

        Ramps linearly from 0 at epoch 0 to ``beta`` at epoch ``anneal_epochs``,
        then holds at ``beta``. A non-positive ``anneal_epochs`` disables annealing.
        """
        if self.anneal_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, epoch / self.anneal_epochs)

    def _train_mode(self) -> None:
        """Put all three networks in train mode."""
        self._recognition.train() # type: ignore[union-attr]
        self._prior.train() # type: ignore[union-attr]
        self._decoder.train() # type: ignore[union-attr]

    def _eval_mode(self) -> None:
        """Put all three networks in eval mode."""
        self._recognition.eval() # type: ignore[union-attr]
        self._prior.eval() # type: ignore[union-attr]
        self._decoder.eval() # type: ignore[union-attr]

    def _prepare_inputs(self, X, X_cond, X_params):
        """Standardise ``X`` and the context and return them as device tensors."""
        assert self._x_mean is not None and self._x_std is not None
        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        context = self._context(X_cond, X_params)
        cond_t = self._to_tensor(context) if context is not None else None
        return x_t, cond_t

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
        with its stored statistics) fed to all three networks, or None if neither
        modality is in use. Presence is assumed already validated by
        :meth:`_check_modality`.

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
