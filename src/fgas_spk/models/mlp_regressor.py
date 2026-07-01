"""Deterministic, bottlenecked MLP regressor for f_gas(R) to SP(k) (PyTorch).

This is the **supervised skeleton** the later conditional VAE (CVAE) inherits, so
the bottleneck is mandatory and central, not incidental. The architecture is an
encoder/decoder pair around a narrow deterministic latent:

The conditioning *context* is whichever of two optional modalities are present:
``X_cond`` (observable scalars -- number density, optional mean halo mass) and
``X_params`` (the simulation's CAMELS parameters). Each is standardised on its own
scale and the two are concatenated into a single context vector; both are wired in
identically, so ``X_params`` is used exactly as ``X_cond`` is.

1. **encoder** maps ``concat(X, context)`` to a latent of size ``latent_dim``
   (default 4). ``X`` is the f_gas(R) profile; the conditioning context is
   concatenated to the input exactly as :mod:`~fgas_spk.models.pca_linear`
   concatenates ``X_cond``. With no conditioning the encoder reads the profile
   alone.
2. **decoder** maps ``concat(latent, context)`` to the SP(k) target of width
   ``n_k``. The decoder is conditioned on the context too -- mirroring the Lin
   et al. trick of conditioning the decoder on cosmology (the CAMELS parameters
   make that literal) -- so the latent is pushed to carry the residual feedback
   information rather than re-encoding the number density or the known
   parameters. With no conditioning the decoder reads the latent alone.

The latent is a *plain deterministic bottleneck*: there is no sampling here. The
stochastic latent is the CVAE's job in the next plugin; this model is the
point-prediction skeleton it builds on.

**Normalisation.** Raw f_gas, SP(k), and the conditioning modalities live on very
different scales, so the regressor standardises internally. At :meth:`fit` time it
fits a per-column mean and standard deviation on the training batch for the
profile ``X``, each conditioning modality present (``X_cond`` and ``X_params``),
and the target ``y``; the network sees and predicts standardised quantities, and
:meth:`predict` inverts the target standardisation on the way out. The stored
statistics are reused at predict time so callers always pass and receive raw-scale
arrays. Zero-variance columns are given unit scale to avoid division by zero.

**Determinism.** :meth:`fit` seeds torch via :func:`torch.manual_seed`, which
gives run-to-run stability on a *fixed* backend (same machine, same device). It
is **not** bit-reproducible across machines or backends: CUDA/MPS kernels are not
guaranteed identical to CPU or to each other. Pass ``device="cpu"`` for a
backend-stable comparison. No global cuDNN-deterministic flags are set (they tank
throughput for no cross-machine guarantee anyway).

**Per-epoch history.** :attr:`history` accumulates one ``{"epoch", "train_mse"}``
dict per epoch (standardised-space training MSE). The reference runner
(:mod:`fgas_spk.train`) currently passes ``metrics=None`` for the non-iterative
``pca_linear`` and is *not* changed by this plugin; ``history`` is exposed so a
later, separate change can thread the per-epoch trace into ``write_run_record``.

In a run, the model is constructed from the
:class:`~fgas_spk.experiment.RunConfig` as
``MlpRegressor(seed=run_config.seed, **run_config.model_params)`` and looked up by
name via ``fgas_spk.models.REGISTRY["mlp_regressor"]``.

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


@register("mlp_regressor")
class MlpRegressor:
    """Bottlenecked encoder/decoder MLP mapping f_gas(R) (+ conditioning) to SP(k).

    The model standardises its inputs and target internally (see the module
    docstring) and trains an encoder -> narrow latent -> decoder network with Adam
    (or AdamW when ``weight_decay > 0``) to minimise the mean-squared error
    against the SP(k) target. Input and output widths are inferred from the first
    :meth:`fit` batch; nothing about the data dimensions is hardcoded.

    Args:
        latent_dim (int): Width of the deterministic bottleneck. Defaults to 4.
        hidden (int): Hidden-layer width on each side of the bottleneck.
            Defaults to 256.
        n_layers (int): Number of hidden layers in the encoder and in the decoder
            (each). Defaults to 2.
        epochs (int): Number of training epochs. Defaults to 200.
        lr (float): Adam/AdamW learning rate. Defaults to 1e-3.
        weight_decay (float): Weight decay. When > 0 the optimiser is AdamW
            (decoupled weight decay); when 0 it is plain Adam. Defaults to 0.0.
        batch_size (int): Minibatch size; capped at the training-set size at fit
            time. Defaults to 128.
        seed (int): Reproducibility seed; from ``RunConfig.seed`` in a real run.
            Threaded into :func:`torch.manual_seed`. Defaults to 0.
        device (str | None): Torch device string (e.g. ``"cpu"``, ``"cuda"``,
            ``"mps"``). None auto-selects cuda -> mps -> cpu, mirroring
            :func:`fgas_spk.train.pick_device`. Defaults to None.
        **_: Extra keyword arguments are accepted and ignored, so a stray
            ``model_params`` key never breaks construction.

    Attributes:
        history (list[dict]): One ``{"epoch", "train_mse"}`` dict per epoch,
            populated by :meth:`fit`. Exposed for a later change that threads the
            per-epoch trace into the run record; the runner does not read it yet.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden: int = 256,
        n_layers: int = 2,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
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
        self.batch_size = batch_size
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
        """Fit the encoder/decoder network on a TrainingData bundle.

        Consumes ``training_data.X`` (profiles), ``training_data.y`` (SP(k)
        target, the curve of shape ``(n_examples, n_k)``; a 1-D ``single_k``
        target is supported and inverted back to 1-D at predict time), and --
        when present -- the two conditioning modalities ``training_data.X_cond``
        (observable scalars) and ``training_data.X_params`` (CAMELS parameters).
        Both conditioning modalities are standardised on their own scale and
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
            print('Using conditioning scalars (X_cond) in MlpRegressor.fit.')
            X_cond = np.asarray(X_cond, dtype=np.float64)

        self._uses_params = X_params is not None
        if self._uses_params:
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
        loss_fn = nn.MSELoss()

        n = Xs.shape[0]
        batch_size = min(self.batch_size, n)
        self.history = []
        for epoch in range(self.epochs):
            self._encoder.train()
            self._decoder.train()
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb = Xs[idx]
                yb = ys[idx]
                cb = conds[idx] if conds is not None else None
                optimizer.zero_grad()
                pred = self._decode(self._encode(xb, cb), cb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            # Per-epoch metric: full-batch standardised-space training MSE.
            self._encoder.eval()
            self._decoder.eval()
            with torch.no_grad():
                pred_all = self._decode(self._encode(Xs, conds), conds)
                train_mse = float(loss_fn(pred_all, ys).item())
            self.history.append({"epoch": epoch, "train_mse": train_mse})

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict SP(k) for profiles ``X`` (the deterministic point prediction).

        numpy in, torch on the device, forward pass, numpy out, with the target
        standardisation inverted. The conditioning modalities must match
        :meth:`fit`: whichever of ``X_cond`` / ``X_params`` were used at fit are
        required here, and ones unused at fit must stay None.

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
            raise RuntimeError("MlpRegressor.predict called before fit.")
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
            out = self._decode(self._encode(x_t, cond_t), cond_t)
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
        """Return the bottleneck codes for profiles ``X`` (for analysis/plotting).

        Not part of the :class:`~fgas_spk.models.base.ProfileToSpk` protocol -- the
        runner never calls it -- but it lets the latent structure be inspected
        after training. The conditioning modalities must match :meth:`fit`, exactly
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
            np.ndarray: Bottleneck latent codes, shape (n_examples, latent_dim).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit.
        """
        import torch

        if self._encoder is None:
            raise RuntimeError("MlpRegressor.latents called before fit.")
        self._check_modality(X_cond, X_params)
        assert self._x_mean is not None and self._x_std is not None

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        context = self._context(X_cond, X_params)
        cond_t = self._to_tensor(context) if context is not None else None

        self._encoder.eval()
        with torch.no_grad():
            z = self._encode(x_t, cond_t)
        return z.detach().cpu().numpy()

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_profile: int, n_context: int, n_k: int) -> None:
        """Build the encoder and decoder (torch imported lazily here).

        ``n_context`` is the combined width of the conditioning modalities
        (``X_cond`` plus ``X_params``); it widens both the encoder input and the
        decoder input.
        """
        self._encoder = self._mlp(n_profile + n_context, self.latent_dim)
        self._decoder = self._mlp(self.latent_dim + n_context, n_k)

    def _mlp(self, in_dim: int, out_dim: int):
        """Return an ``n_layers``-deep GELU MLP ``in_dim -> hidden... -> out_dim``."""
        from torch import nn

        layers: list = []
        d = in_dim
        for _ in range(self.n_layers):
            layers.append(nn.Linear(d, self.hidden))
            layers.append(nn.GELU())
            d = self.hidden
        layers.append(nn.Linear(d, out_dim))
        return nn.Sequential(*layers)

    def _encode(self, x, cond):
        """Encode ``concat(x, cond)`` (or ``x`` alone) to the latent."""
        import torch

        enc_in = x if cond is None else torch.cat([x, cond], dim=1)
        assert self._encoder is not None
        return self._encoder(enc_in)

    def _decode(self, z, cond):
        """Decode ``concat(z, cond)`` (or ``z`` alone) to standardised SP(k)."""
        import torch

        dec_in = z if cond is None else torch.cat([z, cond], dim=1)
        assert self._decoder is not None
        return self._decoder(dec_in)

    def _resolve_device(self):
        """Resolve the torch device (cuda -> mps -> cpu), honouring ``device``."""
        import torch

        if self.device is not None:
            print(f"Using user-specified device '{self.device}' for MlpRegressor.")
            return torch.device(self.device)
        if torch.cuda.is_available():
            print("Using CUDA device for MlpRegressor.")
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            print("Using MPS device for MlpRegressor.")
            return torch.device("mps")
        print("Using CPU device for MlpRegressor.")
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
