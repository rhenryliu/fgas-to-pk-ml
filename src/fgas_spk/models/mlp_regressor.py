"""Deterministic, bottlenecked MLP regressor for f_gas(R) to SP(k) (PyTorch).

This is the **supervised skeleton** the later conditional VAE (CVAE) inherits, so
the bottleneck is mandatory and central, not incidental. The architecture is an
encoder/decoder pair around a narrow deterministic latent:

1. **encoder** maps ``concat(X, X_cond)`` to a latent of size ``latent_dim``
   (default 4). ``X`` is the f_gas(R) profile; the conditioning scalars
   ``X_cond`` (number density, optional mean halo mass) are concatenated to the
   input exactly as :mod:`~fgas_spk.models.pca_linear` concatenates them. With
   ``X_cond=None`` the encoder reads the profile alone.
2. **decoder** maps ``concat(latent, X_cond)`` to the SP(k) target of width
   ``n_k``. The decoder is conditioned on ``X_cond`` too -- mirroring the Lin
   et al. trick of conditioning the decoder on cosmology -- so the latent is
   pushed to carry feedback information rather than re-encoding the number
   density. With ``X_cond=None`` the decoder reads the latent alone.

The latent is a *plain deterministic bottleneck*: there is no sampling here. The
stochastic latent is the CVAE's job in the next plugin; this model is the
point-prediction skeleton it builds on.

**Normalisation.** Raw f_gas and SP(k) live on very different scales, so the
regressor standardises internally. At :meth:`fit` time it fits a per-column mean
and standard deviation on the training batch for the profile ``X``, the
conditioning ``X_cond`` (if present), and the target ``y``; the network sees and
predicts standardised quantities, and :meth:`predict` inverts the target
standardisation on the way out. The stored statistics are reused at predict time
so callers always pass and receive raw-scale arrays. Zero-variance columns are
given unit scale to avoid division by zero.

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
        self._y_was_1d: bool = False
        # Standardisation statistics, fitted on the train batch in fit.
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._cond_mean: np.ndarray | None = None
        self._cond_std: np.ndarray | None = None
        self._y_mean: np.ndarray | None = None
        self._y_std: np.ndarray | None = None

    # --- public API --------------------------------------------------------

    def fit(self, training_data: "TrainingData") -> None:
        """Fit the encoder/decoder network on a TrainingData bundle.

        Consumes ``training_data.X`` (profiles), ``training_data.X_cond``
        (conditioning scalars, if present), and ``training_data.y`` (SP(k)
        target, the curve of shape ``(n_examples, n_k)``; a 1-D ``single_k``
        target is supported and inverted back to 1-D at predict time).
        ``X_params`` is ignored. Standardisation statistics are fitted here and
        stored for predict (see the module docstring). torch is imported lazily
        inside this method.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.
        """
        import torch
        from torch import nn

        X = np.asarray(training_data.X, dtype=np.float64)
        y = np.asarray(training_data.y, dtype=np.float64)
        X_cond = training_data.X_cond

        self._uses_cond = X_cond is not None
        if self._uses_cond:
            X_cond = np.asarray(X_cond, dtype=np.float64)

        # Keep the target 2-D internally; remember a 1-D (single_k) input so the
        # prediction can be squeezed back to the caller's shape.
        self._y_was_1d = y.ndim == 1
        y2d = y[:, None] if self._y_was_1d else y

        # Fit normalisation on the training batch.
        self._x_mean, self._x_std = self._fit_norm(X)
        self._y_mean, self._y_std = self._fit_norm(y2d)
        if self._uses_cond:
            self._cond_mean, self._cond_std = self._fit_norm(X_cond)

        # Infer widths from this first batch -- no hardcoded dimensions.
        n_profile = X.shape[1]
        n_cond = X_cond.shape[1] if self._uses_cond else 0
        n_k = y2d.shape[1]

        # Seed, resolve the device, and build the modules now that dims are known.
        torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._build_modules(n_profile, n_cond, n_k)
        self._encoder.to(self._device)
        self._decoder.to(self._device)

        # Standardise and move to the device as float32 tensors.
        Xs = self._to_tensor(self._apply_norm(X, self._x_mean, self._x_std))
        ys = self._to_tensor(self._apply_norm(y2d, self._y_mean, self._y_std))
        conds = (
            self._to_tensor(self._apply_norm(X_cond, self._cond_mean, self._cond_std))
            if self._uses_cond
            else None
        )

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
        standardisation inverted. The modality must match :meth:`fit`: if the
        model was fit with conditioning, ``X_cond`` is required, and vice versa.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                conditioning. Defaults to None.
            X_params (np.ndarray | None): Ignored by this model (the CVAE
                successor may use it). Defaults to None.

        Returns:
            np.ndarray: Predicted SP(k), shape (n_examples, n_k) for a curve
                target or (n_examples,) for a 1-D ``single_k`` target.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X_cond`` presence does not match how the model was
                fit (the encoder/decoder input widths would not line up).
        """
        import torch

        if self._encoder is None or self._decoder is None:
            raise RuntimeError("MlpRegressor.predict called before fit.")
        self._check_cond_modality(X_cond)

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        cond_t = None
        if self._uses_cond:
            cond_t = self._to_tensor(
                self._apply_norm(
                    np.asarray(X_cond, dtype=np.float64),
                    self._cond_mean,
                    self._cond_std,
                )
            )

        self._encoder.eval()
        self._decoder.eval()
        with torch.no_grad():
            out = self._decode(self._encode(x_t, cond_t), cond_t)
        out = out.detach().cpu().numpy()

        y = self._invert_norm(out, self._y_mean, self._y_std)
        if self._y_was_1d:
            y = y[:, 0]
        return y

    def latents(
        self, X: np.ndarray, X_cond: np.ndarray | None = None
    ) -> np.ndarray:
        """Return the bottleneck codes for profiles ``X`` (for analysis/plotting).

        Not part of the :class:`~fgas_spk.models.base.ProfileToSpk` protocol -- the
        runner never calls it -- but it lets the latent structure be inspected
        after training. The modality check matches :meth:`predict`.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                conditioning. Defaults to None.

        Returns:
            np.ndarray: Bottleneck latent codes, shape (n_examples, latent_dim).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X_cond`` presence does not match how the model was
                fit.
        """
        import torch

        if self._encoder is None:
            raise RuntimeError("MlpRegressor.latents called before fit.")
        self._check_cond_modality(X_cond)

        x_t = self._to_tensor(
            self._apply_norm(np.asarray(X, dtype=np.float64), self._x_mean, self._x_std)
        )
        cond_t = None
        if self._uses_cond:
            cond_t = self._to_tensor(
                self._apply_norm(
                    np.asarray(X_cond, dtype=np.float64),
                    self._cond_mean,
                    self._cond_std,
                )
            )

        self._encoder.eval()
        with torch.no_grad():
            z = self._encode(x_t, cond_t)
        return z.detach().cpu().numpy()

    # --- internals ---------------------------------------------------------

    def _build_modules(self, n_profile: int, n_cond: int, n_k: int) -> None:
        """Build the encoder and decoder (torch imported lazily here)."""
        self._encoder = self._mlp(n_profile + n_cond, self.latent_dim)
        self._decoder = self._mlp(self.latent_dim + n_cond, n_k)

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
        return self._encoder(enc_in)

    def _decode(self, z, cond):
        """Decode ``concat(z, cond)`` (or ``z`` alone) to standardised SP(k)."""
        import torch

        dec_in = z if cond is None else torch.cat([z, cond], dim=1)
        return self._decoder(dec_in)

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

    def _check_cond_modality(self, X_cond: np.ndarray | None) -> None:
        """Raise if ``X_cond`` presence does not match how the model was fit."""
        if self._uses_cond and X_cond is None:
            raise ValueError(
                "This model was fit with conditioning (X_cond); predict/latents "
                "requires X_cond as well."
            )
        if not self._uses_cond and X_cond is not None:
            raise ValueError(
                "This model was fit profile-only but X_cond was given. Pass "
                "X_cond=None to match the fitted modality."
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
