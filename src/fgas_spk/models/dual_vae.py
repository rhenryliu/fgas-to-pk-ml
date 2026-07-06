"""Codec-agnostic dual-codec composite for f_gas(R) to SP(k) (``dual_vae``).

The Stage 5 registered model of the dual-VAE staged spec (F0.8/F0.9, as
amended): ``fit`` trains a **y-codec** on SP(k), an **x-codec** on f_gas(R),
and a **latent mapping rung** on the frozen posterior-mean/code pairs
``(mu1, mu2)`` (decision F0.5), end to end in one process, so a single
``(DataConfig, RunConfig)`` pair reproduces the staged pipeline through
``scripts/run.py``.

**Codec-agnostic design (amendment B7).** ``model_params`` carries ``codec_x``
and ``codec_y``, each ``"vae"`` (a :class:`~fgas_spk.models.dual_vae_components.GaussianVAE`)
or ``"pca"`` (a :class:`~fgas_spk.models.dual_vae_components.PcaCodec`: PCA
scores as codes, inverse transform as decode, per-bin train-fold
reconstruction-residual standard deviations serving the ``obs_sigma`` role,
honestly interpreted as code-truncation error). Both default to ``"vae"``, so
the registry name stays accurate. The mapping rung is selected by
``mapping`` in {"ridge", "mlp", "mdn"}.

**Prediction paths** (the Stage 4 composite helpers, shared verbatim):

* :meth:`predict` -- ``x -> encode_x -> mapping.predict -> decode_y``
  (rungs ridge/mlp: the deterministic map; mdn: the mixture mean).
* :meth:`predict_samples` (mdn only) -- ``z2 ~ p(z2 | mu1)`` decoded per draw
  plus per-bin decoder-Y observation noise. **Spread semantics (F0.5):** the
  spread is the mapping density plus decoder-Y observation noise and
  deliberately EXCLUDES the encoder-X posterior spread and any input
  measurement noise (input-noise propagation is a later, separate addition).

The conditioning context follows the cvae ``_context`` pattern: whichever of
``X_cond`` / ``X_params`` are present at :meth:`fit` are concatenated (each
codec standardises internally; the PCA codec ignores context) and required at
predict. ``history`` concatenates the phases' per-epoch traces with a
``phase`` key (``"codec_y"`` / ``"codec_x"`` / ``"mapping"``); single-shot
phases (PCA, ridge) contribute no rows.

Dependencies:
    torch and scikit-learn, both imported lazily inside the composed
    components -- importing this module (and ``fgas_spk.models``) stays
    torch-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register
from fgas_spk.models.dual_vae_components import (
    GaussianVAE,
    LatentMapMDN,
    LatentMapMLP,
    LatentMapRidge,
    PcaCodec,
    composite_predict,
    composite_samples,
)

if TYPE_CHECKING:  # type hints only
    from fgas_spk.loader import TrainingData

_CODECS = ("vae", "pca")
_MAPPINGS = ("ridge", "mlp", "mdn")


@register("dual_vae")
class DualVae:
    """Codec-agnostic dual-codec composite mapping f_gas(R) to SP(k).

    See the module docstring for the architecture. Hyperparameters mirror the
    Stage 2-4 stage scripts; the defaults below are the staged selections on
    the pinned context (tag 20260706, R < 10 crop).

    Args:
        codec_x (str): ``"vae"`` or ``"pca"`` for the f_gas codec. Defaults
            to ``"vae"``.
        codec_y (str): ``"vae"`` or ``"pca"`` for the SP(k) codec. Defaults
            to ``"vae"``.
        latent_dim_x (int): x-codec code width. Defaults to 2 (F0.3).
        latent_dim_y (int): y-codec code width. Defaults to 3 (Stage 2
            selection).
        beta_x (float): x-codec terminal KL weight (VAE codec only).
            Defaults to 1.0 (Stage 3 selection).
        beta_y (float): y-codec terminal KL weight (VAE codec only).
            Defaults to 1e-4 (Stage 2 selection).
        hidden (int): Hidden width of every trained network. Defaults to 128.
        n_layers (int): Hidden layers of every trained network. Defaults to 2.
        epochs (int): VAE codec training epochs. Defaults to 5000.
        lr (float): Learning rate (codecs and torch rungs). Defaults to 1e-3.
        weight_decay (float): Weight decay (>0 selects AdamW). Defaults to 1e-4.
        dropout (float): Codec dropout. Defaults to 0.0.
        batch_size (int): Minibatch size. Defaults to 128.
        anneal_epochs (int | None): VAE KL warm-up; None -> epochs // 4.
        val_frac (float): Internal holdout for best-epoch restore. Defaults
            to 0.1.
        mapping (str): ``"ridge"`` / ``"mlp"`` / ``"mdn"``. Defaults to
            ``"mdn"`` (the probabilistic rung).
        mdn_components (int): MDN mixture size K. Defaults to 3.
        map_epochs (int): Torch-rung training epochs. Defaults to 2000.
        seed (int): Reproducibility seed, threaded into every component.
            Defaults to 0.
        device (str | None): Torch device string; None auto-selects.
        **_: Extra keyword arguments are accepted and ignored.

    Attributes:
        history (list[dict]): Concatenated per-epoch traces with a ``phase``
            key.
    """

    def __init__(
        self,
        codec_x: str = "vae",
        codec_y: str = "vae",
        latent_dim_x: int = 2,
        latent_dim_y: int = 3,
        beta_x: float = 1.0,
        beta_y: float = 1e-4,
        hidden: int = 128,
        n_layers: int = 2,
        epochs: int = 5000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        batch_size: int = 128,
        anneal_epochs: int | None = None,
        val_frac: float = 0.1,
        mapping: str = "mdn",
        mdn_components: int = 3,
        map_epochs: int = 2000,
        seed: int = 0,
        device: str | None = None,
        **_: object,
    ) -> None:
        if codec_x not in _CODECS or codec_y not in _CODECS:
            raise ValueError(
                f"codec_x / codec_y must be in {_CODECS}; got "
                f"{codec_x!r} / {codec_y!r}."
            )
        if mapping not in _MAPPINGS:
            raise ValueError(f"mapping must be in {_MAPPINGS}; got {mapping!r}.")
        self.codec_x = codec_x
        self.codec_y = codec_y
        self.latent_dim_x = latent_dim_x
        self.latent_dim_y = latent_dim_y
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.anneal_epochs = anneal_epochs
        self.val_frac = val_frac
        self.mapping = mapping
        self.mdn_components = mdn_components
        self.map_epochs = map_epochs
        self.seed = seed
        self.device = device

        self.history: list[dict] = []
        self._x_codec = None
        self._y_codec = None
        self._map = None
        self._uses_cond = False
        self._uses_params = False

    # --- public API --------------------------------------------------------

    def fit(self, training_data: "TrainingData") -> None:
        """Train y-codec, x-codec, and the mapping rung in one process.

        Args:
            training_data (TrainingData): The model-ready arrays; ``X_cond`` /
                ``X_params`` (whichever are present) become the codecs'
                conditioning context, required again at predict.
        """
        X = np.asarray(training_data.X, dtype=float)
        y = np.asarray(training_data.y, dtype=float)
        self._uses_cond = training_data.X_cond is not None
        self._uses_params = training_data.X_params is not None
        ctx = self._context(training_data.X_cond, training_data.X_params)

        vae_kw = dict(
            hidden=self.hidden, n_layers=self.n_layers, epochs=self.epochs,
            lr=self.lr, weight_decay=self.weight_decay, dropout=self.dropout,
            batch_size=self.batch_size, anneal_epochs=self.anneal_epochs,
            val_frac=self.val_frac, seed=self.seed, device=self.device,
        )
        self._y_codec = (
            GaussianVAE(latent_dim=self.latent_dim_y, beta=self.beta_y, **vae_kw)
            if self.codec_y == "vae"
            else PcaCodec(latent_dim=self.latent_dim_y, seed=self.seed)
        )
        self._y_codec.fit(y, ctx)

        self._x_codec = (
            GaussianVAE(latent_dim=self.latent_dim_x, beta=self.beta_x, **vae_kw)
            if self.codec_x == "vae"
            else PcaCodec(latent_dim=self.latent_dim_x, seed=self.seed)
        )
        self._x_codec.fit(X, ctx)

        mu1 = self._x_codec.encode(X, ctx)
        mu2 = self._y_codec.encode(y, ctx)
        map_kw = dict(
            hidden=self.hidden, n_layers=self.n_layers, epochs=self.map_epochs,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, val_frac=self.val_frac,
            seed=self.seed, device=self.device,
        )
        if self.mapping == "ridge":
            self._map = LatentMapRidge(seed=self.seed)
        elif self.mapping == "mlp":
            self._map = LatentMapMLP(**map_kw)
        else:
            self._map = LatentMapMDN(n_components=self.mdn_components, **map_kw)
        self._map.fit(mu1, mu2)

        self.history = [
            {"phase": phase, **row}
            for phase, component in (
                ("codec_y", self._y_codec),
                ("codec_x", self._x_codec),
                ("mapping", self._map),
            )
            for row in getattr(component, "history", [])
        ]

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Composite point prediction (Stage 4 task 5 path).

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars; required iff
                fitted with them. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters; required iff
                fitted with them. Defaults to None.

        Returns:
            np.ndarray: Predicted SP(k), shape (n_examples, n_k).

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: On a conditioning-modality mismatch with :meth:`fit`.
        """
        self._require_fitted("predict")
        ctx = self._checked_context(X_cond, X_params)
        return composite_predict(self._x_codec, self._map, self._y_codec, X, ctx)

    def predict_samples(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
        n_samples: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Composite predictive samples (MDN mapping only), seeded and MPS-safe.

        **F0.5 spread semantics:** the across-sample spread is the mapping
        density plus decoder-Y observation noise (``obs_sigma``); it EXCLUDES
        the encoder-X posterior spread and any input measurement noise.

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars; as :meth:`predict`.
            X_params (np.ndarray | None): CAMELS parameters; as :meth:`predict`.
            n_samples (int): Draws per example. Defaults to 100.
            seed (int | None): Sampling seed; None uses the model seed.

        Returns:
            np.ndarray: Samples, shape (n_samples, n_examples, n_k).

        Raises:
            RuntimeError: If called before :meth:`fit`, or with a non-MDN
                mapping (rungs 1-2 define no predictive distribution).
            ValueError: On a conditioning-modality mismatch with :meth:`fit`.
        """
        self._require_fitted("predict_samples")
        if self.mapping != "mdn":
            raise RuntimeError(
                "predict_samples requires the MDN mapping (rung 3); this model "
                f"was configured with mapping={self.mapping!r}."
            )
        ctx = self._checked_context(X_cond, X_params)
        return composite_samples(
            self._x_codec, self._map, self._y_codec, X, ctx,
            n_samples=n_samples,
            seed=self.seed if seed is None else seed,
        )

    def latents(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the x-codec codes ``mu1`` (the runner's latent diagnostics).

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): As :meth:`predict`.
            X_params (np.ndarray | None): As :meth:`predict`.

        Returns:
            np.ndarray: Codes, shape (n_examples, latent_dim_x).
        """
        self._require_fitted("latents")
        ctx = self._checked_context(X_cond, X_params)
        return self._x_codec.encode(np.asarray(X, dtype=float), ctx)

    def latents_y(
        self,
        y: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the y-codec codes ``mu2`` for inspection (non-protocol).

        Args:
            y (np.ndarray): SP(k) curves, shape (n_examples, n_k).
            X_cond (np.ndarray | None): As :meth:`predict`.
            X_params (np.ndarray | None): As :meth:`predict`.

        Returns:
            np.ndarray: Codes, shape (n_examples, latent_dim_y).
        """
        self._require_fitted("latents_y")
        ctx = self._checked_context(X_cond, X_params)
        return self._y_codec.encode(np.asarray(y, dtype=float), ctx)

    def obs_sigma(self) -> np.ndarray:
        """Decoder-Y per-bin noise on the raw target scale.

        For the PCA y-codec this is the train-fold truncation-residual std
        (see :class:`~fgas_spk.models.dual_vae_components.PcaCodec`).

        Returns:
            np.ndarray: Noise sigma per k bin, shape (n_k,).
        """
        self._require_fitted("obs_sigma")
        return self._y_codec.obs_sigma()

    # --- internals ---------------------------------------------------------

    def _context(
        self, X_cond: np.ndarray | None, X_params: np.ndarray | None
    ) -> np.ndarray | None:
        """Concatenate the present conditioning modalities (cvae pattern)."""
        blocks = [
            np.asarray(b, dtype=float)
            for b in (X_cond, X_params)
            if b is not None
        ]
        return np.hstack(blocks) if blocks else None

    def _checked_context(
        self, X_cond: np.ndarray | None, X_params: np.ndarray | None
    ) -> np.ndarray | None:
        """Validate modality presence against fit and build the context."""
        if self._uses_cond != (X_cond is not None):
            raise ValueError(
                "X_cond presence must match fit: fitted "
                f"{'with' if self._uses_cond else 'without'} X_cond."
            )
        if self._uses_params != (X_params is not None):
            raise ValueError(
                "X_params presence must match fit: fitted "
                f"{'with' if self._uses_params else 'without'} X_params."
            )
        return self._context(X_cond, X_params)

    def _require_fitted(self, method: str) -> None:
        """Raise if the composite has not been fitted yet."""
        if self._x_codec is None or self._y_codec is None or self._map is None:
            raise RuntimeError(f"DualVae.{method} called before fit.")
