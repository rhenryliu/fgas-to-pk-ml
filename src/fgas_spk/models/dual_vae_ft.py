"""Stage 6 fine-tuned dual-codec composite (``dual_vae_ft``).

Amendment 7 (J2): the end-to-end fine-tune of amendment 6's Arm V, testing
whether the F-1 deficit — VAE-X codes carrying less task-relevant information
than PCA scores — is objective-induced and recoverable under task
supervision, or structural. Registered separately; the frozen-pipeline
:class:`~fgas_spk.models.dual_vae.DualVae` remains untouched as the
diagnosable baseline.

**Two phases** (maintainer design choice, J2.1):

1. **Joint fine-tune.** Starting from the trained Arm V components
   (VAE-Y; VAE-X; the rung-2 MLP map fitted on the frozen ``(mu1, mu2)``
   pairs), the encoder-X and the MLP map are fine-tuned **jointly** with a
   y-space Gaussian NLL through the **frozen decoder-Y** — every decoder-Y
   parameter frozen, including ``obs_logvar``, so the y-manifold and noise
   semantics are preserved exactly. Small learning rate, internal seeded
   holdout, best-epoch restore; the G3.4 training-sanity criterion applies
   (:attr:`ft_best_epoch`).
2. **MDN re-fit.** The fine-tuned encoder is frozen; ``(mu1', mu2)`` pairs
   are rebuilt on the training data; the MDN (rung 3, same K and protocol as
   Arm V) is re-fit on them. This isolates the recoverability question from
   density estimation and re-establishes calibration through a standard MDN
   fit rather than a joint mixture-through-decoder objective.

Prediction paths, spread semantics (F0.5), and the public surface are
inherited from :class:`~fgas_spk.models.dual_vae.DualVae`; after ``fit`` the
composite is ``x -> fine-tuned encoder-X -> MDN -> frozen decoder-Y``.
``history`` gains a ``"finetune"`` phase (per-epoch train/holdout y-space
NLL) between the initial-map and final-MDN phases.

Dependencies: torch, lazily imported inside methods, as everywhere else.
"""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import numpy as np

from fgas_spk.models.base import register
from fgas_spk.models.dual_vae import DualVae
from fgas_spk.models.dual_vae_components import (
    GaussianVAE,
    LatentMapMDN,
    LatentMapMLP,
    _LOG_2PI,
)

if TYPE_CHECKING:  # type hints only
    from fgas_spk.loader import TrainingData


@register("dual_vae_ft")
class DualVaeFt(DualVae):
    """Two-phase fine-tuned Arm V composite (see the module docstring).

    Args:
        ft_epochs (int): Phase-1 fine-tune epochs. Defaults to 1000.
        ft_lr (float): Phase-1 learning rate (small by design). Defaults to
            1e-4.
        ft_val_frac (float): Phase-1 internal holdout fraction for best-epoch
            restore. Defaults to 0.1.
        **kwargs: Everything :class:`~fgas_spk.models.dual_vae.DualVae`
            accepts. ``codec_x`` / ``codec_y`` are forced to ``"vae"`` and
            ``mapping`` to ``"mdn"`` (the Stage 6 design is Arm V only).

    Attributes:
        ft_history (list[dict]): Phase-1 per-epoch trace (also merged into
            ``history`` with ``phase="finetune"``).
        ft_best_epoch (int | None): Phase-1 restored epoch (G3.4 sanity:
            must not be 0/None).
    """

    def __init__(
        self,
        ft_epochs: int = 1000,
        ft_lr: float = 1e-4,
        ft_val_frac: float = 0.1,
        **kwargs: object,
    ) -> None:
        kwargs.pop("codec_x", None), kwargs.pop("codec_y", None)
        kwargs.pop("mapping", None)
        super().__init__(codec_x="vae", codec_y="vae", mapping="mdn", **kwargs)
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.ft_val_frac = ft_val_frac
        self.ft_history: list[dict] = []
        self.ft_best_epoch: int | None = None

    def fit(self, training_data: "TrainingData") -> None:
        """Train Arm V, then fine-tune (phase 1), then re-fit the MDN (phase 2).

        Args:
            training_data (TrainingData): The model-ready arrays; ``X_cond`` /
                ``X_params`` become the conditioning context as in the parent.
        """
        import torch

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
        map_kw = dict(
            hidden=self.hidden, n_layers=self.n_layers, epochs=self.map_epochs,
            lr=self.lr, weight_decay=self.weight_decay,
            batch_size=self.batch_size, val_frac=self.val_frac,
            seed=self.seed, device=self.device,
        )

        # --- Arm V components ------------------------------------------------
        self._y_codec = GaussianVAE(
            latent_dim=self.latent_dim_y, beta=self.beta_y, **vae_kw
        )
        self._y_codec.fit(y, ctx)
        self._x_codec = GaussianVAE(
            latent_dim=self.latent_dim_x, beta=self.beta_x, **vae_kw
        )
        self._x_codec.fit(X, ctx)
        mu1 = self._x_codec.encode(X, ctx)
        mu2 = self._y_codec.encode(y, ctx)
        mlp_map = LatentMapMLP(**map_kw)
        mlp_map.fit(mu1, mu2)

        # --- Phase 1: joint fine-tune through the frozen decoder-Y -----------
        self._finetune(X, y, ctx, mlp_map)

        # --- Phase 2: freeze the encoder; re-fit the MDN on (mu1', mu2) ------
        mu1_ft = self._x_codec.encode(X, ctx)
        self._map = LatentMapMDN(n_components=self.mdn_components, **map_kw)
        self._map.fit(mu1_ft, mu2)

        self.history = [
            {"phase": phase, **row}
            for phase, rows in (
                ("codec_y", self._y_codec.history),
                ("codec_x", self._x_codec.history),
                ("mapping_init", mlp_map.history),
                ("finetune", self.ft_history),
                ("mapping", self._map.history),
            )
            for row in rows
        ]

    # --- internals -----------------------------------------------------------

    def _finetune(
        self, X: np.ndarray, y: np.ndarray, ctx: np.ndarray | None, mlp_map
    ) -> None:
        """Phase 1: fine-tune encoder-X + MLP map with y-space NLL (J2.1).

        The full differentiable path is: standardise x and context with the
        x-codec's fitted statistics -> encoder mu head -> the MLP map's fitted
        input/output affine standardisation and trunk -> z2 -> the frozen
        decoder-Y (with the y-codec's context standardisation) -> standardised
        y -> Gaussian NLL against the frozen per-bin ``obs_logvar``. Only the
        encoder and map trunks receive gradient updates; every affine
        statistic and every decoder-Y parameter is a frozen constant.
        """
        import torch

        vx, vy = self._x_codec, self._y_codec
        device = vx._device
        t = lambda a: torch.as_tensor(np.asarray(a, float), dtype=torch.float32,
                                      device=device)

        x_std = t(GaussianVAE._apply_norm(X, vx._v_mean, vx._v_std))
        y_std_t = t(GaussianVAE._apply_norm(y, vy._v_mean, vy._v_std))
        ctx_x = (t(GaussianVAE._apply_norm(ctx, vx._ctx_mean, vx._ctx_std))
                 if ctx is not None else None)
        ctx_y = (t(GaussianVAE._apply_norm(ctx, vy._ctx_mean, vy._ctx_std))
                 if ctx is not None else None)
        in_mean, in_std = t(mlp_map._in_mean), t(mlp_map._in_std)
        out_mean, out_std = t(mlp_map._out_mean), t(mlp_map._out_std)
        obs_lv = vy._obs_logvar.detach().clamp(-8.0, 8.0)

        # Freeze the decoder-Y wholesale (gradients still flow THROUGH it to
        # the encoder and map; its own parameters receive none).
        for p in vy._decoder.parameters():
            p.requires_grad_(False)

        encoder, mapper = vx._encoder, mlp_map._net
        params = list(encoder.parameters()) + list(mapper.parameters())

        def forward(idx):
            xb = x_std[idx]
            enc_in = xb if ctx_x is None else torch.cat([xb, ctx_x[idx]], dim=1)
            mu1_b = torch.chunk(encoder(enc_in), 2, dim=1)[0]
            z2_std = mapper((mu1_b - in_mean) / in_std)
            z2 = z2_std * out_std + out_mean
            dec_in = z2 if ctx_y is None else torch.cat([z2, ctx_y[idx]], dim=1)
            v_hat = vy._decoder(dec_in)
            per_ex = 0.5 * (
                (v_hat - y_std_t[idx]).pow(2) / obs_lv.exp() + obs_lv + _LOG_2PI
            ).sum(dim=1)
            return per_ex.mean()

        # Seeded internal holdout (component convention).
        n_total = x_std.shape[0]
        n_val = math.ceil(self.ft_val_frac * n_total) if self.ft_val_frac > 0 else 0
        use_val = 0 < n_val < n_total
        if use_val:
            shuffled = np.random.default_rng(self.seed).permutation(n_total)
            val_rows = torch.as_tensor(shuffled[:n_val], dtype=torch.long,
                                       device=device)
            train_rows = torch.as_tensor(shuffled[n_val:], dtype=torch.long,
                                         device=device)
        else:
            train_rows = torch.arange(n_total, device=device)

        torch.manual_seed(self.seed)
        optimizer = (
            torch.optim.AdamW(params, lr=self.ft_lr,
                              weight_decay=self.weight_decay)
            if self.weight_decay > 0
            else torch.optim.Adam(params, lr=self.ft_lr)
        )
        n_train = train_rows.shape[0]
        batch_size = min(self.batch_size, n_train)
        self.ft_history = []
        self.ft_best_epoch = None
        best_val = math.inf
        best_state = None
        for epoch in range(self.ft_epochs):
            encoder.train(), mapper.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, batch_size):
                idx = train_rows[perm[start:start + batch_size]]
                optimizer.zero_grad()
                loss = forward(idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
            encoder.eval(), mapper.eval()
            with torch.no_grad():
                train_nll = float(forward(train_rows).item())
                val_nll = float(forward(val_rows).item()) if use_val else None
            self.ft_history.append(
                {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll}
            )
            if val_nll is not None and val_nll < best_val:
                best_val = val_nll
                self.ft_best_epoch = epoch
                best_state = (copy.deepcopy(encoder.state_dict()),
                              copy.deepcopy(mapper.state_dict()))
        if best_state is not None:
            encoder.load_state_dict(best_state[0])
            mapper.load_state_dict(best_state[1])
