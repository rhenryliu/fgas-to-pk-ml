"""Model-plugin interface and registry for f_gas(R) to SP(k) models.

This module defines the contract every model plugin implements -- the
:class:`ProfileToSpk` protocol -- and the registry machinery that lets a run
look a model up by name. Plugins register themselves with the :func:`register`
decorator; importing :mod:`fgas_spk.models` imports the plugin modules so those
decorators actually run (see that package's ``__init__``).

The architecture is deliberately model-agnostic: a plugin is any object with the
two methods below. New models are added against this interface; nothing here
knows about any specific model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # import only for type hints; avoids pulling the loader at runtime
    from fgas_spk.loader import TrainingData


@runtime_checkable
class ProfileToSpk(Protocol):
    """Structural interface for a model mapping f_gas(R) to SP(k).

    A plugin is any object providing these two methods; it need not subclass
    anything. The input modalities follow :class:`~fgas_spk.loader.TrainingData`:
    ``X`` is the gas-fraction profile, ``X_cond`` the observable-derived
    conditioning scalars, and ``X_params`` the CAMELS parameters. A model is
    free to use or ignore ``X_cond`` / ``X_params``.

    **Construction contract.** The runner builds a model as
    ``Model(seed=run_config.seed, **run_config.model_params)`` (see
    :func:`fgas_spk.train.run_training`). So beyond the two methods below, a
    conforming plugin's ``__init__`` must accept a ``seed`` keyword (the
    reproducibility seed) plus its own hyperparameters as keywords, and it should
    tolerate unexpected keys (e.g. via ``**_``) so a stray ``model_params`` entry
    never breaks construction. All model hyperparameters live in ``model_params``;
    the runner passes nothing else to the constructor.

    **Optional ``history`` attribute.** A model that trains iteratively may
    expose ``history`` -- a list of per-epoch metric mappings (e.g.
    ``{"epoch": i, "train_mse": ...}``). The runner records it verbatim into the
    run's ``metrics.jsonl`` when present; a model without it (or with an empty
    one) records no per-epoch trace. It is optional and deliberately not a method
    on this protocol, so non-iterative models need not provide it.
    """

    def fit(self, training_data: "TrainingData") -> None:
        """Fit the model on a :class:`~fgas_spk.loader.TrainingData` bundle.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on
                (``X``, ``X_cond``, ``X_params``, ``y``, ...).
        """
        ...

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict SP(k) for profiles ``X`` (with optional conditioning).

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars, shape
                (n_examples, n_cond), or None. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params), or None. Defaults to None.

        Returns:
            np.ndarray: Predicted SP(k), shape (n_examples, n_k) for a curve
                target or (n_examples,) for a single-k target.
        """
        ...


# Registry of model name -> class, populated by the @register decorator at
# import time. It is empty until the plugin modules are imported; see
# fgas_spk.models.__init__ for why those imports are explicit.
REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Return a class decorator that registers the class under ``name``.

    Args:
        name (str): The registry key (e.g. the value of ``RunConfig.model``).

    Returns:
        Callable[[type], type]: A decorator that inserts the class into
            :data:`REGISTRY` and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered to a different class.
    """

    def decorator(cls: type) -> type:
        existing = REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model name {name!r} is already registered to "
                f"{existing.__name__}; choose a different name."
            )
        REGISTRY[name] = cls
        return cls

    return decorator
