"""API-reference model: PCA-compress the profile, then linear-map to SP(k).

This is the **reference plugin** for the
:class:`~fgas_spk.models.base.ProfileToSpk` interface -- a worked example to copy
when building real models, not a tuned scientific baseline. It does the simplest
end-to-end multimodal thing:

1. compress the f_gas(R) profile ``X`` to a few principal components (PCA);
2. concatenate the conditioning scalars ``X_cond`` and the CAMELS parameters
   ``X_params`` alongside those components;
3. map the result to the SP(k) target ``y`` with an ordinary linear regressor.

Step 2 is the point: it shows the multimodal wiring (profile + conditioning +
parameters) you would reuse in a real model. Each side input is optional and
folded in the same way: whichever of ``X_cond`` / ``X_params`` the loader
provides is placed beside the components, in that fixed order, and a model fit
with a modality requires it again at predict time. Both patterns are documented
here -- to build a **profile-only** model instead, ignore the side inputs; the
one-line change at the feature-assembly site is::

    # multimodal (this reference): components + side inputs side by side
    features = self._design_matrix(components, X_cond, X_params)
    # profile-only: ignore side inputs, use the components alone
    features = components

In a run, the model is constructed from the :class:`~fgas_spk.experiment.RunConfig`
as ``PcaLinear(seed=run_config.seed, **run_config.model_params)`` and looked up by
name via ``fgas_spk.models.REGISTRY["pca_linear"]``.

Dependencies:
    scikit-learn (:class:`sklearn.decomposition.PCA`,
    :class:`sklearn.linear_model.LinearRegression`) -- already in the project
    environment (it is one of the analysis/ML pins, not part of the core
    numpy+pyyaml import surface). Determinism comes from ``seed``, threaded into
    PCA's ``random_state``; the linear regressor has no randomness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from fgas_spk.models.base import register

if TYPE_CHECKING:  # import only for type hints
    from fgas_spk.loader import TrainingData


@register("pca_linear")
class PcaLinear:
    """Reference model: PCA on the profile, linear map (with side inputs) to SP(k).

    The PCA components are mapped to SP(k) by an ordinary linear regressor; any
    conditioning scalars (``X_cond``) and CAMELS parameters (``X_params``) the
    loader provides are concatenated beside the components, in that fixed order,
    before the map.

    Args:
        n_components (int): Number of principal components of ``X`` to keep.
            Defaults to 8. Capped at ``min(n_samples, n_radii)`` at fit time.
        seed (int): Reproducibility seed; from ``RunConfig.seed`` in a real run.
            Threaded into PCA's ``random_state``. Defaults to 0.

    Attributes:
        n_components (int): Requested number of PCA components.
        seed (int): Reproducibility seed.
    """

    def __init__(self, n_components: int = 8, seed: int = 0) -> None:
        self.n_components = n_components
        self.seed = seed
        self._pca: PCA | None = None
        self._linear: LinearRegression | None = None
        self._uses_cond: bool = False
        self._uses_params: bool = False

    def fit(self, training_data: "TrainingData") -> None:
        """Fit PCA on the profiles and a linear map to the SP(k) target.

        Consumes ``training_data.X`` (profiles), ``training_data.X_cond``
        (conditioning scalars, if present), ``training_data.X_params`` (CAMELS
        parameters, if present), and ``training_data.y`` (target). ``X_cond`` and
        ``X_params`` are each folded into the linear map's design matrix when the
        loader supplies them; whichever modalities are present here are then
        required again at predict time. Arrays are taken from the loader as given
        -- no reshaping or re-binning here.

        Args:
            training_data (TrainingData): The model-ready arrays to fit on.
        """
        X = np.asarray(training_data.X)
        y = np.asarray(training_data.y)
        X_cond = training_data.X_cond
        X_params = training_data.X_params

        n_samples, n_radii = X.shape
        n_components = min(self.n_components, n_samples, n_radii)
        self._pca = PCA(n_components=n_components, random_state=self.seed)
        components = self._pca.fit_transform(X)        # (n_samples, n_components)

        self._uses_cond = X_cond is not None
        self._uses_params = X_params is not None
        features = self._design_matrix(components, X_cond, X_params)
        self._linear = LinearRegression().fit(features, y)

    def predict(
        self,
        X: np.ndarray,
        X_cond: np.ndarray | None = None,
        X_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict SP(k) for profiles ``X`` (using the same modalities as fit).

        Args:
            X (np.ndarray): Gas-fraction profiles, shape (n_examples, n_radii).
            X_cond (np.ndarray | None): Conditioning scalars, shape
                (n_examples, n_cond). Required iff the model was fit with
                conditioning. Defaults to None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params). Required iff the model was fit with
                parameters. Defaults to None.

        Returns:
            np.ndarray: Predicted SP(k), shape (n_examples, n_k) for a curve
                target or (n_examples,) for a single-k target.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the presence of ``X_cond`` or ``X_params`` does not
                match how the model was fit (the linear map's feature layout
                would not line up).
        """
        if self._pca is None or self._linear is None:
            raise RuntimeError("PcaLinear.predict called before fit.")
        if self._uses_cond and X_cond is None:
            raise ValueError(
                "This model was fit with conditioning (X_cond); predict requires "
                "X_cond as well."
            )
        if not self._uses_cond and X_cond is not None:
            raise ValueError(
                "This model was fit without conditioning but predict was given "
                "X_cond. Pass X_cond=None to match the fitted feature layout."
            )
        if self._uses_params and X_params is None:
            raise ValueError(
                "This model was fit with CAMELS parameters (X_params); predict "
                "requires X_params as well."
            )
        if not self._uses_params and X_params is not None:
            raise ValueError(
                "This model was fit without CAMELS parameters but predict was "
                "given X_params. Pass X_params=None to match the fitted feature "
                "layout."
            )
        components = self._pca.transform(np.asarray(X))
        features = self._design_matrix(components, X_cond, X_params)
        return self._linear.predict(features)

    @staticmethod
    def _design_matrix(
        components: np.ndarray,
        X_cond: np.ndarray | None,
        X_params: np.ndarray | None,
    ) -> np.ndarray:
        """Assemble the linear map's input matrix.

        The multimodal wiring: the profile-derived PCA ``components``, then the
        observable conditioning scalars ``X_cond``, then the CAMELS parameters
        ``X_params``, placed side by side in that fixed order. Whichever side
        inputs are None are skipped; with both None the components are returned
        unchanged -- the same code the profile-only one-line change (see the
        module docstring) produces. Fit and predict must pass the same modalities
        so the column layout lines up.

        Args:
            components (np.ndarray): PCA components, shape (n_examples, n_comp).
            X_cond (np.ndarray | None): Conditioning scalars, shape
                (n_examples, n_cond), or None.
            X_params (np.ndarray | None): CAMELS parameters, shape
                (n_examples, n_params), or None.

        Returns:
            np.ndarray: The feature matrix fed to the linear regressor.
        """
        blocks = [components]
        if X_cond is not None:
            blocks.append(np.asarray(X_cond))
        if X_params is not None:
            blocks.append(np.asarray(X_params))
        if len(blocks) == 1:
            return components
        return np.concatenate(blocks, axis=1)
