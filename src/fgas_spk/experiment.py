"""Trainer-side configuration for f_gas(R) to SP(k) model runs.

This module defines :class:`RunConfig` -- the model-and-training half of a run's
configuration -- mirroring the :class:`~fgas_spk.loader.DataConfig` style
(dataclass + ``from_yaml`` / ``to_yaml`` round-trip + ``__post_init__``
validation).

The two configs are kept strictly separate. :class:`RunConfig` owns **only**
model and training concerns; it carries **no** data-selection fields -- those
live in :class:`~fgas_spk.loader.DataConfig` (the data YAML the loader already
consumes), whose ``project_root`` is the read root. The only path-like field
here is :attr:`RunConfig.write_root`, the scratch-root override for run outputs.

There is no cross-suite logic anywhere: the data is a single CAMELS IllustrisTNG
suite, so the only split is a grouped split by ``sim_index`` within it (whole
simulations to one side), never by ``(sim, nd)`` row -- grouping avoids leaking a
simulation's shared SP(k) target across folds. See :class:`SplitSpec`.

:func:`load_configs` reads the data YAML and the run YAML **independently** and
does not merge them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from fgas_spk.loader import DataConfig


@dataclass
class SplitSpec:
    """Grouped-by-simulation train/val/test split fractions and seed.

    The split is always grouped by ``sim_index`` -- whole simulations go to one
    side -- never by ``(sim, nd)`` row, so a simulation's shared SP(k) target
    cannot leak across folds. Set ``val_frac = 0`` for a two-way train/test
    split; downstream code then simply has no validation fold.

    Attributes:
        train_frac (float): Fraction of simulations used for training. Must be
            > 0.
        val_frac (float): Fraction used for validation; ``0`` means no validation
            fold (i.e. a two-way train/test split).
        test_frac (float): Fraction used for testing.
        seed (int): Seed for the grouped shuffle, so the split is reproducible.
    """

    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        fracs = {
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "test_frac": self.test_frac,
        }
        out_of_range = {k: v for k, v in fracs.items() if not 0.0 <= v <= 1.0}
        if out_of_range:
            raise ValueError(
                f"split fractions must lie in [0, 1]; out of range: {out_of_range}."
            )
        total = self.train_frac + self.val_frac + self.test_frac
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(
                f"split fractions must sum to 1.0; got {total} "
                f"(train={self.train_frac}, val={self.val_frac}, "
                f"test={self.test_frac})."
            )
        if self.train_frac <= 0.0:
            raise ValueError(
                f"train_frac must be > 0 (a split needs training data); "
                f"got {self.train_frac}."
            )


@dataclass
class EvaluationSpec:
    """Held-out evaluation-diagnostic settings (no effect on training).

    These knobs configure the runner's post-fit diagnostics only; they do not
    change what or how a model trains. Kept on :class:`RunConfig` so the choice
    is recorded verbatim in ``config_run.yaml`` alongside the run it describes.

    Attributes:
        suppressed_thresholds (list[float]): TRUE SP(k) thresholds ``t`` for the
            suppressed-regime pooled RMSE -- the held-out RMSE restricted to the
            k bins where the true suppression falls below ``t``. Defaults to
            ``[0.95, 0.9, 0.8]`` (progressively deeper suppression).
    """

    suppressed_thresholds: list[float] = field(
        default_factory=lambda: [0.95, 0.9, 0.8]
    )

    def __post_init__(self) -> None:
        # A None (e.g. an explicit ``suppressed_thresholds: null`` in YAML) means
        # "use the default". Coerce entries to float so downstream comparisons and
        # the ``str(t)`` record keys are stable regardless of how the YAML wrote
        # them (0.9 vs "0.9").
        if self.suppressed_thresholds is None:
            self.suppressed_thresholds = [0.95, 0.9, 0.8]
        try:
            self.suppressed_thresholds = [
                float(t) for t in self.suppressed_thresholds
            ]
        except TypeError as exc:
            raise TypeError(
                "suppressed_thresholds must be a sequence of numbers, got "
                f"{type(self.suppressed_thresholds).__name__}."
            ) from exc


@dataclass
class RunConfig:
    """Model and training recipe for a single run.

    Owns only model and training concerns -- no data selection (that is
    :class:`~fgas_spk.loader.DataConfig`). The ``model`` name is validated
    against the model registry later, at run time, not here.

    All model hyperparameters -- including the iterative-training knobs
    (``epochs``, ``batch_size``, learning rate, optimizer choice, ...) -- live in
    :attr:`model_params`, which the runner passes straight through as
    ``REGISTRY[model](seed=seed, **model_params)``. The config carries **no**
    top-level training fields: each model owns and documents its own
    hyperparameters (e.g. the MLP reads ``epochs`` / ``lr`` / ``batch_size`` /
    ``weight_decay`` from ``model_params``, the PCA reference reads
    ``n_components``), so the recorded ``config_run.yaml`` reflects exactly what
    trained with no unused defaults.

    Attributes:
        model (str): Registry name of the model to train. Not validated here.
        model_params (dict): Free-form hyperparameters passed verbatim to the
            model constructor. This is where epochs / learning rate / batch size /
            etc. go for models that train iteratively.
        seed (int): Reproducibility seed for model construction (passed to the
            model as ``seed``). The grouped data split has its own seed,
            ``split.seed``.
        split (SplitSpec): Grouped-by-``sim_index`` train/val/test split. A plain
            mapping is coerced to :class:`SplitSpec` on construction.
        evaluation (EvaluationSpec): Held-out evaluation-diagnostic settings (does
            not affect training). A plain mapping is coerced to
            :class:`EvaluationSpec` on construction; a ``None`` uses the defaults.
        write_root (str | None): Scratch-root override for run outputs. ``None``
            defers to :func:`fgas_spk.paths.resolve_scratch_root`. This is the
            only path-like field on the config.
    """

    model: str
    model_params: dict = field(default_factory=dict)
    seed: int = 0
    split: SplitSpec = field(default_factory=SplitSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    write_root: str | None = None

    def __post_init__(self) -> None:
        # YAML round-trips the nested split as a plain mapping; restore the
        # SplitSpec so downstream code can rely on the declared type (mirrors
        # DataConfig's tuple restoration). A None is treated as "use defaults".
        if self.model_params is None:
            self.model_params = {}
        if not isinstance(self.model_params, dict):
            raise TypeError(
                f"model_params must be a mapping, got "
                f"{type(self.model_params).__name__}."
            )
        if self.split is None:
            self.split = SplitSpec()
        elif isinstance(self.split, dict):
            self.split = SplitSpec(**self.split)
        elif not isinstance(self.split, SplitSpec):
            raise TypeError(
                f"split must be a mapping or SplitSpec, got "
                f"{type(self.split).__name__}."
            )
        # Same coercion for the evaluation-diagnostic settings.
        if self.evaluation is None:
            self.evaluation = EvaluationSpec()
        elif isinstance(self.evaluation, dict):
            self.evaluation = EvaluationSpec(**self.evaluation)
        elif not isinstance(self.evaluation, EvaluationSpec):
            raise TypeError(
                f"evaluation must be a mapping or EvaluationSpec, got "
                f"{type(self.evaluation).__name__}."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Construct a config from a YAML file.

        Args:
            path (str | Path): Path to a YAML mapping of config fields.

        Returns:
            RunConfig: The parsed configuration.

        Raises:
            TypeError: If the YAML top level is not a mapping.
        """
        data = yaml.safe_load(Path(path).read_text())
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(
                f"Config YAML must be a mapping of fields, got {type(data).__name__}."
            )
        return cls(**data)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the config to a YAML file.

        Args:
            path (str | Path): Destination path.

        Returns:
            Path: The written path.
        """
        out = Path(path)
        out.write_text(yaml.safe_dump(asdict(self), sort_keys=False))
        return out


def load_configs(
    data_yaml_path: str | Path, run_yaml_path: str | Path
) -> tuple[DataConfig, RunConfig]:
    """Load the data config and the run config independently (no merge).

    The two YAMLs are parsed separately and returned as distinct objects; nothing
    is copied between them. The read root stays inside the
    :class:`~fgas_spk.loader.DataConfig` (its ``project_root`` field); the write
    root stays inside :class:`RunConfig` (its ``write_root`` field).

    Args:
        data_yaml_path (str | Path): Path to the data-selection YAML
            (:class:`~fgas_spk.loader.DataConfig`).
        run_yaml_path (str | Path): Path to the run YAML (:class:`RunConfig`).

    Returns:
        tuple[DataConfig, RunConfig]: The two configs, in that order.
    """
    data_config = DataConfig.from_yaml(data_yaml_path)
    run_config = RunConfig.from_yaml(run_yaml_path)
    return data_config, run_config
