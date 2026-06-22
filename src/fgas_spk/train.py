"""Training runner: a (DataConfig, RunConfig) pair through to a recorded run.

End-to-end driver that ties the package together:

1. take a resolved ``(DataConfig, RunConfig)`` pair (plus optional root overrides);
2. **read-root fill** -- if the data config is in store-field mode with no
   ``project_root``, fill it from :func:`fgas_spk.paths.resolve_data_root` so the
   loader receives a fully-specified config (the loader stays untouched);
3. resolve the **write root** via :func:`fgas_spk.paths.resolve_scratch_root`;
4. load the model-ready arrays via :func:`fgas_spk.loader.load_training_data`;
5. split **grouped by ``sim_index``** per ``RunConfig.split`` -- whole
   simulations go to one fold, never split by ``(sim, nd)`` row (that would leak
   a simulation's shared SP(k) target across folds);
6. instantiate the model from :data:`fgas_spk.models.REGISTRY`, fit on the train
   split, and evaluate on the held-out split;
7. write the git-tracked run record, save a checkpoint to scratch under
   ``<scratch_root>/models/<run_id>/``, and write a diagnostic figure via
   :func:`fgas_spk.paths.figure_dir` with the run id in the filename.

There is no cross-suite logic: the data is a single CAMELS IllustrisTNG suite, so
the only partition is the grouped-by-``sim_index`` split.

Physics note (CLAUDE.md rule 3): the suppression-target definition, the f_gas
observable, and the mass conventions are upstream choices, recorded for
provenance in the run record's ``env.json`` -- this runner only partitions and
fits, and resolves no physics choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from fgas_spk.loader import TrainingData, load_training_data
from fgas_spk.models import REGISTRY
from fgas_spk.paths import (
    figure_dir,
    fill_data_root,
    resolve_data_root,
    resolve_scratch_root,
)
from fgas_spk.run_record import write_run_record

if TYPE_CHECKING:  # type hints only
    from fgas_spk.experiment import RunConfig, SplitSpec
    from fgas_spk.loader import DataConfig


@dataclass
class RunResult:
    """Outcome of a training run.

    Attributes:
        run_id (str): The run identifier.
        run_dir (Path): The git-tracked run-record directory.
        summary (dict): Headline metrics (split sizes, per-split RMSE, ``rmse``).
        checkpoint_path (Path): The saved model checkpoint on scratch.
        split_masks (dict[str, np.ndarray]): Boolean row masks per split
            (``train`` / ``val`` / ``test``), grouped by ``sim_index``.
    """

    run_id: str
    run_dir: Path
    summary: dict
    checkpoint_path: Path
    split_masks: dict = field(default_factory=dict)


def pick_device() -> str:
    """Return the best available torch device as a string (cuda / mps / cpu).

    Mirrors the README ``pick_device`` pattern. torch is imported lazily and
    treated as optional: if it is not installed, the device is reported as
    ``"cpu"``.

    Returns:
        str: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is in the env
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def grouped_split(sim_index: np.ndarray, split: "SplitSpec") -> dict[str, np.ndarray]:
    """Return train/val/test row masks grouped by simulation.

    Every row of a given simulation goes to exactly one fold: the unique
    ``sim_index`` values are shuffled (seeded) and partitioned by the split
    fractions, then rows are masked by simulation membership. Splitting by
    ``(sim, nd)`` row would leak a simulation's shared SP(k) target across folds;
    grouping by simulation avoids that. With ``val_frac == 0`` the validation
    fold is empty by construction.

    Args:
        sim_index (np.ndarray): Originating simulation id per row, shape
            (n_rows,).
        split (SplitSpec): Train/val/test fractions and the split seed.

    Returns:
        dict[str, np.ndarray]: Boolean masks ``{"train": ..., "val": ...,
            "test": ...}`` over the rows; the three are disjoint and cover all
            rows.
    """
    sim_index = np.asarray(sim_index)
    unique_sims = np.unique(sim_index)
    shuffled = np.random.default_rng(split.seed).permutation(unique_sims)

    n = len(shuffled)
    # Cumulative boundaries from the fractions: this honours the fractions
    # exactly, keeps the three folds disjoint and covering, and -- crucially --
    # makes val_frac == 0 yield an empty val fold by construction (the previous
    # round-each-independently scheme could leak a simulation into val). Any
    # rounding slack falls into the test fold.
    c_train = int(round(split.train_frac * n))
    c_val = int(round((split.train_frac + split.val_frac) * n))
    train_sims = shuffled[:c_train]
    val_sims = shuffled[c_train:c_val]
    test_sims = shuffled[c_val:]

    return {
        "train": np.isin(sim_index, train_sims),
        "val": np.isin(sim_index, val_sims),
        "test": np.isin(sim_index, test_sims),
    }


def _subset(td: TrainingData, mask: np.ndarray) -> TrainingData:
    """Return a TrainingData holding only the rows selected by ``mask``."""
    mask = np.asarray(mask)
    return TrainingData(
        X=td.X[mask],
        X_cond=td.X_cond[mask] if td.X_cond is not None else None,
        X_params=td.X_params[mask] if td.X_params is not None else None,
        y=td.y[mask],
        nd=td.nd[mask],
        sim_index=td.sim_index[mask],
        k=td.k,
        radii_mpch=td.radii_mpch,
        source_path=td.source_path,
        meta=td.meta,
        config=td.config,
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error over all elements (handles 1-D and 2-D targets)."""
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(diff ** 2)))


def _write_pred_vs_true_figure(
    model, td: TrainingData, masks: dict, run_id: str, summary: dict
) -> Path | None:
    """Write a held-out predicted-vs-true SP(k) scatter; run id in the filename.

    matplotlib is imported lazily (Agg backend) so the runner imports without it.
    Returns None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    held = summary["held_out_split"]
    sub = _subset(td, masks[held])
    y_true = np.asarray(sub.y).ravel()
    y_pred = np.asarray(model.predict(sub.X, sub.X_cond, sub.X_params)).ravel()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.4)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("true SP(k)")
    ax.set_ylabel("predicted SP(k)")
    ax.set_title(f"{run_id}\n{held} RMSE = {summary['rmse']:.4g}")

    out = figure_dir() / f"{run_id}__pred_vs_true.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def run_training(
    data_config: "DataConfig",
    run_config: "RunConfig",
    *,
    data_root_override: str | Path | None = None,
    scratch_root_override: str | Path | None = None,
    experiments_root: str | Path | None = None,
    write_figures: bool = True,
    device: str | None = None,
) -> RunResult:
    """Run training end to end and write the run record.

    Args:
        data_config (DataConfig): Resolved data-selection config.
        run_config (RunConfig): Resolved run config (model, split, seed, ...).
        data_root_override (str | Path | None): Override for the read root; used
            only to fill ``project_root`` when the data config leaves it null in
            store-field mode. Defaults to None.
        scratch_root_override (str | Path | None): Override for the write root.
            Takes precedence over ``RunConfig.write_root``. Defaults to None.
        experiments_root (str | Path | None): Root of the git-tracked record tree.
            Defaults to ``<repo_root>/experiments`` (see
            :func:`fgas_spk.run_record.write_run_record`).
        write_figures (bool): Write the diagnostic figure. Defaults to True.
        device (str | None): Device string for the record; defaults to
            :func:`pick_device`.

    Returns:
        RunResult: The run id, record directory, summary metrics, checkpoint
            path, and split masks.

    Raises:
        KeyError: If ``RunConfig.model`` is not in the registry.
    """
    # 2. READ-ROOT FILL via the shared entry-point helper (store-field mode only;
    # a path-mode config is returned untouched).
    data_config = fill_data_root(data_config, explicit=data_root_override)

    # 3. WRITE ROOT.
    scratch_root = resolve_scratch_root(
        explicit=scratch_root_override or run_config.write_root
    )

    # 4. LOAD DATA (loader resolves the .npz and records it on td.source_path).
    td = load_training_data(data_config)

    # 5. GROUPED SPLIT by sim_index.
    masks = grouped_split(td.sim_index, run_config.split)
    train_td = _subset(td, masks["train"])

    # 6. MODEL: instantiate from the registry, fit, evaluate on the held-out fold.
    if run_config.model not in REGISTRY:
        raise KeyError(
            f"Unknown model {run_config.model!r}; registered: {sorted(REGISTRY)}."
        )
    model = REGISTRY[run_config.model](seed=run_config.seed, **run_config.model_params)
    model.fit(train_td)

    summary: dict = {
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "train_rmse": _rmse(
            train_td.y,
            model.predict(train_td.X, train_td.X_cond, train_td.X_params),
        ),
    }
    for name in ("val", "test"):
        if masks[name].any():
            sub = _subset(td, masks[name])
            summary[f"{name}_rmse"] = _rmse(
                sub.y, model.predict(sub.X, sub.X_cond, sub.X_params)
            )
    # Headline held-out metric: prefer test, then val, then train.
    held = "test" if "test_rmse" in summary else "val" if "val_rmse" in summary else "train"
    summary["held_out_split"] = held
    summary["rmse"] = summary[f"{held}_rmse"]

    device = device if device is not None else pick_device()

    # 7. RECORD (assigns run_id), then checkpoint + figure under that run_id.
    effective_data_root = (
        data_config.project_root
        if data_config.project_root is not None
        else str(resolve_data_root(explicit=data_root_override))
    )
    record = write_run_record(
        data_config,
        run_config,
        dataset_path=td.source_path,
        data_root=effective_data_root,
        scratch_root=scratch_root,
        summary=summary,
        metrics=None,  # PCA reference is non-iterative: no per-epoch trace
        device=device,
        experiments_root=experiments_root,
    )

    # Checkpoint -> <scratch_root>/models/<run_id>/ (the big artifact on scratch).
    ckpt_dir = Path(scratch_root) / "models" / record.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "model.joblib"
    joblib.dump(model, checkpoint_path)

    if write_figures:
        _write_pred_vs_true_figure(model, td, masks, record.run_id, summary)

    return RunResult(
        run_id=record.run_id,
        run_dir=record.run_dir,
        summary=summary,
        checkpoint_path=checkpoint_path,
        split_masks=masks,
    )
