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
   ``<scratch_root>/models/<run_id>/``, and write the diagnostic figures via
   :func:`fgas_spk.paths.figure_dir` with the run id in the filename.

The run record's ``summary.json`` also carries additive held-out evaluation
diagnostics (per-curve and suppressed-regime RMSE for every model; empirical
coverage and latent-code norms for models that expose ``predict_samples`` /
``latents``); the greppable ``runs.jsonl`` ledger keeps only the headline
scalars. These diagnostics are computed post-fit and resolve no physics choice.

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
        summary (dict): Headline metrics (split sizes, per-split RMSE, ``rmse``)
            plus additive held-out evaluation diagnostics written to
            ``summary.json`` (``per_curve_rmse`` and ``suppressed_rmse`` for every
            model; ``coverage`` and ``latent_norms`` for models exposing
            ``predict_samples`` / ``latents``).
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
        param_names=td.param_names,
        meta=td.meta,
        config=td.config,
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error over all elements (handles 1-D and 2-D targets)."""
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sqrt(np.mean(diff ** 2)))


def _figure_label(run_id: str, model_name: str) -> str:
    """Human-readable run label for figure filenames and titles.

    Replaces the trailing git-sha segment of ``run_id`` -- useful for machine
    bookkeeping but not for a human reading a plot -- with the model name,
    yielding ``<timestamp>__<config-hash>__<model>``. The full ``run_id`` (git
    sha included) remains the key for the run record and checkpoint.

    Args:
        run_id (str): The run identifier, ``<ts>__<config-hash8>__<sha7>``.
        model_name (str): The registered model name (``RunConfig.model``).

    Returns:
        str: The label with the git-sha segment replaced by ``model_name``.
    """
    base = run_id.rsplit("__", 1)[0]  # drop the trailing git-sha7 segment
    return f"{base}__{model_name}"


def _write_pred_vs_true_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict, extension: str = "png"
) -> Path | None:
    """Write a held-out predicted-vs-true SP(k) scatter; ``label`` in the filename.

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
    ax.set_title(f"{label}\n{held} RMSE = {summary['rmse']:.4g}")

    out = figure_dir() / f"{label}__pred_vs_true.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_rmse_vs_k_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict, extension: str = "png"
) -> Path | None:
    """Write the k-dependent RMSE of SP(k) for every non-empty split.

    RMSE is computed *per k bin* (across rows of a split) and plotted against k
    on a log-k axis, one curve per non-empty split (train / val / test); ``val``
    is omitted when ``val_frac == 0``. This complements the held-out scatter from
    :func:`_write_pred_vs_true_figure` by showing where, in k, the model errs.
    ``label`` is used in the filename and title.

    Skipped for ``single_k`` targets: ``y`` is then 1-D (a single k bin), which is
    not a curve. matplotlib is imported lazily (Agg backend). Returns the written
    path, or None if matplotlib is unavailable or the figure was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    # single_k yields a 1-D target (one k bin); a per-k RMSE curve is undefined.
    if np.asarray(td.y).ndim < 2:
        return None

    k = np.asarray(td.k)
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for name in ("train", "val", "test"):
        mask = masks.get(name)
        if mask is None or not np.asarray(mask).any():
            continue
        sub = _subset(td, mask)
        y_true = np.asarray(sub.y)
        y_pred = np.asarray(
            model.predict(sub.X, sub.X_cond, sub.X_params)
        ).reshape(y_true.shape)
        rmse_k = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))  # (n_k,)
        ax.plot(k, rmse_k, marker=".", label=f"{name} (n={int(np.asarray(mask).sum())})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("RMSE of SP(k)")
    ax.set_title(f"{label}\nper-k RMSE")
    ax.legend()

    out = figure_dir() / f"{label}__rmse_vs_k.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# Nominal central-interval levels for the predictive-coverage diagnostic.
_COVERAGE_LEVELS = (0.68, 0.90, 0.95)
# Posterior samples drawn per example for the probabilistic diagnostics.
_N_PREDICTIVE_SAMPLES = 200


def _held_out_true_pred(
    model, td: TrainingData, masks: dict, summary: dict
) -> tuple[TrainingData, np.ndarray, np.ndarray]:
    """Return the held-out ``(subset, y_true, y_pred)`` for the evaluation fold.

    The held-out fold is ``summary["held_out_split"]`` (test, else val, else
    train). ``y_pred`` is the model's deterministic point prediction on that
    fold, reshaped to ``y_true``'s shape so a single squared-error array serves
    every held-out metric consistently.

    Args:
        model: The fitted model (its :meth:`predict` is called).
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks from :func:`grouped_split`.
        summary (dict): The run summary, read for ``held_out_split``.

    Returns:
        tuple[TrainingData, np.ndarray, np.ndarray]: The held-out subset, the
            true target, and the point prediction (shapes matched).
    """
    sub = _subset(td, masks[summary["held_out_split"]])
    y_true = np.asarray(sub.y, dtype=float)
    y_pred = np.asarray(
        model.predict(sub.X, sub.X_cond, sub.X_params), dtype=float
    ).reshape(y_true.shape)
    return sub, y_true, y_pred


def _per_curve_rmse_metrics(
    model, td: TrainingData, masks: dict, summary: dict, *, top_n: int = 10
) -> dict | None:
    """Per-curve RMSE distribution on the held-out split and the worst curves.

    Each held-out ``(sim, nd)`` row is one predicted SP(k) curve; ``rmse_i`` is
    the root-mean-square error of that curve across the k bins. Returns the
    distribution's ``mean`` / ``median`` / ``p90`` / ``max`` plus the ``top_n``
    worst curves as ``worst_sim_indices`` (their originating simulation ids, from
    ``td.sim_index``, so they can be cross-referenced against the CAMELS
    parameters) and the aligned ``worst_sim_rmse``, both in descending RMSE.

    Returns None for a 1-D (``single_k``) target -- a per-k RMSE per curve is
    undefined -- mirroring :func:`_write_rmse_vs_k_figure`.

    Args:
        model: The fitted model.
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.
        summary (dict): The run summary (read for ``held_out_split``).
        top_n (int): Number of worst curves to list (capped at the fold size).
            Defaults to 10.

    Returns:
        dict | None: The per-curve RMSE record, or None for a 1-D target.
    """
    sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2:
        return None
    rmse_per_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))  # (n_curves,)
    sim_ids = np.asarray(sub.sim_index)
    order = np.argsort(rmse_per_curve)[::-1]  # descending rmse_i
    worst = order[: min(top_n, rmse_per_curve.shape[0])]
    return {
        "mean": float(np.mean(rmse_per_curve)),
        "median": float(np.median(rmse_per_curve)),
        "p90": float(np.percentile(rmse_per_curve, 90)),
        "max": float(np.max(rmse_per_curve)),
        "worst_sim_indices": [int(sim_ids[i]) for i in worst],
        "worst_sim_rmse": [float(rmse_per_curve[i]) for i in worst],
    }


def _suppressed_rmse_metrics(
    model, td: TrainingData, masks: dict, summary: dict, thresholds
) -> dict:
    """Pooled held-out RMSE over bins where the TRUE SP(k) < t, per threshold.

    For each ``t`` the mask ``y_true < t`` selects the suppressed bins across all
    held-out rows and k bins; the RMSE pools their squared errors. An empty mask
    records ``{"rmse": None, "n_bins": 0}`` rather than raising. Works for both
    curve and ``single_k`` targets.

    Args:
        model: The fitted model.
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.
        summary (dict): The run summary (read for ``held_out_split``).
        thresholds (Iterable[float]): The SP(k) thresholds ``t``.

    Returns:
        dict: ``{str(t): {"rmse": float | None, "n_bins": int}}``.
    """
    _sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    se = (y_pred - y_true) ** 2
    out: dict = {}
    for t in thresholds:
        mask = y_true < t
        n_bins = int(np.count_nonzero(mask))
        if n_bins == 0:
            out[str(t)] = {"rmse": None, "n_bins": 0}
        else:
            out[str(t)] = {
                "rmse": float(np.sqrt(np.mean(se[mask]))),
                "n_bins": n_bins,
            }
    return out


def _coverage_metrics(
    model,
    td: TrainingData,
    masks: dict,
    summary: dict,
    *,
    seed: int,
    n_samples: int = _N_PREDICTIVE_SAMPLES,
    levels=_COVERAGE_LEVELS,
) -> dict:
    """Empirical coverage of central predictive intervals on the held-out fold.

    Draws ``n_samples`` posterior samples per held-out example (seeded), forms
    the central interval at each nominal ``level`` per ``(row, k bin)`` from the
    sample quantiles, and records the fraction of true values inside. Requires
    ``model.predict_samples`` (guarded by the caller).

    Args:
        model: The fitted model (its :meth:`predict_samples` is called).
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.
        summary (dict): The run summary (read for ``held_out_split``).
        seed (int): Sampling seed (the run's seed).
        n_samples (int): Posterior samples per example. Defaults to 200.
        levels (Iterable[float]): Nominal central-interval levels.

    Returns:
        dict: ``{str(level): fraction}``, one per nominal level.
    """
    sub = _subset(td, masks[summary["held_out_split"]])
    y_true = np.asarray(sub.y, dtype=float)
    samples = np.asarray(
        model.predict_samples(
            sub.X, sub.X_cond, sub.X_params, n_samples=n_samples, seed=seed
        ),
        dtype=float,
    )  # (n_samples, n_curves[, n_k])
    out: dict = {}
    for level in levels:
        lo = np.quantile(samples, (1.0 - level) / 2.0, axis=0)
        hi = np.quantile(samples, (1.0 + level) / 2.0, axis=0)
        inside = (y_true >= lo) & (y_true <= hi)
        out[str(level)] = float(np.mean(inside))
    return out


def _latent_norm_metrics(model, td: TrainingData, masks: dict) -> dict:
    """Per-split summary of the encoder code norm ``||mu||_2``.

    For every non-empty split (train/val/test) returns ``{"median", "p99",
    "max"}`` of the L2 norm of each example's posterior-mean latent. This
    localises encoder blow-up on out-of-distribution examples. Requires
    ``model.latents`` (guarded by the caller).

    Args:
        model: The fitted model (its :meth:`latents` is called).
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.

    Returns:
        dict: ``{split: {"median", "p99", "max"}}`` over non-empty splits.
    """
    out: dict = {}
    for name in ("train", "val", "test"):
        mask = masks.get(name)
        if mask is None or not np.asarray(mask).any():
            continue
        sub = _subset(td, mask)
        mu = np.asarray(model.latents(sub.X, sub.X_cond, sub.X_params), dtype=float)
        norms = np.linalg.norm(mu, axis=1)
        out[name] = {
            "median": float(np.median(norms)),
            "p99": float(np.percentile(norms, 99)),
            "max": float(np.max(norms)),
        }
    return out


def _write_per_curve_rmse_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict, extension: str = "png"
) -> Path | None:
    """Write a histogram of held-out per-curve RMSE with median / p90 markers.

    Each held-out ``(sim, nd)`` curve contributes one RMSE (across k bins);
    vertical lines mark the median and the 90th percentile. Skipped for a 1-D
    ``single_k`` target (no per-k curve), mirroring :func:`_write_rmse_vs_k_figure`.
    matplotlib is imported lazily (Agg backend). Returns the written path, or None
    if matplotlib is unavailable or the figure was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    _sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2:
        return None
    rmse_per_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
    median = float(np.median(rmse_per_curve))
    p90 = float(np.percentile(rmse_per_curve, 90))

    held = summary["held_out_split"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rmse_per_curve, bins="auto", color="C0", alpha=0.8)
    ax.axvline(median, color="k", linewidth=1, label=f"median = {median:.4g}")
    ax.axvline(p90, color="k", linestyle="--", linewidth=1, label=f"p90 = {p90:.4g}")
    ax.set_xlabel("per-curve RMSE of SP(k)")
    ax.set_ylabel("count")
    ax.set_title(f"{label}\n{held} per-curve RMSE ({rmse_per_curve.shape[0]} curves)")
    ax.legend()

    out = figure_dir() / f"{label}__per_curve_rmse.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_worst_curves_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict,
    extension: str = "png", n_worst: int = 5,
) -> Path | None:
    """Plot true vs predicted SP(k) for the worst held-out curves.

    Selects the ``n_worst`` held-out ``(sim, nd)`` curves with the largest
    per-curve RMSE and plots each true SP(k) (solid) against its prediction
    (dashed) versus k on a log-k axis, one colour per curve. The legend labels
    each by its originating simulation id and RMSE. Skipped for a 1-D
    ``single_k`` target. matplotlib is imported lazily (Agg backend). Returns the
    written path, or None if matplotlib is unavailable or the figure was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2:
        return None
    rmse_per_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
    sim_ids = np.asarray(sub.sim_index)
    k = np.asarray(td.k)
    order = np.argsort(rmse_per_curve)[::-1]
    worst = order[: min(n_worst, rmse_per_curve.shape[0])]
    cmap = plt.get_cmap("tab10")

    held = summary["held_out_split"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for j, i in enumerate(worst):
        colour = cmap(j % 10)
        ax.plot(
            k, y_true[i], color=colour, linestyle="-",
            label=f"sim {int(sim_ids[i])} (rmse={rmse_per_curve[i]:.4g})",
        )
        ax.plot(k, y_pred[i], color=colour, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("SP(k)")
    ax.set_title(f"{label}\n{len(worst)} worst {held} curves (solid=true, dashed=pred)")
    ax.legend()

    out = figure_dir() / f"{label}__worst_curves.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_history_figure(model, label: str, extension: str = "png") -> Path | None:
    """Write the training-history figure from a model's ``history`` trace.

    For a CVAE-style history (keys ``epoch`` / ``recon`` / ``kl`` / ``beta_eff`` /
    ``train_loss``) a two-panel figure: the top panel plots ``recon`` and
    ``train_loss`` versus epoch (log-y), the bottom panel plots ``kl`` versus
    epoch (log-y) with ``beta_eff`` on a twin (linear) axis. For any other history
    every numeric column is plotted versus epoch on a single log-y panel.
    matplotlib is imported lazily (Agg backend). Returns the written path, or None
    if matplotlib is unavailable or the history is empty.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    history = getattr(model, "history", None)
    if not history:
        return None

    epochs = [row.get("epoch", i) for i, row in enumerate(history)]
    cvae_keys = {"epoch", "recon", "kl", "beta_eff", "train_loss"}

    if cvae_keys.issubset(history[0]):
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
        ax_top.plot(epochs, [r["recon"] for r in history], marker=".", label="recon")
        ax_top.plot(
            epochs, [r["train_loss"] for r in history], marker=".", label="train_loss"
        )
        ax_top.set_yscale("log")
        ax_top.set_ylabel("recon / train_loss")
        ax_top.legend()
        ax_top.set_title(f"{label}\ntraining history")

        ax_bot.plot(epochs, [r["kl"] for r in history], marker=".", color="C2", label="kl")
        ax_bot.set_yscale("log")
        ax_bot.set_xlabel("epoch")
        ax_bot.set_ylabel("KL")
        ax_twin = ax_bot.twinx()
        ax_twin.plot(
            epochs, [r["beta_eff"] for r in history], color="C3", linestyle="--",
            label="beta_eff",
        )
        ax_twin.set_ylabel("beta_eff (linear)")
        # Merge the twin-axis handle into the bottom panel's legend.
        lines, labels = ax_bot.get_legend_handles_labels()
        lines_tw, labels_tw = ax_twin.get_legend_handles_labels()
        ax_bot.legend(lines + lines_tw, labels + labels_tw, loc="best")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for key, value in history[0].items():
            if key == "epoch" or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            ax.plot(epochs, [r[key] for r in history], marker=".", label=key)
            plotted = True
        if not plotted:
            plt.close(fig)
            return None
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("value")
        ax.set_title(f"{label}\ntraining history")
        ax.legend()

    out = figure_dir() / f"{label}__history.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_spread_vs_error_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict,
    *, seed: int, extension: str = "png",
) -> Path | None:
    """Scatter the across-sample spread against the point-prediction error.

    Per held-out ``(row, k bin)``: x is the across-sample standard deviation of
    the posterior samples, y is the absolute error of the posterior-mean point
    prediction (:meth:`predict`). A calibrated model tracks ``y ~ x`` on average
    (the plotted ``y = x`` line). This is a diagnostic instrument: the current
    CVAE's spread is known to be miscalibrated, and this figure exists to make
    that visible -- it does not correct anything. Requires ``model.predict_samples``
    (guarded by the caller). matplotlib is imported lazily (Agg backend). Returns
    the written path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    samples = np.asarray(
        model.predict_samples(
            sub.X, sub.X_cond, sub.X_params,
            n_samples=_N_PREDICTIVE_SAMPLES, seed=seed,
        ),
        dtype=float,
    )
    spread = np.std(samples, axis=0).reshape(y_true.shape)
    abs_err = np.abs(y_pred - y_true)

    x = spread.ravel()
    y = abs_err.ravel()
    held = summary["held_out_split"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x, y, s=6, alpha=0.3)
    hi = float(max(x.max(), y.max())) if x.size else 1.0
    ax.plot([0.0, hi], [0.0, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("across-sample std of SP(k)")
    ax.set_ylabel("|posterior-mean error|")
    ax.set_title(f"{label}\n{held} spread vs error")
    ax.legend()

    out = figure_dir() / f"{label}__spread_vs_error.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_coverage_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict,
    *, seed: int, extension: str = "png",
) -> Path | None:
    """Plot empirical vs nominal coverage of central predictive intervals.

    For each nominal level the empirical coverage is the fraction of held-out true
    values inside the central interval formed from the posterior samples (per row
    and k bin). Points on the diagonal indicate calibration; below it,
    over-confident intervals. A diagnostic instrument (see
    :func:`_write_spread_vs_error_figure`); it reveals miscalibration, it does not
    correct it. Requires ``model.predict_samples`` (guarded by the caller).
    matplotlib is imported lazily (Agg backend). Returns the written path, or None
    if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    coverage = _coverage_metrics(model, td, masks, summary, seed=seed)
    levels = [float(key) for key in coverage]  # insertion order = _COVERAGE_LEVELS
    empirical = [coverage[key] for key in coverage]

    held = summary["held_out_split"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", linewidth=1, label="ideal")
    ax.plot(levels, empirical, marker="o", label="empirical")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{label}\n{held} coverage")
    ax.legend()

    out = figure_dir() / f"{label}__coverage.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_latent_norm_figure(
    model, td: TrainingData, masks: dict, label: str, extension: str = "png"
) -> Path | None:
    """Overlay per-split histograms of the encoder code norm ``||mu||_2``.

    One translucent histogram per non-empty split (train/val/test) of the L2 norm
    of each example's posterior-mean latent, with the split name and example count
    in the legend. This localises encoder blow-up on out-of-distribution held-out
    examples. Requires ``model.latents`` (guarded by the caller). matplotlib is
    imported lazily (Agg backend). Returns the written path, or None if matplotlib
    is unavailable or no split has examples.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for name in ("train", "val", "test"):
        mask = masks.get(name)
        if mask is None or not np.asarray(mask).any():
            continue
        sub = _subset(td, mask)
        mu = np.asarray(model.latents(sub.X, sub.X_cond, sub.X_params), dtype=float)
        norms = np.linalg.norm(mu, axis=1)
        ax.hist(norms, bins="auto", alpha=0.5, label=f"{name} (n={norms.shape[0]})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel(r"$||\mu||_2$")
    ax.set_ylabel("count")
    ax.set_title(f"{label}\nlatent code norms per split")
    ax.legend()

    out = figure_dir() / f"{label}__latent_norms.{extension}"
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

    # The ledger keeps only these headline scalars; the richer evaluation
    # diagnostics added below go to summary.json but not into the greppable
    # one-line ledger entry (kept lean by the ledger_metrics argument).
    ledger_metrics = dict(summary)

    # Held-out evaluation diagnostics (additive run-record keys, written to
    # summary.json). All are computed on the held-out fold from the same point
    # predictions behind summary["rmse"]; the probabilistic and latent ones are
    # duck-typed on the model, so a model without them simply omits those keys.
    per_curve = _per_curve_rmse_metrics(model, td, masks, summary)
    if per_curve is not None:  # None for a 1-D single_k target
        summary["per_curve_rmse"] = per_curve
    summary["suppressed_rmse"] = _suppressed_rmse_metrics(
        model, td, masks, summary, run_config.evaluation.suppressed_thresholds
    )
    if hasattr(model, "predict_samples"):
        summary["coverage"] = _coverage_metrics(
            model, td, masks, summary, seed=run_config.seed
        )
    if hasattr(model, "latents"):
        summary["latent_norms"] = _latent_norm_metrics(model, td, masks)

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
        ledger_metrics=ledger_metrics,
        # Record a per-epoch trace if the model keeps one (duck-typed, so the
        # runner names no model): a model that trains iteratively may expose a
        # ``history`` list of per-epoch metric rows; one that fits in a single
        # shot has none and records an empty metrics.jsonl, as before.
        metrics=getattr(model, "history", None) or None,
        device=device,
        experiments_root=experiments_root,
    )

    # Checkpoint -> <scratch_root>/models/<run_id>/ (the big artifact on scratch).
    ckpt_dir = Path(scratch_root) / "models" / record.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "model.joblib"
    joblib.dump(model, checkpoint_path)

    if write_figures:
        label = _figure_label(record.run_id, run_config.model)
        _write_pred_vs_true_figure(model, td, masks, label, summary)
        _write_rmse_vs_k_figure(model, td, masks, label, summary)
        # Part 1: per-curve / worst-curve diagnostics for every model (both
        # self-skip for a 1-D single_k target, as _write_rmse_vs_k_figure does).
        _write_per_curve_rmse_figure(model, td, masks, label, summary)
        _write_worst_curves_figure(model, td, masks, label, summary)
        # Part 2: training-history figure for any model with a non-empty history
        # (duck-typed, as with the metrics.jsonl per-epoch trace).
        if getattr(model, "history", None):
            _write_history_figure(model, label)
        # Part 3: probabilistic diagnostics for models exposing predict_samples.
        if hasattr(model, "predict_samples"):
            _write_spread_vs_error_figure(
                model, td, masks, label, summary, seed=run_config.seed
            )
            _write_coverage_figure(
                model, td, masks, label, summary, seed=run_config.seed
            )
        # Part 4: latent-norm diagnostic for models exposing latents.
        if hasattr(model, "latents"):
            _write_latent_norm_figure(model, td, masks, label)

    return RunResult(
        run_id=record.run_id,
        run_dir=record.run_dir,
        summary=summary,
        checkpoint_path=checkpoint_path,
        split_masks=masks,
    )
