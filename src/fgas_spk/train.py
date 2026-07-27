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
diagnostics (per-curve and suppressed-regime RMSE, the per-curve-error/​
suppression-depth correlation ``rmse_vs_depth``, and the ``residual_map``
provenance for the residual-map figure's ladder picks, for every model;
empirical coverage and latent-code norms for models that expose
``predict_samples`` / ``latents``); the greppable ``runs.jsonl`` ledger keeps
only the headline scalars. These diagnostics are computed post-fit and resolve
no physics choice.

**Residual sign convention.** Every residual diagnostic here -- the
``rmse_vs_k`` figure's envelope panel and the ``residual_map`` heatmap -- uses
``pred - true``. The two figures are meant to be read together; a mixed
convention would make them silently contradict each other.

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
    """Write per-k RMSE per split, over a held-out signed-residual envelope.

    Two panels sharing the log-k axis:

    - **top** -- RMSE computed *per k bin* (across rows of a split), one curve per
      non-empty split (train / val / test); ``val`` is omitted when
      ``val_frac == 0``. This shows where, in k, the model errs, and a train/test
      crossing is the overfitting signature.
    - **bottom** -- the *signed* residual ``pred - true`` on the held-out fold
      only: median with 16-84 and 5-95 percentile bands across curves, per k bin,
      against a ``y = 0`` reference.

    The two panels answer different questions and are only meaningful together:
    RMSE conflates bias with scatter, so a large RMSE cannot distinguish "unbiased
    but noisy" from "systematically low". The envelope separates them -- its median
    is the bias, its width the scatter. The percentile levels follow the existing
    ``fig10_spk_range`` convention in ``scripts/lin_analogue_figures.py``.

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
    fig, (ax, ax_env) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 7),
        gridspec_kw={"height_ratios": [1.0, 0.85]},
    )
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

    ax.set_ylabel("RMSE of SP(k)")
    ax.set_title(f"{label}\nper-k RMSE (top) and held-out residual spread (bottom)")
    ax.legend()

    # Bottom: signed-residual percentile envelope on the held-out fold.
    held = summary["held_out_split"]
    _sub, y_true_h, y_pred_h = _held_out_true_pred(model, td, masks, summary)
    resid = y_pred_h - y_true_h  # sign convention: pred - true (module docstring)
    p05, p16, p50, p84, p95 = np.percentile(resid, [5, 16, 50, 84, 95], axis=0)
    ax_env.fill_between(k, p05, p95, color="C0", alpha=0.18, linewidth=0,
                        label="5-95%")
    ax_env.fill_between(k, p16, p84, color="C0", alpha=0.35, linewidth=0,
                        label="16-84%")
    ax_env.plot(k, p50, color="C0", marker=".", label="median (bias)")
    ax_env.axhline(0.0, color="k", linestyle=":", linewidth=1)
    ax_env.set_xscale("log")
    ax_env.set_xlabel("k [h/Mpc]")
    ax_env.set_ylabel("residual (pred - true)")
    ax_env.set_title(f"{held} signed residual ({resid.shape[0]} curves)", fontsize=9)
    ax_env.legend(fontsize=8)

    out = figure_dir() / f"{label}__rmse_vs_k.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# Nominal central-interval levels for the predictive-coverage diagnostic.
_COVERAGE_LEVELS = (0.68, 0.90, 0.95)
# Posterior samples drawn per example for the probabilistic diagnostics.
_N_PREDICTIVE_SAMPLES = 200
# Percentiles of the suppression-depth ordering at which the residual-map figure
# draws its curve panels. A fixed ladder, not a random or worst-N draw: it is
# reproducible, spans the whole population, and cannot be cherry-picked.
_LADDER_PERCENTILES = (5, 25, 50, 75, 95)
# Robust colour limit for the residual heatmap: the symmetric range is set at
# this percentile of |residual| so a handful of outlier curves saturate rather
# than washing the whole map out.
_RESIDUAL_MAP_CLIP_PCT = 99.0
# Latent draws per example when estimating an aggregate latent density. Matches
# QZ_SAMPLES_PER_EXAMPLE in scripts/lin_analogue_figures.py, so the runner's
# density and that script's Lin-Fig.-1 analogue are estimated the same way.
_QZ_SAMPLES_PER_EXAMPLE = 20
# 2-D histogram resolution for the aggregate-density panels (as in that script).
_QZ_BINS = 60
# Latent-traversal figure. The profiles held fixed while z is swept, given as
# percentiles of the suppression-depth ordering: weak, median, and strong
# feedback, so the traversal shows whether the latent acts the same way across
# the population. Real rows, never an averaged synthetic profile -- the decoder
# reads the profile, so an averaged one is an input it never trained on.
_TRAVERSAL_PROFILE_PERCENTILES = (10, 50, 90)
# Sweep half-range, in units of that profile's OWN prior sd sigma_p(x). Wider
# than the +/-0.9 of the reference figure in scripts/lin_analogue_figures.py:
# that script sweeps a population sigma, and a weakly-informative latent needs
# a wider walk before its effect on the decode is visible.
_TRAVERSAL_SIGMA = 2.0
# Curves drawn across the sweep (odd, so the unperturbed code is included).
_TRAVERSAL_N_CURVES = 13


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


def _suppression_depth(y_true: np.ndarray) -> np.ndarray:
    """Per-curve ordering scalar: the TRUE SP(k) in the highest-k selected bin.

    The ordering variable behind the residual-map figure and the
    ``rmse_vs_depth`` correlation (maintainer decision, 2026-07-23). One number
    per curve, monotone in feedback strength for essentially every curve, and it
    separates the two tails that matter: strongly suppressed curves at one end,
    the rare *enhanced* (SP > 1) curves at the other.

    This orders and labels curves; it defines no physics and changes no target.

    Args:
        y_true (np.ndarray): True SP(k), shape (n_curves, n_k).

    Returns:
        np.ndarray: SP(k_max) per curve, shape (n_curves,).
    """
    return np.asarray(y_true, dtype=float)[:, -1]


def _rank(a: np.ndarray) -> np.ndarray:
    """Ordinal ranks of ``a`` (ties broken arbitrarily), as floats.

    The ``argsort(argsort(·))`` idiom already used in
    ``scripts/stage4_latent_map.py``; numpy-only, so no scipy import enters the
    training layer. Ties receive arbitrary distinct ranks rather than midranks --
    immaterial for the continuous quantities correlated here (suppression depth,
    per-curve RMSE), but not a general-purpose ranking.
    """
    return np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation of ``a`` and ``b``, or None if it is undefined.

    Returns None -- not NaN -- for fewer than two points or a constant input, so
    the value stays valid strict JSON in the run record.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation: Pearson on the ordinal ranks (see :func:`_rank`).

    The degeneracy guard is applied to the **raw** inputs, not to the ranks: a
    constant array carries no ordering information, yet :func:`_rank` breaks its
    ties into ``[0, 1, 2, ...]``, which would make it correlate perfectly with any
    monotone partner. Checking the ranks alone would therefore report ``r = 1``
    for a model whose predictions never vary. Returns None when either input is
    constant or has fewer than two points.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or a.std() == 0.0 or b.std() == 0.0:
        return None
    return _corr(_rank(a), _rank(b))


def _thin_log_ticks(ax, values: np.ndarray) -> None:
    """Put sparse, plainly-formatted ticks on a narrow log x axis.

    Over a window spanning a decade or two a log axis carries few decade ticks
    and a crowd of minor ones, whose default ``2 x 10^0``-style labels collide in
    a narrow panel. This pins a handful of round values with plain labels and
    silences the minor labels. Used for both the k axes and the radial axis of
    the profile panels. Cosmetic only.
    """
    from matplotlib.ticker import NullFormatter

    lo, hi = float(np.min(values)), float(np.max(values))
    ticks = [t for t in (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0) if lo <= t <= hi]
    if len(ticks) >= 2:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _ladder_picks(depth: np.ndarray, percentiles=_LADDER_PERCENTILES):
    """Row indices at fixed percentiles of the ascending suppression-depth order.

    Deterministic stratified selection: the curves are ordered by ``depth`` and
    the rows sitting at each requested percentile are taken. Where two
    percentiles collide on the same row -- unavoidable on a small held-out fold --
    the *lower* percentile keeps it and the duplicate is dropped, so the returned
    lists stay aligned and strictly increasing in depth.

    Args:
        depth (np.ndarray): The ordering scalar per curve, shape (n_curves,).
        percentiles (Iterable[int]): Percentiles of the ordering to sample.

    Returns:
        tuple[np.ndarray, list[int], np.ndarray]: ``(rows, labels, positions)`` --
            row indices into the *unsorted* curves, the percentile label kept for
            each, and each pick's rank within the ascending-depth ordering (the
            y coordinate in the sorted heatmap).
    """
    order = np.argsort(np.asarray(depth, dtype=float))
    n = order.shape[0]
    if n == 0:
        return np.empty(0, dtype=int), [], np.empty(0, dtype=int)
    raw = np.clip(
        np.rint(np.asarray(percentiles, dtype=float) / 100.0 * (n - 1)).astype(int),
        0, n - 1,
    )
    kept: dict[int, int] = {}
    for pct, pos in zip(percentiles, raw):
        kept.setdefault(int(pos), int(pct))  # first (lowest) percentile wins
    positions = np.array(sorted(kept), dtype=int)
    labels = [kept[int(p)] for p in positions]
    return order[positions], labels, positions


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


def _rmse_vs_depth_metrics(model, td: TrainingData, masks: dict, summary: dict) -> dict | None:
    """Correlation of per-curve RMSE with suppression depth on the held-out fold.

    Turns "the worst curves happen to be the extreme ones" -- a five-point
    impression from :func:`_write_worst_curves_figure` -- into a population
    statement over every held-out curve. ``spearman_r`` is the headline (rank
    correlation: robust, and the statistic matching the rank axis the figure's
    marginal panel is drawn on); ``pearson_r`` is reported alongside because it
    is the one that would move if the relation were linear in depth rather than
    merely monotone. Either is None when undefined (see :func:`_corr`).

    Returns None for a 1-D (``single_k``) target, where a per-curve RMSE and a
    highest-k bin are both undefined.

    Args:
        model: The fitted model.
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.
        summary (dict): The run summary (read for ``held_out_split``).

    Returns:
        dict | None: ``{"order_by", "spearman_r", "pearson_r", "n_curves"}``, or
            None for a 1-D target.
    """
    _sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2:
        return None
    rmse_per_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))
    depth = _suppression_depth(y_true)
    return {
        "order_by": "spk_at_k_max",
        "spearman_r": _spearman(depth, rmse_per_curve),
        "pearson_r": _corr(depth, rmse_per_curve),
        "n_curves": int(rmse_per_curve.shape[0]),
    }


def _residual_map_metrics(model, td: TrainingData, masks: dict, summary: dict) -> dict | None:
    """Provenance for the residual-map figure's stratified ladder picks.

    Records which held-out curves the figure drew, and why: the ordering
    variable, the percentile each panel sits at, and the originating simulation
    id and depth of each pick. This makes a rendered figure traceable back to
    rows -- the same purpose ``per_curve_rmse.worst_sim_indices`` serves for the
    worst-curve list -- without needing the figure itself.

    Returns None for a 1-D (``single_k``) target, matching the figure's own skip.

    Args:
        model: The fitted model.
        td (TrainingData): The full loaded data.
        masks (dict): Split row masks.
        summary (dict): The run summary (read for ``held_out_split``).

    Returns:
        dict | None: ``{"order_by", "picked_percentiles", "picked_sim_indices",
            "picked_depths"}``, or None for a 1-D target.
    """
    sub, y_true, _y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2:
        return None
    depth = _suppression_depth(y_true)
    rows, labels, _positions = _ladder_picks(depth)
    sim_ids = np.asarray(sub.sim_index)
    return {
        "order_by": "spk_at_k_max",
        "picked_percentiles": [int(p) for p in labels],
        "picked_sim_indices": [int(sim_ids[i]) for i in rows],
        "picked_depths": [float(depth[i]) for i in rows],
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


def _write_residual_map_figure(
    model, td: TrainingData, masks: dict, label: str, summary: dict,
    *, seed: int, extension: str = "png",
) -> Path | None:
    """Every held-out residual as one map, plus a stratified ladder of curves.

    Three linked regions, all ordered by suppression depth
    (:func:`_suppression_depth`, ``SP(k_max)``):

    - **heatmap (top left)** -- ``pred - true`` for *every* held-out curve: rows
      are curves sorted by depth (most suppressed at the bottom), columns are k,
      colour is the signed residual on a diverging scale centred exactly at zero.
      This is the display that needs no sub-selection at all: bias that grows with
      depth, bias that grows with k, and individual bad simulations all read off
      directly. Colour limits are symmetric and robust
      (:data:`_RESIDUAL_MAP_CLIP_PCT` of ``|residual|``), so outliers saturate
      instead of flattening the map.
    - **marginal (top right)** -- per-curve RMSE against the same depth ordering,
      one point per curve, sharing the heatmap's y axis. It is the heatmap's
      projection: the quantitative form of the trend the colours show. Because the
      shared axis is a *rank*, the statistic quoted is Spearman's, which is
      exactly a rank correlation -- no distortion from plotting against rank.
    - **ladder (bottom)** -- the curves at fixed percentiles of that ordering
      (:data:`_LADDER_PERCENTILES`), true (solid black) against the prediction
      (dashed) with 68% and 95% predictive bands where available. Colour-matched
      markers on the heatmap's left spine show where each panel sits in the
      population.

    The ladder is the point of the figure: a *deterministic stratified* sample,
    not a random draw and not the worst-N of
    :func:`_write_worst_curves_figure`. It spans the whole population -- including
    the median curve, which no other figure in the runner shows -- and it cannot
    be cherry-picked.

    The band edges are sample quantiles; the dashed central line is
    :meth:`predict` (the deterministic point prediction behind ``summary["rmse"]``),
    **not** the sample median, so the curve shown is the one every RMSE number in
    the run refers to. For a non-linear decoder the two differ by Jensen's
    inequality. Samples are drawn for the picked rows only -- an independent
    seeded draw from the same predictive distribution, far cheaper than sampling
    the whole fold a third time.

    Models without ``predict_samples`` get the figure without bands, so the
    heatmap and the marginal are available to every model. Skipped for a 1-D
    ``single_k`` target. matplotlib is imported lazily (Agg backend). Returns the
    written path, or None if matplotlib is unavailable or the figure was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.lines import Line2D
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    sub, y_true, y_pred = _held_out_true_pred(model, td, masks, summary)
    if y_true.ndim < 2 or y_true.shape[0] == 0:
        return None

    k = np.asarray(td.k, dtype=float)
    held = summary["held_out_split"]
    resid = y_pred - y_true  # sign convention: pred - true (module docstring)
    rmse_per_curve = np.sqrt(np.mean(resid ** 2, axis=1))
    depth = _suppression_depth(y_true)
    order = np.argsort(depth)
    sim_ids = np.asarray(sub.sim_index)

    rows, labels, positions = _ladder_picks(depth)
    n_picks = rows.shape[0]
    cmap = plt.get_cmap("tab10")

    # Predictive bands for the picked rows only (models exposing predict_samples).
    samples = None
    if hasattr(model, "predict_samples"):
        X_cond = np.asarray(sub.X_cond)[rows] if sub.X_cond is not None else None
        X_params = np.asarray(sub.X_params)[rows] if sub.X_params is not None else None
        samples = np.asarray(
            model.predict_samples(
                np.asarray(sub.X)[rows], X_cond, X_params,
                n_samples=_N_PREDICTIVE_SAMPLES, seed=seed,
            ),
            dtype=float,
        ).reshape(_N_PREDICTIVE_SAMPLES, n_picks, y_true.shape[1])

    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1.0],
                          width_ratios=[3.4, 1.0, 0.07])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_marg = fig.add_subplot(gs[0, 1], sharey=ax_map)
    cax = fig.add_subplot(gs[0, 2])

    # --- heatmap: every held-out curve, sorted by depth ---------------------
    # pcolormesh, not imshow: the k grid is not uniform in log k, and imshow
    # would place columns at equal spacing regardless of their true k.
    vmax = float(np.percentile(np.abs(resid), _RESIDUAL_MAP_CLIP_PCT))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(resid)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0  # a perfect fit: any symmetric range renders a flat map
    mesh = ax_map.pcolormesh(
        k, np.arange(resid.shape[0]), resid[order], shading="nearest",
        cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
    )
    ax_map.set_xscale("log")
    ax_map.set_ylabel(r"held-out curves, sorted by SP($k_{max}$)")
    ax_map.set_xlabel("k [h/Mpc]")
    ax_map.set_title(
        f"{held} residual map -- all {resid.shape[0]} curves (pred - true)",
        fontsize=10,
    )
    # Label the y axis by depth VALUE, not row index.
    n_rows = resid.shape[0]
    ticks = np.unique(np.linspace(0, n_rows - 1, min(6, n_rows)).round().astype(int))
    ax_map.set_yticks(ticks)
    ax_map.set_yticklabels([f"{depth[order][int(t)]:.3f}" for t in ticks])
    fig.colorbar(mesh, cax=cax, label="pred - true")

    # Colour-matched markers linking each ladder panel to its row in the map.
    for j, pos in enumerate(positions):
        ax_map.plot(
            -0.012, pos, marker=">", color=cmap(j % 10), markersize=8,
            transform=ax_map.get_yaxis_transform(), clip_on=False,
        )

    # --- marginal: per-curve RMSE against the same ordering -----------------
    ax_marg.scatter(rmse_per_curve[order], np.arange(n_rows), s=9, alpha=0.45,
                    color="0.35")
    for j, (row_idx, pos) in enumerate(zip(rows, positions)):
        ax_marg.scatter(rmse_per_curve[row_idx], pos, s=45, color=cmap(j % 10),
                        edgecolor="k", linewidth=0.4, zorder=3)
    ax_marg.set_xlabel("per-curve RMSE")
    ax_marg.tick_params(labelleft=False)
    spearman = _spearman(depth, rmse_per_curve)
    ax_marg.set_title(
        "error vs depth\n"
        + (f"Spearman r = {spearman:.2f}" if spearman is not None else "r undefined"),
        fontsize=9,
    )

    # --- ladder: stratified curves with predictive bands --------------------
    gs_bot = gs[1, :].subgridspec(1, max(n_picks, 1), wspace=0.10)
    ax_first = None
    for j, row_idx in enumerate(rows):
        ax = fig.add_subplot(gs_bot[0, j], sharey=ax_first)
        if ax_first is None:
            ax_first = ax
        colour = cmap(j % 10)
        if samples is not None:
            lo95, lo68, hi68, hi95 = np.quantile(
                samples[:, j, :], [0.025, 0.16, 0.84, 0.975], axis=0
            )
            ax.fill_between(k, lo95, hi95, color=colour, alpha=0.18, linewidth=0)
            ax.fill_between(k, lo68, hi68, color=colour, alpha=0.38, linewidth=0)
        ax.plot(k, y_true[row_idx], color="k", linewidth=1.3)
        ax.plot(k, y_pred[row_idx], color=colour, linestyle="--", linewidth=1.3)
        ax.set_xscale("log")
        _thin_log_ticks(ax, k)
        ax.set_xlabel("k [h/Mpc]")
        ax.set_title(
            f"p{labels[j]} · sim {int(sim_ids[row_idx])}\n"
            f"rmse = {rmse_per_curve[row_idx]:.4g}", fontsize=8,
        )
        if j == 0:
            ax.set_ylabel("SP(k)")
            handles = [
                Line2D([], [], color="k", linewidth=1.3, label="true"),
                Line2D([], [], color="0.4", linestyle="--", linewidth=1.3,
                       label="pred"),
            ]
            if samples is not None:
                handles.append(
                    Line2D([], [], color="0.4", linewidth=6, alpha=0.38,
                           label="68 / 95%")
                )
            ax.legend(handles=handles, fontsize=7, loc="best")
        else:
            ax.tick_params(labelleft=False)

    band_note = "68/95% bands" if samples is not None else "no predictive samples"
    fig.suptitle(
        f"{label}\nresidual map + depth-stratified ladder ({band_note})",
        fontsize=11,
    )

    out = figure_dir() / f"{label}__residual_map.{extension}"
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
        # A history may mix per-phase schemas (e.g. the dual_vae composite
        # concatenates codec and mapping traces with different keys), so
        # collect the numeric keys across ALL rows and plot each series over
        # the rows that carry it, rather than indexing every row by the first
        # row's keys.
        numeric_keys: list = []
        for row in history:
            for key, value in row.items():
                if key == "epoch" or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and key not in numeric_keys:
                    numeric_keys.append(key)
        for key in numeric_keys:
            points = [
                (epoch, row[key])
                for epoch, row in zip(epochs, history)
                if isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
            ]
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    marker=".", label=key)
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


def _aggregate_draws(mu: np.ndarray, std: np.ndarray, rng) -> np.ndarray:
    """Pool ``mu + std * eps`` draws over all examples into one latent cloud.

    Estimates the *aggregate* distribution ``(1/N) sum_i N(mu_i, std_i)`` by
    drawing :data:`_QZ_SAMPLES_PER_EXAMPLE` samples from each example's own
    Gaussian and stacking them. Pooling the means alone would instead give a
    systematically narrower cloud that is not the aggregate distribution -- which
    is why the figure needs the widths, not just ``latents()``.

    Args:
        mu (np.ndarray): Per-example means, shape (n_examples, latent_dim).
        std (np.ndarray): Per-example standard deviations, same shape.
        rng (np.random.Generator): Seeded generator.

    Returns:
        np.ndarray: Pooled draws, shape (n_examples * samples, latent_dim).
    """
    n, d = mu.shape
    eps = rng.standard_normal((_QZ_SAMPLES_PER_EXAMPLE, n, d))
    return (mu[None, ...] + std[None, ...] * eps).reshape(-1, d)


def _mass_contour_levels(hist: np.ndarray, fracs=(0.68, 0.95)) -> list:
    """Density thresholds enclosing the given fractions of a histogram's mass.

    The standard corner-plot convention: sort the bin densities descending, walk
    the cumulative sum, and take the density at which the requested fraction of
    total mass has been accumulated. Contours drawn at these levels enclose that
    share of the distribution, which is interpretable in a way that raw density
    levels are not. Returns levels ascending (as ``contour`` requires), dropping
    duplicates and any non-positive level.
    """
    flat = np.sort(hist.ravel())[::-1]
    csum = np.cumsum(flat)
    if csum.size == 0 or csum[-1] <= 0:
        return []
    levels = []
    for frac in fracs:
        idx = min(int(np.searchsorted(csum, frac * csum[-1])), flat.size - 1)
        if flat[idx] > 0:
            levels.append(float(flat[idx]))
    return sorted(set(levels))


def _write_latent_density_figure(
    model, td: TrainingData, label: str, *, seed: int, extension: str = "png",
) -> Path | None:
    """Aggregate latent densities: recognition ``q(z)`` filled, prior ``p(z)`` contoured.

    One panel per pair of latent dimensions, over the **full dataset**. The filled
    ``log10`` density is the aggregate recognition posterior
    ``(1/N) sum_i q(z | x_i, y_i, ctx_i)``; the overlaid white contours are the
    aggregate conditional prior ``(1/N) sum_i p(z | x_i, ctx_i)``, drawn at the
    68% and 95% enclosed-mass levels. Both are estimated from
    :data:`_QZ_SAMPLES_PER_EXAMPLE` seeded draws per example, on shared bin edges
    so the two are directly comparable.

    **What the comparison says.** ``q`` sees the true SP(k); ``p`` sees only the
    profile, and is what :meth:`predict_samples` actually draws from. Their
    difference is what f_gas(R) cannot determine about SP(k). Contours sitting on
    top of the filled density mean the target taught the encoder nothing beyond
    the profile -- posterior collapse, or a genuinely conditionally-deterministic
    target. A tighter ``q`` inside a broader ``p`` is the healthy, informative
    case, and the excess width of ``p`` is the model's estimate of the residual
    SP(k) uncertainty.

    Two readings to avoid. ``p`` here is **learned and input-dependent**, so the
    aggregate prior is a mixture and is *not* ``N(0, I)`` -- the unit-Gaussian
    reference of a plain VAE does not apply. And this is an *aggregate*
    comparison: two aggregates can coincide while individual examples differ. The
    per-example mismatch is the ``kl`` already in the training history; this
    figure shows the shape and extent of the code space instead, which the KL
    cannot.

    Requires ``model.latent_dists`` (guarded by the caller) -- ``latents`` alone
    is insufficient, since the means carry no width. Returns None for a latent of
    fewer than 2 dimensions (no plane to draw). matplotlib is imported lazily
    (Agg backend); returns None if it is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    mu_q, std_q = model.latent_dists(td.X, td.X_cond, td.X_params, y=td.y)
    mu_p, std_p = model.latent_dists(td.X, td.X_cond, td.X_params)
    mu_q = np.asarray(mu_q, dtype=float)
    d = mu_q.shape[1]
    if d < 2:  # a single latent dimension has no plane to plot
        return None

    rng = np.random.default_rng(seed)
    z_q = _aggregate_draws(mu_q, np.asarray(std_q, dtype=float), rng)
    z_p = _aggregate_draws(np.asarray(mu_p, dtype=float),
                           np.asarray(std_p, dtype=float), rng)

    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    # Floor the width: at ncols == 1 (a 2-D latent, one pair) a panel-sized
    # figure is narrower than the caption, which would clip it.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(max(4.8 * ncols, 7.5), 4.2 * nrows), squeeze=False,
    )

    for idx, (a, b) in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        # Shared, robust bin edges over BOTH clouds: q and p must be binned
        # identically or the filled density and the contours are not comparable.
        both = np.concatenate([z_q[:, [a, b]], z_p[:, [a, b]]], axis=0)
        lo = np.percentile(both, 0.5, axis=0)
        hi = np.percentile(both, 99.5, axis=0)
        # Degenerate (collapsed) dimensions can give lo == hi; pad so the bin
        # edges stay strictly increasing.
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        lo, hi = lo - 0.02 * span, hi + 0.02 * span
        edges = [np.linspace(lo[0], hi[0], _QZ_BINS + 1),
                 np.linspace(lo[1], hi[1], _QZ_BINS + 1)]

        hist_q, xe, ye = np.histogram2d(z_q[:, a], z_q[:, b], bins=edges, density=True)
        hist_p, _xe, _ye = np.histogram2d(z_p[:, a], z_p[:, b], bins=edges, density=True)

        log_q = np.full_like(hist_q, np.nan)
        np.log10(hist_q, out=log_q, where=hist_q > 0)
        pcm = ax.pcolormesh(xe, ye, log_q.T, cmap="viridis", shading="auto")
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10}$ aggregate $q(z)$")

        levels = _mass_contour_levels(hist_p)
        if levels:
            centres_x = 0.5 * (xe[:-1] + xe[1:])
            centres_y = 0.5 * (ye[:-1] + ye[1:])
            ax.contour(centres_x, centres_y, hist_p.T, levels=levels,
                       colors="white", linewidths=1.1, alpha=0.9)
        ax.set_xlabel(f"latent {a}")
        ax.set_ylabel(f"latent {b}")

    for idx in range(len(pairs), nrows * ncols):  # blank any unused grid cell
        axes[idx // ncols][idx % ncols].axis("off")

    # The contour key lives in the caption, not an in-panel legend: the density
    # fills the axes, so any legend corner would sit on top of the data.
    fig.suptitle(
        f"{label}\n"
        r"aggregate latent density -- filled: $q(z|x,y)$, "
        r"white contours: $p(z|x)$ at 68/95% mass" "\n"
        f"full dataset ({mu_q.shape[0]} rows x {_QZ_SAMPLES_PER_EXAMPLE} draws)",
        fontsize=9,
    )

    out = figure_dir() / f"{label}__latent_density.{extension}"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_latent_traversal_figure(
    model, td: TrainingData, label: str, extension: str = "png",
) -> Path | None:
    """Decoder response to sweeping one latent at a time, at fixed real profiles.

    A grid of panels: **rows are profiles**, and the columns are the row's
    **input f_gas(R) profile** followed by one panel per **latent dimension**
    (so ``n_dims + 1`` columns in total).

    The leading column is the conditioning input the rest of that row is decoded
    at, over the dataset's 16-84% f_gas band for scale. It is not decoration: the
    decoder reads the profile, and the latent only ever perturbs around what that
    profile already implies, so the size and shape of the fan to its right is not
    interpretable without seeing which profile produced it.

    In each traversal cell the code starts at that profile's own conditional-prior
    mean ``mu_p(x)``, one component is swept across
    ``+/-``:data:`_TRAVERSAL_SIGMA` of that profile's own prior sd
    ``sigma_p(x)`` while the others are held, and every perturbed code is decoded
    **at that same profile and its own context**. Curves are coloured by the
    perturbation; the unperturbed decode is drawn in black and the dataset's
    16-84% SP(k) band in grey for scale.

    This is the **decoder-side** counterpart to the KL trace and the aggregate
    density. Those ask whether the *encoder* uses ``z``; this asks whether ``z``
    changes the *output*, and by how much against the spread of real data. A
    visible fan means the latent carries real, decodable information about SP(k);
    curves that collapse onto the black line mean it does not, whatever the KL
    says.

    Two adaptations are forced by this decoder reading the profile directly
    (``p(y | z, x, ctx)``), unlike the ``p(y | z, ctx)`` codec that the analogous
    figure in ``scripts/lin_analogue_figures.py`` traverses:

    - **Real (profile, context) pairs only.** The reference sweeps around a
      *population*-mean code at a *median* context. Here both would be off-
      manifold input combinations the decoder never trained on, so the profiles
      are real dataset rows (at :data:`_TRAVERSAL_PROFILE_PERCENTILES` of the
      suppression-depth ordering, via :func:`_ladder_picks`) and each is decoded
      with its own context.
    - **Per-profile sweep units.** The step is that profile's own
      ``sigma_p(x)``, not a population sigma, so the sweep spans exactly the
      range the model itself considers plausible *for that profile* -- making the
      fan the spine of the predictive band drawn in the residual-map ladder.

    Requires ``model.decode`` and ``model.latent_dists`` (guarded by the caller).
    Deterministic -- nothing is sampled, so no seed is taken. Skipped for a 1-D
    ``single_k`` target (no curve to trace, and the depth ordering is undefined).
    matplotlib is imported lazily (Agg backend); returns None if it is
    unavailable or the figure was skipped.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:  # pragma: no cover - matplotlib is in the env
        return None

    y_all = np.asarray(td.y, dtype=float)
    if y_all.ndim < 2 or y_all.shape[0] == 0:
        return None

    k = np.asarray(td.k, dtype=float)
    sim_ids = np.asarray(td.sim_index)
    depth = _suppression_depth(y_all)
    rows, labels, _positions = _ladder_picks(
        depth, percentiles=_TRAVERSAL_PROFILE_PERCENTILES
    )
    if rows.size == 0:
        return None

    deltas = np.linspace(-_TRAVERSAL_SIGMA, _TRAVERSAL_SIGMA, _TRAVERSAL_N_CURVES)
    lo, med, hi = np.percentile(y_all, [16, 50, 84], axis=0)

    def _row(arr, i, repeat=1):
        """The i-th row of an optional modality, repeated, or None."""
        if arr is None:
            return None
        return np.repeat(np.asarray(arr)[[i]], repeat, axis=0)

    # Probe the latent width from the first picked profile.
    mu_probe, _std_probe = model.latent_dists(
        _row(td.X, int(rows[0])), _row(td.X_cond, int(rows[0])),
        _row(td.X_params, int(rows[0])),
    )
    n_dims = np.asarray(mu_probe).shape[1]

    # The conditioning profile that each row's traversal is decoded at, shown as
    # the leading column. Fall back to a bin index if the stored radii do not
    # line up with the profile width (e.g. a cropped or augmented feature axis).
    X_all = np.asarray(td.X, dtype=float)
    radii = np.asarray(td.radii_mpch, dtype=float)
    radii_are_real = radii.size == X_all.shape[1]
    r_axis = radii if radii_are_real else np.arange(X_all.shape[1], dtype=float)
    p_lo, p_med, p_hi = np.percentile(X_all, [16, 50, 84], axis=0)

    nrows, ncols = rows.size, n_dims + 1  # +1 for the input-profile column
    cmap = plt.get_cmap("RdBu_r")
    fig = plt.figure(figsize=(4.3 * ncols, 3.4 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    # Sharing is set per column group, not uniformly: the profile column carries
    # f_gas and shares y down the column so profiles are comparable, while each
    # traversal ROW shares y in SP(k). One `sharey` rule cannot express both.
    axes = np.empty((nrows, ncols), dtype=object)
    ax_profile_ref = None
    ax_k_ref = None
    for r in range(nrows):
        ax_p = fig.add_subplot(gs[r, 0], sharex=ax_profile_ref, sharey=ax_profile_ref)
        if ax_profile_ref is None:
            ax_profile_ref = ax_p
        axes[r, 0] = ax_p
        row_ref = None
        for j in range(n_dims):
            ax = fig.add_subplot(gs[r, j + 1], sharex=ax_k_ref, sharey=row_ref)
            if ax_k_ref is None:
                ax_k_ref = ax
            if row_ref is None:
                row_ref = ax
            axes[r, j + 1] = ax

    for r, row_idx in enumerate(rows):
        i = int(row_idx)
        X_i = _row(td.X, i)
        cond_i = _row(td.X_cond, i)
        params_i = _row(td.X_params, i)
        mu_p, std_p = model.latent_dists(X_i, cond_i, params_i)
        base = np.asarray(mu_p, dtype=float)[0]
        sigma = np.asarray(std_p, dtype=float)[0]

        # Column 0: the conditioning input this row's whole traversal is decoded
        # at. The latent only ever perturbs around what THIS profile implies, so
        # the fan to its right is unreadable without it.
        ax_p = axes[r, 0]
        ax_p.fill_between(r_axis, p_lo, p_hi, color="grey", alpha=0.22,
                          linewidth=0, label="data 16-84%")
        ax_p.plot(r_axis, p_med, color="grey", linestyle=":", linewidth=1)
        ax_p.plot(r_axis, X_all[i], color="black", linewidth=1.5,
                  label=r"this row's $f_{gas}$")
        if radii_are_real:
            ax_p.set_xscale("log")
            _thin_log_ticks(ax_p, r_axis)
        ax_p.set_ylabel(
            f"p{labels[r]} · sim {int(sim_ids[i])}\n" + r"$f_{gas}(R)$",
            fontsize=9,
        )
        if r == nrows - 1:
            ax_p.set_xlabel("R [Mpc/h]" if radii_are_real else "profile bin")
        if r == 0:
            ax_p.set_title("input profile (conditioning)", fontsize=10)
            ax_p.legend(fontsize=7, loc="best")

        # The profile and context, repeated once per swept code.
        X_rep = _row(td.X, i, _TRAVERSAL_N_CURVES)
        cond_rep = _row(td.X_cond, i, _TRAVERSAL_N_CURVES)
        params_rep = _row(td.X_params, i, _TRAVERSAL_N_CURVES)
        fiducial = np.asarray(
            model.decode(base[None, :], X_i, cond_i, params_i), dtype=float
        )[0]

        for j in range(n_dims):
            ax = axes[r, j + 1]
            z = np.tile(base, (_TRAVERSAL_N_CURVES, 1))
            z[:, j] = base[j] + deltas * sigma[j]
            curves = np.asarray(
                model.decode(z, X_rep, cond_rep, params_rep), dtype=float
            )

            ax.fill_between(k, lo, hi, color="grey", alpha=0.22, linewidth=0,
                            label="data 16-84%")
            ax.plot(k, med, color="grey", linestyle=":", linewidth=1)
            for delta, curve in zip(deltas, curves):
                ax.plot(k, curve, linewidth=1.2,
                        color=cmap(0.5 + 0.5 * delta / _TRAVERSAL_SIGMA))
            ax.plot(k, fiducial, color="black", linewidth=1.5,
                    label=r"unperturbed ($z=\mu_p$)")
            ax.plot(k, y_all[i], color="black", linestyle="--", linewidth=1.0,
                    label="true SP(k)")
            ax.set_xscale("log")
            _thin_log_ticks(ax, k)
            if r == nrows - 1:
                ax.set_xlabel("k [h/Mpc]")
            if r == 0:
                ax.set_title(f"latent {j}", fontsize=10)
            if j == 0:
                ax.set_ylabel("SP(k)", fontsize=9)
            if r == 0 and j == 0:
                ax.legend(fontsize=7, loc="best")

    scalar_map = plt.cm.ScalarMappable(
        cmap=cmap, norm=Normalize(-_TRAVERSAL_SIGMA, _TRAVERSAL_SIGMA)
    )
    fig.colorbar(
        scalar_map, ax=axes.ravel().tolist(),
        label=r"perturbation [profile's own $\sigma_p(x)$]",
    )
    fig.suptitle(
        f"{label}\nlatent traversal -- one component swept at a time, decoded "
        "at each row's own input profile (first column)\n"
        f"rows: profiles at p{'/p'.join(str(p) for p in labels)} of "
        r"SP($k_{max}$)",
        fontsize=9,
    )

    out = figure_dir() / f"{label}__latent_traversal.{extension}"
    fig.savefig(out, bbox_inches="tight")
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
    # Ordering-based diagnostics: how error tracks suppression depth over the
    # whole held-out fold, and which curves the residual-map figure drew.
    depth_corr = _rmse_vs_depth_metrics(model, td, masks, summary)
    if depth_corr is not None:  # None for a 1-D single_k target
        summary["rmse_vs_depth"] = depth_corr
    residual_map = _residual_map_metrics(model, td, masks, summary)
    if residual_map is not None:
        summary["residual_map"] = residual_map
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
        # Residual map + depth-stratified ladder. Always-on: the bands need
        # predict_samples, but the heatmap and marginal do not, so the function
        # degrades internally rather than being gated away from point models.
        _write_residual_map_figure(
            model, td, masks, label, summary, seed=run_config.seed
        )
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
        # Part 5: aggregate latent densities for models exposing the latent
        # DISTRIBUTIONS (means alone cannot estimate an aggregate density).
        if hasattr(model, "latent_dists"):
            _write_latent_density_figure(model, td, label, seed=run_config.seed)
        # Part 6: latent traversal -- the decoder-side counterpart, for models
        # that can also decode a caller-supplied code.
        if hasattr(model, "decode") and hasattr(model, "latent_dists"):
            _write_latent_traversal_figure(model, td, label)

    return RunResult(
        run_id=record.run_id,
        run_dir=record.run_dir,
        summary=summary,
        checkpoint_path=checkpoint_path,
        split_masks=masks,
    )
