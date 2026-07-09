"""Lin et al. (2026) analogue figures from existing checkpoints and records.

Post-close-out, comparison-only figure generation: regenerates every figure of
Lin et al. (2026, arXiv:2509.01881) that has an analogue on this project's own
data and can be made **without training or new data** — from the committed
tag-20260706 checkpoints, the pinned dataset, and the Stage 2 run records.

Provenance statement (test-fold discipline): the fold-scored figures below
(latent-parameter correlations, the Fig. 3 heatmaps, the Fig. 12
reconstruction curves) are evaluated on the **test fold** by maintainer
decision (2026-07-09). This is an additional test-fold look at models whose
test rows were already declared and recorded (see
``experiments/notes/dual_vae_test_declaration.md``); it produces figures
only, no new selection or adoption decisions.

Figure map (paper figure -> analogue here; files under
``figures/<YYYY-MM>/<MM-DD>/lin2026_analogues/``):

* ``fig01_midleft_latent_traversal`` — decoder-Y response to perturbing each
  y-latent by up to +/-0.9 population-sigma (their Fig. 1 mid-left; one
  snapshot instead of their three redshifts).
* ``fig01_midright_latent_pdf`` — aggregate posterior q(z) density, pairwise
  panels for the 3 latent dimensions (their Fig. 1 mid-right).
* ``fig01_bottom_latent_param_corr`` — Pearson r of the y-latents against all
  35 SB35 parameters (their Fig. 1 bottom; the model stays conditioned on the
  5 cosmological parameters only).
* ``fig03_param_heatmaps_L<a>L<b>`` — parameter-value heatmaps projected onto
  each latent plane: the six strongest-|r| parameters plus two null
  references (their Fig. 3).
* ``fig05_dimensional_stability`` — active-dimension counts and the
  rate-distortion curve over the Stage 2 (ld x beta) sweep records — this
  project's evidence for the effective dimensionality (their Fig. 5's
  higher-dimensional stability question, answered from records rather than
  exemplar failures).
* ``fig06_training_curves_vae_y`` / ``fig06_training_curves_dual_vae`` —
  per-epoch training/validation traces from the stored histories (their
  Fig. 6).
* ``fig07_sweep_front`` — reconstruction-vs-KL scatter over the Stage 2
  sweep, selection starred (their Fig. 7 Pareto front; the sweep is a fixed
  grid, not an Optuna search).
* ``fig10_spk_range`` — median / 16-84% / min-max bands of SP(k) over the
  dataset (their Fig. 10, single suite).
* ``fig12_pca_vs_vae_recon`` — per-k-bin reconstruction RMSE, PCA at
  n = 1..5 (train-fold fit) against the VAE-Y posterior-mean decode (their
  Fig. 12).
* ``recon_rmse_vs_k`` — per-k **reconstruction** RMSE of the VAE-Y, one
  curve per split (train/val/test), in the style of ``train.py``'s
  ``__rmse_vs_k`` diagnostic. Not a Lin figure: the companion to the
  composite runs' *prediction* rmse_vs_k plots, and deliberately titled
  "reconstruction" so the two error types are never conflated (an
  autoencoder's y -> y error is not the cross-modal f_gas -> SP(k) error).

Deliberately excluded (recorded, not forgotten): Fig. 1 top (architecture
schematic, not a data figure); Fig. 2 (needs the CAMELS CV set — not in the
store); Fig. 4 (needs black-hole catalogue reads); Figs. 8-9 (multi-suite
cross-training; this is a single-suite programme); Fig. 11 (reproducibility
across 10 reseeded retrainings — deferred: it is the one figure requiring
training compute).

The data tag defaults to **20260706** — the tag every loaded checkpoint was
trained on — regardless of the working config's tag, so figure provenance
matches checkpoint provenance (override deliberately via ``--tag``).

Usage::

    python scripts/lin_analogue_figures.py \\
        [--config-data scripts/configs/data/config_dual_vae.yaml] \\
        [--tag 20260706] [--fold test] [--seed 0] \\
        [--vae-y-run-id 20260706T190601Z__8a715039__8522dd4] \\
        [--dual-vae-run-id 20260706T215627Z__cfc56666__43ea387] \\
        [--figures fig01_traversal,fig10,...] [--scratch-root PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from fgas_spk.loader import DataConfig, load_training_data
from fgas_spk.paths import figure_dir, fill_data_root, repo_root, resolve_scratch_root
from fgas_spk.train import grouped_split

from stage1_pca_dimensionality import PINNED_SPLIT, pca_dimensionality_table
from stage2_vae_y import COLLAPSE_KL_MIN, latent_param_correlations

DEFAULT_DATA_CONFIG = "scripts/configs/data/config_dual_vae.yaml"

# The tag the loaded checkpoints were trained on. Pinned here on purpose:
# the working data config may move to a newer tag, but decoding
# 20260706-trained models against another build would mix provenances.
DEFAULT_TAG = "20260706"

# Stage 2 selected VAE-Y (ld=3, beta=1e-4) and the primary dual_vae composite
# (Arm P, (pca d4, vae ld3)); both from the committed test declarations.
DEFAULT_VAE_Y_RUN_ID = "20260706T190601Z__8a715039__8522dd4"
DEFAULT_DUAL_VAE_RUN_ID = "20260706T215627Z__cfc56666__43ea387"

# Category subfolder under the dated figure tree (figure_dir convention).
OUT_SUBDIR = "lin2026_analogues"

# Lin et al. perturb each latent by +/-0.9 around its mean against a unit
# Gaussian prior. The selected VAE-Y runs at beta=1e-4 (high rate), so its
# posterior means spread far beyond N(0, 1); the faithful analogue is +/-0.9
# **population standard deviations** of each latent. Presentation choice,
# documented in the figure title.
TRAVERSAL_SIGMA = 0.9
TRAVERSAL_N_CURVES = 13

# Aggregate-posterior sampling (their Fig. 1 mid-right): samples per example.
QZ_SAMPLES_PER_EXAMPLE = 20

FIGURE_KEYS = (
    "fig01_traversal", "fig01_qz", "fig01_corr", "fig03",
    "fig05", "fig06", "fig07", "fig10", "fig12", "recon_k",
)

# The five conditioning (cosmological) parameter names; everything else in
# the 35-parameter set is astrophysical/baryonic (used to pick null panels).
COSMO_PARAM_NAMES = ("Omega0", "sigma8", "OmegaBaryon", "HubbleParam", "n_s")


def _plt():
    """Return pyplot on the Agg backend (lazy, matching the stage scripts)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_checkpoint(scratch_root: str | Path, run_id: str):
    """Load ``<scratch>/models/<run_id>/model.joblib`` (read-only)."""
    path = Path(scratch_root) / "models" / run_id / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return joblib.load(path)


def collect_stage2_records(tag: str) -> list[dict]:
    """Collect the Stage 2 sweep records for ``tag`` from the run-record tree.

    Scans ``<repo>/experiments/runs/*/summary.json`` for
    ``stage == "dual_vae_stage2"`` whose sibling ``config_data.yaml`` carries
    the requested tag. On 20260706 this yields the 12-run (ld x beta) sweep
    plus the four beta=1e-4 extension runs (ld in {2, 3, 4, 6}); the SP(k)
    modality is untouched by the radial crop, so the two batches share folds
    and are directly comparable (F5 repin).

    Args:
        tag (str): Dataset tag the records must be pinned to.

    Returns:
        list[dict]: One dict per record with ``run_id``, ``latent_dim``,
            ``beta``, ``val_recon_rmse``, ``kl_per_dim_train``.
    """
    records = []
    for run_dir in sorted((repo_root() / "experiments" / "runs").iterdir()):
        summary_path = run_dir / "summary.json"
        config_path = run_dir / "config_data.yaml"
        if not summary_path.exists() or not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("stage") != "dual_vae_stage2":
            continue
        if yaml.safe_load(config_path.read_text()).get("tag") != tag:
            continue
        records.append({
            "run_id": run_dir.name,
            "latent_dim": summary["latent_dim"],
            "beta": summary["beta"],
            "val_recon_rmse": summary["val_recon_rmse"],
            "kl_per_dim_train": summary["kl_per_dim_train"],
        })
    return records


def fig01_traversal(
    vae, y_all: np.ndarray, ctx_all: np.ndarray, k: np.ndarray,
    out_dir: Path, run_id: str,
) -> Path:
    """Fig. 1 mid-left analogue: decoder-Y response to single-latent sweeps.

    Each panel perturbs one latent dimension of the population-mean code by
    up to +/-0.9 population-sigma (others held at their means) and decodes at
    the median cosmological context; the data's median and 16-84% band give
    the reference spread.

    Args:
        vae: The fitted ``GaussianVAE`` y-codec.
        y_all (np.ndarray): All SP(k) rows, shape (n, n_k).
        ctx_all (np.ndarray): The 5-parameter context rows, shape (n, 5).
        k (np.ndarray): The k grid, shape (n_k,).
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id (filename + title).

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    mu = vae.encode(y_all, ctx_all)                       # (n, d)
    base, sigma = mu.mean(axis=0), mu.std(axis=0)
    ctx_fid = np.median(ctx_all, axis=0, keepdims=True)
    deltas = np.linspace(-TRAVERSAL_SIGMA, TRAVERSAL_SIGMA, TRAVERSAL_N_CURVES)

    lo, med, hi = np.percentile(y_all, [16, 50, 84], axis=0)
    n_dims = mu.shape[1]
    cmap = plt.get_cmap("RdBu_r")
    fig, axes = plt.subplots(
        1, n_dims, figsize=(4.2 * n_dims, 3.6), sharey=True
    )
    for j, ax in enumerate(np.atleast_1d(axes)):
        z = np.tile(base, (len(deltas), 1))
        z[:, j] = base[j] + deltas * sigma[j]
        curves = vae.decode(z, np.tile(ctx_fid, (len(deltas), 1)))
        for d, curve in zip(deltas, curves):
            ax.plot(k, curve, color=cmap(0.5 + 0.5 * d / TRAVERSAL_SIGMA),
                    linewidth=1.2)
        ax.plot(k, vae.decode(base[None, :], ctx_fid)[0], color="black",
                linewidth=1.5, label="fiducial decode")
        ax.fill_between(k, lo, hi, color="grey", alpha=0.25,
                        label="data 16-84%")
        ax.plot(k, med, color="grey", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h$/Mpc]")
        ax.set_title(f"latent {j}")
        if j == 0:
            ax.set_ylabel("SP(k)")
            ax.legend(fontsize=7)
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(-TRAVERSAL_SIGMA, TRAVERSAL_SIGMA),
    )
    fig.colorbar(sm, ax=axes, label=r"perturbation [population $\sigma_j$]")
    fig.suptitle(
        f"{run_id}\nVAE-Y latent traversal at median context "
        f"(Lin Fig. 1 mid-left analogue; snap74)", fontsize=9, y=1.12,
    )
    out = out_dir / f"fig01_midleft_latent_traversal__{run_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig01_qz(
    vae, y_all: np.ndarray, ctx_all: np.ndarray,
    out_dir: Path, run_id: str, seed: int,
) -> Path:
    """Fig. 1 mid-right analogue: aggregate posterior q(z), pairwise panels.

    Draws ``QZ_SAMPLES_PER_EXAMPLE`` posterior samples per example over the
    full dataset and shows log10 density in each latent plane (three panels
    for ld=3, against the paper's single 2D panel).

    Args:
        vae: The fitted ``GaussianVAE`` y-codec.
        y_all (np.ndarray): All SP(k) rows.
        ctx_all (np.ndarray): Context rows.
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.
        seed (int): Sampling seed.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    mu, std = vae.encode(y_all, ctx_all, return_std=True)
    rng = np.random.default_rng(seed)
    n, d = mu.shape
    z = (mu[None] + std[None] * rng.standard_normal(
        (QZ_SAMPLES_PER_EXAMPLE, n, d))).reshape(-1, d)

    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    fig, axes = plt.subplots(
        1, len(pairs), figsize=(4.4 * len(pairs), 3.8)
    )
    for (a, b), ax in zip(pairs, np.atleast_1d(axes)):
        hist, xe, ye = np.histogram2d(z[:, a], z[:, b], bins=60, density=True)
        log_q = np.full_like(hist, np.nan)
        np.log10(hist, out=log_q, where=hist > 0)
        pcm = ax.pcolormesh(
            xe, ye, log_q.T, cmap="RdBu_r", shading="auto"
        )
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10} q(z)$")
        ax.set_xlabel(f"latent {a}")
        ax.set_ylabel(f"latent {b}")
    fig.suptitle(
        f"{run_id}\naggregate posterior q(z), {QZ_SAMPLES_PER_EXAMPLE} "
        "samples/example, full dataset (Lin Fig. 1 mid-right analogue)",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / f"fig01_midright_latent_pdf__{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig01_corr(
    corr: np.ndarray, param_names: list[str],
    out_dir: Path, run_id: str, fold: str,
) -> Path:
    """Fig. 1 bottom analogue: latent-parameter Pearson heatmap (35 params).

    Args:
        corr (np.ndarray): Pearson matrix, shape (n_latents, n_params).
        param_names (list[str]): The 35 SB35 parameter names, column order.
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.
        fold (str): Which fold the latents were computed on (title).

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    n_dims, n_params = corr.shape
    fig, ax = plt.subplots(
        figsize=(max(10, 0.32 * n_params), 1.6 + 0.6 * n_dims)
    )
    vmax = max(0.45, float(np.abs(corr).max()))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(param_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels([f"latent {j}" for j in range(n_dims)])
    for j in range(n_dims):
        for p in range(n_params):
            ax.text(p, j, f"{corr[j, p]:+.2f}", ha="center", va="center",
                    fontsize=5)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(
        f"{run_id}\nmu2 vs all 35 SB35 parameters, {fold} fold "
        "(Lin Fig. 1 bottom analogue)", fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / f"fig01_bottom_latent_param_corr__{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig03_heatmaps(
    mu: np.ndarray, params: np.ndarray, param_names: list[str],
    corr: np.ndarray, out_dir: Path, run_id: str, fold: str,
) -> list[Path]:
    """Fig. 3 analogue: parameter heatmaps projected onto each latent plane.

    For every latent pair, hexbin panels of the six strongest-|r| parameters
    plus two null references (Omega0 and the weakest-|r| baryonic parameter),
    each bin coloured by the mean parameter value.

    Args:
        mu (np.ndarray): Latent posterior means on the chosen fold, (n, d).
        params (np.ndarray): All 35 parameter values on the fold, (n, 35).
        param_names (list[str]): Parameter names, column order.
        corr (np.ndarray): Pearson matrix from :func:`fig01_corr`, (d, 35).
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.
        fold (str): Fold name (title).

    Returns:
        list[Path]: One written figure per latent pair.
    """
    plt = _plt()
    strength = np.abs(corr).max(axis=0)                    # per-parameter
    top = list(np.argsort(strength)[::-1][:6])
    baryonic = [i for i, nm in enumerate(param_names)
                if nm not in COSMO_PARAM_NAMES]
    cosmo = [i for i, nm in enumerate(param_names)
             if nm in COSMO_PARAM_NAMES]
    # Null references are the WEAKEST-|r| parameters of each family, not a
    # fixed name: unlike Lin et al. (whose latents are cosmology-null by
    # construction), this VAE-Y carries residual cosmology at low beta
    # (Stage 2 finding), so e.g. Omega0 can be a top correlate here and must
    # not be mislabelled as a null.
    null_baryonic = min((i for i in baryonic if i not in top),
                        key=lambda i: strength[i])
    null_cosmo = min((i for i in cosmo if i not in top),
                     key=lambda i: strength[i])
    panel_idx = top + [null_baryonic, null_cosmo]

    d = mu.shape[1]
    outs = []
    for a in range(d):
        for b in range(a + 1, d):
            fig, axes = plt.subplots(2, 4, figsize=(15, 7),
                                     sharex=True, sharey=True)
            for p, ax in zip(panel_idx, axes.ravel()):
                hb = ax.hexbin(mu[:, a], mu[:, b], C=params[:, p],
                               reduce_C_function=np.mean, gridsize=10,
                               cmap="RdBu_r")
                fig.colorbar(hb, ax=ax)
                tag = (f"null ref, |r|={strength[p]:.2f}"
                       if p in (null_baryonic, null_cosmo)
                       else f"|r|={strength[p]:.2f}")
                ax.set_title(f"{param_names[p]} ({tag})", fontsize=8)
                ax.set_xlabel(f"latent {a}")
                ax.set_ylabel(f"latent {b}")
            fig.suptitle(
                f"{run_id}\nparameter heatmaps in the (latent {a}, latent "
                f"{b}) plane, {fold} fold (Lin Fig. 3 analogue)", fontsize=10,
            )
            fig.tight_layout()
            out = out_dir / f"fig03_param_heatmaps_L{a}L{b}__{run_id}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            outs.append(out)
    return outs


def fig05_dim_stability(
    records: list[dict], y_tr: np.ndarray, y_va: np.ndarray, out_dir: Path,
) -> Path:
    """Fig. 5 analogue: latent-dimension stability over the Stage 2 sweep.

    Left: active dimensions (per-dim train KL >= 0.01 nats, the G2.2'
    criterion) against beta per latent_dim — the collapse behaviour their
    Fig. 5 probes with exemplar 3D failures. Right: the rate-distortion
    curve (val recon RMSE vs beta) with matched-dimension PCA-on-Y floors.

    Args:
        records (list[dict]): Stage 2 sweep records (:func:`collect_stage2_records`).
        y_tr (np.ndarray): Train-fold SP(k) (for the PCA reference).
        y_va (np.ndarray): Val-fold SP(k) (for the PCA reference).
        out_dir (Path): Output directory.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    lds = sorted({r["latent_dim"] for r in records})
    _, pca_table = pca_dimensionality_table(
        np.asarray(y_tr, dtype=float), np.asarray(y_va, dtype=float),
        tuple(lds), seed=0,
    )
    pca_at = dict(zip(pca_table["n_components"], pca_table["val_recon_rmse"]))

    fig, (ax_act, ax_rd) = plt.subplots(1, 2, figsize=(11, 4))
    for i, ld in enumerate(lds):
        pts = sorted((r for r in records if r["latent_dim"] == ld),
                     key=lambda r: r["beta"])
        betas = [r["beta"] for r in pts]
        active = [sum(k >= COLLAPSE_KL_MIN for k in r["kl_per_dim_train"])
                  for r in pts]
        rmse = [r["val_recon_rmse"] for r in pts]
        ax_act.plot(betas, active, marker="o", color=f"C{i}",
                    label=f"ld={ld}")
        ax_rd.plot(betas, rmse, marker="o", color=f"C{i}", label=f"ld={ld}")
        if ld in pca_at:
            ax_rd.axhline(pca_at[ld], color=f"C{i}", linestyle=":",
                          linewidth=1)
    ax_act.set_xscale("log")
    ax_act.set_xlabel(r"$\beta$")
    ax_act.set_ylabel(f"active dims (train KL >= {COLLAPSE_KL_MIN} nats)")
    ax_act.set_title("latent activity vs rate", fontsize=9)
    ax_act.legend(fontsize=8)
    ax_rd.set_xscale("log")
    ax_rd.set_yscale("log")
    ax_rd.set_xlabel(r"$\beta$")
    ax_rd.set_ylabel("val recon RMSE of SP(k)")
    ax_rd.set_title("rate-distortion (dotted: PCA-on-Y at matched dim)",
                    fontsize=9)
    ax_rd.legend(fontsize=8)
    fig.suptitle(
        "Stage 2 sweep records, tag 20260706\nlatent-dimension stability "
        "(Lin Fig. 5 analogue, from records)", fontsize=10,
    )
    fig.tight_layout()
    out = out_dir / "fig05_dimensional_stability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig06_training_curves(
    history: list[dict], out_dir: Path, run_id: str, label: str,
    best_epoch: int | None = None,
) -> Path:
    """Fig. 6 analogue: per-epoch training/validation traces.

    Plots every numeric series present in the history; a multi-phase history
    (the composite) gets one panel row per phase.

    Args:
        history (list[dict]): Per-epoch rows (a ``phase`` key splits panels).
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.
        label (str): Filename token (``vae_y`` / ``dual_vae``).
        best_epoch (int | None): Marked with a vertical line when given.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    phases = list(dict.fromkeys(r.get("phase", label) for r in history))
    fig, axes = plt.subplots(
        len(phases), 1, figsize=(7, 3.0 * len(phases)), squeeze=False
    )
    for phase, ax in zip(phases, axes.ravel()):
        rows = [r for r in history if r.get("phase", label) == phase]
        keys = [k for k in rows[0]
                if k not in ("epoch", "phase", "kl_per_dim", "activity")
                and isinstance(rows[0][k], (int, float))
                and not isinstance(rows[0][k], bool)]
        for key in keys:
            pts = [(r["epoch"], r[key]) for r in rows
                   if isinstance(r.get(key), (int, float))
                   and not isinstance(r.get(key), bool)]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    linewidth=1, label=key)
        if best_epoch is not None and phase == phases[0]:
            ax.axvline(best_epoch, color="grey", linestyle=":",
                       label=f"best epoch {best_epoch}")
        ax.set_xscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(phase, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle(f"{run_id}\ntraining traces (Lin Fig. 6 analogue)",
                 fontsize=9)
    fig.tight_layout()
    out = out_dir / f"fig06_training_curves_{label}__{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig07_sweep_front(
    records: list[dict], selected_run_id: str, out_dir: Path,
) -> Path:
    """Fig. 7 analogue: reconstruction-vs-KL scatter over the Stage 2 sweep.

    The paper's Pareto front is an Optuna search; the analogue here is the
    fixed (ld x beta) grid, coloured by log10(beta), one marker shape per
    latent_dim, the selected configuration starred.

    Args:
        records (list[dict]): Stage 2 sweep records.
        selected_run_id (str): The starred (selected) run id.
        out_dir (Path): Output directory.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    markers = {2: "o", 3: "s", 4: "^", 6: "D"}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    log_betas = [np.log10(r["beta"]) for r in records]
    norm = plt.Normalize(min(log_betas), max(log_betas))
    cmap = plt.get_cmap("viridis")
    for r in records:
        ax.scatter(r["val_recon_rmse"], sum(r["kl_per_dim_train"]),
                   marker=markers.get(r["latent_dim"], "x"), s=60,
                   color=cmap(norm(np.log10(r["beta"]))),
                   edgecolor="black", linewidth=0.4)
    sel = next(r for r in records if r["run_id"] == selected_run_id)
    ax.scatter(sel["val_recon_rmse"], sum(sel["kl_per_dim_train"]),
               marker="*", s=380, facecolor="none", edgecolor="red",
               linewidth=1.5,
               label=f"this set's checkpoint (ld={sel['latent_dim']}, "
                     f"beta={sel['beta']:g})")
    for ld, m in markers.items():
        if any(r["latent_dim"] == ld for r in records):
            ax.scatter([], [], marker=m, color="grey", label=f"ld={ld}")
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                 label=r"$\log_{10} \beta$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("val recon RMSE of SP(k)")
    ax.set_ylabel("total train KL [nats]")
    ax.set_title(
        "Stage 2 sweep, tag 20260706\nreconstruction vs rate "
        "(Lin Fig. 7 analogue; fixed grid, not a search)", fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "fig07_sweep_front.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig10_spk_range(y_all: np.ndarray, k: np.ndarray, out_dir: Path) -> Path:
    """Fig. 10 analogue: median / 16-84% / min-max bands of SP(k).

    Args:
        y_all (np.ndarray): All SP(k) rows, shape (n, n_k).
        k (np.ndarray): The k grid.
        out_dir (Path): Output directory.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    lo, med, hi = np.percentile(y_all, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(k, y_all.min(axis=0), y_all.max(axis=0),
                    color="C0", alpha=0.25, label="min-max")
    ax.fill_between(k, lo, hi, color="C3", alpha=0.35, label="16-84%")
    ax.plot(k, med, color="black", linewidth=1.2, label="median")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel("SP(k)")
    ax.set_title(
        "SP(k) range, IllustrisTNG SB35, snap74, tag 20260706\n"
        "(Lin Fig. 10 analogue; single suite)", fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "fig10_spk_range.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig12_pca_vs_vae(
    vae, y_tr: np.ndarray, y_ev: np.ndarray, ctx_ev: np.ndarray,
    k: np.ndarray, out_dir: Path, run_id: str, fold: str,
) -> tuple[Path, dict]:
    """Fig. 12 analogue: per-k reconstruction RMSE, PCA n=1..5 vs VAE-Y.

    PCA is fit on the train fold only (full SVD, matching ``PcaCodec``);
    both reconstructions are scored per k-bin on the evaluation fold. The
    VAE uses the posterior-mean decode (amendment B4).

    Args:
        vae: The fitted ``GaussianVAE`` y-codec.
        y_tr (np.ndarray): Train-fold SP(k).
        y_ev (np.ndarray): Evaluation-fold SP(k).
        ctx_ev (np.ndarray): Evaluation-fold context.
        k (np.ndarray): The k grid.
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.
        fold (str): Evaluation fold name (title).

    Returns:
        tuple[Path, dict]: The written figure and the per-curve mean RMSEs
            (for the printed sanity block).
    """
    from sklearn.decomposition import PCA

    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = plt.get_cmap("Reds")
    means = {}
    for n_pc in range(1, 6):
        pca = PCA(n_components=n_pc, svd_solver="full", random_state=0)
        pca.fit(np.asarray(y_tr, dtype=float))
        recon = pca.inverse_transform(pca.transform(y_ev))
        per_k = np.sqrt(np.mean((recon - y_ev) ** 2, axis=0))
        # Darker = fewer components, matching the paper's Fig. 12.
        ax.plot(k, per_k, color=cmap(1.0 - 0.15 * (n_pc - 1)),
                linewidth=1.2, label=f"n_pc = {n_pc}")
        means[f"pca_{n_pc}"] = float(per_k.mean())
    vae_per_k = np.sqrt(
        np.mean((vae.reconstruct(y_ev, ctx_ev) - y_ev) ** 2, axis=0)
    )
    ax.plot(k, vae_per_k, color="C0", linestyle="--", linewidth=1.6,
            label=f"VAE-Y (ld={vae.latent_dim})")
    means["vae"] = float(vae_per_k.mean())
    ax.axhline(0.01, color="black", linestyle=":", linewidth=1,
               label="1% RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel(f"reconstruction RMSE of SP(k), {fold} fold")
    ax.set_title(
        f"{run_id}\nPCA (train-fold fit) vs VAE-Y per-k reconstruction "
        "(Lin Fig. 12 analogue)", fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / f"fig12_pca_vs_vae_recon__{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, means


def recon_rmse_vs_k(
    vae, y_all: np.ndarray, ctx_all: np.ndarray, masks: dict,
    k: np.ndarray, out_dir: Path, run_id: str,
) -> Path:
    """Per-k RECONSTRUCTION RMSE of the VAE-Y, one curve per split.

    Same layout as ``train.py``'s ``__rmse_vs_k`` diagnostic, but the error
    plotted is the autoencoding residual (SP(k) -> latent -> SP(k),
    posterior-mean decode), NOT the composite's cross-modal prediction error
    — the title says so explicitly.

    Args:
        vae: The fitted ``GaussianVAE`` y-codec.
        y_all (np.ndarray): All SP(k) rows, shape (n, n_k).
        ctx_all (np.ndarray): Context rows, shape (n, 5).
        masks (dict): Boolean fold masks from ``grouped_split``.
        k (np.ndarray): The k grid.
        out_dir (Path): Output directory.
        run_id (str): Checkpoint run id.

    Returns:
        Path: The written figure.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in ("train", "val", "test"):
        mask = masks[name]
        if not mask.any():
            continue
        recon = vae.reconstruct(y_all[mask], ctx_all[mask])
        rmse_k = np.sqrt(np.mean((recon - y_all[mask]) ** 2, axis=0))
        ax.plot(k, rmse_k, marker=".", label=f"{name} (n={int(mask.sum())})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [$h$/Mpc]")
    ax.set_ylabel("reconstruction RMSE of SP(k)")
    ax.set_title(
        f"{run_id}\nper-k RECONSTRUCTION RMSE (VAE-Y autoencoding, "
        "posterior-mean decode;\nnot the composite's f_gas -> SP(k) "
        "prediction error)", fontsize=9,
    )
    ax.legend()
    fig.tight_layout()
    out = out_dir / f"recon_rmse_vs_k__{run_id}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    """Generate the Lin et al. analogue figure set.

    Args:
        argv (list[str] | None): Command-line arguments; None uses ``sys.argv``.

    Returns:
        int: Process exit status (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Lin et al. (2026) analogue figures from existing "
        "checkpoints and records (post-close-out, comparison-only)."
    )
    parser.add_argument("--config-data", default=DEFAULT_DATA_CONFIG,
                        help="Path to the pinned DataConfig YAML.")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help="Dataset tag; defaults to the checkpoints' "
                        "training tag (20260706), overriding the config.")
    parser.add_argument("--fold", choices=("test", "val", "full"),
                        default="test",
                        help="Evaluation fold for the fold-scored figures "
                        "(maintainer decision 2026-07-09: test).")
    parser.add_argument("--vae-y-run-id", default=DEFAULT_VAE_Y_RUN_ID,
                        help="Stage 2 selected VAE-Y checkpoint run id.")
    parser.add_argument("--dual-vae-run-id", default=DEFAULT_DUAL_VAE_RUN_ID,
                        help="Primary dual_vae checkpoint run id (Fig. 6 "
                        "composite traces only).")
    parser.add_argument("--figures", default=None,
                        help="Comma-separated subset of "
                        f"{','.join(FIGURE_KEYS)}; default all.")
    parser.add_argument("--out-subdir", default=OUT_SUBDIR,
                        help="Category subfolder under the dated figure "
                        "directory (e.g. lin2026_analogues_2 for an "
                        "alternative-checkpoint set).")
    parser.add_argument("--scratch-root", default=None,
                        help="Override the model store root (else "
                        "$FGAS_SCRATCH_ROOT).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed (q(z) figure).")
    args = parser.parse_args(argv)

    wanted = tuple(args.figures.split(",")) if args.figures else FIGURE_KEYS
    unknown = set(wanted) - set(FIGURE_KEYS)
    if unknown:
        raise SystemExit(f"Unknown figure keys: {sorted(unknown)}")

    scratch_root = resolve_scratch_root(explicit=args.scratch_root)
    out_dir = figure_dir(args.out_subdir)

    # Pinned context, tag forced to the checkpoints' training tag.
    data_config = fill_data_root(
        dataclasses.replace(
            DataConfig.from_yaml(args.config_data), tag=args.tag
        )
    )
    td = load_training_data(data_config)
    assert td.X_params is not None, "pinned config must load the 5 cosmo params"
    masks = grouped_split(td.sim_index, PINNED_SPLIT)
    ev_mask = (np.ones(len(td.sim_index), dtype=bool) if args.fold == "full"
               else masks[args.fold])
    y_all, ctx_all, k = np.asarray(td.y, float), np.asarray(td.X_params, float), td.k
    y_tr, y_va = y_all[masks["train"]], y_all[masks["val"]]
    y_ev, ctx_ev = y_all[ev_mask], ctx_all[ev_mask]

    # Full 35-parameter load for the correlation/heatmap figures (the model
    # itself stays conditioned on the 5 cosmological parameters).
    td_full = load_training_data(
        dataclasses.replace(data_config, camels_param_names=None)
    )
    if not np.array_equal(td_full.sim_index, td.sim_index):
        raise RuntimeError("full-parameter load returned different rows")
    assert td_full.param_names is not None
    params_ev = np.asarray(td_full.X_params, float)[ev_mask]

    vae = _load_checkpoint(scratch_root, args.vae_y_run_id)
    mu_ev = vae.encode(y_ev, ctx_ev)
    corr = latent_param_correlations(mu_ev, params_ev)

    records = collect_stage2_records(args.tag)
    written: list[Path] = []
    checks: dict = {
        "tag": args.tag,
        "fold": args.fold,
        "n_rows": {"total": int(len(td.sim_index)),
                   "train": int(masks["train"].sum()),
                   "val": int(masks["val"].sum()),
                   "eval": int(ev_mask.sum())},
        "corr_shape": list(corr.shape),
        "corr_max_abs": float(np.abs(corr).max()),
        "n_stage2_records": len(records),
    }

    if "fig01_traversal" in wanted:
        written.append(fig01_traversal(
            vae, y_all, ctx_all, k, out_dir, args.vae_y_run_id))
        base = vae.encode(y_all, ctx_all).mean(axis=0)
        fid = vae.decode(base[None, :], np.median(ctx_all, axis=0,
                                                  keepdims=True))[0]
        checks["fiducial_decode_range"] = [float(fid.min()), float(fid.max())]
    if "fig01_qz" in wanted:
        written.append(fig01_qz(
            vae, y_all, ctx_all, out_dir, args.vae_y_run_id, args.seed))
    if "fig01_corr" in wanted:
        written.append(fig01_corr(
            corr, list(td_full.param_names), out_dir, args.vae_y_run_id,
            args.fold))
    if "fig03" in wanted:
        written.extend(fig03_heatmaps(
            mu_ev, params_ev, list(td_full.param_names), corr, out_dir,
            args.vae_y_run_id, args.fold))
    if "fig05" in wanted:
        written.append(fig05_dim_stability(records, y_tr, y_va, out_dir))
    if "fig06" in wanted:
        written.append(fig06_training_curves(
            vae.history, out_dir, args.vae_y_run_id, "vae_y",
            best_epoch=vae.best_epoch))
        composite = _load_checkpoint(scratch_root, args.dual_vae_run_id)
        written.append(fig06_training_curves(
            composite.history, out_dir, args.dual_vae_run_id, "dual_vae"))
    if "fig07" in wanted:
        written.append(fig07_sweep_front(
            records, args.vae_y_run_id, out_dir))
    if "fig10" in wanted:
        written.append(fig10_spk_range(y_all, k, out_dir))
    if "recon_k" in wanted:
        written.append(recon_rmse_vs_k(
            vae, y_all, ctx_all, masks, k, out_dir, args.vae_y_run_id))
    if "fig12" in wanted:
        out, means = fig12_pca_vs_vae(
            vae, y_tr, y_ev, ctx_ev, k, out_dir, args.vae_y_run_id, args.fold)
        written.append(out)
        pca_means = [means[f"pca_{n}"] for n in range(1, 6)]
        checks["fig12_mean_rmse"] = means
        checks["fig12_pca_monotone"] = bool(
            all(a >= b for a, b in zip(pca_means, pca_means[1:]))
        )

    checks["figures_written"] = [str(p) for p in written]
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
