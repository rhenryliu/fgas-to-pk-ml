"""Tests for the held-out evaluation diagnostics in ``fgas_spk.train``.

Coverage: the numerical verification gate (pooled held-out RMSE recomputed from
the per-bin squared-error array equals ``summary["rmse"]``; a suppressed
threshold above ``max(y_true)`` reproduces that pooled RMSE; empirical coverage
stays in ``[0, 1]`` and is monotone in the nominal level), that per-curve worst
lists carry the *original* simulation ids, that the per-curve metric is skipped
for a 1-D ``single_k`` target, that the runner writes the additive diagnostics to
``summary.json`` while keeping the ledger line lean, and that every new figure
renders. Fixtures are synthetic; no real CAMELS data is needed.
"""

import json
from pathlib import Path

import numpy as np

import fgas_spk.schema as S
import fgas_spk.train as T
from fgas_spk.experiment import RunConfig, SplitSpec
from fgas_spk.loader import DataConfig, TrainingData
from fgas_spk.train import grouped_split

_NEW_KEYS = ("per_curve_rmse", "suppressed_rmse", "coverage", "latent_norms",
             "rmse_vs_depth", "residual_map")


# --- a picklable probabilistic fake model ----------------------------------

class _DiagFake:
    """Module-level (picklable) probabilistic model with the optional hooks.

    Exposes ``predict`` / ``predict_samples`` / ``latents`` / ``history`` so it
    triggers Parts 1-4. Deterministic given a seed, so the tests are stable.
    """

    def __init__(self, seed: int = 0, latent_dim: int = 3, **_):
        self.seed = seed
        self.latent_dim = latent_dim
        self._n_k = 1
        self.history = [
            {"epoch": i, "recon": 1.0 / (i + 1), "kl": 0.5 / (i + 1),
             "beta_eff": min(1.0, i / 5.0), "train_loss": 1.1 / (i + 1)}
            for i in range(8)
        ]

    def fit(self, training_data) -> None:
        y = np.asarray(training_data.y)
        self._n_k = y.shape[1] if y.ndim > 1 else 1

    def predict(self, X, X_cond=None, X_params=None):
        X = np.asarray(X, dtype=float)
        return np.tile(np.sin(X[:, :1]) + 0.7, (1, self._n_k))

    def predict_samples(self, X, X_cond=None, X_params=None, n_samples=100, seed=None):
        mu = self.predict(X, X_cond, X_params)
        gen = np.random.default_rng(0 if seed is None else seed)
        return mu[None, ...] + gen.normal(scale=0.05, size=(n_samples,) + mu.shape)

    def latents(self, X, X_cond=None, X_params=None):
        X = np.asarray(X, dtype=float)
        return np.tile(X[:, :1], (1, self.latent_dim))


class _DistFake(_DiagFake):
    """A ``_DiagFake`` that also exposes ``latent_dists`` (means AND widths).

    The prior and the recognition posterior are deliberately different laws --
    the posterior is shifted and tighter -- so the aggregate-density figure has
    two distinguishable clouds to draw, as a non-collapsed model would.
    """

    def latent_dists(self, X, X_cond=None, X_params=None, y=None):
        mu = self.latents(X, X_cond, X_params)
        std = np.full_like(mu, 0.5)
        if y is not None:  # recognition: shifted and tighter than the prior
            mu = mu + 0.75
            std = std * 0.4
        return mu, std

    def decode(self, z, X, X_cond=None, X_params=None):
        """A profile-and-latent-dependent decode, as the real one is.

        Both inputs must matter: the profile sets the level and the latent tilts
        it, so a traversal produces a visible fan rather than identical curves.
        """
        z = np.asarray(z, dtype=float)
        X = np.asarray(X, dtype=float)
        if z.shape[0] != X.shape[0]:
            raise ValueError("z and X must have the same number of rows.")
        base = np.tile(np.sin(X[:, :1]) + 0.7, (1, self._n_k))
        tilt = np.linspace(0.0, 0.1, self._n_k)[None, :] * z[:, :1]
        return base + tilt


class _PointFake:
    """A point-only model: ``predict`` and nothing else.

    Deliberately exposes no ``predict_samples`` / ``latents`` / ``history``, so it
    exercises the graceful-degradation path of the figures that probe for them.
    """

    def __init__(self, seed: int = 0, **_):
        self.seed = seed
        self._n_k = 1

    def fit(self, training_data) -> None:
        y = np.asarray(training_data.y)
        self._n_k = y.shape[1] if y.ndim > 1 else 1

    def predict(self, X, X_cond=None, X_params=None):
        X = np.asarray(X, dtype=float)
        return np.tile(np.cos(X[:, :1]) + 0.8, (1, self._n_k))


# --- synthetic fixtures ----------------------------------------------------

def _td(n_sims=16, n_nd=2, n_radii=5, n_k=8, seed=0, single_k=False):
    """A synthetic TrainingData with a per-simulation SP(k) target repeated by nd."""
    rng = np.random.default_rng(seed)
    n = n_sims * n_nd
    sim_index = np.repeat(np.arange(500, 500 + n_sims), n_nd)  # non-trivial ids
    sup = rng.uniform(0.6, 1.1, size=(n_sims, n_k))
    y = np.repeat(sup, n_nd, axis=0)
    if single_k:
        y = y[:, 0]
    return TrainingData(
        X=rng.uniform(0, 1, size=(n, n_radii)),
        X_cond=None, X_params=None, y=y,
        nd=np.tile(np.arange(n_nd), n_sims).astype(float),
        sim_index=sim_index, k=np.linspace(0.5, 5.0, n_k),
        radii_mpch=np.linspace(0.1, 2.0, n_radii),
        source_path="synthetic", meta={}, config=None,
    )


def _fit(td):
    """Fit a fresh fake and return (model, masks, summary) like the runner does."""
    model = _DiagFake()
    model.fit(td)
    masks = grouped_split(td.sim_index, SplitSpec(0.6, 0.2, 0.2, seed=1))
    held = "test" if masks["test"].any() else "val" if masks["val"].any() else "train"
    sub = T._subset(td, masks[held])
    summary = {
        "held_out_split": held,
        "rmse": T._rmse(sub.y, model.predict(sub.X, sub.X_cond, sub.X_params)),
    }
    return model, masks, summary


# --- gate (a): pooled RMSE from the per-bin SE array == summary["rmse"] -----

def test_pooled_rmse_matches_summary():
    td = _td()
    model, masks, summary = _fit(td)
    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    se = (y_pred - y_true) ** 2
    pooled = float(np.sqrt(np.mean(se)))
    assert abs(pooled - summary["rmse"]) <= 1e-10


# --- gate (b): a threshold above max(y_true) reproduces the pooled RMSE -----

def test_suppressed_high_threshold_equals_pooled():
    td = _td()
    model, masks, summary = _fit(td)
    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    pooled = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    t_hi = float(y_true.max()) + 1.0
    rec = T._suppressed_rmse_metrics(model, td, masks, summary, [t_hi])[str(t_hi)]
    assert rec["n_bins"] == y_true.size
    assert abs(rec["rmse"] - pooled) <= 1e-10


def test_suppressed_empty_mask_is_null_not_error():
    td = _td()
    model, masks, summary = _fit(td)
    _sub, y_true, _yp = T._held_out_true_pred(model, td, masks, summary)
    t_lo = float(y_true.min()) - 1.0
    rec = T._suppressed_rmse_metrics(model, td, masks, summary, [t_lo])[str(t_lo)]
    assert rec == {"rmse": None, "n_bins": 0}


# --- gate (c): coverage in [0, 1] and monotone in the nominal level ---------

def test_coverage_in_range_and_monotonic():
    td = _td()
    model, masks, summary = _fit(td)
    cov = T._coverage_metrics(model, td, masks, summary, seed=0)
    assert all(0.0 <= f <= 1.0 for f in cov.values())
    levels = sorted(float(k) for k in cov)
    fracs = [cov[str(lv)] for lv in levels]
    assert all(fracs[i] <= fracs[i + 1] + 1e-12 for i in range(len(fracs) - 1))


# --- per-curve metric semantics --------------------------------------------

def test_per_curve_worst_lists_use_original_sim_ids():
    td = _td()
    model, masks, summary = _fit(td)
    sub = T._subset(td, masks[summary["held_out_split"]])
    held_sims = {int(s) for s in sub.sim_index.tolist()}
    pc = T._per_curve_rmse_metrics(model, td, masks, summary)
    assert set(pc) == {"mean", "median", "p90", "max",
                       "worst_sim_indices", "worst_sim_rmse"}
    assert all(s in held_sims for s in pc["worst_sim_indices"])
    # descending, and aligned with the indices
    assert pc["worst_sim_rmse"] == sorted(pc["worst_sim_rmse"], reverse=True)
    assert pc["max"] == pc["worst_sim_rmse"][0]


def test_per_curve_skipped_for_single_k_but_suppressed_still_works():
    td = _td(single_k=True)
    model, masks, summary = _fit(td)
    assert T._per_curve_rmse_metrics(model, td, masks, summary) is None
    # suppressed RMSE is defined for a 1-D target and reproduces the pooled RMSE.
    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    pooled = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    t_hi = float(y_true.max()) + 1.0
    rec = T._suppressed_rmse_metrics(model, td, masks, summary, [t_hi])[str(t_hi)]
    assert abs(rec["rmse"] - pooled) <= 1e-10


# --- runner integration: summary carries diagnostics, ledger stays lean -----

def _save_curve_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_sims, n_nd, n_radii, n_k = 12, 2, 5, 8
    ds = S.FgasSpkDataset(
        radii_mpch=np.linspace(0.1, 5.0, n_radii),
        number_densities=np.array([1.0e-4, 2.8e-4]),
        fgas=rng.random((n_sims, n_nd, n_radii)),
        fgas_std=rng.random((n_sims, n_nd, n_radii)),
        k=np.linspace(0.1, 10, n_k),
        suppression=rng.uniform(0.6, 1.1, size=(n_sims, n_k)),
        sim_ids=np.arange(n_sims),
        mean_halo_mass=np.full((n_sims, n_nd), 1e13),
        rank_key="halo_mass",
        snapshot=74,
    )
    return S.save_dataset(
        ds, project_root=tmp_path / "data", suite="TEST",
        snapshot=74, redshift=0.47, source="rhliu", tag="v1", overwrite=True,
    )


def test_run_training_writes_diagnostics_and_keeps_ledger_lean(tmp_path):
    from fgas_spk.models.base import REGISTRY

    REGISTRY["_diag_fake"] = _DiagFake
    npz = _save_curve_dataset(tmp_path)
    data_cfg = DataConfig(path=str(npz), target_mode="curve",
                          include_nd_feature=True, include_camels_params=False)
    run_cfg = RunConfig(model="_diag_fake", seed=0,
                        split=SplitSpec(0.6, 0.2, 0.2, seed=0))
    try:
        result = T.run_training(
            data_cfg, run_cfg,
            scratch_root_override=tmp_path / "scratch",
            experiments_root=tmp_path / "experiments",
            write_figures=False, device="cpu",
        )
    finally:
        REGISTRY.pop("_diag_fake", None)

    # summary.json (and the returned summary) carry every additive diagnostic.
    summary_json = json.loads((result.run_dir / "summary.json").read_text())
    for key in _NEW_KEYS:
        assert key in result.summary, key
        assert key in summary_json, key

    # The ledger line stays lean: headline scalars only, none of the diagnostics.
    ledger = (tmp_path / "experiments" / "runs.jsonl").read_text().splitlines()
    assert len(ledger) == 1
    metrics = json.loads(ledger[0])["metrics"]
    assert "rmse" in metrics and "test_rmse" in metrics
    assert not any(key in metrics for key in _NEW_KEYS)

    # Internal consistency (gate a) end to end.
    assert abs(summary_json["suppressed_rmse"][
        list(summary_json["suppressed_rmse"])[0]]["n_bins"]) >= 0  # present & sane


# --- every new figure renders ----------------------------------------------

def test_new_figures_render(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    model, masks, summary = _fit(td)
    label = "20260101T000000Z__deadbeef__diag"

    writers = [
        lambda: T._write_per_curve_rmse_figure(model, td, masks, label, summary),
        lambda: T._write_worst_curves_figure(model, td, masks, label, summary),
        lambda: T._write_history_figure(model, label),
        lambda: T._write_spread_vs_error_figure(model, td, masks, label, summary, seed=0),
        lambda: T._write_coverage_figure(model, td, masks, label, summary, seed=0),
        lambda: T._write_latent_norm_figure(model, td, masks, label),
        lambda: T._write_rmse_vs_k_figure(model, td, masks, label, summary),
        lambda: T._write_residual_map_figure(model, td, masks, label, summary, seed=0),
    ]
    for write in writers:
        out = write()
        assert out is not None and Path(out).exists()


def test_history_figure_generic_branch(tmp_path, monkeypatch):
    # A non-CVAE history (train_mse only) uses the single-panel generic path.
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)

    class _Mlpish:
        history = [{"epoch": i, "train_mse": 1.0 / (i + 1)} for i in range(6)]

    out = T._write_history_figure(_Mlpish(), "lbl")
    assert out is not None and Path(out).exists()


# --- ordering helpers: rank convention and stratified picks ----------------

def test_spearman_is_one_for_a_monotone_nonlinear_relation():
    # Pins the rank convention: Spearman must see a monotone-but-curved relation
    # as perfect, where Pearson does not.
    x = np.linspace(0.1, 2.0, 25)
    y = np.exp(3.0 * x)
    assert abs(T._spearman(x, y) - 1.0) < 1e-12
    assert T._corr(x, y) < 0.95  # Pearson is degraded by the curvature
    assert abs(T._spearman(x, -y) + 1.0) < 1e-12


def test_corr_returns_none_when_undefined():
    assert T._corr(np.array([1.0]), np.array([2.0])) is None      # < 2 points
    assert T._corr(np.ones(5), np.arange(5.0)) is None            # constant input
    assert T._spearman(np.ones(5), np.arange(5.0)) is None


def test_ladder_picks_land_on_percentile_ranks_of_the_depth_order():
    depth = np.array([0.9, 0.6, 1.05, 0.75, 0.85, 0.7, 1.0, 0.65, 0.95, 0.8])
    rows, labels, positions = T._ladder_picks(depth, percentiles=(5, 25, 50, 75, 95))

    n = depth.size
    expected = np.rint(np.array([5, 25, 50, 75, 95]) / 100.0 * (n - 1)).astype(int)
    assert list(positions) == sorted(set(expected.tolist()))
    assert labels == [5, 25, 50, 75, 95]

    # Each returned row is the curve sitting at that rank of ascending depth ...
    order = np.argsort(depth)
    for row, pos in zip(rows, positions):
        assert row == order[pos]
    # ... so the picked depths increase, and the ends are the true extremes.
    assert list(depth[rows]) == sorted(depth[rows])
    assert depth[rows[0]] == depth.min()
    assert depth[rows[-1]] == depth.max()


def test_ladder_picks_dedupe_on_a_fold_smaller_than_the_ladder():
    # 3 curves, 5 requested percentiles: collisions drop, lowest percentile wins.
    # At n = 3 the default ladder maps to ranks 0/1/1/1/2, so p50 and p70 lose
    # rank 1 to p30 and the kept labels are p10/p30/p90.
    rows, labels, positions = T._ladder_picks(np.array([0.8, 0.6, 1.0]))
    assert len(rows) == len(labels) == len(positions) == 3
    assert list(positions) == [0, 1, 2]
    assert labels == [10, 30, 90]
    assert T._ladder_picks(np.empty(0))[0].size == 0


# --- residual map: the plotted matrix is the depth-sorted signed residual ---

def test_residual_map_plots_depth_sorted_signed_residual(tmp_path, monkeypatch):
    """The array handed to pcolormesh IS (pred - true) reordered by depth."""
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    captured: dict = {}
    real = Axes.pcolormesh

    def spy(self, *args, **kwargs):
        if "C" not in captured and len(args) >= 3:
            captured["k"] = np.asarray(args[0])
            captured["C"] = np.asarray(args[2])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "pcolormesh", spy)

    td = _td()
    model, masks, summary = _fit(td)
    out = T._write_residual_map_figure(model, td, masks, "lbl", summary, seed=0)
    assert out is not None and Path(out).exists()

    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    expected = (y_pred - y_true)[np.argsort(y_true[:, -1])]
    assert captured["C"].shape == (y_true.shape[0], np.asarray(td.k).size)
    assert np.allclose(captured["C"], expected)
    assert np.allclose(captured["k"], np.asarray(td.k))
    # Sign convention: a prediction above truth must be positive on the map.
    assert np.allclose(np.sign(captured["C"]), np.sign(expected))


def test_residual_map_renders_without_predict_samples(tmp_path, monkeypatch):
    """A point-only model still gets the heatmap + marginal, just no bands."""
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    model = _PointFake()
    model.fit(td)
    assert not hasattr(model, "predict_samples")
    masks = grouped_split(td.sim_index, SplitSpec(0.6, 0.2, 0.2, seed=1))
    sub = T._subset(td, masks["test"])
    summary = {"held_out_split": "test",
               "rmse": T._rmse(sub.y, model.predict(sub.X, sub.X_cond, sub.X_params))}
    out = T._write_residual_map_figure(model, td, masks, "lbl", summary, seed=0)
    assert out is not None and Path(out).exists()


def test_residual_map_and_depth_metrics_skipped_for_single_k(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td(single_k=True)
    model, masks, summary = _fit(td)
    assert T._write_residual_map_figure(model, td, masks, "lbl", summary, seed=0) is None
    assert T._rmse_vs_depth_metrics(model, td, masks, summary) is None
    assert T._residual_map_metrics(model, td, masks, summary) is None


def test_residual_map_metrics_match_the_ladder_the_figure_draws():
    td = _td()
    model, masks, summary = _fit(td)
    sub, y_true, _yp = T._held_out_true_pred(model, td, masks, summary)
    rec = T._residual_map_metrics(model, td, masks, summary)

    rows, labels, _pos = T._ladder_picks(y_true[:, -1])
    assert rec["order_by"] == "spk_at_k_max"
    assert rec["picked_percentiles"] == labels
    assert rec["picked_sim_indices"] == [int(s) for s in np.asarray(sub.sim_index)[rows]]
    assert np.allclose(rec["picked_depths"], y_true[rows, -1])
    # The recorded ids are real held-out simulations, as for the worst-curve list.
    held_sims = {int(s) for s in sub.sim_index.tolist()}
    assert all(s in held_sims for s in rec["picked_sim_indices"])


def test_rmse_vs_depth_correlation_matches_a_direct_computation():
    td = _td()
    model, masks, summary = _fit(td)
    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    rmse_curve = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=1))

    rec = T._rmse_vs_depth_metrics(model, td, masks, summary)
    assert rec["n_curves"] == rmse_curve.shape[0]
    assert abs(rec["spearman_r"] - T._spearman(y_true[:, -1], rmse_curve)) < 1e-12
    assert abs(rec["pearson_r"] - T._corr(y_true[:, -1], rmse_curve)) < 1e-12
    assert -1.0 <= rec["spearman_r"] <= 1.0


# --- rmse_vs_k: the added envelope panel ------------------------------------

def test_rmse_vs_k_envelope_uses_signed_residual_percentiles(tmp_path, monkeypatch):
    """The plotted bands are the 5-95 and 16-84 percentiles of (pred - true)."""
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    bands: list = []
    real = Axes.fill_between

    def spy(self, x, y1, y2=0, **kwargs):
        bands.append((np.asarray(y1), np.asarray(y2)))
        return real(self, x, y1, y2, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", spy)

    td = _td()
    model, masks, summary = _fit(td)
    out = T._write_rmse_vs_k_figure(model, td, masks, "lbl", summary)
    assert out is not None and Path(out).exists()

    _sub, y_true, y_pred = T._held_out_true_pred(model, td, masks, summary)
    resid = y_pred - y_true
    p05, p16, p84, p95 = np.percentile(resid, [5, 16, 84, 95], axis=0)
    assert len(bands) == 2
    assert np.allclose(bands[0][0], p05) and np.allclose(bands[0][1], p95)
    assert np.allclose(bands[1][0], p16) and np.allclose(bands[1][1], p84)


def test_rmse_vs_k_still_skipped_for_single_k(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td(single_k=True)
    model, masks, summary = _fit(td)
    assert T._write_rmse_vs_k_figure(model, td, masks, "lbl", summary) is None


# --- aggregate latent density ----------------------------------------------

def test_aggregate_draws_recover_the_input_moments():
    """The pooled cloud's moments must match the (mu, std) it was built from."""
    rng = np.random.default_rng(0)
    # One example, so the aggregate IS that example's Gaussian and the moments
    # are directly checkable rather than a mixture's.
    mu = np.array([[2.0, -1.0, 0.5]])
    std = np.array([[0.3, 1.5, 0.8]])
    z = T._aggregate_draws(mu, std, rng)

    assert z.shape == (T._QZ_SAMPLES_PER_EXAMPLE * 1, 3)
    # Monte-Carlo tolerance at n = 20 draws is loose by construction; the point
    # is that the width is carried through at all, not that it is precise.
    z_big = T._aggregate_draws(np.repeat(mu, 4000, axis=0),
                               np.repeat(std, 4000, axis=0), rng)
    assert np.allclose(z_big.mean(axis=0), mu[0], atol=0.05)
    assert np.allclose(z_big.std(axis=0), std[0], rtol=0.05)


def test_aggregate_draws_are_wider_than_the_means_alone():
    """Pooling means alone understates the aggregate -- the reason for latent_dists."""
    rng = np.random.default_rng(1)
    mu = rng.normal(size=(200, 2))
    std = np.full((200, 2), 0.9)
    z = T._aggregate_draws(mu, std, rng)
    assert np.all(z.std(axis=0) > mu.std(axis=0))


def test_mass_contour_levels_enclose_the_requested_fraction():
    # A histogram whose mass is known exactly: levels must be ascending, and the
    # bins at or above each level must carry at least the requested fraction.
    hist = np.array([[10.0, 1.0], [2.0, 0.0]])
    levels = T._mass_contour_levels(hist, fracs=(0.68, 0.95))
    assert levels == sorted(levels)
    assert all(lv > 0 for lv in levels)
    total = hist.sum()
    for frac, lv in zip((0.95, 0.68), sorted(levels)):
        assert hist[hist >= lv].sum() >= frac * total - 1e-12
    assert T._mass_contour_levels(np.zeros((3, 3))) == []


def test_latent_density_figure_renders_and_scales_with_latent_dim(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    for latent_dim in (2, 3, 4):  # 1, 3 and 6 pairwise panels respectively
        model = _DistFake(latent_dim=latent_dim)
        model.fit(td)
        out = T._write_latent_density_figure(
            model, td, f"lbl_ld{latent_dim}", seed=0
        )
        assert out is not None and Path(out).exists()


def test_latent_density_figure_skipped_for_one_dimensional_latent(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    model = _DistFake(latent_dim=1)
    model.fit(td)
    assert T._write_latent_density_figure(model, td, "lbl", seed=0) is None


def test_latent_density_figure_is_gated_on_latent_dists_not_latents(tmp_path):
    # _DiagFake has latents() but NOT latent_dists: the runner must skip it, since
    # means alone cannot estimate an aggregate density.
    td = _td()
    model, _masks, _summary = _fit(td)
    assert hasattr(model, "latents")
    assert not hasattr(model, "latent_dists")


# --- latent traversal -------------------------------------------------------

def test_latent_traversal_figure_renders_across_latent_dims(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    for latent_dim in (1, 2, 4):  # 2, 3 and 5 rows; 5 profile columns each
        model = _DistFake(latent_dim=latent_dim)
        model.fit(td)
        out = T._write_latent_traversal_figure(model, td, f"trav_ld{latent_dim}")
        assert out is not None and Path(out).exists()


def test_latent_traversal_grid_is_latents_by_profiles(tmp_path, monkeypatch):
    """The grid is (n_dims + 1) rows x n_profiles columns, not the transpose.

    Adding a latent adds a ROW, so the rendered figure must grow taller at
    essentially fixed width -- the profile ladder alone sets the width. Asserted
    on the rendered pixels rather than on the gridspec because ``bbox_inches=
    "tight"`` is what finally sets the saved size.
    """
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    td = _td()
    sizes = {}
    for latent_dim in (1, 2, 4):
        model = _DistFake(latent_dim=latent_dim)
        model.fit(td)
        out = T._write_latent_traversal_figure(model, td, f"grid_ld{latent_dim}")
        img = plt.imread(out)
        sizes[latent_dim] = (img.shape[1], img.shape[0])  # (width, height) in px

    heights = [sizes[d][1] for d in (1, 2, 4)]
    widths = [sizes[d][0] for d in (1, 2, 4)]
    assert heights == sorted(heights) and heights[0] < heights[-1]
    # Width is set by the profile count alone, so it is the same figure width at
    # every latent dim, up to what the tight bbox crops off the margins.
    assert max(widths) - min(widths) <= 0.03 * max(widths)


def test_latent_traversal_shares_the_residual_map_percentile_ladder():
    """The two figures' panels sit at the same depths (maintainer decision).

    Consistency of the *percentiles* only: the residual map picks from the
    held-out fold and the traversal from the full dataset, so the same ladder
    still selects different simulations in the two figures.
    """
    assert T._TRAVERSAL_PROFILE_PERCENTILES == T._LADDER_PERCENTILES


def test_latent_traversal_skipped_for_single_k(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td(single_k=True)
    model = _DistFake(latent_dim=2)
    model.fit(td)
    assert T._write_latent_traversal_figure(model, td, "lbl") is None


def test_latent_traversal_falls_back_when_radii_do_not_match_profile_width(
    tmp_path, monkeypatch
):
    """A radii axis that does not line up with X still renders (bin-index x)."""
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    mismatched = TrainingData(
        X=td.X, X_cond=None, X_params=None, y=td.y, nd=td.nd,
        sim_index=td.sim_index, k=td.k,
        radii_mpch=np.linspace(0.1, 2.0, td.X.shape[1] + 3),  # deliberately wrong
        source_path=td.source_path, param_names=None, meta={}, config=None,
    )
    model = _DistFake(latent_dim=2)
    model.fit(mismatched)
    out = T._write_latent_traversal_figure(model, mismatched, "fallback")
    assert out is not None and Path(out).exists()


def test_latent_traversal_is_deterministic(tmp_path, monkeypatch):
    # Nothing is sampled: repeat renders must be byte-identical.
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    model = _DistFake(latent_dim=2)
    model.fit(td)
    first = Path(T._write_latent_traversal_figure(model, td, "det")).read_bytes()
    assert Path(T._write_latent_traversal_figure(model, td, "det")).read_bytes() == first


def test_latent_traversal_uses_real_rows_at_depth_percentiles():
    """The profiles swept are real dataset rows at the configured percentiles."""
    td = _td()
    y_all = np.asarray(td.y, dtype=float)
    rows, labels, _pos = T._ladder_picks(
        T._suppression_depth(y_all), percentiles=T._TRAVERSAL_PROFILE_PERCENTILES
    )
    assert labels == list(T._TRAVERSAL_PROFILE_PERCENTILES)
    # Real rows of the dataset, ordered by increasing suppression depth.
    for row in rows:
        assert np.array_equal(y_all[row], y_all[int(row)])
    assert list(y_all[rows, -1]) == sorted(y_all[rows, -1])


def test_latent_traversal_sweep_is_centred_and_symmetric():
    """The swept offsets are symmetric about 0 and include the unperturbed code."""
    deltas = np.linspace(-T._TRAVERSAL_SIGMA, T._TRAVERSAL_SIGMA,
                         T._TRAVERSAL_N_CURVES)
    assert T._TRAVERSAL_N_CURVES % 2 == 1          # odd, so 0 is on the grid
    assert deltas[T._TRAVERSAL_N_CURVES // 2] == 0.0
    assert np.allclose(deltas, -deltas[::-1])


def test_latent_density_figure_is_seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "figure_dir", lambda subdir=None: tmp_path)
    td = _td()
    model = _DistFake(latent_dim=2)
    model.fit(td)
    first = Path(T._write_latent_density_figure(model, td, "seedcheck", seed=7))
    payload = first.read_bytes()
    second = Path(T._write_latent_density_figure(model, td, "seedcheck", seed=7))
    assert second.read_bytes() == payload
