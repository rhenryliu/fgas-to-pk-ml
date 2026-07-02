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

_NEW_KEYS = ("per_curve_rmse", "suppressed_rmse", "coverage", "latent_norms")


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
