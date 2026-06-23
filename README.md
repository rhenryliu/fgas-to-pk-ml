# fgas-to-pk-ml

Tools for assembling **f_gas(R) → SP(k)** training data from CAMELS stacking
products, and for storing it under a consistent, self-describing scheme.

The scientific goal is to learn the mapping from projected gas-fraction profiles
`f_gas(R)` (the model-independent observable, built from ionized-gas ΔΣ over
total ΔΣ) to matter power-spectrum suppression `SP(k) = P_total(k) / P_DM(k)`,
using the CAMELS IllustrisTNG SB35 suite as the training set. This repo is the
machine-learning / data-assembly side; it consumes the per-simulation profile
products written by `SimulationStacker` and turns them into model-ready arrays.

## Contents

The library is the installed `fgas_spk` package (distribution name
`fgas-to-pk-ml`), laid out under `src/` and installed editable with
`pip install -e .`. It has two layers: a **data-assembly core**
(schema → builder → loader) that turns CAMELS stacking products into saved,
model-ready datasets, and a thin **training / experiment layer** (paths,
experiment, models, run_record, train) that takes a saved dataset through a
recorded model run.

**Data-assembly core** — imports only numpy + pyyaml:

| File | Purpose |
| --- | --- |
| `src/fgas_spk/schema.py` | **The on-disk data contract** — single source of truth: the `FgasSpkDataset` container, the filename/directory convention, `save_dataset`/`load_dataset`/`resolve_dataset_path`/`dataset_dir`/`ensure_store_dirs`, and `SCHEMA_VERSION`. |
| `src/fgas_spk/builder.py` | Builder: load profiles, fix the halo-ordering bug, assemble `(profile, nd) → SP(k)` arrays. Imports the contract from `fgas_spk.schema`. |
| `src/fgas_spk/loader.py` | Configurable training-data loader: read a saved dataset and serve model-ready numpy arrays per a `DataConfig`. Importable by training scripts and runnable as a CLI dry-run. numpy + pyyaml only (no torch). |
| `src/fgas_spk/__init__.py` | Package init: re-exports the common API (`FgasSpkDataset`, `save_dataset`/`load_dataset`, `build_fgas_spk_dataset`, `DataConfig`/`load_training_data`, `RunConfig`/`SplitSpec`/`load_configs`, …) as a flat `fgas_spk.*` surface. |

**Training / experiment layer** — opt-in; the models and runner pull
scikit-learn, joblib (and lazily torch/matplotlib) from the pinned environment:

| File | Purpose |
| --- | --- |
| `src/fgas_spk/paths.py` | Machine- and layout-aware path resolution: `resolve_data_root` (reads — defaults to the repo), `resolve_scratch_root` (writes — no default; set `$FGAS_SCRATCH_ROOT`), `repo_root`, `figure_dir`, and the `fill_data_root` entry-point helper. No hardcoded paths, no filesystem detection. |
| `src/fgas_spk/experiment.py` | Trainer-side `RunConfig` (model + training fields only — no data selection), the grouped-by-`sim_index` `SplitSpec`, and `load_configs` (reads the data YAML and run YAML independently — no merge). |
| `src/fgas_spk/models/` | Model-plugin layer: the `ProfileToSpk` interface, the name→class `REGISTRY`, and the `register` decorator (`base.py`), plus the `pca_linear` reference plugin. Importing the package runs each plugin's `@register` for its side effect. |
| `src/fgas_spk/run_record.py` | Git-tracked text record + ledger for a run: writes `experiments/runs/<run_id>/` and appends to `experiments/runs.jsonl`. Never imports torch (the device string is passed in). |
| `src/fgas_spk/train.py` | End-to-end runner: a `(DataConfig, RunConfig)` pair → read-root fill → grouped split → fit a model from the registry → write the run record, checkpoint to scratch, and a diagnostic figure. |

**Backup and provenance** (not maintained):

| File | Purpose |
| --- | --- |
| `src/_frozen/fgas_spk_dataset_v0.py` | Frozen pre-refactor backup of the loader, quarantined outside the package (no `__init__.py`). Do not extend; kept only because callers may still source it. |
| `notebooks/test_CAMELS_sixth_gen.ipynb` | Reference notebook (collaborator-authored) the loader was distilled from. Kept for provenance; not the source of truth. |
| `notebooks/test_loader.ipynb` | Scratch notebook exercising `fgas_spk.loader`. Not authoritative. |

Configs, docs, and the small text run records live **here**, in this git-tracked
repo. Large datasets and model checkpoints live separately on scratch; generated
figures go to a gitignored `figures/` tree (see *Data store*).

### Installing and importing

Install the package editable once per environment (see *Environment* below for
the full pinned dependency set):

```bash
pip install -e .
```

Then import it as the `fgas_spk` package from anywhere — no `sys.path` shimming:

```python
import fgas_spk as F                            # flat API: F.build_fgas_spk_dataset, F.save_dataset, ...
from fgas_spk import schema, builder, loader    # or reach into the submodules
```

- **Tests** import `fgas_spk.*`; the root `conftest.py` also keeps `src/` on
  `sys.path`, so they run even before the editable install and the frozen
  backup under `src/_frozen/` stays sourceable.
- **The loader CLI** runs as a module: `python -m fgas_spk.loader --config
  scripts/configs/data/config.yaml`.

## Requirements

The package is **tiered** by import weight:

- **Core + flat API** (`import fgas_spk`, and `fgas_spk.schema` / `builder` /
  `loader` / `experiment`) — Python ≥ 3.10, `numpy`, and `pyyaml`. `pyyaml` is
  optional for `fgas_spk.schema`/`fgas_spk.builder` (without it the sidecar
  manifest is written as JSON instead) but **required** for `fgas_spk.loader` and
  `fgas_spk.experiment`, which read and write config YAML. Importing the
  top-level package stays at this weight — it does **not** pull the training
  submodules.
- **Training / experiment layer** (opt-in, imported explicitly) —
  `fgas_spk.models` and `fgas_spk.train` additionally need **scikit-learn** and
  **joblib** (the PCA reference model and the checkpoint writer), and `train`
  imports **torch** and **matplotlib** lazily (device pick + diagnostic figure),
  treating both as optional. `fgas_spk.run_record` stays light (numpy + stdlib).

Only the core's two dependencies are declared in `pyproject.toml` (unpinned
floors); the exact, cross-platform pins — including the training layer's
scikit-learn / joblib and the wider cosmology/ML stack (pyccl, colossus,
SP(k)/BCemu, torch/sbi, …) — live in `requirements.txt` / `environment.yml`.
Keeping them out of the package metadata is deliberate: the data-assembly core
must stay installable with just numpy + pyyaml, while the heavier stack serves
the training, gate, and analysis scripts.

## Environment

`environment.yml` and `requirements.txt` (repo root) define one pinned environment
that installs matched package versions on both target platforms — NERSC Perlmutter
(Linux x86_64, CUDA) and an Apple-Silicon Mac (macOS arm64, MPS). Python is 3.12
(3.11 also works; numpy 2.4 sets the ≥3.11 floor). Almost everything installs from
pre-built wheels; the one compile-from-source exception is noted below.

**Per-machine roots.** `fgas_spk.paths` resolves where data is read and written:
export `$FGAS_SCRATCH_ROOT` once per machine for run outputs (writes raise a clear
error if it is unset — the code never guesses a writable location), and optionally
`$FGAS_DATA_ROOT` for reads (it defaults to the repo, since the datasets are
git-tracked).

Two ways to create it:

```bash
# Route A — environment.yml (its pip: block installs requirements.txt for you):
conda env create -f environment.yml
conda activate fgas-ml

# Route B — bare conda env, then install deps explicitly:
conda create -n fgas-ml -c conda-forge python=3.12
conda activate fgas-ml
pip install -r requirements.txt
```

Then register the env as a Jupyter kernel on each machine:

```bash
python -m ipykernel install --user --name fgas-ml --display-name "fgas-ml"
```

Cross-platform rules:

- **Do not** regenerate `requirements.txt` with `pip freeze`. A freeze on Perlmutter
  pins the `nvidia-*` CUDA stack (no arm64-mac wheels); a freeze on the Mac pins a
  CPU/MPS torch (no CUDA on NERSC). The file pins top-level packages only, on purpose.
- Keep `torch` a clean upstream pin — no `+cuXXX` local version, no `--index-url` in
  the shared file. PyPI serves the CUDA wheel on Linux and the MPS wheel on macOS from
  the same version string.
- Select the compute device at runtime so the code is identical on every machine:

```python
  import torch

  def pick_device() -> torch.device:
      """Return the best available device (CUDA on NERSC, MPS on Apple Silicon, else CPU)."""
      if torch.cuda.is_available():
          return torch.device("cuda")
      if torch.backends.mps.is_available():
          return torch.device("mps")
      return torch.device("cpu")
```

- **Pylians is NERSC-side and commented out by default.** Its released sdist mis-quotes
  the macOS OpenMP flag and fails to build on Apple Silicon (even with `libomp`); it
  builds cleanly on Linux/gcc. Uncomment and `pip install` it on NERSC when you need
  CAMELS field/P(k) tools. `pypower` (installed by default, pure-Python wheel) covers
  P(k) estimation in the meantime. When adopting compile-from-source packages, keep them
  in a separate `requirements-nersc.txt` so the base file stays wheel-only and
  cross-platform.

For lockfile-grade reproducibility across machines — and the future ARM-Linux NERSC-10
("Doudna") system, where pyccl currently has no aarch64 wheel and would come from
conda-forge — generate a multi-platform lock with `uv` or `conda-lock` rather than
relying on these top-level pins.

## Quickstart

```python
from pathlib import Path
import fgas_spk as F

DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
SNAPSHOT, REDSHIFT = 74, 0.47        # snap82 -> z=0.21

# Correct path: rebuild number-density bins from raw per-halo profiles,
# re-ranking halos by mass so the bins are the N most massive distinct halos,
# all projections averaged per halo (see Caveats).
dataset = F.build_fgas_spk_dataset(
    base_path_template=BASE,
    suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
    snapshot=SNAPSHOT,
    rank_key="halo_mass",                              # M_500c ranking
    params_path=Path(DATA_DIR) / "camels_params_matrix.npy",   # optional
)

# Quick in-memory check (NOT the training path): flatten the (sim, nd) grid to
# arrays. This convenience emits a UserWarning — it has no subsetting, radial
# crop, or X_cond/X_params split. For real training, save the dataset and load it
# via fgas_spk.loader (below), then run it through fgas_spk.train (see "Running a
# model"). k_target picks SP(k) at one wavenumber; omit it for the full curve.
X, nd, y = dataset.to_training_arrays(k_target=3.0)
# X:  (n_examples, n_radii)   f_gas profiles
# nd: (n_examples,)           number-density conditioning, (Mpc/h)^-3
# y:  (n_examples,)           SP(k≈3 h/Mpc) per example; omit k_target for the
#                             full SP(k) curve, shape (n_examples, n_k)

# Save into the scratch store under a consistent, self-describing name.
PROJECT = "/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml"
F.ensure_store_dirs(PROJECT)
out = F.save_dataset(
    dataset, project_root=PROJECT,
    suite="CAMELS-IllustrisTNG-L50n512-SB35", snapshot=SNAPSHOT, redshift=REDSHIFT,
    source="lindajin",
)
reloaded = F.load_dataset(out)
```

For quick iteration on the pre-aggregated ratio files instead of raw profiles,
use the fast path (see Caveats — it does **not** re-rank):

```python
fast = F.build_fgas_spk_from_compiled(
    data_dir=DATA_DIR,
    suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
    snapshot=SNAPSHOT,
    name_extra="_sixth_gen_makeMap",   # or whatever the files on disk are stamped
)
```

## API

`FgasSpkDataset` — dataclass holding an assembled dataset:
`radii_mpch (n_radii,)`, `number_densities (n_nd,)`, `fgas (n_sims, n_nd, n_radii)`,
`fgas_std`, `k (n_k,)`, `suppression (n_sims, n_k)`, `sim_ids (n_sims,)`,
`mean_halo_mass (n_sims, n_nd)`, `rank_key`, optional `camels_params (n_sims, n_params)`,
and optional `snapshot` (the snapshot the dataset was built for, e.g. 74 → z=0.47;
populated by the loaders and carried through `save_dataset`/`load_dataset`).
Method: `to_training_arrays(k_target=None) -> (X, nd, y)` — an in-memory
quick-check that emits a `UserWarning`; use `fgas_spk.loader` for training.

| Function | What it does |
| --- | --- |
| `build_fgas_spk_dataset(base_path_template, suppression_path, snapshot, ...)` | Build a dataset from raw per-simulation profile files for a given snapshot; re-ranks halo columns by `rank_key` before the number-density cut. **Preferred for science.** |
| `build_fgas_spk_from_compiled(data_dir, suppression_path, snapshot, ...)` | Fast path: load pre-aggregated per-nd ratio files for a given snapshot. Inherits whatever binning was frozen at production; cannot re-rank. |
| `save_dataset(dataset, project_root, suite, snapshot, redshift, source, ...)` | Write a `.npz` (metadata embedded under `__meta__`) plus a human-readable `.yaml` sidecar. Refuses to overwrite by default. |
| `load_dataset(npz_path)` | Reconstruct a `FgasSpkDataset` from a saved `.npz`. |
| `dataset_dir(project_root, suite, snapshot, redshift, source)` | Canonical directory for a provenance bucket. |
| `resolve_dataset_path(project_root, suite, snapshot, redshift, source, rank, tag)` | Build a saved `.npz` path from its fields; `tag="latest"` auto-selects the newest (max date or max `vN`; mixed tags raise). Prefer explicit tags for reproducibility. |
| `ensure_store_dirs(project_root, make_models=True)` | Create `datasets/` and `models/` in the scratch store. |

`save_dataset`, `load_dataset`, `dataset_dir`, `resolve_dataset_path`,
`ensure_store_dirs`, and `FgasSpkDataset` are defined in `fgas_spk.schema`,
re-exported from `fgas_spk.builder` for backward compatibility, and also exposed
on the top-level `fgas_spk` flat API.

Both builders take a required `snapshot` argument (e.g. 74 → z=0.47, 82 → z=0.21),
which threads into the on-disk filenames. Pass the matching
`Ptot_Pdm_ratio_snap{NN}.npz` as `suppression_path`.

### Training-data loader (`fgas_spk.loader`)

`fgas_spk.loader` reads a saved dataset (via `fgas_spk.schema`) and serves
model-ready numpy arrays per a `DataConfig`. It does **no** preprocessing,
normalisation, or splitting — those belong to the training script — and never
imports a deep-learning framework.

```python
from fgas_spk import loader as L

cfg = L.DataConfig(
    project_root=PROJECT, suite="CAMELS-IllustrisTNG-L50n512-SB35",
    snapshot=74, redshift=0.47, source="lindajin", rank="m500", tag="latest",
    number_density_indices=[0, 2, 4],   # subset of the five nds (None = all)
    radial_range_mpch=(0.3, 2.5),       # crop profile radii (None = all)
    include_nd_feature=True,            # number density -> X_cond
    include_camels_params=True,         # CAMELS params -> X_params (on by default)
    target_mode="curve",               # or "single_k" (+k_target) / "k_range" (+k_range)
)
td = L.load_training_data(cfg)
# td.y, td.nd, td.sim_index, td.k, td.radii_mpch, td.source_path, td.meta, td.config
```

Inputs are served as **separate modalities** rather than one concatenated array,
so a multimodal model can route each branch independently (a single-input model
can concatenate them itself):

| Attribute | Shape | Contents | `None` when |
| --- | --- | --- | --- |
| `td.X` | `(n_examples, n_radii_sel)` | the `f_gas(R)` profile alone (columns are `td.radii_mpch`) | — |
| `td.X_cond` | `(n_examples, n_cond)` | observable-derived scalars: number density (if `include_nd_feature`) then mean `M_500c` (if `include_mean_halo_mass`) | no conditioning feature requested |
| `td.X_params` | `(n_examples, n_params)` | the simulation's CAMELS parameters, aligned to `td.sim_index` | `include_camels_params=False` |

`include_camels_params` defaults to `True`; with it on, a dataset that carries no
CAMELS params **raises** rather than silently dropping them — set it `False` to
load a param-less dataset.

Each row is one `(simulation, number-density)` pair. `td.sim_index` gives the
originating simulation id per row so a downstream script can **split by
simulation** (splitting by row would leak a simulation's shared `SP(k)` target
across folds). Dry-run summary from the CLI:

```
python -m fgas_spk.loader --config scripts/configs/data/config.yaml
```

A full example `DataConfig` YAML is in the `fgas_spk.loader` module docstring.

## Running a model

The training / experiment layer takes a saved dataset through to a recorded run.
A run is configured by **two independent YAMLs** — the data selection
(`DataConfig`, above) and the model/training recipe (`RunConfig`) — which
`load_configs` reads separately and never merges. The split is deliberate: the
read root lives in the `DataConfig` (its `project_root`), the write root lives in
the `RunConfig` (its `write_root`), and nothing is copied between them.

### Configuring a run (`RunConfig`)

`RunConfig` (`fgas_spk.experiment`) owns **only** model and training concerns; it
carries no data-selection fields. Its only path-like field is `write_root` (the
scratch-root override for run outputs).

| Field | Meaning |
| --- | --- |
| `model` | Registry name (validated at run time against `fgas_spk.models.REGISTRY`, not at config-load). |
| `model_params` | Free-form hyperparameter mapping passed to the model constructor as `**kwargs`. **All** model hyperparameters live here, including the iterative-training knobs (`epochs`, `batch_size`, learning rate, weight decay, ...) for models that train iteratively — each model owns and documents its own (e.g. the MLP reads `epochs`/`lr`/`batch_size`/`weight_decay`; the PCA reference reads `n_components`). Must be a mapping; `None` is coerced to `{}`. |
| `seed` | Global reproducibility seed (model init, shuffles). |
| `split` | A `SplitSpec`: `train_frac`/`val_frac`/`test_frac` (must lie in `[0, 1]`, sum to 1, with `train_frac > 0`) and a `seed`. `val_frac = 0` gives a two-way train/test split. A plain YAML mapping here is coerced to a `SplitSpec`. |
| `write_root` | Scratch-root override; `null` defers to `$FGAS_SCRATCH_ROOT` (or `--scratch-root`). |

`RunConfig` mirrors `DataConfig`'s style — `from_yaml`/`to_yaml` round-trip and
`__post_init__` validation — so the resolved config can be re-serialized verbatim
into the run record. The bundled `scripts/configs/run/pca_reference.yaml`:

```yaml
model: pca_linear          # registry name (fgas_spk.models.REGISTRY)
model_params:
  n_components: 8          # PCA components of the f_gas profile (capped at fit time)
seed: 0                    # threaded into PCA's random_state
split:                     # grouped-by-sim_index; val_frac: 0.0 => two-way train/test
  train_frac: 0.8
  val_frac: 0.0
  test_frac: 0.2
  seed: 0
write_root: null           # defers to $FGAS_SCRATCH_ROOT (or --scratch-root)
```

The split is **always grouped by `sim_index`** — whole simulations go to one fold,
never split by `(sim, nd)` row — so a simulation's shared SP(k) target cannot leak
across folds. `grouped_split` seeds the shuffle from `SplitSpec.seed`, so the
partition is reproducible.

### Models (`fgas_spk.models`)

A model is any object satisfying the `ProfileToSpk` interface — `fit(training_data)`
and `predict(X, X_cond=None, X_params=None)`; it need not subclass anything.
Plugins register themselves under a name with the `@register("name")` decorator;
importing `fgas_spk.models` imports the plugin modules so those decorators run and
populate `REGISTRY` (name → class). The runner looks a model up with
`REGISTRY[run_config.model]` and builds it as
`Model(seed=run_config.seed, **run_config.model_params)`.

The bundled reference plugin is `pca_linear` (`PcaLinear`): PCA-compress the
profile `X`, concatenate the conditioning scalars `X_cond` alongside the
components, then linear-map to the SP(k) target with an ordinary least-squares
regressor. It is a worked **multimodal** example to copy, not a tuned baseline (a
one-line change at the feature-assembly site makes it profile-only — see its
module docstring; `X_params` is ignored by this reference). To add a model: write
a plugin against `ProfileToSpk`, decorate it with `@register`, and add its import
to `fgas_spk/models/__init__.py` (a registry that never imports its plugins is
silently empty).

### The runner (`fgas_spk.train`)

`run_training(data_config, run_config, ...)` ties the package together, in order:

1. **Read-root fill** — if the data config is in store-field mode with
   `project_root` null, fill it from `resolve_data_root` so the loader receives a
   fully-specified config (a path-mode config is left untouched).
2. **Write root** — resolve via `resolve_scratch_root`.
3. **Load** the arrays via `load_training_data` (records the resolved `.npz` on
   `td.source_path`).
4. **Split** grouped by `sim_index` per `RunConfig.split`.
5. **Fit** the model — `REGISTRY[run_config.model](seed=run_config.seed,
   **run_config.model_params)`, `.fit(train_td)` — and **evaluate** the held-out
   fold.
6. **Record + persist** — write the run record, checkpoint the model to
   `<scratch_root>/models/<run_id>/model.joblib` (via `joblib`), and write a
   predicted-vs-true figure (run id in the filename, `write_figures=True`).

It returns a `RunResult` (`run_id`, `run_dir`, `summary`, `checkpoint_path`,
`split_masks`) and raises `KeyError` if `RunConfig.model` is not registered.

**Root resolution precedence** (the two roots are independent):

- *Read root* — `data_root_override` (`--data-root`) → `DataConfig.project_root`
  (if set) → `$FGAS_DATA_ROOT` → the repo. The override and env var only take
  effect when `project_root` is left null in store-field mode.
- *Write root* — `scratch_root_override` (`--scratch-root`) → `RunConfig.write_root`
  → `$FGAS_SCRATCH_ROOT`. There is **no** default: writes raise a clear error if
  none is set, rather than guessing a writable location.

The device string is resolved by `pick_device` (`cuda` → `mps` → `cpu`, treating
torch as optional) and recorded for provenance; the runner itself is numpy/sklearn
and does no GPU work.

**`summary`** (also written to `summary.json` and carried in the ledger) holds
`n_train`/`n_val`/`n_test` (row counts), `train_rmse`, `val_rmse` and `test_rmse`
(only for folds that have rows), the chosen `held_out_split` (preference:
`test` → `val` → `train`), and the headline `rmse` for that fold.

A minimal programmatic run:

```python
from fgas_spk import load_configs
from fgas_spk.train import run_training

data_cfg, run_cfg = load_configs(
    "scripts/configs/data/config.yaml",
    "scripts/configs/run/pca_reference.yaml",
)
result = run_training(data_cfg, run_cfg)   # roots from the configs / env vars
print(result.run_id, result.summary["held_out_split"], result.summary["rmse"])
```

### The CLI

Two scripts drive the runner:

```bash
# Generic: any (data config, run config) pair.
python scripts/run.py --config-data scripts/configs/data/config.yaml \
                      --config-run  scripts/configs/run/pca_reference.yaml \
                      [--data-root PATH] [--scratch-root PATH]

# Worked example: the PCA reference against the bundled configs.
python scripts/run_pca_demo.py [--data-root PATH] [--scratch-root PATH]
```

Both print the `run_id`, the run-record directory, the held-out split, and its
RMSE (`run_pca_demo.py` also prints the checkpoint path). `--data-root` fills the
read root only when the data config leaves `project_root` null; `--scratch-root`
overrides the write root (else `RunConfig.write_root`, else `$FGAS_SCRATCH_ROOT`).
`scripts/demo.sh` is a SLURM wrapper that runs `run_pca_demo.py` on a Perlmutter
GPU node.

### What a run writes

One run touches all three storage locations (see *Data store* for the full
layout):

- **Git-tracked** `experiments/runs/<run_id>/` (`config_data.yaml`,
  `config_run.yaml`, `env.json`, `metrics.jsonl`, `summary.json`) plus one line
  appended to `experiments/runs.jsonl`.
- **Scratch** `<scratch_root>/models/<run_id>/model.joblib` — the checkpoint.
- **Gitignored** `figures/<YYYY-MM>/<MM-DD>/<run_id>__pred_vs_true.pdf` — the
  diagnostic figure.

## Data store

Outputs land in **three** places, split by size and by whether they are
provenance:

1. **Scratch store** (`$FGAS_SCRATCH_ROOT`, data-only — no configs/docs) holds
   the large artifacts: saved datasets and model checkpoints.
2. **Git-tracked `experiments/`** holds the small, greppable text record of every
   run.
3. **Gitignored `figures/`** holds generated diagnostic figures.

### Scratch store (datasets + checkpoints)

```
<scratch_root>/                       # $FGAS_SCRATCH_ROOT, e.g. /pscratch/sd/r/rhliu/projects/fgas-to-pk-ml
├── datasets/
│   └── <suite>/snap<NNN>_z<z>/src-<source>/
│       ├── fgas_spk__<suite>__snap<NNN>_z<z>__src-<source>__rank-<rank>__<tag>.npz
│       └── fgas_spk__...__<tag>.yaml          # human-readable manifest
└── models/
    └── <run_id>/model.joblib                   # one checkpoint dir per run
```

The dataset path nests by the axes that define a dataset — **suite →
snapshot/redshift → source** — so products from different producers (e.g.
`src-lindajin` vs `src-rhliu`) sit side by side and never collide. The leaf
filename is fully self-describing, so a file stays identifiable if copied out of
the tree:

- `<source>` — who produced the underlying profiles (`lindajin`, `rhliu`, …).
- `<rank>` — the halo ordering used for the number-density cut: `m500`
  (M_500c), `mstar` (stellar mass), or `frozen` (whatever a compiled file baked in).
- `<tag>` — a date (`YYYYMMDD`, default) or explicit version (`v1`, `v2`, …).

Each `.npz` embeds its full metadata under `__meta__`; the `.yaml` sidecar is the
same metadata in human-readable form.

### Run records (git-tracked `experiments/`)

Every run writes a small text record under the repo, plus one ledger line:

```
<repo>/experiments/
├── runs/<run_id>/
│   ├── config_data.yaml     # the resolved DataConfig actually used
│   ├── config_run.yaml      # the resolved RunConfig actually used
│   ├── env.json             # git sha + dirty flag, seed, device, roots, dataset
│   │                        #   __meta__, suppression-target definition, pkg versions
│   ├── metrics.jsonl        # per-epoch rows (empty for non-iterative models)
│   └── summary.json         # headline metrics (split sizes, per-split + held-out RMSE)
└── runs.jsonl               # one greppable line per run (the ledger)
```

A run is identified by

```
run_id = "<UTC-timestamp>__<config-hash8>__<git-sha7>"
```

where the config hash is a stable hash over the resolved `(DataConfig, RunConfig)`
pair and the git sha is the current `HEAD` (with a dirty flag when the tree has
uncommitted or untracked changes). The record captures the suppression-target
definition verbatim (`run_record.DEFAULT_SUPPRESSION_DEFINITION`) so runs built
against different target definitions are never silently compared (see *Caveats*).

### Figures (gitignored `figures/`)

`fgas_spk.paths.figure_dir` returns a dated directory
`<repo>/figures/<YYYY-MM>/<MM-DD>/`; the run id goes in the **filename** (e.g.
`<run_id>__pred_vs_true.pdf`), not a subfolder. `figures/` is gitignored — figures
are regenerated, not version-controlled.

## Caveats (read before trusting a dataset)

- **Halo-ordering fix.** The upstream CAMELS-fork selection returns halos in FoF
  catalogue order, not sorted by the selection proxy. `build_fgas_spk_dataset`
  re-ranks halos by `rank_key` (default `halo_mass` = M_500c) before slicing the
  top-N, so number-density bins are genuinely the N most massive **distinct**
  halos (each halo's three projections are averaged before the bin statistics,
  so `N = int(nd · V)` counts distinct halos, not halo-projection samples). The
  compiled fast path cannot do this — it inherits the frozen ordering.
- **`rank_key="stellar"` is not yet wired.** The producer does not save
  `SubhaloMStar`, so stellar-mass ranking (the observationally-matched choice)
  requires a catalogue re-read and currently raises. `halo_mass` (M_500c) is the
  working default.
- **Provenance guard.** The loader requires the three-profile products
  (it checks for `prof2_ionized_gas_*`). It deliberately fails on older
  generations rather than silently producing mislabelled output.
- **Snapshots.** Data exists for snap74 (z=0.47, the DESI LRG bin-1 match) and
  snap82 (z=0.21). The `snapshot` argument threads into every on-disk filename —
  the per-sim profiles (`..._snap{NN}.npz`), the compiled ratio files
  (`..._profiles_snap{NN}_nd_{i}_n_{N}.npz`), and the suppression file
  (`Ptot_Pdm_ratio_snap{NN}.npz`). Pass a `suppression_path` that matches `snapshot`.
- **Suppression target.** `SP(k)` here is `P_total / P_DM` computed *within* the
  hydro run (the DM component as a DMO proxy), **not** the paired-DMO
  `P_hydro / P_DMO`. This is a known approximation; it is small at low k but
  grows and becomes feedback-correlated at k ≳ 5 h/Mpc. Treat current `suppression`
  values accordingly until regenerated against the matched CAMELS N-body runs.
- **Units.** Radii are comoving Mpc/h, `k` is h/Mpc, `mean_halo_mass` is M_500c in
  M⊙/h, `fgas` and `suppression` are dimensionless. CAMELS cosmology varies per
  simulation (Sobol set); per-sim `f_b`/`h` live in the source profiles, not in
  the assembled dataset.

## Provenance of current data

Datasets built today derive from profiles in a collaborator's scratch (the
`Klinjin/SimulationStacker` fork). They are intended to be regenerated from
`rhliu/SimulationStacker` and re-saved under `src-rhliu`. Record the source repo,
commit, and producer script via the `provenance=` argument to `save_dataset`.