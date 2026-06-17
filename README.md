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

The library modules live under `src/` as loose top-level modules (not an
installed package).

| File | Purpose |
| --- | --- |
| `src/fgas_spk_schema.py` | **The on-disk data contract** — single source of truth: the `FgasSpkDataset` container, the filename/directory convention, `save_dataset`/`load_dataset`/`resolve_dataset_path`, and `SCHEMA_VERSION`. |
| `src/fgas_spk_builder.py` | Builder: load profiles, fix the halo-ordering bug, assemble `(profile, nd) → SP(k)` arrays. Imports the contract from `fgas_spk_schema`. |
| `src/fgas_spk_loader.py` | Configurable training-data loader: read a saved dataset and serve model-ready numpy arrays per a `DataConfig`. Importable by training scripts and runnable as a CLI dry-run. numpy + pyyaml only (no torch). |
| `src/fgas_spk_dataset_v0.py` | Frozen pre-refactor backup of the loader. Do not extend; kept only because callers may still source it. |
| `notebooks/test_CAMELS_sixth_gen.ipynb` | Reference notebook (collaborator-authored) the loader was distilled from. Kept for provenance; not the source of truth. |

Configs, figures, and documentation live **here**, in this git-tracked repo.
Large data and model checkpoints live separately on scratch (see *Data store*).

### Importing

The modules are imported by their flat names (`import fgas_spk_builder`,
`import fgas_spk_schema`, `import fgas_spk_loader`), so `src/` must be on
`sys.path`:

- **Tests** get it automatically via the root `conftest.py`.
- **Scripts** can either run a module directly — `python src/fgas_spk_loader.py
  --config config.yaml` puts `src/` on the path for free — or set
  `PYTHONPATH=src` (e.g. `PYTHONPATH=src python my_train.py`).

## Requirements

The library itself (the `src/` modules) imports only:

- Python ≥ 3.10
- `numpy`
- `pyyaml` — optional for `fgas_spk_schema`/`fgas_spk_builder` (without it the
  sidecar manifest is written as JSON instead), but **required** for
  `fgas_spk_loader`, which reads and writes `DataConfig` YAML.

The cosmology and ML dependencies (pyccl, colossus, SP(k)/BCemu, scikit-learn,
torch/sbi, …) are **not** part of the library's import surface — they live in the
pinned environment files below, for the gate and training scripts.

## Environment

`environment.yml` and `requirements.txt` define one exact-pinned environment that
installs identically on both target platforms — NERSC Perlmutter (Linux x86_64,
CUDA) and an Apple-Silicon Mac (macOS arm64, MPS) — entirely from pre-built wheels
(nothing compiles from source on either). The exact `==` pins are intentional:
this is an *environment* spec meant to reproduce a known-good setup across
machines, not a library dependency declaration. If this ever becomes an installed
package, its `install_requires` should use floors, not these pins.

```bash
conda env create -f environment.yml     # Python 3.12 from conda-forge + pip deps
conda activate fgas-ml
pip install -r requirements.txt
python -m ipykernel install --user --name fgas-ml --display-name "fgas-ml"
```

Cross-platform notes:

- **Do not** regenerate `requirements.txt` with `pip freeze`. A freeze on
  Perlmutter pins the `nvidia-*` CUDA stack (no arm64-mac wheels); a freeze on the
  Mac pins a CPU/MPS torch (no CUDA on NERSC). Pin top-level packages only, as the
  file does.
- Keep `torch` as a clean upstream pin — no `+cuXXX` local version, no
  `--index-url` in the shared file. PyPI then serves the CUDA wheel on Linux and
  the MPS wheel on macOS from the same version string.
- Choose the compute device at runtime so the code is identical on every machine:

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

- The pinned env requires Python ≥ 3.12 (driven by numpy 2.4, not by the code,
  which runs on ≥ 3.10). For lockfile-grade reproducibility across machines — and
  the future ARM-Linux NERSC-10 ("Doudna") system, where pyccl currently lacks an
  aarch64 wheel and would come from conda-forge — generate a multi-platform lock
  with `uv` or `conda-lock` rather than relying on these top-level pins.

## Quickstart

```python
from pathlib import Path
import fgas_spk_builder as F

DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
SNAPSHOT, REDSHIFT = 74, 0.47        # snap82 -> z=0.21

# Correct path: rebuild number-density bins from raw per-halo profiles,
# re-ranking halos by mass so the bins are the N most massive (see Caveats).
dataset = F.build_fgas_spk_dataset(
    base_path_template=BASE,
    suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
    snapshot=SNAPSHOT,
    rank_key="halo_mass",                              # M_500c ranking
    params_path=Path(DATA_DIR) / "camels_params_matrix.npy",   # optional
)

# Flatten to model-ready arrays. k_target picks SP(k) at one wavenumber;
# omit it to keep the full SP(k) curve as the target.
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
Method: `to_training_arrays(k_target=None) -> (X, nd, y)`.

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
`ensure_store_dirs`, and `FgasSpkDataset` are defined in `fgas_spk_schema` and
re-exported from `fgas_spk_builder` for convenience.

Both builders take a required `snapshot` argument (e.g. 74 → z=0.47, 82 → z=0.21),
which threads into the on-disk filenames. Pass the matching
`Ptot_Pdm_ratio_snap{NN}.npz` as `suppression_path`.

### Training-data loader (`fgas_spk_loader`)

`fgas_spk_loader` reads a saved dataset (via `fgas_spk_schema`) and serves
model-ready numpy arrays per a `DataConfig`. It does **no** preprocessing,
normalisation, or splitting — those belong to the training script — and never
imports a deep-learning framework.

```python
import fgas_spk_loader as L

cfg = L.DataConfig(
    project_root=PROJECT, suite="CAMELS-IllustrisTNG-L50n512-SB35",
    snapshot=74, redshift=0.47, source="lindajin", rank="m500", tag="latest",
    number_density_indices=[0, 2, 4],   # subset of the five nds (None = all)
    radial_range_mpch=(0.3, 2.5),       # crop profile radii (None = all)
    include_nd_feature=True,            # append the number density as a feature
    target_mode="curve",               # or "single_k" (+k_target) / "k_range" (+k_range)
)
td = L.load_training_data(cfg)
# td.X (n_examples, n_features), td.y, td.nd, td.sim_index, td.k, td.radii_mpch,
# td.source_path, td.meta, td.config
```

Each row is one `(simulation, number-density)` pair. `td.sim_index` gives the
originating simulation id per row so a downstream script can **split by
simulation** (splitting by row would leak a simulation's shared `SP(k)` target
across folds). Dry-run summary from the CLI:

```
python src/fgas_spk_loader.py --config config.yaml
```

A full example `DataConfig` YAML is in the `fgas_spk_loader` module docstring.

## Data store

The script writes to a **data-only** scratch store (no configs/figures/docs):

```
<project_root>/                       # e.g. /pscratch/sd/r/rhliu/projects/fgas-to-pk-ml
├── datasets/
│   └── <suite>/snap<NNN>_z<z>/src-<source>/
│       ├── fgas_spk__<suite>__snap<NNN>_z<z>__src-<source>__rank-<rank>__<tag>.npz
│       └── fgas_spk__...__<tag>.yaml          # human-readable manifest
└── models/                                     # large checkpoints only
```

The path nests by the axes that define a dataset — **suite → snapshot/redshift →
source** — so products from different producers (e.g. `src-lindajin` vs
`src-rhliu`) sit side by side and never collide. The leaf filename is fully
self-describing, so a file stays identifiable if copied out of the tree:

- `<source>` — who produced the underlying profiles (`lindajin`, `rhliu`, …).
- `<rank>` — the halo ordering used for the number-density cut: `m500`
  (M_500c), `mstar` (stellar mass), or `frozen` (whatever a compiled file baked in).
- `<tag>` — a date (`YYYYMMDD`, default) or explicit version (`v1`, `v2`, …).

Each `.npz` embeds its full metadata under `__meta__`; the `.yaml` sidecar is the
same metadata in human-readable form.

## Caveats (read before trusting a dataset)

- **Halo-ordering fix.** The upstream CAMELS-fork selection returns halos in FoF
  catalogue order, not sorted by the selection proxy. `build_fgas_spk_dataset`
  re-ranks columns by `rank_key` (default `halo_mass` = M_500c) before slicing
  the top-N, so number-density bins are genuinely the N most massive. The
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