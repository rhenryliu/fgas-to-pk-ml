# CLAUDE.md

Guidance for AI agents (Claude Code and similar) working in this repository.
Read this before editing code, writing files, or running anything.

## What this repo is

`fgas-to-pk-ml` is the machine-learning / data-assembly side of a cosmology
project that learns the mapping from projected gas-fraction profiles
**f_gas(R)** to matter power-spectrum suppression **SP(k) = P_total(k) / P_DM(k)**,
using the CAMELS IllustrisTNG **SB35** suite. It consumes the per-simulation
profile products written by `SimulationStacker` and produces model-ready
training data.

The library is the installed `fgas_spk` package (distribution name
`fgas-to-pk-ml`, src layout under `src/`, `pip install -e .`). It has two layers.

**Data-assembly core** (imports only numpy + pyyaml):

- `src/fgas_spk/schema.py` — the on-disk data contract and **single source of
  truth**: the `FgasSpkDataset` container, the filename/directory convention,
  `save_dataset`/`load_dataset`/`resolve_dataset_path`/`dataset_dir`/
  `ensure_store_dirs`, and `SCHEMA_VERSION`. Do not redefine the contract
  elsewhere.
- `src/fgas_spk/builder.py` — the builder: load profiles, fix the halo-ordering
  bug, assemble `(profile, nd) → SP(k)` arrays. Imports and re-exports the
  contract from `fgas_spk.schema`.
- `src/fgas_spk/loader.py` — the configurable training-data loader: read a saved
  dataset and serve model-ready numpy arrays per a `DataConfig`. numpy + pyyaml
  only (no torch); runnable as a CLI dry-run (`python -m fgas_spk.loader`).
- `src/fgas_spk/__init__.py` — re-exports the common API as a flat `fgas_spk.*`
  surface (the schema names, the builders, the loader's `DataConfig` /
  `load_training_data`, and the experiment's `RunConfig`/`SplitSpec`/
  `load_configs`). Importing the top-level package stays numpy + pyyaml — it
  does not pull the heavier training submodules below.

**Training / experiment layer** (opt-in; the models and runner pull scikit-learn,
joblib, and lazily torch/matplotlib from the pinned environment — import these
explicitly):

- `src/fgas_spk/paths.py` — machine- and layout-aware path resolution:
  `resolve_data_root` (reads — defaults to the repo), `resolve_scratch_root`
  (writes — no default; raises unless `$FGAS_SCRATCH_ROOT` is set or a root is
  passed), `repo_root`, `figure_dir`, and the `fill_data_root` entry-point
  helper. No hardcoded paths, no filesystem detection, no username in code. A
  leaf module: it never imports the rest of the package at runtime.
- `src/fgas_spk/experiment.py` — the trainer-side `RunConfig` (model + training
  fields **only**, no data selection), the grouped-by-`sim_index` `SplitSpec`,
  and `load_configs` (reads the data YAML and run YAML independently — no merge).
- `src/fgas_spk/models/` — the model-plugin layer: the `ProfileToSpk` interface,
  the name→class `REGISTRY`, and the `register` decorator (`base.py`), plus the
  `pca_linear` reference plugin. Importing the package runs each plugin's
  `@register` for its side effect; add new plugins' imports to its `__init__`.
- `src/fgas_spk/run_record.py` — the git-tracked text record + ledger for a run:
  writes `experiments/runs/<run_id>/` and appends to `experiments/runs.jsonl`.
  Never imports torch (the device string is passed in). Holds
  `DEFAULT_SUPPRESSION_DEFINITION`, the recorded suppression-target string.
- `src/fgas_spk/train.py` — the end-to-end runner (`run_training`): a
  `(DataConfig, RunConfig)` pair → read-root fill → grouped split → fit a model
  from the registry → write the run record, checkpoint to scratch, and a
  diagnostic figure. Resolves no physics choice (rule 3); records provenance.

**Backup** (do not extend or fix): `src/_frozen/fgas_spk_dataset_v0.py` — a
**frozen, pre-refactor backup** of the loader, quarantined outside the package
(no `__init__.py`), kept only because callers may still source it; all changes go
to the maintained modules above.

A copy of the collaborator-authored reference notebook lives at
`notebooks/test_CAMELS_sixth_gen.ipynb` (and a scratch loader notebook at
`notebooks/test_loader.ipynb`). These are provenance, not the source of truth —
the library is the corrected, maintained version of their loading logic.

## How to work here

- **Iterative, with verification gates.** Prefer small, staged changes each
  followed by an explicit numerical check (shapes, round-trips, value ranges)
  over large one-shot rewrites. Add a check that proves the change before moving on.
- **Conventions.** Google-style docstrings. Canadian English in prose and
  comments. Type hints on public functions. The code is the installed `fgas_spk`
  package under `src/` (`schema` holds the contract; `builder` and `loader`
  import from it via absolute intra-package imports — `from fgas_spk.schema
  import ...`). Keep the dependency direction one-way (loader/builder → schema);
  do not add back-imports from `schema` to its consumers.
- **Units and constants.** Public data uses comoving **Mpc/h** for radii, **h/Mpc**
  for `k`, **M⊙/h** for masses; `fgas`/`suppression` are dimensionless. CAMELS
  cosmology varies per simulation (Sobol set) — never assume one cosmology for
  the suite; per-sim `f_b`/`h` live in the source profiles. For any
  fixed-cosmology context elsewhere in the project, use Planck-2018
  (h = 0.6736, Ω_m = 0.3153, Ω_b = 0.0493) and state it inline.
- **h-factor boundary rule.** If you wrap a library that works internally in
  h-factored units (e.g. `colossus`, BCemu, SP(k)), convert explicitly at the
  boundary and document it. Do not let h-factors leak silently between layers.

## Environment

The canonical environment is two git-tracked files at the repo root: `environment.yml`
(conda-forge Python 3.12 + pip) and `requirements.txt` (exact, top-level pins). They
install matched versions on NERSC Perlmutter (Linux x86_64, CUDA) and an Apple-Silicon
Mac (macOS arm64, MPS), almost entirely from wheels.

Rules for the environment (extending binding rule 2 — *do not modify the environment
without explicit permission*):

- The pins are exact (`==`) on purpose: this is an environment spec for cross-machine
  reproduction, not a library dependency declaration. Do not relax them to floors here,
  and do not regenerate the file with `pip freeze` — a freeze bakes in platform-specific
  transitive deps (the `nvidia-*` CUDA stack on Linux, a CPU/MPS torch on macOS) that
  break the other platform.
- Keep `torch` pinned by upstream version only — never a `+cuXXX` local version or an
  `--index-url` in the shared file. The correct wheel (CUDA on Linux, MPS on macOS) is
  resolved per platform; device selection happens at runtime, not install time.
- The base `requirements.txt` is wheel-only and must stay installable on both platforms.
  Compile-from-source / single-platform packages do not belong in it. **Pylians** is the
  current example: it fails to build on Apple Silicon (the released sdist mis-quotes the
  macOS OpenMP flag) and is commented out; it is NERSC-side (Linux/gcc) and, when adopted,
  belongs in a separate `requirements-nersc.txt`. `pypower` covers P(k) estimation
  cross-platform in the meantime.
- Bumping a pin is a deliberate, reviewed edit, not a side effect of reinstalling —
  consistent with the gated workflow.
- The package is tiered by import weight. The **data-assembly core** and the
  **flat API** (`import fgas_spk`, plus `schema`/`builder`/`loader`/`experiment`) import
  only numpy + pyyaml (+ stdlib) — keep them that way. The **opt-in training submodules**
  may pull more: `fgas_spk.models` and `fgas_spk.train` use scikit-learn and joblib, and
  `train` imports torch and matplotlib lazily (both optional); `fgas_spk.run_record` stays
  light (numpy + stdlib). Only the core's two deps are declared in `pyproject.toml`; the
  rest live in the env files. The wider cosmology/ML pins (pyccl, colossus, SP(k)/BCemu,
  torch/sbi, pandas, astropy, pypower, corner, getdist, …) are for gate, analysis, and
  training scripts, not the data-assembly modules.

## Binding rules

1. **Do not call external libraries from memory.** Verify a function's real
   signature and behaviour (read the installed source or docs) before using it.
   If unsure, say so and check; do not guess.
2. **Do not modify the environment** (install, upgrade, change the venv) without
   explicit permission. State what you would change and why, and wait.
3. **Physics decisions stay with the maintainer.** Mass-definition choices, which
   f_gas observable to fit, the suppression-target definition, mass extrapolation,
   and similar are flagged for a human, never silently resolved in code.
4. **Do not assume — ask.** If a path, key, convention, or scientific choice is
   ambiguous, surface it and request a decision rather than incorporating a
   silent default. This is a hard rule for this project.
5. **Judge code on merit.** When comparing implementations, evaluate the code
   itself, not who wrote it. Flag the maintainer's own mistakes too.
6. **The scratch store is data-only.** Outputs split three ways by size and
   provenance: large artifacts (datasets and model checkpoints) go to the scratch root
   (`$FGAS_SCRATCH_ROOT`); the small text run records go to the git-tracked
   `experiments/` tree; configs, READMEs, and docs stay in this git-tracked repo.
   Generated figures go to a `figures/` tree under the repo that is **gitignored** —
   regenerated, not version-controlled. Do not write large artifacts into the repo.

## Known landmines (verified context, not speculation)

- **Upstream halo-ordering bug.** The CAMELS fork selects halos by stellar mass
  but returns them in FoF catalogue order, so a naive `[:, :n_halos]` slice takes
  the top-N by FoF mass, not by the selection proxy. `build_fgas_spk_dataset`
  corrects this by re-ranking columns via `rank_key` before slicing. Any new
  binning code must preserve that fix. The compiled fast path
  (`build_fgas_spk_from_compiled`) inherits the frozen, uncorrected ordering by
  design — use it only for quick iteration.
- **`rank_key="stellar"` is stubbed.** `SubhaloMStar` is not saved by the
  producer; stellar-mass ranking (the observationally-matched choice) needs a
  catalogue re-read and currently raises. Default is `halo_mass` (M_500c).
- **Generation / provenance.** Source files must be the three-profile products
  (`prof2_ionized_gas_*` present). The reference notebook never executed its
  `NAME_EXTRA` cell, so older on-disk products may be mis-stamped; the loader
  guards against this by requiring `prof2`. Do not relax that guard.
- **Suppression target is a proxy.** `suppression` is `P_total/P_DM` within the
  hydro run (DM-as-DMO proxy), not paired-DMO `P_hydro/P_DMO`. It is acceptable at
  low k but biases at k ≳ 5 h/Mpc in a feedback-correlated way. CAMELS provides
  matched N-body (`*_DM`) counterparts; computing the true target is a known
  to-do. Flag, don't silently "fix", and don't conflate the two in metadata.
- **Snapshot handling.** Data spans snap74 (z=0.47) and snap82 (z=0.21). The
  loaders take a required `snapshot` argument that threads into all on-disk
  filenames: `_PROFILE_FILENAME_TEMPLATE` for per-sim profiles, the compiled
  `..._profiles_snap{NN}_nd_{i}_n_{N}.npz` pattern, and the caller-supplied
  `Ptot_Pdm_ratio_snap{NN}.npz`. Keep `snapshot` consistent with the
  `suppression_path` and with `snapshot`/`redshift` in `save_dataset`; never
  hardcode a snapshot back into a filename.

## Data store layout

Scratch store (`$FGAS_SCRATCH_ROOT`) — large artifacts only:

```
<scratch_root>/datasets/<suite>/snap<NNN>_z<z>/src-<source>/
    fgas_spk__<suite>__snap<NNN>_z<z>__src-<source>__rank-<rank>__<tag>.npz
    fgas_spk__...__<tag>.yaml
<scratch_root>/models/<run_id>/model.joblib        # one checkpoint dir per run
```

`save_dataset` builds the dataset paths; never hardcode them elsewhere. Set
`source`, `provenance`, and `notes` honestly (e.g. `source="lindajin"` with a
provenance note when data derives from the fork; `source="rhliu"` for
self-generated). `save_dataset` refuses to overwrite by default — change `tag`
or pass `overwrite=True` deliberately.

Git-tracked run records — small text only, written by `fgas_spk.run_record`:

```
<repo>/experiments/runs/<run_id>/
    config_data.yaml   config_run.yaml   env.json   metrics.jsonl   summary.json
<repo>/experiments/runs.jsonl                      # one greppable ledger line per run
```

`run_id = "<UTC-timestamp>__<config-hash8>__<git-sha7>"` (config hash over the
resolved `(DataConfig, RunConfig)` pair; git sha of `HEAD` with a dirty flag).
`env.json` captures the dataset `__meta__`, the roots, package versions, and the
suppression-target definition verbatim, so runs against different targets are
never silently compared. Build these paths via `paths.repo_root` /
`run_record.write_run_record`; do not hardcode them.

## Verification expectations

Before reporting a change as done:
- Parse/​import cleanly and run any touched function on small synthetic input.
- Round-trip test save/load when touching I/O.
- Check array shapes and value ranges against the docstrings.
- For anything touching selection, binning, or units, show a numerical check
  that the result matches the intended definition — do not assert correctness
  from inspection alone.

## Out of scope without explicit request

A thin **reference** training layer now lives in the repo (the `pca_linear`
reference model, the `run_training` runner, and the run-record machinery) — it
exists to demonstrate the `ProfileToSpk` interface and the recorded-run workflow,
not as a modelling effort. Still out of scope without an explicit request: real
model architectures and hyperparameter tuning, new training loops beyond the
reference runner, environment changes, regenerating profiles, and any rewrite of
the upstream `SimulationStacker`. Keep changes scoped to data assembly, storage,
the reference training layer, and their tests.