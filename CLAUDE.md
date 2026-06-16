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

Current contents: `fgas_spk_dataset.py` (the library) and a copy of the
collaborator-authored reference notebook `test_CAMELS_sixth_gen.ipynb`. The
notebook is provenance, not the source of truth — the library is the corrected,
maintained version of its loading logic.

## How to work here

- **Iterative, with verification gates.** Prefer small, staged changes each
  followed by an explicit numerical check (shapes, round-trips, value ranges)
  over large one-shot rewrites. Add a check that proves the change before moving on.
- **Conventions.** Google-style docstrings. Canadian English in prose and
  comments. Type hints on public functions. Single-module layout for now; if it
  grows into a package, use a `src/` layout.
- **Units and constants.** Public data uses comoving **Mpc/h** for radii, **h/Mpc**
  for `k`, **M⊙/h** for masses; `fgas`/`suppression` are dimensionless. CAMELS
  cosmology varies per simulation (Sobol set) — never assume one cosmology for
  the suite; per-sim `f_b`/`h` live in the source profiles. For any
  fixed-cosmology context elsewhere in the project, use Planck-2018
  (h = 0.6736, Ω_m = 0.3153, Ω_b = 0.0493) and state it inline.
- **h-factor boundary rule.** If you wrap a library that works internally in
  h-factored units (e.g. `colossus`, BCemu, SP(k)), convert explicitly at the
  boundary and document it. Do not let h-factors leak silently between layers.

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
6. **The scratch store is data-only.** Datasets and large model checkpoints go to
   the scratch `project_root`; configs, figures, READMEs, and docs stay in this
   git-tracked repo.

## Known landmines (verified context, not speculation)

- **Upstream halo-ordering bug.** The CAMELS fork selects halos by stellar mass
  but returns them in FoF catalogue order, so a naive `[:, :n_halos]` slice takes
  the top-N by FoF mass, not by the selection proxy. `load_fgas_spk_dataset`
  corrects this by re-ranking columns via `rank_key` before slicing. Any new
  binning code must preserve that fix. The compiled fast path
  (`load_fgas_spk_from_compiled`) inherits the frozen, uncorrected ordering by
  design — use it only for quick iteration.
- **`rank_key="stellar"` is stubbed.** `SubhaloMStar` is not saved by the
  producer; stellar-mass ranking (the observationally-matched choice) needs a
  catalogue re-read and currently raises. Default is `halo_mass` (M_500c).
- **Generation / provenance.** Source files must be the three-profile `snap74`
  products (`_PROFILE_FILENAME`, with `prof2_ionized_gas_*`). The reference
  notebook never executed its `NAME_EXTRA` cell, so older on-disk products may be
  mis-stamped; the loader guards against this by requiring `prof2`. Do not relax
  that guard.
- **Suppression target is a proxy.** `suppression` is `P_total/P_DM` within the
  hydro run (DM-as-DMO proxy), not paired-DMO `P_hydro/P_DMO`. It is acceptable at
  low k but biases at k ≳ 5 h/Mpc in a feedback-correlated way. CAMELS provides
  matched N-body (`*_DM`) counterparts; computing the true target is a known
  to-do. Flag, don't silently "fix", and don't conflate the two in metadata.
- **Filename coupling.** `_PROFILE_FILENAME` is pinned to the snapshot
  (`..._snap74.npz`). Changing snapshot means changing this constant *and* the
  `snapshot`/`redshift` arguments to `save_dataset`; keep them consistent.

## Data store layout

```
<project_root>/datasets/<suite>/snap<NNN>_z<z>/src-<source>/
    fgas_spk__<suite>__snap<NNN>_z<z>__src-<source>__rank-<rank>__<tag>.npz
    fgas_spk__...__<tag>.yaml
<project_root>/models/
```

`save_dataset` builds these paths; never hardcode them elsewhere. Set
`source`, `provenance`, and `notes` honestly (e.g. `source="lindajin"` with a
provenance note when data derives from the fork; `source="rhliu"` for
self-generated). `save_dataset` refuses to overwrite by default — change `tag`
or pass `overwrite=True` deliberately.

## Verification expectations

Before reporting a change as done:
- Parse/​import cleanly and run any touched function on small synthetic input.
- Round-trip test save/load when touching I/O.
- Check array shapes and value ranges against the docstrings.
- For anything touching selection, binning, or units, show a numerical check
  that the result matches the intended definition — do not assert correctness
  from inspection alone.

## Out of scope without explicit request

Training-loop code, model architectures, environment changes, regenerating
profiles, and any rewrite of the upstream `SimulationStacker` belong to separate,
explicitly-requested tasks. Keep changes here scoped to data assembly, storage,
and their tests.