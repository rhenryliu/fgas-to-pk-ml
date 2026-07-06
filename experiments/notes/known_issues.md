# Known issues (out of scope for the dual-VAE line of work)

- **Tag 20260706's stamped guard range is the escape-hatch value (F3.2
  ledger note).** That build's `__meta__` records `fgas_sanity_range`
  (-15, 130): the documented one-off widening from the Branch-A rebuild,
  applied before the amendment-4 two-tier guard existed. The dataset is NOT
  rebuilt for the guard redesign (the guard changes no computed values; a
  rebuild would be bitwise-identical fgas under a new tag). Two-tier
  semantics — hard-fail within the modelled radial window, census-and-warn
  outside it — apply from the next build onward.

- **Stale dataset variants under the superseded f_gas statistic.** The
  Branch-A rebuild (2026-07-06) corrected the gas-fraction statistic to
  ratio-of-stacked-profiles and produced tag `20260706` for the pinned
  variant only (snap74, lindajin, m500). Every other variant — snap82, other
  ranks, and ALL tags before 20260706 (20260616/20260617/20260702, both
  snapshots) — still carries the superseded per-halo-ratio statistic, which
  is singular at per-halo ΔΣ_total zero-crossings (see
  `dual_vae_stage3_forensics.md`). **Do not train on them.** The
  discriminator is the `fgas_definition` key in `__meta__`: present = new
  statistic; absent = old (E7.3: superseded by definition change, not merely
  by tag).

- **`tests/test_fgas_spk_builder.py`: 2 pre-existing failures**
  (`test_build_selects_top_n_distinct_halos`,
  `test_build_clamps_when_density_exceeds_halo_count`), both raising
  `KeyError: 'r12_mpch is not a file in the archive'` when the synthetic
  fixture archive is read. Present on clean HEAD (`5f7f776` and earlier),
  before and independent of the dual-VAE work. Per the 2026-07-05 spec
  amendment (D1), these are deliberately **not** fixed in the dual-VAE line of
  commits; fix on a separate branch. Suspected fixture/schema drift around the
  builder's expected archive keys — diagnose there, not in the tests alone.
