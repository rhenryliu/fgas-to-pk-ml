# Known issues (out of scope for the dual-VAE line of work)

- **`tests/test_fgas_spk_builder.py`: 2 pre-existing failures**
  (`test_build_selects_top_n_distinct_halos`,
  `test_build_clamps_when_density_exceeds_halo_count`), both raising
  `KeyError: 'r12_mpch is not a file in the archive'` when the synthetic
  fixture archive is read. Present on clean HEAD (`5f7f776` and earlier),
  before and independent of the dual-VAE work. Per the 2026-07-05 spec
  amendment (D1), these are deliberately **not** fixed in the dual-VAE line of
  commits; fix on a separate branch. Suspected fixture/schema drift around the
  builder's expected archive keys — diagnose there, not in the tests alone.
