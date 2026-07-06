"""Branch A (E3): rebuild the pinned dataset under tag 20260706.

Rebuilds the pinned variant (CAMELS-IllustrisTNG-L50n512-SB35, snap 74,
source lindajin, rank m500, all five number densities) with the corrected
ratio-of-stacked-profiles statistic (:data:`fgas_spk.builder.FGAS_DEFINITION`),
stamps the definition and realized per-bin range into ``__meta__`` (E2.2), and
then runs the E3.2 **identity assertions** against tag 20260702: the
suppression array, k grid, radial grid, CAMELS parameter block (and names),
sim ids, number densities, and mean halo masses must be bitwise-identical;
ONLY ``fgas`` / ``fgas_std`` may differ (and ``fgas`` must actually differ --
an identical fgas would mean the fix did not take effect). Any other
difference aborts with a non-zero exit (STOP condition).

Other variants (snapshot 82, other ranks) are NOT rebuilt in this pass; the
stale-variant warning lives in ``experiments/notes/known_issues.md``.

Run from the repository root (in the ``fgas-ml`` env)::

    python scripts/rebuild_dataset_20260706.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fgas_spk.builder import (
    DEFAULT_FGAS_SANITY_RANGE,
    build_fgas_spk_dataset,
    fgas_meta_stamp,
)
from fgas_spk.schema import ensure_store_dirs, resolve_dataset_path, save_dataset

DATA_DIR = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data/"
BASE = "/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{}/data/"
PROJECT = "/pscratch/sd/r/rhliu/projects/fgas-to-pk-ml"
SUITE = "CAMELS-IllustrisTNG-L50n512-SB35"
SNAPSHOT, REDSHIFT = 74, 0.47
NEW_TAG, REF_TAG = "20260706", "20260702"

# Deliberate widening of the PROVISIONAL default sanity range ([-1, 3]) for
# this build, per the guard's own escape hatch: the first build attempt
# hard-failed on 22 cells (0.02%), all at R >= 9.4 Mpc/h and concentrated at
# the sparsest stacks (nd 0/1; worst 117.8 at nd 1, R = 20) -- the stacked
# Delta Sigma_total legitimately becomes small at the outermost radii, a mild
# physically-driven tail, not the per-halo singularity the guard exists to
# catch (the old corruption reached +-8540 across 992 cells at all radii).
# The pinned nd-index-2 slice spans [-1.32, 4.70]; the realized per-bin range
# is stamped into __meta__ either way. RESOLVED (amendment 7, J5): [-1, 3] is
# confirmed as the production default WITHIN the modelled radial window under
# the later two-tier guard; this build's whole-grid widening predates that
# guard and stands as the recorded escape hatch for tag 20260706.
BUILD_SANITY_RANGE = (-15.0, 130.0)

# E3.2: arrays that must be bitwise-identical to the reference tag.
IDENTICAL_KEYS = (
    "suppression", "k", "radii_mpch", "camels_params", "camels_param_names",
    "sim_ids", "number_densities", "mean_halo_mass",
)


def main() -> int:
    """Rebuild, stamp, save, and assert identity against the reference tag.

    Returns:
        int: 0 on success; 1 on any E3.2 identity violation.
    """
    dataset = build_fgas_spk_dataset(
        base_path_template=BASE,
        suppression_path=Path(DATA_DIR) / f"Ptot_Pdm_ratio_snap{SNAPSHOT}.npz",
        snapshot=SNAPSHOT,
        rank_key="halo_mass",
        params_path=Path(DATA_DIR) / "camels_params_matrix.npy",
        fgas_sanity_range=BUILD_SANITY_RANGE,
        progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True),
    )
    print()
    print("fgas grid:", dataset.fgas.shape, "(n_sims, n_nd, n_radii)")
    print(f"NOTE: sanity range widened deliberately to {BUILD_SANITY_RANGE} "
          f"(default {DEFAULT_FGAS_SANITY_RANGE} is provisional; see comment).")
    stamp = fgas_meta_stamp(dataset.fgas, BUILD_SANITY_RANGE)
    print("realized fgas range:",
          min(stamp["fgas_realized_range_per_bin"]["min"]),
          max(stamp["fgas_realized_range_per_bin"]["max"]))

    ensure_store_dirs(PROJECT)
    out = save_dataset(
        dataset,
        project_root=PROJECT,
        suite=SUITE,
        snapshot=SNAPSHOT,
        redshift=REDSHIFT,
        source="lindajin",
        tag=NEW_TAG,
        provenance={
            "profiles_repo": "github.com/Klinjin/SimulationStacker (fork)",
            "producer": "CAMELS_training_data_second_generation.py",
            "note": (
                "Profiles from lindajin scratch; re-ranked by M_500c on load. "
                "Statistic corrected to ratio-of-stacked-profiles per Branch A "
                "(see experiments/notes/dual_vae_stage3_forensics.md); "
                "supersedes all earlier tags by DEFINITION change."
            ),
        },
        notes=(
            "Branch-A rebuild: ratio-of-stacks fgas. E2.1 guard run with a "
            f"deliberately widened range {BUILD_SANITY_RANGE} (default [-1, 3] "
            "provisional): 22 cells (0.02%), all R >= 9.4 Mpc/h, mostly nd 0/1 "
            "(sparse stacks; stacked DSigma_total small at outer radii), lie "
            "outside the default; pinned nd-2 slice spans [-1.32, 4.70]. "
            "Maintainer confirmation of the range pending."
        ),
        extra_meta=stamp,
    )
    print("saved:", out)

    # --- E3.2 identity assertions vs the reference tag ----------------------
    ref_path = resolve_dataset_path(
        PROJECT, SUITE, SNAPSHOT, REDSHIFT, "lindajin", "m500", REF_TAG
    )
    ref = np.load(ref_path, allow_pickle=False)
    new = np.load(out, allow_pickle=False)
    failures: list[str] = []
    for key in IDENTICAL_KEYS:
        if not np.array_equal(ref[key], new[key]):
            failures.append(key)
        else:
            print(f"identity OK : {key}")
    if np.array_equal(ref["fgas"], new["fgas"]):
        failures.append("fgas UNCHANGED (fix did not take effect)")
    else:
        diff_frac = float(np.mean(~np.isclose(ref["fgas"], new["fgas"])))
        print(f"fgas differs as required (fraction of cells changed: "
              f"{diff_frac:.3f})")

    if failures:
        print("E3.2 IDENTITY ASSERTIONS FAILED:", failures)
        return 1
    meta = json.loads(str(new["__meta__"]))
    print("fgas_definition stamped:", "fgas_definition" in meta)
    print("E3.2 identity assertions: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
