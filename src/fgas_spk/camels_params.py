"""Per-set registry of CAMELS input-parameter names, and selection resolution.

This module is the **single source of the parameter-name ordering** for the
CAMELS suites this project consumes. The on-disk ``camels_params`` matrix (see
:class:`~fgas_spk.schema.FgasSpkDataset`) is an unlabelled ``(n_sims, n_params)``
float array; the column order is fixed by the producer's
``camels_params_matrix.npy`` and is *suite-specific*. The mapping from a column
index to a physical parameter name therefore lives here, keyed by the suite
string the dataset already records in its ``__meta__``.

Keying by the full suite string (not a derived ``"SB35"`` token) means the lookup
is an exact match against recorded metadata with no parsing, and a future CAMELS
set with a different parameter set simply gets its own entry -- the labels are
never assumed across suites. A guard at use sites checks that the registered name
count matches the data's actual column count, so a mismatch fails loud rather
than silently mislabelling columns.

The ordering for ``CAMELS-IllustrisTNG-L50n512-SB35`` was transcribed verbatim
from, and validated cell-for-cell against, the authoritative CAMELS file
``docs/params/IllustrisTNG/CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt`` (the
``seed`` column is excluded -- it is not part of ``camels_params``).

This is a leaf module: pure standard library, no intra-package imports, so the
schema, builder, and loader can all import it without pulling weight or creating
cycles.
"""

from __future__ import annotations

# Registry: suite string -> ordered tuple of parameter names, one per column of
# that suite's ``camels_params`` matrix (in matrix-column order). Add a new entry
# when a new CAMELS set is adopted; do not reuse another suite's list.
CAMELS_PARAM_NAMES: dict[str, tuple[str, ...]] = {
    "CAMELS-IllustrisTNG-L50n512-SB35": (
        "Omega0",
        "sigma8",
        "WindEnergyIn1e51erg",
        "RadioFeedbackFactor",
        "VariableWindVelFactor",
        "RadioFeedbackReiorientationFactor",
        "OmegaBaryon",
        "HubbleParam",
        "n_s",
        "MaxSfrTimescale",
        "FactorForSofterEQS",
        "IMFslope",
        "SNII_MinMass_Msun",
        "ThermalWindFraction",
        "VariableWindSpecMomentum",
        "WindFreeTravelDensFac",
        "MinWindVel",
        "WindEnergyReductionFactor",
        "WindEnergyReductionMetallicity",
        "WindEnergyReductionExponent",
        "WindDumpFactor",
        "SeedBlackHoleMass",
        "BlackHoleAccretionFactor",
        "BlackHoleEddingtonFactor",
        "BlackHoleFeedbackFactor",
        "BlackHoleRadiativeEfficiency",
        "QuasarThreshold",
        "QuasarThresholdPower",
        "UVBH0beta",
        "UVBH0Deltaz",
        "UVBHepbeta",
        "UVBHepDeltaz",
        "SNIa_Rate_Norm",
        "SNIa_Rate_DTD_power",
        "SofteningComovingType01",
    ),
}

# Self-check: every registered name list must have no duplicate names (a
# duplicate would make name-based selection ambiguous). This runs at import.
for _suite, _names in CAMELS_PARAM_NAMES.items():
    if len(set(_names)) != len(_names):
        raise ValueError(
            f"CAMELS_PARAM_NAMES[{_suite!r}] has duplicate parameter names; "
            "names must be unique for name-based selection to be unambiguous."
        )


def param_names_for(suite: str) -> tuple[str, ...]:
    """Return the ordered parameter names for a suite.

    Args:
        suite (str): The suite string as recorded in a dataset's ``__meta__``
            (e.g. ``'CAMELS-IllustrisTNG-L50n512-SB35'``).

    Returns:
        tuple[str, ...]: The parameter names in ``camels_params`` column order.

    Raises:
        KeyError: If the suite has no registered name list. The message lists the
            known suites so a missing entry is obvious.
    """
    try:
        return CAMELS_PARAM_NAMES[suite]
    except KeyError:
        raise KeyError(
            f"No CAMELS parameter-name list registered for suite {suite!r}. "
            f"Known suites: {sorted(CAMELS_PARAM_NAMES)}. Add an entry to "
            "fgas_spk.camels_params.CAMELS_PARAM_NAMES (in matrix-column order)."
        ) from None


def resolve_param_columns(
    n_params: int,
    available_names: "tuple[str, ...] | list[str] | None",
    indices: "list[int] | None",
    names: "list[str] | None",
) -> tuple[list[int], list[str] | None]:
    """Resolve an index/name selection into sorted, de-duplicated columns.

    Selection may be given by 0-based column ``indices``, by physical ``names``,
    or by both; the two are unioned, de-duplicated, and returned **sorted
    ascending by column index**, so any combination that picks the same physical
    set yields identical columns (and identical ``X_params`` semantics). With both
    ``indices`` and ``names`` None, all columns are selected.

    Name-based selection requires ``available_names``; index-based selection does
    not. ``available_names``, when given, must have length ``n_params`` (the data's
    actual column count) -- a mismatch is a labelling error and is rejected.

    Args:
        n_params (int): Number of columns in the ``camels_params`` matrix.
        available_names (tuple[str, ...] | list[str] | None): The column names in
            order (stamped on the dataset, or from the registry), or None if no
            name list is available.
        indices (list[int] | None): Requested 0-based column indices, or None.
        names (list[str] | None): Requested parameter names, or None.

    Returns:
        tuple[list[int], list[str] | None]: ``(col_indices, col_names)`` where
            ``col_indices`` is the sorted, de-duplicated column list and
            ``col_names`` is the matching names (or None if no name list was
            available to label them).

    Raises:
        ValueError: If a selection is empty, an index is out of range, a name is
            unknown, names are requested without an available name list, or
            ``available_names`` length disagrees with ``n_params``.
    """
    if available_names is not None and len(available_names) != n_params:
        raise ValueError(
            f"available_names has {len(available_names)} entries but the "
            f"camels_params matrix has {n_params} columns; the name list does "
            "not match the data. Check the registry entry or the stamped names."
        )

    # No selection given -> all columns in natural order.
    if indices is None and names is None:
        col_indices = list(range(n_params))
        col_names = list(available_names) if available_names is not None else None
        return col_indices, col_names

    selected: set[int] = set()

    if indices is not None:
        if len(indices) == 0:
            raise ValueError(
                "camels_param_indices is empty; select at least one column or "
                "set include_camels_params=False."
            )
        bad = [i for i in indices if not (0 <= int(i) < n_params)]
        if bad:
            raise ValueError(
                f"camels_param_indices {bad} out of range for n_params={n_params}."
            )
        selected.update(int(i) for i in indices)

    if names is not None:
        if len(names) == 0:
            raise ValueError(
                "camels_param_names is empty; select at least one parameter or "
                "set include_camels_params=False."
            )
        if available_names is None:
            raise ValueError(
                "camels_param_names was given but no parameter-name list is "
                "available for this dataset (it carries no stamped names and its "
                "suite is not in the registry). Select by camels_param_indices "
                "instead, or register the suite's names."
            )
        lookup = {nm: i for i, nm in enumerate(available_names)}
        missing = [nm for nm in names if nm not in lookup]
        if missing:
            raise ValueError(
                f"camels_param_names {missing} not found. Available names: "
                f"{list(available_names)}."
            )
        selected.update(lookup[nm] for nm in names)

    col_indices = sorted(selected)
    col_names = (
        [available_names[i] for i in col_indices]
        if available_names is not None
        else None
    )
    return col_indices, col_names
