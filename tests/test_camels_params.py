"""Tests for the CAMELS parameter-name registry and selection resolution.

Covers the registry lookup (known / unknown suite), the SB35 entry's integrity,
and :func:`resolve_param_columns` -- index / name / mixed selection, the
sorted-and-de-duplicated contract, and every validation failure.
"""

import pytest

import fgas_spk.camels_params as cp


# --- registry --------------------------------------------------------------

def test_sb35_entry_has_35_unique_names():
    names = cp.param_names_for("CAMELS-IllustrisTNG-L50n512-SB35")
    assert len(names) == 35
    assert len(set(names)) == 35           # no duplicates
    assert names[0] == "Omega0"
    assert names[-1] == "SofteningComovingType01"


def test_param_names_for_unknown_suite_raises():
    with pytest.raises(KeyError, match="No CAMELS parameter-name list"):
        cp.param_names_for("CAMELS-Some-Other-Suite")


# --- resolve_param_columns: selection --------------------------------------

NAMES = ("a", "b", "c", "d", "e")


def test_no_selection_returns_all_columns():
    idx, names = cp.resolve_param_columns(5, NAMES, None, None)
    assert idx == [0, 1, 2, 3, 4]
    assert names == list(NAMES)


def test_no_selection_without_namelist_returns_all_indices_no_names():
    idx, names = cp.resolve_param_columns(5, None, None, None)
    assert idx == [0, 1, 2, 3, 4]
    assert names is None


def test_index_selection_is_sorted_and_deduplicated():
    idx, names = cp.resolve_param_columns(5, NAMES, [3, 0, 3], None)
    assert idx == [0, 3]
    assert names == ["a", "d"]


def test_index_selection_without_namelist_labels_none():
    idx, names = cp.resolve_param_columns(5, None, [2, 0], None)
    assert idx == [0, 2]
    assert names is None


def test_name_selection_resolves_to_sorted_columns():
    idx, names = cp.resolve_param_columns(5, NAMES, None, ["d", "a"])
    assert idx == [0, 3]
    assert names == ["a", "d"]


def test_mixed_index_and_name_union_dedup_sorted():
    # index 3 == name "d"; the union collapses and sorts ascending by column.
    idx, names = cp.resolve_param_columns(5, NAMES, [3, 1], ["d", "a"])
    assert idx == [0, 1, 3]
    assert names == ["a", "b", "d"]


# --- resolve_param_columns: validation -------------------------------------

def test_out_of_range_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        cp.resolve_param_columns(5, NAMES, [5], None)


def test_negative_index_raises():
    with pytest.raises(ValueError, match="out of range"):
        cp.resolve_param_columns(5, NAMES, [-1], None)


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="not found"):
        cp.resolve_param_columns(5, NAMES, None, ["zzz"])


def test_name_selection_without_namelist_raises():
    with pytest.raises(ValueError, match="no parameter-name list"):
        cp.resolve_param_columns(5, None, None, ["a"])


def test_empty_index_selection_raises():
    with pytest.raises(ValueError, match="empty"):
        cp.resolve_param_columns(5, NAMES, [], None)


def test_empty_name_selection_raises():
    with pytest.raises(ValueError, match="empty"):
        cp.resolve_param_columns(5, NAMES, None, [])


def test_namelist_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        cp.resolve_param_columns(4, NAMES, [0], None)  # 5 names vs 4 columns
