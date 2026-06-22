"""Tests for the path-resolution helpers in ``fgas_spk.paths``.

Each resolver is exercised with its environment variable both set and unset
(via ``monkeypatch``), the no-default write root is checked to fail loudly, and
``figure_dir`` is verified to build the dated tree. ``repo_root`` is checked to
be stable regardless of the process working directory.
"""

from datetime import date
from pathlib import Path

import pytest

import fgas_spk.paths as paths


# --- repo_root -------------------------------------------------------------

def test_repo_root_is_the_repository():
    root = paths.repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src" / "fgas_spk").is_dir()


def test_repo_root_stable_from_subdirectory(tmp_path, monkeypatch):
    # Anchored to the module location, not the CWD: chdir elsewhere and it is
    # unchanged.
    before = paths.repo_root()
    monkeypatch.chdir(tmp_path)
    assert paths.repo_root() == before


def test_repo_root_fallback_matches_layout():
    # The git answer must agree with the relative-layout fallback for this
    # editable checkout (paths.py is three parents below the repo root).
    assert paths.repo_root() == paths._REPO_ROOT_FALLBACK


# --- resolve_data_root -----------------------------------------------------

def test_data_root_defaults_to_repo_when_unset(monkeypatch):
    monkeypatch.delenv("FGAS_DATA_ROOT", raising=False)
    assert paths.resolve_data_root() == paths.repo_root()


def test_data_root_uses_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FGAS_DATA_ROOT", str(tmp_path))
    assert paths.resolve_data_root() == tmp_path


def test_data_root_empty_env_falls_back_to_repo(monkeypatch):
    monkeypatch.setenv("FGAS_DATA_ROOT", "")
    assert paths.resolve_data_root() == paths.repo_root()


def test_data_root_explicit_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FGAS_DATA_ROOT", str(tmp_path / "from_env"))
    explicit = tmp_path / "from_arg"
    assert paths.resolve_data_root(explicit) == explicit


# --- resolve_scratch_root --------------------------------------------------

def test_scratch_root_uses_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FGAS_SCRATCH_ROOT", str(tmp_path))
    assert paths.resolve_scratch_root() == tmp_path


def test_scratch_root_explicit_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FGAS_SCRATCH_ROOT", str(tmp_path / "from_env"))
    explicit = tmp_path / "from_arg"
    assert paths.resolve_scratch_root(explicit) == explicit


def test_scratch_root_raises_when_unset(monkeypatch):
    monkeypatch.delenv("FGAS_SCRATCH_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="FGAS_SCRATCH_ROOT"):
        paths.resolve_scratch_root()


def test_scratch_root_empty_env_raises(monkeypatch):
    monkeypatch.setenv("FGAS_SCRATCH_ROOT", "")
    with pytest.raises(RuntimeError, match="FGAS_SCRATCH_ROOT"):
        paths.resolve_scratch_root()


# --- figure_dir ------------------------------------------------------------

def test_figure_dir_creates_dated_tree(tmp_path, monkeypatch):
    # Redirect the repo root so the test never writes into the real repo.
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    out = paths.figure_dir()

    today = date.today()
    expected = tmp_path / "figures" / today.strftime("%Y-%m") / today.strftime("%m-%d")
    assert out == expected
    assert out.is_dir()


def test_figure_dir_appends_optional_subdir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    out = paths.figure_dir("diagnostics")

    today = date.today()
    expected = (
        tmp_path / "figures" / today.strftime("%Y-%m")
        / today.strftime("%m-%d") / "diagnostics"
    )
    assert out == expected
    assert out.is_dir()


def test_figure_dir_idempotent(tmp_path, monkeypatch):
    # Calling twice must not raise (mkdir exist_ok) and must return the same dir.
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    first = paths.figure_dir()
    second = paths.figure_dir()
    assert first == second
    assert first.is_dir()
