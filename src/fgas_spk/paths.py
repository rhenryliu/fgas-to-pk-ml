"""Machine- and layout-aware path resolution for ``fgas-to-pk-ml``.

This module is the single home for resolving where things live, so no other
module hardcodes a path or walks a brittle relative ``../../``. Reading and
writing are deliberately kept as separate concerns:

* :func:`resolve_data_root` -- where datasets are **read** from. It has a
  sensible default (the repository) because the current datasets are
  git-tracked under it.
* :func:`resolve_scratch_root` -- where run outputs are **written**. It has no
  default: an unconfigured machine must stop loudly rather than guess where to
  write, so the per-machine path lives in ``$FGAS_SCRATCH_ROOT`` (set once per
  machine) or is passed explicitly.

Per-machine configuration is via environment variables only -- there is no
filesystem detection, no hardcoded path, and no username baked into the code.
The per-machine value belongs in the environment, set once per machine.
"""

from __future__ import annotations

import os
import subprocess
from datetime import date
from pathlib import Path

# paths.py lives at <repo_root>/src/fgas_spk/paths.py, so the repo root is three
# parents up. Used only as a fallback when the git top-level cannot be resolved.
_REPO_ROOT_FALLBACK = Path(__file__).resolve().parents[2]

# Environment variables holding the per-machine roots (see the module docstring).
_DATA_ROOT_ENV = "FGAS_DATA_ROOT"
_SCRATCH_ROOT_ENV = "FGAS_SCRATCH_ROOT"


def repo_root() -> Path:
    """Return the repository root, anchored to this module's own location.

    Resolution is anchored to this file's directory rather than the process
    working directory, so the answer is identical no matter where a script is
    run from. The git top-level is queried first (``git -C <pkg dir> rev-parse
    --show-toplevel``); if that is unavailable -- ``git`` is not installed, or
    this is not a git checkout (e.g. a non-editable install) -- the location of
    this module is used as a fallback.

    Returns:
        Path: Absolute path to the repository root.
    """
    anchor = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(anchor), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        # OSError/FileNotFoundError -> git missing; CalledProcessError -> not a
        # git checkout. Either way, fall back to the known layout relative to
        # this file.
        return _REPO_ROOT_FALLBACK
    return Path(result.stdout.strip())


def resolve_data_root(explicit: str | Path | None = None) -> Path:
    """Resolve the root that datasets are read from.

    Precedence: the ``explicit`` argument, then ``$FGAS_DATA_ROOT``, then
    :func:`repo_root`. The repository is a sensible default because the current
    datasets are git-tracked under it. The returned path is the data root -- the
    directory beneath which ``datasets/`` lives -- not the ``datasets/``
    directory itself.

    Args:
        explicit (str | Path | None): An explicit root that overrides both the
            environment variable and the default. Defaults to None.

    Returns:
        Path: The data root.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get(_DATA_ROOT_ENV)
    if env:  # an empty value is treated as unset
        return Path(env).expanduser()
    return repo_root()


def resolve_scratch_root(explicit: str | Path | None = None) -> Path:
    """Resolve the root that run outputs are written to.

    Precedence: the ``explicit`` argument, then ``$FGAS_SCRATCH_ROOT``. There is
    deliberately **no** default -- an unconfigured machine must stop loudly
    rather than guess a writable location. Set ``$FGAS_SCRATCH_ROOT`` once per
    machine, or pass the root explicitly.

    Args:
        explicit (str | Path | None): An explicit write root that overrides the
            environment variable. Defaults to None.

    Returns:
        Path: The scratch root, beneath which run outputs are written.

    Raises:
        RuntimeError: If neither ``explicit`` nor ``$FGAS_SCRATCH_ROOT`` resolves
            to a value.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get(_SCRATCH_ROOT_ENV)
    if env:  # an empty value is treated as unset
        return Path(env).expanduser()
    raise RuntimeError(
        "No scratch root configured: set $FGAS_SCRATCH_ROOT or pass "
        "--scratch-root / write_root. This must be set per machine; the code "
        "does not guess a writable location."
    )


def figure_dir(subdir: str | None = None) -> Path:
    """Return (creating if needed) today's dated figure directory.

    Reproduces the existing convention exactly:
    ``<repo_root>/figures/<YYYY-MM>/<MM-DD>/``, grouped by the current local
    date. Figures are grouped by date; a run identifier belongs in the
    **filename**, not a subfolder. The optional ``subdir`` is for grouping a
    *category* of figures under the dated directory, not for a run id.

    Args:
        subdir (str | None): Optional extra directory appended beneath the dated
            path (e.g. ``"diagnostics"``). Defaults to None. Do not pass a run id
            here -- that goes in the filename.

    Returns:
        Path: The created directory.
    """
    today = date.today()
    out = repo_root() / "figures" / today.strftime("%Y-%m") / today.strftime("%m-%d")
    if subdir is not None:
        out = out / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out
