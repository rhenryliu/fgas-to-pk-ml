"""Git-tracked text record and ledger for model runs.

This module writes only the small, human- and grep-friendly **text** record of a
run -- the resolved configs, an environment/provenance blob, per-epoch metrics,
and a headline summary -- under ``<repo>/experiments/runs/<run_id>/``, and
appends one line to ``<repo>/experiments/runs.jsonl``. Large artifacts
(checkpoints, per-step traces) do **not** belong here; they go to scratch under
``<scratch_root>/models/<run_id>/``. Only the scratch *root* is recorded here, as
provenance.

A run is identified by::

    run_id = "<UTC-timestamp>__<config-hash8>__<git-sha7>"

where the config hash is a short hash over the resolved ``(DataConfig,
RunConfig)`` pair and the git sha is the current ``HEAD`` (with a dirty flag).

Hard imports are kept light (numpy + standard library; the config objects pull in
pyyaml via their own ``to_yaml``). In particular this module never imports torch:
the device string is passed in by the caller (see :func:`write_run_record`).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from fgas_spk.paths import repo_root

if TYPE_CHECKING:  # only for type hints; not imported at runtime
    from fgas_spk.experiment import RunConfig
    from fgas_spk.loader import DataConfig

# The suppression target currently in use. Recorded verbatim so runs built
# against different target definitions are never silently compared. This is the
# known low-k-accurate / high-k-biased approximation flagged in CLAUDE.md; when
# the true paired-DMO target is computed, pass the new string as
# ``suppression_definition`` rather than editing this default.
DEFAULT_SUPPRESSION_DEFINITION = (
    "P_total(k)/P_DM(k) computed within the hydro run (the DM component as a "
    "DMO proxy); NOT the paired-DMO P_hydro/P_DMO. Acceptable at low k, "
    "feedback-correlated bias at k >~ 5 h/Mpc."
)

# Packages whose versions are recorded for provenance (best effort).
_RECORDED_PACKAGES = ("fgas-to-pk-ml", "numpy", "pyyaml", "scikit-learn", "torch")


@dataclass
class RunRecord:
    """Locations written by :func:`write_run_record`.

    Attributes:
        run_id (str): The run identifier (``<ts>__<hash8>__<sha7>``).
        run_dir (Path): The per-run directory holding the text record.
        ledger_path (Path): The ``runs.jsonl`` ledger appended to.
    """

    run_id: str
    run_dir: Path
    ledger_path: Path


def _run_git(args: list[str], repo: Path) -> str | None:
    """Run a git command in ``repo`` and return stdout, or None on failure.

    Args:
        args (list[str]): Git arguments after ``git`` (e.g. ``["rev-parse",
            "HEAD"]``).
        repo (Path): Directory to run git in.

    Returns:
        str | None: Stripped stdout, or None if git is missing or errors (e.g.
            not a checkout).
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_info(repo: str | Path | None = None) -> tuple[str, bool]:
    """Return the current git ``HEAD`` sha and whether the tree is dirty.

    Args:
        repo (str | Path | None): Repository directory. Defaults to
            :func:`fgas_spk.paths.repo_root`.

    Returns:
        tuple[str, bool]: ``(full_sha, dirty)``. ``full_sha`` is ``"unknown"`` if
            this is not a git checkout. ``dirty`` is True when the working tree
            has uncommitted or untracked changes (and False when the sha is
            unknown).
    """
    repo = Path(repo) if repo is not None else repo_root()
    sha = _run_git(["rev-parse", "HEAD"], repo)
    if sha is None:
        return "unknown", False
    # Default porcelain includes untracked files: any output means the code that
    # ran is not exactly the recorded sha.
    status = _run_git(["status", "--porcelain"], repo)
    dirty = bool(status)
    return sha, dirty


def compute_config_hash(
    data_config: "DataConfig", run_config: "RunConfig", length: int = 8
) -> str:
    """Return a short, stable hash over the resolved config pair.

    The hash is taken over a canonical JSON serialization of both dataclasses
    (keys sorted), so the same configs always hash the same regardless of field
    order.

    Args:
        data_config (DataConfig): The resolved data-selection config.
        run_config (RunConfig): The resolved run config.
        length (int): Number of leading hex characters to keep. Defaults to 8.

    Returns:
        str: The first ``length`` hex characters of the SHA-256 digest.
    """
    payload = {"data": asdict(data_config), "run": asdict(run_config)}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def make_run_id(
    config_hash: str, git_sha: str, when: datetime | None = None
) -> str:
    """Build a run id from a config hash and a git sha.

    Args:
        config_hash (str): The short config hash (see :func:`compute_config_hash`).
        git_sha (str): The git sha; the first 7 characters are used.
        when (datetime | None): Timestamp to stamp; defaults to the current UTC
            time. A timezone-naive value is treated as UTC.

    Returns:
        str: ``"<YYYYMMDDTHHMMSSZ>__<config_hash>__<sha7>"``.
    """
    when = when or datetime.now(timezone.utc)
    timestamp = when.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}__{config_hash}__{git_sha[:7]}"


def _read_dataset_meta(dataset_path: str | Path) -> dict | None:
    """Return the embedded ``__meta__`` blob from a saved dataset ``.npz``.

    Args:
        dataset_path (str | Path): Path to a dataset ``.npz`` (see
            :func:`fgas_spk.schema.save_dataset`).

    Returns:
        dict | None: The parsed metadata, or None if the file is absent or
            carries no ``__meta__`` (provenance is best effort -- a missing meta
            must not break record writing).
    """
    path = Path(dataset_path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "__meta__" in data:
            return json.loads(str(data["__meta__"]))
    return None


def _package_versions() -> dict[str, str | None]:
    """Return recorded package versions plus the Python version (best effort)."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _RECORDED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _build_env(
    *,
    data_config: "DataConfig",
    run_config: "RunConfig",
    dataset_path: str | Path,
    data_root: str | Path,
    scratch_root: str | Path,
    device: str | None,
    suppression_definition: str,
    git_sha: str,
    dirty: bool,
) -> dict:
    """Assemble the ``env.json`` provenance blob (see module docstring)."""
    return {
        "git": {"sha": git_sha, "dirty": dirty},
        "seed": run_config.seed,
        "device": device,
        "roots": {
            "data_root": str(data_root),       # read from
            "scratch_root": str(scratch_root),  # written to
        },
        "dataset": {
            "path": str(dataset_path),
            "meta": _read_dataset_meta(dataset_path),
        },
        "suppression_target": {"definition": suppression_definition},
        "target_selection": {
            "target_mode": data_config.target_mode,
            "k_target": data_config.k_target,
            "k_range": (
                list(data_config.k_range)
                if data_config.k_range is not None
                else None
            ),
        },
        "packages": _package_versions(),
    }


def write_run_record(
    data_config: "DataConfig",
    run_config: "RunConfig",
    *,
    dataset_path: str | Path,
    data_root: str | Path,
    scratch_root: str | Path,
    summary: Mapping,
    metrics: Iterable[Mapping] | None = None,
    device: str | None = None,
    suppression_definition: str = DEFAULT_SUPPRESSION_DEFINITION,
    experiments_root: str | Path | None = None,
    when: datetime | None = None,
) -> RunRecord:
    """Write the git-tracked text record for a run and append to the ledger.

    Creates ``<experiments_root>/runs/<run_id>/`` containing ``config_data.yaml``,
    ``config_run.yaml``, ``env.json``, ``metrics.jsonl`` and ``summary.json``,
    then appends one line to ``<experiments_root>/runs.jsonl``.

    Args:
        data_config (DataConfig): The resolved data-selection config actually
            used. Written verbatim to ``config_data.yaml``.
        run_config (RunConfig): The resolved run config actually used. Written
            verbatim to ``config_run.yaml``.
        dataset_path (str | Path): Path to the dataset ``.npz`` that was read; its
            embedded ``__meta__`` is captured in ``env.json``.
        data_root (str | Path): The resolved root that data was read from.
        scratch_root (str | Path): The resolved root that run outputs are written
            to (the big artifacts live under ``<scratch_root>/models/<run_id>/``).
        summary (Mapping): Headline metrics (e.g. final / test RMSE). Written to
            ``summary.json`` and carried in the ledger line.
        metrics (Iterable[Mapping] | None): Per-epoch metric rows, one JSON object
            per line in ``metrics.jsonl``. None or empty leaves the file empty
            (the case for non-iterative models like the PCA reference).
        device (str | None): Compute device string (e.g. ``"cuda"`` / ``"mps"`` /
            ``"cpu"``). Passed in so this module need not import torch. Defaults
            to None.
        suppression_definition (str): Which suppression target the data uses.
            Defaults to :data:`DEFAULT_SUPPRESSION_DEFINITION`; override when the
            target changes so runs are never silently compared across targets.
        experiments_root (str | Path | None): Root of the git-tracked record tree.
            Defaults to ``<repo_root>/experiments``.
        when (datetime | None): Timestamp for the run id; defaults to now (UTC).

    Returns:
        RunRecord: The run id and the paths written.
    """
    experiments_root = (
        Path(experiments_root)
        if experiments_root is not None
        else repo_root() / "experiments"
    )

    git_sha, dirty = git_info()
    config_hash = compute_config_hash(data_config, run_config)
    run_id = make_run_id(config_hash, git_sha, when)

    run_dir = experiments_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resolved configs, written through their own serializers.
    data_config.to_yaml(run_dir / "config_data.yaml")
    run_config.to_yaml(run_dir / "config_run.yaml")

    # Environment / provenance.
    env = _build_env(
        data_config=data_config,
        run_config=run_config,
        dataset_path=dataset_path,
        data_root=data_root,
        scratch_root=scratch_root,
        device=device,
        suppression_definition=suppression_definition,
        git_sha=git_sha,
        dirty=dirty,
    )
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    # Per-epoch metrics (one JSON object per line; possibly empty).
    with (run_dir / "metrics.jsonl").open("w") as handle:
        for row in metrics or []:
            handle.write(json.dumps(dict(row)) + "\n")

    # Headline summary.
    (run_dir / "summary.json").write_text(json.dumps(dict(summary), indent=2))

    # One greppable ledger line per run. The run-dir path is recorded relative to
    # the experiments tree's parent (the repo for a real run), so the ledger
    # stays portable.
    ledger_path = experiments_root / "runs.jsonl"
    ledger_line = {
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha[:7],
        "dirty": dirty,
        "hparams": {
            "model": run_config.model,
            "model_params": run_config.model_params,
            "seed": run_config.seed,
        },
        "metrics": dict(summary),
        "run_dir": str(run_dir.relative_to(experiments_root.parent)),
    }
    with ledger_path.open("a") as handle:
        handle.write(json.dumps(ledger_line) + "\n")

    return RunRecord(run_id=run_id, run_dir=run_dir, ledger_path=ledger_path)
