"""Pytest configuration for the single-module layout.

Tests import the top-level ``fgas_spk_dataset`` module directly (per CLAUDE.md,
this repo is single-module for now). Placing this ``conftest.py`` at the repo
root puts the root on ``sys.path`` so that import resolves regardless of
pytest's import mode.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
