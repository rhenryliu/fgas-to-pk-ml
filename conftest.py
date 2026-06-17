"""Pytest configuration: put the ``src/`` module directory on ``sys.path``.

The library modules live in ``src/`` (``fgas_spk_schema``, ``fgas_spk_dataset``,
``fgas_spk_loader``) as loose top-level modules -- not an installed package.
Prepending ``src/`` here lets the tests import them by name regardless of
pytest's import mode, with no editable install required. Running the loader CLI
directly (``python src/fgas_spk_loader.py``) works without this shim, because
Python already puts the script's own directory on ``sys.path``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
