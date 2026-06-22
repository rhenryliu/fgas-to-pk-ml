"""Pytest configuration: keep the ``src/`` directory on ``sys.path``.

The library is now the installed ``fgas_spk`` package (``pip install -e .``),
so the tests import it as ``fgas_spk.schema`` / ``fgas_spk.builder`` /
``fgas_spk.loader``. This shim is no longer required for that -- it is kept for
two reasons: tests still run if the editable install is momentarily absent
(``src/`` on the path makes ``import fgas_spk`` resolve to ``src/fgas_spk/``),
and it keeps the quarantined frozen backup sourceable via
``import _frozen.fgas_spk_dataset_v0`` (``src/_frozen/`` carries no
``__init__.py`` and is deliberately outside the package).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
