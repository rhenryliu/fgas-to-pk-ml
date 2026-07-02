"""Model-plugin layer for f_gas(R) to SP(k) models.

The public surface is the :class:`~fgas_spk.models.base.ProfileToSpk` interface,
the :data:`~fgas_spk.models.base.REGISTRY` (model name -> class), and the
:func:`~fgas_spk.models.base.register` decorator. Look a model up by name with
``fgas_spk.models.REGISTRY[name]``.

Importing this package imports every plugin module below **for its side effect**:
each plugin's ``@register(...)`` decorator runs at import time and inserts the
class into ``REGISTRY``. A registry that never imports its plugins is silently
empty, so these imports are deliberate -- add a line here for every new plugin.

This layer pulls in scikit-learn (via the plugins) and so is opt-in: it is *not*
imported by the top-level :mod:`fgas_spk` package, which keeps its core import
surface to numpy + pyyaml. Import :mod:`fgas_spk.models` explicitly to use it.
"""

from fgas_spk.models.base import REGISTRY, ProfileToSpk, register

# Side-effect imports: running each module executes its @register decorator,
# which is what populates REGISTRY. Add new plugins to this list. These modules
# import their heavy deps (torch, sklearn) lazily inside methods, so importing
# this package stays torch-free -- only the @register side effect runs here.
from fgas_spk.models import mlp  # noqa: F401  (registers "mlp")
from fgas_spk.models import mlp_regressor  # noqa: F401  (registers "mlp_regressor")
from fgas_spk.models import pca_linear  # noqa: F401  (registers "pca_linear")
from fgas_spk.models import vib_regressor  # noqa: F401  (registers "vib_regressor")
from fgas_spk.models import cvae  # noqa: F401  (registers "cvae")

__all__ = [
    "REGISTRY",
    "ProfileToSpk",
    "register",
    "mlp",
    "mlp_regressor",
    "pca_linear",
    "vib_regressor",
    "cvae",
]
