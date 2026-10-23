"""Old name of flextool.engine; kept as an alias.

Only the public API of :mod:`flextool.engine` (its ``__all__``) and
``_provider_keys`` (``from flextool.engine_polars import _provider_keys``)
are available here.  Deep private paths such as
``flextool.engine_polars._orchestration`` are not supported: import them
from their place under :mod:`flextool.engine` instead.
"""

from flextool.engine import *  # noqa: F403
from flextool.engine import __all__ as _engine_all
from flextool.engine.preprocess import _provider_keys as _provider_keys

__all__ = list(_engine_all)
