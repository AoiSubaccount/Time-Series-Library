"""Model package initializer.

The experiment classes expect each entry in ``Exp_Basic.model_dict`` to be a
module that exposes a ``Model`` attribute.  Previously ``TimesNetRange`` and
``TimesNetRange_Mamba`` were imported as the ``Model`` class directly which
meant ``self.model_dict['TimesNetRange'].Model`` raised an ``AttributeError``.

To align with the expectation, import the submodules themselves so that the
``Model`` class can be accessed via ``module.Model``.
"""

# Import submodules rather than the classes so callers can access
# ``TimesNetRange.Model`` or ``TimesNetRange_Mamba.Model``.
from . import TimesNetRange
from . import TimesNetRange_Mamba

__all__ = ["TimesNetRange", "TimesNetRange_Mamba"]
