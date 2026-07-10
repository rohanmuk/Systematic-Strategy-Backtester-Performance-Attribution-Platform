"""Asset-allocation backtester.

A config-driven engine for backtesting multi-asset allocation strategies and
analysing them on a benchmark-relative basis. The analytical core
(``backtester.metrics``) implements every formula from first principles so the
methodology is fully auditable.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Number of return observations per year at the engine's native frequency.
# The whole project runs on monthly total returns, so this is 12 everywhere.
MONTHS_PER_YEAR = 12

__all__ = ["__version__", "MONTHS_PER_YEAR"]
