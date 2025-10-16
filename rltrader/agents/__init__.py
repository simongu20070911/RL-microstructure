"""Agent interfaces exposed by the rltrader package.

The core implementations live under :mod:`rltrader.legacy.agents`.  This
module provides clean, well-named access points so new code never needs to
touch the legacy tree directly.
"""

from .base import CachedStates, ValidationStates
from .features import (
    CachedLSTMAttention,
    CachedMultiHeadAttention,
    TimeSeriesEnvWrapper,
)
from .market_maker import create_sac_agent, create_test_env

__all__ = [
    "CachedStates",
    "ValidationStates",
    "CachedMultiHeadAttention",
    "CachedLSTMAttention",
    "TimeSeriesEnvWrapper",
    "create_sac_agent",
    "create_test_env",
]

