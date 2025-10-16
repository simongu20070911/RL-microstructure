"""
rltrader.configs

Expose configuration presets from existing modules.
"""

try:
    from rltrader.lib.envs.env_rebated.final_optimized_config import (  # type: ignore[attr-defined]
        FINAL_OPTIMIZED_CONFIG,
    )
except Exception:
    FINAL_OPTIMIZED_CONFIG = None  # type: ignore

__all__ = [
    "FINAL_OPTIMIZED_CONFIG",
]
