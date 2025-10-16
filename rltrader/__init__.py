"""
rltrader
Lightweight package namespace that provides stable import paths for
agents, environments, and config presets without moving existing files.

Use submodule imports, e.g.:
    from rltrader.agents import CachedLSTMAttention
    from rltrader.envs import RebatedHFTEnv
    from rltrader.configs import FINAL_OPTIMIZED_CONFIG
"""

__all__ = ["agents", "envs", "configs"]
