"""
Rebated HFT Environments Package

This package contains variants of the HFT environment with different rebate structures.
Rebates are modeled as negative transaction costs, simulating market maker rebates
from exchanges like Binance, FTX, and other professional trading venues.

Available environments:
- HFTEnvRebated004: 0.004% rebate (4 bps)
- HFTEnvRebated006: 0.006% rebate (6 bps) 
- HFTEnvRebated008: 0.008% rebate (8 bps)

These rebate rates correspond to high-volume VIP tiers on major exchanges.
"""

from .env_rebated_004 import HFTEnvRebated004
from .env_rebated_006 import HFTEnvRebated006
from .env_rebated_008 import HFTEnvRebated008

__all__ = [
    'HFTEnvRebated004',
    'HFTEnvRebated006', 
    'HFTEnvRebated008'
]