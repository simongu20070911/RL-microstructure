#!/usr/bin/env python3
"""
Market microstructure configuration based on institutional best practices
Optimized for GBP/USD-like major FX pair with HFT market making
"""

MARKET_BEST_PRACTICES_CONFIG = {
    'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed.csv',
    'initial_capital': 1000000,
    'max_steps': 1000,
    'episode_length': 500,
    'tick_size': 1e-05,
    'lot_size': 1000,
    'max_order_volume': 100000,
    'max_inventory': 500000,
    'order_book_levels': 10,
    'max_active_orders': 10,
    'price_offset_ticks': 50,
    'allowed_aggressiveness_ticks': 10,
    'latency_steps_long': 1,
    'latency_steps_short': 1,
    'transaction_cost_long': 5e-05,
    'transaction_cost_short': 5e-05,
    'taker_penalty': 0.0002,
    'inventory_penalty': 1e-06,
    'invalid_order_penalty': 0.001,
    'activity_bonus': 1e-05,
    'quoting_reward_enabled': True,
    'quoting_reward_amount': 5e-06,
    'quoting_reward_max_ticks': 20,
    'explicit_cancel_enabled': True,
    'explicit_cancel_threshold': 0.7,
    'explicit_cancel_penalty': 1e-06,
    'explicit_cancel_clears_pending': True,
    'do_nothing_threshold': 0.8,
    'obs_qty_norm_scale': 100.0,
    'obs_price_norm_scale': 10000.0,
    'rebate_rate_long': 2e-05,
    'rebate_rate_short': 2e-05,
}

# Usage:
# from market_best_practices_config import MARKET_BEST_PRACTICES_CONFIG
# env = RebatedHFTEnv(MARKET_BEST_PRACTICES_CONFIG)
