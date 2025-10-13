#!/usr/bin/env python3
"""
Final optimized configuration for rebated HFT environment
Based on institutional best practices + comprehensive testing
"""

FINAL_OPTIMIZED_CONFIG = {
    'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed.csv',
    'initial_capital': 1000000,
    'max_steps': 1000,
    'episode_length': 1000,
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
    'explicit_cancel_threshold': 0.6,
    'explicit_cancel_penalty': 1e-06,
    'explicit_cancel_clears_pending': True,
    'do_nothing_threshold': 0.8,
    'obs_qty_norm_scale': 1000.0,
    'obs_price_norm_scale': 50000.0,
    'rebate_rate_long': 6e-05,
    'rebate_rate_short': 6e-05,
    'fee_structure': 'rebate_6bps',  # Enable rebated mode (6e-05 = 0.6 bps)
    'post_only_mode': True,  # Reject orders that would become takers
    'liquidation_mode': 'risk_based',  # Options: 'always', 'risk_based', 'never'
    'liquidation_costs': False,  # Whether to charge costs on forced liquidation
}

# This configuration has been tested and validated for:
# - Order placement and cancellation functionality
# - Feature normalization consistency
# - Market microstructure best practices
# - Institutional-scale parameters
