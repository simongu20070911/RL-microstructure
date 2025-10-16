# -*- coding: utf-8 -*-
"""
HFT Environment with 0.004% Rebate (4 bps)

This environment extends the base HFT environment to simulate market maker rebates
of 0.004% (4 basis points). This rebate rate corresponds to high-volume VIP tiers
on exchanges like Binance VIP 5-6 level.

Key changes from base environment:
- transaction_cost_long: -0.00004 (negative = rebate)
- transaction_cost_short: -0.00004 (negative = rebate)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_2sided import HFTEnv
import copy


class HFTEnvRebated004(HFTEnv):
    """
    HFT Environment with 0.004% rebate for market making activities.
    
    Rebates are modeled as negative transaction costs, meaning the trader
    receives money back for providing liquidity rather than paying fees.
    """
    
    def __init__(self, config):
        # Create a copy of config to avoid modifying the original
        rebated_config = copy.deepcopy(config)
        
        # Check if post-only mode is enabled
        self.post_only_mode = rebated_config.get("post_only_mode", False)
        
        if self.post_only_mode:
            # Post-only: reject aggressive orders, only makers get rebates
            rebated_config["allowed_aggressiveness_ticks"] = 0  # No aggressive orders
            rebated_config["transaction_cost_long"] = -0.00004   # 4 bps rebate for makers
            rebated_config["transaction_cost_short"] = -0.00004  # 4 bps rebate for makers
            rebated_config["taker_penalty"] = 0.0  # No taker penalty needed in post-only
        else:
            # Mixed mode: makers get rebates, takers pay fees
            # This is more complex - for now keep original behavior
            rebated_config["transaction_cost_long"] = 0.0001   # 1 bps fee for takers
            rebated_config["transaction_cost_short"] = 0.0001  # 1 bps fee for takers
            # Note: True maker-taker distinction requires core logic changes
        
        # Store rebate information for tracking
        self.rebate_rate = 0.00004
        self.rebate_bps = 4
        
        # Initialize the parent class with rebated config
        super().__init__(rebated_config)
        
        # Track rebate statistics
        self.total_rebates_earned = 0.0
        self.rebated_volume = 0.0
        
    def step(self, action):
        """Override step to track rebate earnings correctly."""
        # Store pre-step state
        pre_step_volume = getattr(self, 'last_executed_volume', 0.0)
        
        # Call parent step method
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Calculate incremental volume this step
        current_volume = getattr(self, 'last_executed_volume', 0.0)
        step_volume = current_volume - pre_step_volume
        
        # Only process if there was actual trading volume this step
        step_rebate = 0.0
        if step_volume > 0:
            if self.post_only_mode:
                # Post-only mode: all executed orders are makers, so all get rebates
                # The negative transaction cost in config already provides the actual rebate
                step_rebate = step_volume * self.rebate_rate * self.midprice
                
                # Update tracking
                self.total_rebates_earned += step_rebate
                self.rebated_volume += step_volume
            else:
                # Mixed mode: wrapper limitation - cannot distinguish maker vs taker
                # For now, no rebate tracking in mixed mode
                # TODO: Would need core logic changes to properly implement maker-taker distinction
                pass
        
        # Add rebate information to info dict
        info['rebate_rate'] = self.rebate_rate
        info['rebate_bps'] = self.rebate_bps
        info['post_only_mode'] = self.post_only_mode
        info['total_rebates_earned'] = self.total_rebates_earned
        info['rebated_volume'] = self.rebated_volume
        info['step_rebate'] = step_rebate
        info['step_volume'] = step_volume
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Override reset to initialize rebate tracking."""
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset rebate tracking
        self.total_rebates_earned = 0.0
        self.rebated_volume = 0.0
        
        # Add rebate information to info dict
        info['rebate_rate'] = self.rebate_rate
        info['rebate_bps'] = self.rebate_bps
        info['post_only_mode'] = self.post_only_mode
        info['total_rebates_earned'] = self.total_rebates_earned
        info['rebated_volume'] = self.rebated_volume
        
        return observation, info
        

def get_default_rebated_config_004():
    """
    Returns a default configuration for 0.004% rebated environment.
    Based on the original environment but with rebated transaction costs.
    """
    config = {
        # Data and Environment
        "csv_path": "/home/gaen/Documents/billions_db/orderbooks/binance/futures/ethusdc/30-Mar-2025/binance_futures_ethusdc_orderbook_30-Mar-2025.csv",
        "initial_capital": 100000.0,
        "order_book_levels": 10,
        "max_steps": 50000,
        "episode_length": 10000,
        
        # Market Microstructure
        "tick_size": 0.01,
        "lot_size": 0.001,
        "max_order_volume": 10.0,
        "max_inventory": 50.0,
        
        # Order Management
        "max_active_orders": 10,
        "latency_steps_long": 2,
        "latency_steps_short": 1,
        
        # Rebated Transaction Costs (4 bps rebate)
        "transaction_cost_long": -0.00004,   # 4 bps rebate for buys
        "transaction_cost_short": -0.00004,  # 4 bps rebate for sells
        
        # Risk Management
        "inventory_penalty": 0.001,
        "invalid_order_penalty": -1.0,
        "taker_penalty": -0.01,
        
        # Action Controls
        "price_offset_ticks": 5,
        "allowed_aggressiveness_ticks": 3,
        
        # Rewards
        "activity_bonus": 0.001,
        "quoting_reward_enabled": True,
        "quoting_reward_amount": 0.0001,
        "quoting_reward_max_ticks": 5,
        
        # Cancellation
        "explicit_cancel_enabled": True,
        "explicit_cancel_threshold": 0.5,
        "explicit_cancel_penalty": 0.1,
        "explicit_cancel_clears_pending": True,
        
        # Do Nothing Action
        "do_nothing_threshold": 0.8,
        
        # Observation Normalization
        "obs_price_norm_scale": 100.0,
        "obs_qty_norm_scale": 10.0
    }
    
    return config


if __name__ == "__main__":
    # Test the rebated environment
    import numpy as np
    
    config = get_default_rebated_config_004()
    env = HFTEnvRebated004(config)
    
    print("Testing HFTEnvRebated004 Environment")
    print(f"Rebate Rate: {env.rebate_rate:.6f} ({env.rebate_bps} bps)")
    print(f"Transaction Cost Long: {env.config['transaction_cost_long']:.6f}")
    print(f"Transaction Cost Short: {env.config['transaction_cost_short']:.6f}")
    
    # Test episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")
    
    # Take a few random actions
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: Reward={reward:.4f}, Rebates={info.get('total_rebates_earned', 0.0):.6f}")
        
        if terminated or truncated:
            break
    
    print("Test completed successfully!")