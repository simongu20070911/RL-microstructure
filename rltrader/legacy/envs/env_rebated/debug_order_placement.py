#!/usr/bin/env python3
"""
Debug the order placement price calculation system
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def debug_order_placement():
    """Debug order placement price calculations."""
    print("🔍 DEBUGGING ORDER PLACEMENT PRICE CALCULATIONS")
    
    # Create simple data
    data = []
    for i in range(3):
        row = {'datetime': i}
        base_bid = 1800.00
        base_ask = 1800.10
        
        for level in range(1, 11):
            row[f'bid{level}'] = base_bid - (level-1) * 0.01
            row[f'bidqty{level}'] = 100.0
            row[f'ask{level}'] = base_ask + (level-1) * 0.01  
            row[f'askqty{level}'] = 100.0
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Test both normal and post-only modes
        for post_only in [False, True]:
            mode = "POST-ONLY" if post_only else "NORMAL"
            print(f"\n{'='*15} {mode} MODE {'='*15}")
            
            config = get_unified_config("rebate_6bps", post_only)
            config["csv_path"] = temp_file.name
            config["episode_length"] = 3
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 0.5
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"Config:")
            print(f"  Post-only: {env.post_only_mode}")
            print(f"  Price offset ticks: {env.config['price_offset_ticks']}")
            print(f"  Allowed aggressiveness ticks: {env.config['allowed_aggressiveness_ticks']}")
            print(f"  Tick size: {env.config['tick_size']}")
            
            # Test different action values for sell orders
            test_actions = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            for sell_action in test_actions:
                print(f"\n--- Testing sell_offset_action = {sell_action} ---")
                
                # Calculate expected price manually
                max_passive_offset_ticks = env.config["price_offset_ticks"]
                tick_size = env.config["tick_size"]
                best_ask = env.best_ask
                
                calculated_tick_offset = sell_action * max_passive_offset_ticks
                expected_price = best_ask + calculated_tick_offset * tick_size
                
                print(f"Manual calculation:")
                print(f"  Reference ask: {best_ask:.2f}")
                print(f"  Tick offset: {sell_action} * {max_passive_offset_ticks} = {calculated_tick_offset:.2f}")
                print(f"  Expected price: {best_ask:.2f} + {calculated_tick_offset:.2f} * {tick_size} = {expected_price:.2f}")
                
                # Apply aggressiveness clipping
                if post_only:
                    min_allowed_sell_price = env.best_ask  # Post-only: at or above ask
                    clipped_price = max(expected_price, min_allowed_sell_price)
                    print(f"  Post-only clipping: max({expected_price:.2f}, {min_allowed_sell_price:.2f}) = {clipped_price:.2f}")
                else:
                    max_aggressive_ticks = env.config["allowed_aggressiveness_ticks"]
                    min_allowed_sell_price = max(tick_size, env.best_bid - max_aggressive_ticks * tick_size)
                    clipped_price = max(expected_price, min_allowed_sell_price)
                    print(f"  Normal clipping: max({expected_price:.2f}, {min_allowed_sell_price:.2f}) = {clipped_price:.2f}")
                
                # Test actual placement
                action = [-1.0, sell_action, 0.0, 0.8, -1.0, -1.0]  # Only sell order
                
                # Reset environment to clean state
                obs, info = env.reset()
                
                pre_pending = len(env.pending_orders)
                obs, reward, term, trunc, info = env.step(action)
                post_pending = len(env.pending_orders)
                
                print(f"  Placement result:")
                print(f"    Pending orders: {pre_pending} → {post_pending}")
                
                if env.pending_orders:
                    actual_order = env.pending_orders[-1]  # Get last placed order
                    actual_price = actual_order['price']
                    print(f"    Actual price: {actual_price:.2f}")
                    print(f"    Expected: {clipped_price:.2f}")
                    print(f"    Match: {abs(actual_price - clipped_price) < 0.001}")
                    
                    # Check if price is executable
                    print(f"    Executable vs bid {env.best_bid:.2f}: {actual_price <= env.best_bid}")
                    print(f"    Distance from ask: {actual_price - env.best_ask:.2f} ticks")
                else:
                    print(f"    ❌ No order placed!")
            
            env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    debug_order_placement()