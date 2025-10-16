#!/usr/bin/env python3
"""
Debug execution by directly calling _execute_orders with manual setup
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def debug_execution_direct():
    """Debug execution by directly calling methods."""
    print("🔍 DEBUGGING EXECUTION DIRECTLY")
    
    # Create minimal data
    data = []
    for i in range(3):
        row = {'datetime': i}
        base_bid = 1800.50  # High bid for execution
        base_ask = 1800.60
        
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
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 3
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Market state: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"LOB shape: bids {env.bids.shape}, asks {env.asks.shape}")
        
        # Manually create a sell order that should execute
        manual_order = {
            'id': 999,
            'price': 1800.00,  # Below current bid of 1800.50
            'volume': 0.5,
            'initial_volume': 0.5,
            'is_buy': False,
            'timestamp_placed': 0,
            'target_step': 0,
            'is_taker_at_activation': False  # Mark as maker
        }
        
        # Add to active orders
        env.active_orders = [manual_order]
        
        print(f"\nManual order created:")
        print(f"  SELL {manual_order['volume']:.2f} @ {manual_order['price']:.2f}")
        print(f"  vs Market bid: {env.best_bid:.2f}")
        print(f"  Should execute: {manual_order['price'] <= env.best_bid}")
        
        # Check execution conditions manually
        print(f"\n=== MANUAL EXECUTION CHECK ===")
        order = manual_order
        is_buy = order['is_buy']
        limit_price = order['price']
        remaining_volume = order['volume']
        
        # Select LOB side
        levels_to_match = env.bids  # For sell order
        print(f"Matching against bids:")
        for i in range(min(3, levels_to_match.shape[0])):
            price, qty = levels_to_match[i]
            print(f"  Level {i}: {price:.2f} x {qty:.1f}")
        
        # Check if can fill
        level_price, level_qty = levels_to_match[0]
        can_fill = limit_price <= level_price + 1e-9
        print(f"\nFill check:")
        print(f"  Limit price: {limit_price:.2f}")
        print(f"  Level price: {level_price:.2f}")
        print(f"  Can fill: {limit_price:.2f} <= {level_price:.2f} + 1e-9 = {can_fill}")
        
        if can_fill:
            fill_qty = min(remaining_volume, level_qty)
            print(f"  Fill quantity: min({remaining_volume:.2f}, {level_qty:.1f}) = {fill_qty:.2f}")
            
            # Check inventory
            current_inventory = env.long_position - env.short_position
            inventory_change = -fill_qty  # Sell
            new_inventory = current_inventory + inventory_change
            max_inv = env.config["max_inventory"]
            
            print(f"  Inventory: {current_inventory:.2f} → {new_inventory:.2f} (max: ±{max_inv:.2f})")
            print(f"  Within limits: {abs(new_inventory) <= max_inv}")
        
        # Now call actual execution
        print(f"\n=== CALLING _execute_orders() ===")
        pre_cash = env.cash
        pre_maker = getattr(env, 'maker_volume', 0)
        pre_active = len(env.active_orders)
        
        print(f"Before execution:")
        print(f"  Cash: ${pre_cash:.2f}")
        print(f"  Maker volume: {pre_maker:.2f}")
        print(f"  Active orders: {pre_active}")
        
        # Call execution method directly
        realized_pnl = env._execute_orders()
        
        post_cash = env.cash
        post_maker = getattr(env, 'maker_volume', 0)
        post_active = len(env.active_orders)
        
        print(f"\nAfter execution:")
        print(f"  Cash: ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
        print(f"  Maker volume: {post_maker:.2f} (Δ{post_maker-pre_maker:+.2f})")
        print(f"  Active orders: {post_active} (Δ{post_active-pre_active:+d})")
        print(f"  Realized PnL: {realized_pnl:+.6f}")
        
        if post_maker > pre_maker or abs(post_cash - pre_cash) > 0.01:
            print(f"\n✅ EXECUTION WORKED!")
        else:
            print(f"\n❌ EXECUTION FAILED - there's a bug in _execute_orders()")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    debug_execution_direct()