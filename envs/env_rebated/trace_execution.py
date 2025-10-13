#!/usr/bin/env python3
"""
Trace the exact execution path step by step
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_execution_data():
    """Create minimal data for testing."""
    data = []
    
    for i in range(6):
        row = {'datetime': i}
        
        if i < 3:
            base_bid = 1799.95
            base_ask = 1800.05
        else:
            base_bid = 1800.20  # Should execute sell orders at 1800.05
            base_ask = 1800.30
        
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
    return temp_file.name

def trace_execution():
    """Trace the execution step by step."""
    print("🕵️ TRACING EXECUTION STEP BY STEP")
    
    test_csv = create_execution_data()
    
    try:
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 6
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        # Place and activate orders
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell order
        env.step(action)
        env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        # Move to execution step
        env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Active orders: {len(env.active_orders)}")
        
        # Show the sell order we want to trace
        sell_order = None
        for order in env.active_orders:
            if not order['is_buy']:
                sell_order = order
                break
        
        if sell_order:
            print(f"\\nTracing SELL order:")
            print(f"  Price: {sell_order['price']:.2f}")
            print(f"  Volume: {sell_order['volume']:.2f}")
            print(f"  Should execute against Bid1 @ {env.bids[0,0]:.2f}")
            
            # Manual execution simulation
            print(f"\\n--- MANUAL EXECUTION SIMULATION ---")
            
            # Copy the execution logic piece by piece
            order = sell_order
            is_buy = order["is_buy"]
            limit_price = order["price"]
            remaining_volume = order["volume"]
            
            print(f"Order details: is_buy={is_buy}, limit_price={limit_price:.2f}, volume={remaining_volume:.2f}")
            
            # Select LOB side to match against
            levels_to_match = env.bids  # For sell order, match against bids
            print(f"Matching against BID levels:")
            for i in range(min(3, levels_to_match.shape[0])):
                price, qty = levels_to_match[i]
                print(f"  Bid{i+1}: {price:.2f} x {qty:.1f}")
            
            # Check if any liquidity exists
            if levels_to_match.shape[0] == 0:
                print(f"❌ No liquidity on bid side!")
            else:
                print(f"✅ Liquidity exists")
                
                # Check first level
                level_price, level_qty = levels_to_match[0]
                print(f"\\nChecking Level 0: Price={level_price:.2f}, Qty={level_qty:.2f}")
                
                # Check if can fill
                can_fill_at_level = limit_price <= level_price + 1e-9  # Sell order condition
                print(f"Can fill check: limit_price ({limit_price:.2f}) <= level_price ({level_price:.2f}) + 1e-9")
                print(f"Can fill result: {can_fill_at_level}")
                
                if can_fill_at_level:
                    fill_qty_possible = min(remaining_volume, level_qty)
                    print(f"Fill quantity possible: min({remaining_volume:.2f}, {level_qty:.2f}) = {fill_qty_possible:.2f}")
                    
                    # Check inventory limits
                    current_net_inventory = env.long_position - env.short_position
                    inventory_change = -fill_qty_possible  # Sell reduces inventory
                    potential_new_inventory = current_net_inventory + inventory_change
                    max_inv = env.config["max_inventory"]
                    
                    print(f"Inventory check:")
                    print(f"  Current: {current_net_inventory:.2f}")
                    print(f"  Change: {inventory_change:.2f}")
                    print(f"  Potential: {potential_new_inventory:.2f}")
                    print(f"  Max allowed: ±{max_inv:.2f}")
                    
                    # Check if inventory limit hit
                    inventory_limit_hit = potential_new_inventory < -max_inv - 1e-9
                    print(f"  Inventory limit hit: {inventory_limit_hit}")
                    
                    if not inventory_limit_hit:
                        print(f"🎉 FILL SHOULD HAPPEN!")
                        print(f"  Fill price: {level_price:.2f}")
                        print(f"  Fill quantity: {fill_qty_possible:.2f}")
                        print(f"  Executed value: {fill_qty_possible * level_price:.2f}")
                    else:
                        print(f"❌ Fill blocked by inventory limit")
                else:
                    print(f"❌ Cannot fill at this level price")
        
        # Now call the actual execution function and see what happens
        print(f"\\n--- CALLING ACTUAL _execute_orders() ---")
        
        pre_cash = env.cash
        pre_inventory = env.inventory
        pre_active = len(env.active_orders)
        
        realized_pnl = env._execute_orders()
        
        post_cash = env.cash
        post_inventory = env.inventory
        post_active = len(env.active_orders)
        
        print(f"Execution results:")
        print(f"  Realized PnL: {realized_pnl:.6f}")
        print(f"  Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
        print(f"  Inventory: {pre_inventory:.3f} → {post_inventory:.3f} (Δ{post_inventory-pre_inventory:+.3f})")
        print(f"  Active orders: {pre_active} → {post_active} (Δ{post_active-pre_active:+d})")
        
        if abs(post_cash - pre_cash) > 0.01 or abs(post_inventory - pre_inventory) > 0.001:
            print(f"✅ EXECUTION ACTUALLY HAPPENED!")
        else:
            print(f"❌ EXECUTION FAILED DESPITE MANUAL CHECK")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    trace_execution()