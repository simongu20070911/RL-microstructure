#!/usr/bin/env python3
"""
Trace the exact execution bug by manually calling _execute_orders
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_execution_test_data():
    """Create simple data for execution test."""
    data = []
    
    for i in range(3):
        row = {'datetime': i}
        
        if i < 2:
            # Phase 1: Order placement
            base_bid = 1799.90
            base_ask = 1800.10
        else:
            # Phase 2: Market up - should execute sell
            base_bid = 1800.50  # WAY above sell orders
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
    return temp_file.name

def trace_execution_bug():
    """Trace the exact execution bug."""
    print("🔍 TRACING EXECUTION BUG - ULTRATHINK MODE")
    
    test_csv = create_execution_test_data()
    
    try:
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 3
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial state:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Bids shape: {env.bids.shape if env.bids is not None else 'None'}")
        print(f"  Asks shape: {env.asks.shape if env.asks is not None else 'None'}")
        
        # Place and activate a sell order
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell order
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        print(f"\nAfter activation:")
        print(f"  Active orders: {len(env.active_orders)}")
        for i, order in enumerate(env.active_orders):
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"    {i}: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Step to execution market
        print(f"\n=== STEPPING TO EXECUTION MARKET ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"After market step:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Bids[0]: {env.bids[0] if env.bids is not None and len(env.bids) > 0 else 'None'}")
        print(f"  Active orders: {len(env.active_orders)}")
        
        # Now manually trace the execution
        print(f"\n=== MANUAL EXECUTION TRACE ===")
        if env.active_orders:
            sell_order = None
            for order in env.active_orders:
                if not order['is_buy']:
                    sell_order = order
                    break
            
            if sell_order:
                print(f"Tracing sell order: {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
                
                # Check LOB state
                current_bids = env.bids.copy() if env.bids is not None and env.bids.size > 0 else np.array([])
                print(f"Current bids shape: {current_bids.shape}")
                
                if current_bids.shape[0] > 0:
                    print(f"Bid levels available:")
                    for i in range(min(3, current_bids.shape[0])):
                        price, qty = current_bids[i]
                        print(f"  Level {i}: {price:.2f} x {qty:.1f}")
                    
                    # Check execution condition
                    level_price, level_qty = current_bids[0]
                    limit_price = sell_order['price']
                    can_fill = limit_price <= level_price + 1e-9
                    
                    print(f"\nExecution check:")
                    print(f"  Sell limit price: {limit_price:.2f}")
                    print(f"  Best bid price: {level_price:.2f}")
                    print(f"  Can fill: {limit_price:.2f} <= {level_price:.2f} + 1e-9 = {can_fill}")
                    
                    if can_fill:
                        print(f"  ✅ ORDER SHOULD EXECUTE!")
                        print(f"  Fill quantity: min({sell_order['volume']:.2f}, {level_qty:.1f}) = {min(sell_order['volume'], level_qty):.2f}")
                        
                        # Check inventory limits
                        current_inventory = env.long_position - env.short_position
                        inventory_change = -min(sell_order['volume'], level_qty)  # Sell reduces inventory
                        new_inventory = current_inventory + inventory_change
                        max_inv = env.config["max_inventory"]
                        
                        print(f"  Inventory check:")
                        print(f"    Current: {current_inventory:.2f}")
                        print(f"    Change: {inventory_change:.2f}")
                        print(f"    New: {new_inventory:.2f}")
                        print(f"    Max: ±{max_inv:.2f}")
                        print(f"    Within limits: {abs(new_inventory) <= max_inv}")
                    else:
                        print(f"  ❌ Order cannot fill at this price")
                else:
                    print(f"❌ NO BIDS AVAILABLE - This is the bug!")
                    print(f"env.bids: {env.bids}")
        
        # Check final state
        print(f"\n=== FINAL STATE ===")
        print(f"Maker volume: {info.get('maker_volume', 0)}")
        print(f"Taker volume: {info.get('taker_volume', 0)}")
        print(f"Cash: {env.cash}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    trace_execution_bug()