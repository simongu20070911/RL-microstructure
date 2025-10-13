#!/usr/bin/env python3
"""
Trace exactly why post-only orders aren't executing
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_simple_movement_data():
    """Create simple data with clear execution opportunity."""
    data = []
    
    for i in range(5):
        row = {'datetime': i}
        
        if i < 2:
            # Phase 1: Order placement market
            base_bid = 1799.95
            base_ask = 1800.05
        else:
            # Phase 2: Market jumps up - should execute sell orders at 1800.xx
            base_bid = 1800.20  # WAY above our sell order
            base_ask = 1800.25
        
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

def trace_post_only_issue():
    """Trace the exact issue with post-only execution."""
    print("🔍 TRACING POST-ONLY EXECUTION ISSUE")
    
    test_csv = create_simple_movement_data()
    
    try:
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 5
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Market at start: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place a sell order at exactly the ask price (should be passive)
        print(f"\n=== PLACING SELL ORDER AT ASK PRICE ===")
        # sell_offset_action = 0.0 should place at ask price (passive)
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Only sell order at ask
        
        obs, reward, term, trunc, info = env.step(action)
        
        if env.pending_orders:
            sell_order = env.pending_orders[0]
            print(f"Sell order placed at: {sell_order['price']:.2f}")
            print(f"Current ask: {env.best_ask:.2f}")
            print(f"Order is at ask: {abs(sell_order['price'] - env.best_ask) < 0.001}")
        
        # Activate order
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        if env.active_orders:
            sell_order = env.active_orders[0]
            print(f"Active sell order: {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
        
        # Now step to market movement
        print(f"\n=== STEPPING TO MARKET MOVEMENT ===")
        print(f"Before: Market {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"After: Market {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        if env.active_orders:
            sell_order = env.active_orders[0]
            print(f"Sell order still active: {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
            print(f"Should execute because: sell_price ({sell_order['price']:.2f}) <= best_bid ({env.best_bid:.2f})")
            
            # Manual execution check
            print(f"\n=== MANUAL EXECUTION CHECK ===")
            can_execute = sell_order['price'] <= env.best_bid + 1e-9
            print(f"Can execute: {can_execute}")
            
            if can_execute:
                print(f"❌ ORDER SHOULD HAVE EXECUTED BUT DIDN'T!")
                print(f"This indicates a bug in the execution system")
            else:
                print(f"✅ Order correctly remains passive")
        else:
            print(f"✅ Order was removed (executed)")
            maker_vol = info.get('maker_volume', 0)
            print(f"Maker volume: {maker_vol}")
        
        # Check if orders were cancelled during episode end
        print(f"\n=== FINAL STEP TO EPISODE END ===")
        pre_active = len(env.active_orders)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        post_active = len(env.active_orders)
        
        print(f"Active orders before final step: {pre_active}")
        print(f"Active orders after final step: {post_active}")
        print(f"Episode terminated: {term}")
        print(f"Episode truncated: {trunc}")
        
        if pre_active > 0 and post_active == 0 and (term or trunc):
            print(f"⚠️  Orders were cancelled during liquidation, not executed")
        
        print(f"\n=== FINAL ANALYSIS ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        
        print(f"Final maker volume: {final_maker}")
        print(f"Final taker volume: {final_taker}")
        
        if final_maker == 0:
            print(f"❌ CONFIRMED: Post-only mode is not executing any trades")
            print(f"Possible causes:")
            print(f"1. Orders are being cancelled at episode end instead of executing")
            print(f"2. Execution logic has an issue with post-only mode")
            print(f"3. Order placement logic has an issue")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    trace_post_only_issue()