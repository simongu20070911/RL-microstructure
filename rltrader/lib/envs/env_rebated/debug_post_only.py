#!/usr/bin/env python3
"""
Debug post-only mode to see why taker orders are still happening
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_simple_data():
    """Create simple test data."""
    data = []
    
    for i in range(20):
        row = {'datetime': i}
        base_bid = 1799.99
        base_ask = 1800.01
        
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

def debug_post_only():
    """Debug post-only mode step by step."""
    print("🔍 DEBUGGING POST-ONLY MODE")
    
    test_csv = create_simple_data()
    
    try:
        config = get_unified_config("rebate_6bps", True)  # Post-only enabled
        config["csv_path"] = test_csv
        config["episode_length"] = 10
        config["latency_steps_long"] = 1
        config["latency_steps_short"] = 1
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment setup:")
        print(f"  Post-only mode: {env.post_only_mode}")
        print(f"  Allowed aggressiveness ticks: {env.config['allowed_aggressiveness_ticks']}")
        print(f"  BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Tick size: {env.config['tick_size']}")
        
        for step in range(5):
            print(f"\n=== STEP {step} ===")
            
            # Try placing very aggressive orders
            action = [0.9, -0.9, 0.8, 0.8, -1.0, -1.0]
            print(f"Action: VERY AGGRESSIVE [0.9, -0.9, 0.8, 0.8, -1.0, -1.0]")
            
            pre_active = len(env.active_orders)
            pre_pending = len(env.pending_orders)
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_active = len(env.active_orders)
            post_pending = len(env.pending_orders)
            
            print(f"Orders: Active {pre_active}→{post_active}, Pending {pre_pending}→{post_pending}")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Examine all orders
            if env.active_orders or env.pending_orders:
                print(f"Active orders:")
                for i, order in enumerate(env.active_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    price = order['price']
                    
                    # Check if order crosses spread
                    if order['is_buy'] and price >= env.best_ask:
                        crosses = f"❌ CROSSES (buy {price:.2f} >= ask {env.best_ask:.2f})"
                    elif not order['is_buy'] and price <= env.best_bid:
                        crosses = f"❌ CROSSES (sell {price:.2f} <= bid {env.best_bid:.2f})"
                    else:
                        crosses = f"✅ PASSIVE"
                    
                    print(f"  {i}: {side} @ {price:.2f} - {crosses}")
                
                print(f"Pending orders:")
                for i, order in enumerate(env.pending_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    price = order['price']
                    
                    # Check if order would cross spread when activated
                    if order['is_buy'] and price >= env.best_ask:
                        would_cross = f"❌ WILL CROSS (buy {price:.2f} >= ask {env.best_ask:.2f})"
                    elif not order['is_buy'] and price <= env.best_bid:
                        would_cross = f"❌ WILL CROSS (sell {price:.2f} <= bid {env.best_bid:.2f})"
                    else:
                        would_cross = f"✅ WILL BE PASSIVE"
                    
                    print(f"  {i}: {side} @ {price:.2f} - {would_cross}")
            else:
                print(f"  No orders placed (expected in post-only with aggressive actions)")
            
            if term or trunc:
                break
        
        print(f"\n=== FINAL CHECK ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        
        print(f"Final maker volume: {final_maker}")
        print(f"Final taker volume: {final_taker}")
        
        if final_taker > 0:
            print(f"❌ POST-ONLY MODE BROKEN: {final_taker} taker volume detected")
        else:
            print(f"✅ POST-ONLY MODE WORKING: No taker orders")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_post_only()