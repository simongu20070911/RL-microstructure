#!/usr/bin/env python3
"""
FINAL DIAGNOSIS: Find the exact bug preventing execution
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

# Enable full debug logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def final_diagnosis():
    """Find the exact execution bug."""
    print("🔬 FINAL DIAGNOSIS: FINDING THE EXECUTION BUG")
    
    # Create simple data with clear execution
    data = []
    for i in range(3):
        row = {'datetime': i}
        if i == 0:
            base_bid, base_ask = 1800.00, 1800.10  # For order placement
        else:
            base_bid, base_ask = 1800.50, 1800.60  # For execution
        
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
        
        print(f"Initial: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place a sell order at ask (should be passive)
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # sell at ask
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # activate
        
        if env.active_orders:
            sell_order = env.active_orders[0]
            print(f"\nActive order: SELL {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
            print(f"Taker status: {sell_order.get('is_taker_at_activation', 'N/A')}")
        
        # Step to execution market and debug the execution call
        print(f"\n🔬 DEBUGGING EXECUTION CALL:")
        print(f"Before step - Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Before step - Active orders: {len(env.active_orders)}")
        
        # I'll patch the _execute_orders method to add debug info
        original_execute = env._execute_orders
        
        def debug_execute_orders():
            print(f"\n--- _execute_orders() CALLED ---")
            print(f"Active orders count: {len(env.active_orders)}")
            
            for i, order in enumerate(env.active_orders):
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"Order {i}: {side} {order['volume']:.2f} @ {order['price']:.2f}")
            
            print(f"LOB bids shape: {env.bids.shape if env.bids is not None else 'None'}")
            print(f"LOB asks shape: {env.asks.shape if env.asks is not None else 'None'}")
            
            if env.bids is not None and env.bids.size > 0:
                print(f"Bid levels:")
                for i in range(min(3, env.bids.shape[0])):
                    price, qty = env.bids[i]
                    print(f"  Bid{i+1}: {price:.2f} x {qty:.1f}")
            
            # Call original method
            result = original_execute()
            
            print(f"Execution returned: {result:.6f}")
            print(f"Active orders after: {len(env.active_orders)}")
            print(f"--- _execute_orders() COMPLETE ---\n")
            
            return result
        
        # Replace the method
        env._execute_orders = debug_execute_orders
        
        # Now step and watch execution
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"After step - Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"After step - Active orders: {len(env.active_orders)}")
        print(f"Volume: {info.get('maker_volume', 0):.2f}")
        print(f"Cash change: {env.cash - 100000:+.2f}")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    final_diagnosis()