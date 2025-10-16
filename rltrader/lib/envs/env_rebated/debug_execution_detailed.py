#!/usr/bin/env python3
"""
Debug execution with detailed logging
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

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def create_simple_data():
    """Create minimal data for execution test."""
    data = []
    
    for i in range(4):
        row = {'datetime': i}
        
        if i < 2:
            # Phase 1: Order placement
            base_bid = 1799.90
            base_ask = 1800.10
        else:
            # Phase 2: Market up - should execute sell
            base_bid = 1800.50  # WAY above any sell order
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

def debug_execution_detailed():
    """Debug execution with full logging."""
    print("🔍 DEBUGGING EXECUTION WITH FULL LOGGING")
    
    test_csv = create_simple_data()
    
    try:
        config = get_unified_config("rebate_6bps", False)  # Normal mode first
        config["csv_path"] = test_csv
        config["episode_length"] = 4
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        config["price_offset_ticks"] = 2  # Smaller offset
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place sell order
        print(f"\n=== PLACING SELL ORDER ===")
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell at ask price
        obs, reward, term, trunc, info = env.step(action)
        
        # Activate
        print(f"\n=== ACTIVATING ORDER ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Active orders after activation:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            taker_status = order.get('is_taker_at_activation', 'N/A')
            print(f"  {side} {order['volume']:.2f} @ {order['price']:.2f} (Taker: {taker_status})")
        
        # Step to execution
        print(f"\n=== STEPPING TO EXECUTION (DEBUG LOGGING ENABLED) ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"\nAfter execution attempt:")
        print(f"Active orders remaining: {len(env.active_orders)}")
        print(f"Maker volume: {info.get('maker_volume', 0)}")
        print(f"Taker volume: {info.get('taker_volume', 0)}")
        print(f"Cash: {env.cash}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_execution_detailed()