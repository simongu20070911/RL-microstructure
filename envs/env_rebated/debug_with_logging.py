#!/usr/bin/env python3
"""
Debug execution with detailed logging enabled
"""

import sys
import os
import tempfile
import pandas as pd
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

# Enable DEBUG logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

def create_simple_execution_data():
    """Create simple data with clear execution opportunity."""
    data = []
    
    for i in range(8):
        row = {'datetime': i}
        
        if i < 3:
            # Phase 1: Stable market
            base_bid = 1799.95
            base_ask = 1800.05
        else:
            # Phase 2: Market jumps UP - should hit sell orders at 1800.05
            base_bid = 1800.20  # WAY above our sell order price
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

def debug_execution_with_logging():
    """Debug execution with full logging."""
    print("🔍 EXECUTION DEBUG WITH LOGGING")
    
    test_csv = create_simple_execution_data()
    
    try:
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 6
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        config["max_active_orders"] = 3
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Step 1: Place order
        print(f"\n=== STEP 1: PLACE ORDER ===")
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Only sell order
        obs, reward, term, trunc, info = env.step(action)
        
        # Step 2: Activate
        print(f"\n=== STEP 2: ACTIVATE ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Active orders: {len(env.active_orders)}")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Step 3: Market movement - should execute
        print(f"\n=== STEP 3: EXECUTION STEP ===")
        print(f"Before: BBO {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"After: BBO {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Cash: {env.cash:.2f}")
        print(f"Inventory: {env.inventory:.3f}")
        print(f"Active orders remaining: {len(env.active_orders)}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_execution_with_logging()