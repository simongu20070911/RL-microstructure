#!/usr/bin/env python3
"""
Debug the actual LOB data being used in execution
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
    """Create data with clear execution."""
    data = []
    
    for i in range(6):
        row = {'datetime': i}
        
        if i < 3:
            # Phase 1: Stable market
            base_bid = 1799.95
            base_ask = 1800.05
        else:
            # Phase 2: Market UP
            base_bid = 1800.20  # Should hit sell orders at 1800.05
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

def debug_lob_data():
    """Debug the LOB data during execution."""
    print("🔍 DEBUGGING LOB DATA DURING EXECUTION")
    
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
        
        # Place orders
        print(f"=== STEP 1: PLACE ORDERS ===")
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell order only
        obs, reward, term, trunc, info = env.step(action)
        
        # Activate orders
        print(f"=== STEP 2: ACTIVATE ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Active orders after activation:")
        for i, order in enumerate(env.active_orders):
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {i}: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Market movement step - debug LOB data
        print(f"\n=== STEP 3: MARKET MOVEMENT (DEBUG) ===")
        
        # Step first to get new market data
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Market BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Show current LOB levels
        print(f"Current BID levels:")
        if env.bids is not None and env.bids.size > 0:
            for i in range(min(5, env.bids.shape[0])):
                price, qty = env.bids[i]
                if not np.isnan(price) and not np.isnan(qty) and qty > 0:
                    print(f"  Bid{i+1}: {price:.2f} x {qty:.1f}")
        else:
            print("  No bid data!")
        
        print(f"Current ASK levels:")
        if env.asks is not None and env.asks.size > 0:
            for i in range(min(5, env.asks.shape[0])):
                price, qty = env.asks[i]
                if not np.isnan(price) and not np.isnan(qty) and qty > 0:
                    print(f"  Ask{i+1}: {price:.2f} x {qty:.1f}")
        else:
            print("  No ask data!")
        
        # Manual execution check
        print(f"\nManual execution check:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            limit_price = order['price']
            
            if order['is_buy']:
                # Buy order - check against ask levels
                if env.asks is not None and env.asks.size > 0:
                    for i in range(env.asks.shape[0]):
                        ask_price, ask_qty = env.asks[i]
                        if not np.isnan(ask_price) and ask_qty > 0:
                            if limit_price >= ask_price:
                                print(f"  {side} order @ {limit_price:.2f} CAN execute against Ask{i+1} @ {ask_price:.2f}")
                                break
                            else:
                                print(f"  {side} order @ {limit_price:.2f} CANNOT execute against Ask{i+1} @ {ask_price:.2f}")
                                break
            else:
                # Sell order - check against bid levels
                if env.bids is not None and env.bids.size > 0:
                    for i in range(env.bids.shape[0]):
                        bid_price, bid_qty = env.bids[i]
                        if not np.isnan(bid_price) and bid_qty > 0:
                            if limit_price <= bid_price:
                                print(f"  {side} order @ {limit_price:.2f} CAN execute against Bid{i+1} @ {bid_price:.2f}")
                                break
                            else:
                                print(f"  {side} order @ {limit_price:.2f} CANNOT execute against Bid{i+1} @ {bid_price:.2f}")
                                break
        
        print(f"\nFinal state:")
        print(f"  Cash: {env.cash:.2f}")
        print(f"  Inventory: {env.inventory:.3f}")
        print(f"  Active orders: {len(env.active_orders)}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_lob_data()