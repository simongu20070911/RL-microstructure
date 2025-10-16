#!/usr/bin/env python3
"""
Deep investigation of taker execution pricing mechanism
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def investigate_taker_pricing():
    """Investigate exactly how taker execution pricing works."""
    print("🔬 DEEP INVESTIGATION OF TAKER EXECUTION PRICING")
    print("=" * 80)
    
    # Create simple scenario with known execution
    data = []
    for i in range(4):
        row = {'datetime': i}
        if i < 2:
            # Initial market for order placement
            base_bid, base_ask = 1800.00, 1800.10
        else:
            # Market moves to trigger execution
            base_bid, base_ask = 1800.20, 1800.30
            
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
        config = get_unified_config("baseline", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 4
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place a simple sell order at 1800.05
        action = [-1.0, 0.25, 0.0, 1.0, -1.0, 0.0]  # Sell order
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        print(f"\nOrder placed:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
            
        order_price = env.active_orders[0]['price'] if env.active_orders else 0
        order_volume = env.active_orders[0]['volume'] if env.active_orders else 0
        
        # Step to next market state
        print(f"\n📈 MARKET MOVES TO TRIGGER EXECUTION:")
        
        pre_cash = env.cash
        pre_inventory = env.long_position - env.short_position
        pre_taker = info.get('taker_volume', 0)
        
        print(f"Before execution:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Cash: ${pre_cash:.2f}")
        print(f"  Inventory: {pre_inventory:.2f}")
        print(f"  Order: SELL {order_volume:.2f} @ {order_price:.2f}")
        print(f"  Expected: Order should execute as taker at bid {env.best_bid:.2f}")
        
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        post_cash = env.cash
        post_inventory = env.long_position - env.short_position
        post_taker = info.get('taker_volume', 0)
        
        cash_change = post_cash - pre_cash
        inventory_change = post_inventory - pre_inventory
        taker_change = post_taker - pre_taker
        
        print(f"\nAfter execution:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Cash: ${post_cash:.2f} (Δ${cash_change:+.2f})")
        print(f"  Inventory: {post_inventory:.2f} (Δ{inventory_change:+.2f})")
        print(f"  Taker volume: {post_taker:.2f} (+{taker_change:.2f})")
        
        if taker_change > 0:
            print(f"\n🎯 TAKER EXECUTION ANALYSIS:")
            
            # Calculate execution details
            print(f"  Volume executed: {taker_change:.2f}")
            print(f"  Cash received: ${cash_change:+.2f}")
            print(f"  Inventory change: {inventory_change:+.2f}")
            
            # For sell order: cash_change = +volume * execution_price (positive = cash received)
            # For inventory: inventory_change = -volume (negative = sold shares)
            
            if abs(inventory_change) > 0.001:  # Avoid division by zero
                gross_execution_price = abs(cash_change) / abs(inventory_change)
                print(f"  Gross execution price: ${gross_execution_price:.2f}")
                
                # Account for transaction costs
                transaction_cost_rate = config.get('transaction_cost_long', 0.0001)
                net_execution_price = gross_execution_price / (1 - transaction_cost_rate)
                print(f"  Transaction cost rate: {transaction_cost_rate:.6f}")
                print(f"  Net execution price (before costs): ${net_execution_price:.2f}")
                
                print(f"\n  📊 PRICE COMPARISON:")
                print(f"    Order price: ${order_price:.2f}")
                print(f"    Market bid: ${env.best_bid:.2f}")
                print(f"    Market ask: ${env.best_ask:.2f}")
                print(f"    Actual execution: ${net_execution_price:.2f}")
                
                # Determine what price was used
                bid_diff = abs(net_execution_price - env.best_bid)
                ask_diff = abs(net_execution_price - env.best_ask)
                order_diff = abs(net_execution_price - order_price)
                
                if bid_diff < 0.01:
                    print(f"    ✅ Executed at BID price ${env.best_bid:.2f} (taker selling)")
                elif ask_diff < 0.01:
                    print(f"    ❓ Executed at ASK price ${env.best_ask:.2f} (unusual for sell)")
                elif order_diff < 0.01:
                    print(f"    ❓ Executed at ORDER price ${order_price:.2f} (unusual for taker)")
                else:
                    print(f"    ⚠️ Execution price ${net_execution_price:.2f} doesn't match any expected price")
                    print(f"      Bid diff: {bid_diff:.4f}, Ask diff: {ask_diff:.4f}, Order diff: {order_diff:.4f}")
        
        # Test multiple order sizes
        print(f"\n" + "=" * 50)
        print(f"🔬 TESTING DIFFERENT ORDER SIZES")
        print(f"=" * 50)
        
        # Reset and test with different volumes
        for test_volume in [0.1, 0.5, 1.0]:
            print(f"\n--- Testing with volume {test_volume} ---")
            
            obs, info = env.reset()
            
            # Place order with specific volume
            volume_signal = test_volume  # Assuming this maps to actual volume
            action = [-1.0, 0.25, 0.0, volume_signal, -1.0, 0.0]
            obs, reward, term, trunc, info = env.step(action)
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            if env.active_orders:
                actual_volume = env.active_orders[0]['volume']
                actual_price = env.active_orders[0]['price']
                print(f"  Order: SELL {actual_volume:.2f} @ {actual_price:.2f}")
                
                # Execute
                pre_cash = env.cash
                pre_inventory = env.long_position - env.short_position
                
                obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
                
                post_cash = env.cash
                post_inventory = env.long_position - env.short_position
                
                cash_change = post_cash - pre_cash
                inventory_change = post_inventory - pre_inventory
                
                if abs(inventory_change) > 0.001:
                    execution_price = abs(cash_change) / abs(inventory_change)
                    print(f"  Result: Cash Δ${cash_change:+.2f}, Inv Δ{inventory_change:+.2f}")
                    print(f"  Execution price: ${execution_price:.2f}")
                else:
                    print(f"  No execution occurred")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    investigate_taker_pricing()