#!/usr/bin/env python3
"""
Test post-only mode with market data that should guarantee execution
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_execution_guaranteed_data():
    """Create data that guarantees passive order execution."""
    data = []
    
    for i in range(10):
        row = {'datetime': i}
        
        if i < 3:
            # Phase 1: Wide spread for order placement
            base_bid = 1799.95
            base_ask = 1800.05  # 10 tick spread
        elif i < 6:
            # Phase 2: Market moves up dramatically - should hit sell orders
            base_bid = 1800.10  # Bid jumps WAY above initial ask
            base_ask = 1800.15
        else:
            # Phase 3: Market crashes down - should hit buy orders 
            base_bid = 1799.85  # Bid drops WAY below initial bid
            base_ask = 1799.90
        
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

def test_post_only_execution():
    """Test if post-only mode can actually execute trades."""
    print("🎯 TESTING POST-ONLY MODE EXECUTION WITH GUARANTEED MARKET MOVEMENT")
    
    test_csv = create_execution_guaranteed_data()
    
    try:
        # Test post-only mode
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 10
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        config["price_offset_ticks"] = 3  # Reduce offset for closer orders
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place orders with moderate passiveness
        print(f"\n=== PLACING ORDERS CLOSER TO MARKET ===")
        # sell_offset_action = 0.5 means less passive (closer to ask)
        # buy_offset_action = -0.5 means less passive (closer to bid)
        action = [-0.5, 0.5, 0.8, 0.8, -1.0, -1.0]  # Both buy and sell orders
        
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"Orders placed:")
        for order in env.pending_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  Pending: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Activate orders
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"\nOrders activated:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  Active: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        execution_log = []
        
        # Step through dramatic market movements
        print(f"\n=== STEPPING THROUGH DRAMATIC MARKET MOVEMENTS ===")
        
        step_count = 0
        while step_count < 8 and not (term or trunc):
            step_count += 1
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            
            print(f"\nStep {step_count}:")
            print(f"  Market before: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            
            print(f"  Market after:  {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"  Volume: Maker {pre_maker:.2f}→{post_maker:.2f}, Taker {pre_taker:.2f}→{post_taker:.2f}")
            print(f"  Cash: ${pre_cash:.2f}→${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
            
            if post_maker > pre_maker or post_taker > pre_taker:
                execution_log.append({
                    'step': step_count,
                    'maker_change': post_maker - pre_maker,
                    'taker_change': post_taker - pre_taker,
                    'cash_change': post_cash - pre_cash,
                    'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                })
                print(f"  🎉 EXECUTION DETECTED!")
        
        print(f"\n=== EXECUTION SUMMARY ===")
        if execution_log:
            print("Executions found:")
            for exec_data in execution_log:
                print(f"  Step {exec_data['step']}: "
                      f"Maker +{exec_data['maker_change']:.2f}, "
                      f"Taker +{exec_data['taker_change']:.2f}, "
                      f"Cash ${exec_data['cash_change']:+.2f}, "
                      f"Market {exec_data['market']}")
        else:
            print("❌ NO EXECUTIONS FOUND")
        
        print(f"\n=== FINAL RESULTS ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"Final Maker Volume: {final_maker:.2f}")
        print(f"Final Taker Volume: {final_taker:.2f}")
        print(f"Total Rebates Earned: ${final_rebates:.6f}")
        print(f"Cash Change: ${final_cash_change:+.2f}")
        
        # Analyze results
        if final_maker > 0:
            print(f"\n✅ SUCCESS: Post-only mode CAN execute trades!")
            print(f"   Total volume: {final_maker:.2f}")
            print(f"   Rebates earned: ${final_rebates:.6f}")
            print(f"   Net PnL: ${final_cash_change:+.2f}")
            if final_cash_change < 0:
                print(f"   ⚠️  Note: Negative PnL as expected for maker-only trading")
        else:
            print(f"\n❌ ISSUE: Post-only mode still shows 0 volume")
            print(f"   This suggests either:")
            print(f"   1. Orders are still too passive even with closer placement")
            print(f"   2. Post-only logic has a bug")
            print(f"   3. Market movement is still insufficient")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    test_post_only_execution()