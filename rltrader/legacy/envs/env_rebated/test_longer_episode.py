#!/usr/bin/env python3
"""
Test post-only mode with longer episodes to allow execution
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_longer_execution_data():
    """Create data with longer episode and clear execution opportunities."""
    data = []
    
    for i in range(12):  # Longer episode
        row = {'datetime': i}
        
        if i < 4:
            # Phase 1: Order placement - stable market
            base_bid = 1799.90
            base_ask = 1800.10
        elif i < 8:
            # Phase 2: Market moves up - execute sell orders  
            base_bid = 1800.50  # Way above sell orders
            base_ask = 1800.60
        else:
            # Phase 3: Market moves down - execute buy orders
            base_bid = 1799.50  # Way below buy orders
            base_ask = 1799.60
        
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

def test_longer_episode():
    """Test with longer episodes to allow execution."""
    print("🧪 TESTING POST-ONLY MODE WITH LONGER EPISODES")
    
    test_csv = create_longer_execution_data()
    
    try:
        # Test post-only mode with longer episode
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 12  # Much longer episode
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Config: Episode length = {config['episode_length']}")
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place orders early in episode
        print(f"\n=== STEP 1-2: PLACE AND ACTIVATE ORDERS ===")
        action = [-0.5, 0.5, 0.8, 0.8, -1.0, -1.0]  # Both buy and sell orders
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        print(f"Orders placed and activated:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        execution_log = []
        
        # Step through the longer episode
        print(f"\n=== STEPPING THROUGH LONGER EPISODE ===")
        step_count = 2
        while step_count < 10 and not (term or trunc):
            step_count += 1
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            pre_active = len(env.active_orders)
            
            print(f"\nStep {step_count}:")
            print(f"  Before: Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active orders: {pre_active}")
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            post_active = len(env.active_orders)
            
            print(f"  After:  Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active orders: {post_active}")
            print(f"  Volume: Maker {pre_maker:.2f}→{post_maker:.2f}, Taker {pre_taker:.2f}→{post_taker:.2f}")
            print(f"  Cash: ${pre_cash:.2f}→${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
            print(f"  Episode status: Term={term}, Trunc={trunc}")
            
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
            print("✅ EXECUTIONS FOUND:")
            total_maker = 0
            total_cash_change = 0
            for exec_data in execution_log:
                print(f"  Step {exec_data['step']}: "
                      f"Maker +{exec_data['maker_change']:.2f}, "
                      f"Taker +{exec_data['taker_change']:.2f}, "
                      f"Cash ${exec_data['cash_change']:+.2f}, "
                      f"Market {exec_data['market']}")
                total_maker += exec_data['maker_change']
                total_cash_change += exec_data['cash_change']
            
            print(f"\nTOTAL RESULTS:")
            print(f"  Total maker volume: {total_maker:.2f}")
            print(f"  Total cash change: ${total_cash_change:+.2f}")
        else:
            print("❌ NO EXECUTIONS FOUND")
        
        print(f"\n=== FINAL ANSWER ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"Final Maker Volume: {final_maker:.2f}")
        print(f"Final Taker Volume: {final_taker:.2f}")
        print(f"Total Rebates Earned: ${final_rebates:.6f}")
        print(f"Net Cash Change: ${final_cash_change:+.2f}")
        
        if final_maker > 0:
            print(f"\n✅ SUCCESS! Post-only mode CAN execute trades!")
            print(f"   Post-only mode works as expected:")
            print(f"   - Allows maker orders to execute when market moves to them")
            print(f"   - Prevents aggressive taker orders (aggressiveness_ticks = 0)")
            print(f"   - Can generate volume and rebates")
            if final_cash_change < 0:
                print(f"   - Shows negative PnL as expected for passive maker trading")
        else:
            print(f"\n❓ Still no volume - may need even longer episodes or different market data")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    test_longer_episode()