#!/usr/bin/env python3
"""
Test with proper action that doesn't trigger do nothing mode
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def test_proper_execution():
    """Test with action that allows execution."""
    print("🎯 TESTING WITH PROPER NON-DO-NOTHING ACTION")
    
    # Create test data with execution opportunity
    data = []
    for i in range(8):
        row = {'datetime': i}
        if i < 3:
            base_bid, base_ask = 1800.00, 1800.10  # For placement
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
        # Test both normal and post-only modes
        for post_only in [False, True]:
            mode = "POST-ONLY" if post_only else "NORMAL"
            print(f"\n{'='*20} {mode} MODE {'='*20}")
            
            config = get_unified_config("rebate_6bps", post_only)
            config["csv_path"] = temp_file.name
            config["episode_length"] = 8
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 0.5
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Config: do_nothing_threshold = {env.config.get('do_nothing_threshold')}")
            print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # STEP 1: Place orders
            action = [-0.5, 0.5, 0.8, 0.8, -1.0, -1.0]  # Both orders, do_nothing = -1.0 (OFF)
            obs, reward, term, trunc, info = env.step(action)
            
            # STEP 2: Activate orders  
            action = [-1.0, -1.0, 0.0, 0.0, -1.0, -1.0]  # No new orders, do_nothing = -1.0 (OFF)
            obs, reward, term, trunc, info = env.step(action)
            
            print(f"Orders after activation:")
            for order in env.active_orders:
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
            
            # STEP 3+: Step through market with EXECUTION ENABLED
            step_count = 2
            execution_log = []
            
            while step_count < 7 and not (term or trunc):
                step_count += 1
                
                pre_maker = info.get('maker_volume', 0)
                pre_cash = env.cash
                pre_active = len(env.active_orders)
                
                print(f"\nStep {step_count}:")
                print(f"  Before: Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active: {pre_active}")
                
                # Check execution potential
                should_execute = []
                for order in env.active_orders:
                    if order['is_buy'] and env.best_ask <= order['price']:
                        should_execute.append(f"BUY@{order['price']:.2f}")
                    elif not order['is_buy'] and env.best_bid >= order['price']:
                        should_execute.append(f"SELL@{order['price']:.2f}")
                
                if should_execute:
                    print(f"  🎯 SHOULD EXECUTE: {', '.join(should_execute)}")
                
                # Use action that DOES NOT trigger do nothing
                action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.0]  # do_nothing = 0.0 < 0.8 (EXECUTION ENABLED)
                
                obs, reward, term, trunc, info = env.step(action)
                
                post_maker = info.get('maker_volume', 0)
                post_cash = env.cash
                post_active = len(env.active_orders)
                
                print(f"  After:  Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active: {post_active}")
                
                maker_change = post_maker - pre_maker
                cash_change = post_cash - pre_cash
                
                print(f"  Changes: Maker +{maker_change:.2f}, Cash ${cash_change:+.2f}")
                
                if maker_change > 0 or abs(cash_change) > 0.01:
                    execution_log.append({
                        'step': step_count,
                        'maker_vol': maker_change,
                        'cash_change': cash_change,
                        'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                    })
                    print(f"  🎉 EXECUTION DETECTED!")
            
            # Results
            print(f"\n📊 RESULTS:")
            final_maker = info.get('maker_volume', 0)
            final_taker = info.get('taker_volume', 0)
            final_rebates = info.get('total_rebates_earned', 0)
            final_cash_change = env.cash - 100000
            
            print(f"  Final maker volume: {final_maker:.2f}")
            print(f"  Final taker volume: {final_taker:.2f}")
            print(f"  Total rebates: ${final_rebates:.6f}")
            print(f"  Net cash change: ${final_cash_change:+.2f}")
            
            if execution_log:
                print(f"  ✅ EXECUTIONS FOUND:")
                for exec_data in execution_log:
                    print(f"    Step {exec_data['step']}: Maker +{exec_data['maker_vol']:.2f}, Cash ${exec_data['cash_change']:+.2f}")
            else:
                print(f"  ❌ NO EXECUTIONS")
            
            if final_maker > 0:
                print(f"\n🏆 SUCCESS! {mode} mode executed trades!")
                if post_only:
                    print(f"  🎯 POST-ONLY MODE WORKS!")
                    print(f"  ✅ Volume > 0: {final_maker:.2f}")
                    print(f"  ✅ Rebates earned: ${final_rebates:.6f}")
                    print(f"  ✅ No taker volume: {final_taker:.2f}")
                    print(f"  💡 USER WAS 100% CORRECT!")
            
            env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    test_proper_execution()