#!/usr/bin/env python3
"""
Debug if we're inadvertently triggering do nothing mode
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def debug_do_nothing():
    """Debug if do nothing mode is being triggered."""
    print("🔍 DEBUGGING DO NOTHING MODE")
    
    # Create simple test data
    data = []
    for i in range(5):
        row = {'datetime': i}
        if i < 2:
            base_bid, base_ask = 1800.00, 1800.10
        else:
            base_bid, base_ask = 1800.50, 1800.60
        
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
        config["episode_length"] = 5
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Configuration:")
        print(f"  do_nothing_threshold: {env.config.get('do_nothing_threshold', 'NOT SET')}")
        
        # Place and activate orders
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell order
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        print(f"\nAfter activation - Active orders: {len(env.active_orders)}")
        
        # Now test the execution step with action analysis
        test_action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Our typical "do nothing" action
        
        print(f"\n🔍 ANALYZING ACTION FOR EXECUTION STEP:")
        print(f"Action: {test_action}")
        
        # Unpack the action like the step method does
        buy_offset_action, sell_offset_action, buy_size_signal, sell_size_signal, explicit_cancel_signal, do_nothing_signal = test_action
        
        print(f"Unpacked action:")
        print(f"  buy_offset_action: {buy_offset_action}")
        print(f"  sell_offset_action: {sell_offset_action}")
        print(f"  buy_size_signal: {buy_size_signal}")
        print(f"  sell_size_signal: {sell_size_signal}")
        print(f"  explicit_cancel_signal: {explicit_cancel_signal}")
        print(f"  do_nothing_signal: {do_nothing_signal}")
        
        # Check do nothing threshold
        do_nothing_threshold = env.config.get("do_nothing_threshold", 1.1)
        do_nothing_triggered = bool(do_nothing_signal > do_nothing_threshold)
        
        print(f"\nDo nothing analysis:")
        print(f"  do_nothing_threshold: {do_nothing_threshold}")
        print(f"  do_nothing_signal: {do_nothing_signal}")
        print(f"  do_nothing_triggered: {do_nothing_triggered}")
        
        if do_nothing_triggered:
            print(f"  ❌ DO NOTHING MODE IS TRIGGERED!")
            print(f"     This prevents _execute_orders() from being called!")
        else:
            print(f"  ✅ Do nothing mode NOT triggered")
        
        # Override the step method to trace execution calls
        original_step = env.step
        
        def traced_step(action):
            print(f"\n--- TRACED STEP START ---")
            
            # Unpack action
            try:
                buy_offset_action, sell_offset_action, buy_size_signal, \
                sell_size_signal, explicit_cancel_signal, do_nothing_signal = action
            except ValueError as e:
                print(f"Action unpacking error: {e}")
                return original_step(action)
            
            # Check do nothing
            do_nothing_threshold = env.config.get("do_nothing_threshold", 1.1)
            do_nothing_triggered = bool(do_nothing_signal > do_nothing_threshold)
            
            print(f"Action unpacked successfully")
            print(f"do_nothing_signal: {do_nothing_signal}, threshold: {do_nothing_threshold}")
            print(f"do_nothing_triggered: {do_nothing_triggered}")
            
            if do_nothing_triggered:
                print(f"🚫 TAKING DO NOTHING PATH - _execute_orders() will NOT be called")
            else:
                print(f"✅ TAKING NORMAL PATH - _execute_orders() will be called")
            
            # Call original step
            result = original_step(action)
            print(f"--- TRACED STEP END ---\n")
            return result
        
        # Replace step method
        env.step = traced_step
        
        # Execute the step
        print(f"\n🚀 EXECUTING STEP WITH TRACING:")
        obs, reward, term, trunc, info = env.step(test_action)
        
        print(f"\n🎯 RESULTS:")
        print(f"Active orders after: {len(env.active_orders)}")
        print(f"Volume: {info.get('maker_volume', 0)}")
        print(f"Cash: {env.cash}")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    debug_do_nothing()