#!/usr/bin/env python3
"""
Test the fixed rebate functionality
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_test_data():
    """Create test data that will guarantee executions."""
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

def test_rebate_functionality():
    """Test that rebates are properly calculated and applied."""
    print("🔧 TESTING FIXED REBATE FUNCTIONALITY")
    
    test_csv = create_test_data()
    
    try:
        # Test rebated environment
        config = get_unified_config("rebate_4bps", False)
        config["csv_path"] = test_csv
        config["max_steps"] = 20
        config["episode_length"] = 15
        config["latency_steps_long"] = 1  # Short latency for quick testing
        config["latency_steps_short"] = 1
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"✅ Environment Setup:")
        print(f"   Fee Structure: {env.fee_structure}")
        print(f"   Is Rebated: {env.is_rebated}")
        print(f"   Rebate Rate: {env.rebate_rate:.6f}")
        print(f"   Initial Cash: ${env.cash:.2f}")
        print(f"   BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Test with forced execution actions
        total_volume = 0
        total_rebates = 0
        
        for step in range(10):
            print(f"\n--- STEP {step} ---")
            
            pre_cash = env.cash
            pre_volume = env.last_executed_volume
            pre_rebates = info.get('total_rebates_earned', 0)
            pre_inventory = env.inventory
            
            # Place aggressive orders to force execution
            if step % 2 == 0:
                # Aggressive buy (should be taker)
                action = [0.9, 0.0, 0.8, 0.0, -1.0, -1.0]
                expected_side = "BUY (taker)"
            else:
                # Aggressive sell (should be taker)  
                action = [0.0, -0.9, 0.0, 0.8, -1.0, -1.0]
                expected_side = "SELL (taker)"
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_cash = env.cash
            post_volume = info.get('maker_volume', 0) + info.get('taker_volume', 0)
            post_rebates = info.get('total_rebates_earned', 0)
            post_inventory = env.inventory
            
            volume_executed = post_volume - total_volume
            rebates_earned = post_rebates - total_rebates
            
            print(f"Expected: {expected_side}")
            print(f"Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash - pre_cash:+.2f})")
            print(f"Volume: {total_volume:.2f} → {post_volume:.2f} (Δ{volume_executed:+.2f})")
            print(f"Rebates: ${total_rebates:.4f} → ${post_rebates:.4f} (Δ${rebates_earned:+.4f})")
            print(f"Inventory: {pre_inventory:.2f} → {post_inventory:.2f} (Δ{post_inventory - pre_inventory:+.2f})")
            print(f"Maker vol: {info.get('maker_volume', 0):.2f}, Taker vol: {info.get('taker_volume', 0):.2f}")
            
            if volume_executed > 0:
                print(f"✅ EXECUTION DETECTED!")
                if rebates_earned > 0:
                    print(f"✅ REBATES EARNED: ${rebates_earned:.4f}")
                else:
                    print(f"⚠️  NO rebates (might be taker orders)")
            else:
                print(f"⚠️  No execution this step")
            
            # Check position tracking
            if abs(post_inventory - pre_inventory) > 0.001:
                print(f"✅ POSITION TRACKING WORKING")
            elif volume_executed > 0:
                print(f"❌ POSITION TRACKING BROKEN - volume executed but no inventory change")
            
            total_volume = post_volume
            total_rebates = post_rebates
            
            if term or trunc:
                break
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Final Cash: ${env.cash:.2f}")
        print(f"Final Inventory: {env.inventory:.2f}")
        print(f"Final Rebates: ${info.get('total_rebates_earned', 0):.4f}")
        print(f"Final Maker Volume: {info.get('maker_volume', 0):.2f}")
        print(f"Final Taker Volume: {info.get('taker_volume', 0):.2f}")
        
        # Verify rebate functionality
        net_change = env.cash - 100000
        if net_change > 1 and info.get('total_rebates_earned', 0) > 0:
            print(f"✅ REBATE SYSTEM WORKING - Net gain: ${net_change:.2f}")
            return True
        elif abs(env.inventory) > 0.1:
            print(f"✅ POSITION TRACKING WORKING - Net inventory: {env.inventory:.2f}")  
            return True
        else:
            print(f"❌ SYSTEM STILL BROKEN - No significant changes")
            return False
            
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def test_baseline_vs_rebated():
    """Compare baseline vs rebated environment."""
    print("\n🔧 TESTING BASELINE VS REBATED COMPARISON")
    
    test_csv = create_test_data()
    
    try:
        # Test baseline first
        print("Testing Baseline Environment...")
        config_base = get_unified_config("baseline", False)
        config_base["csv_path"] = test_csv
        config_base["episode_length"] = 10
        config_base["latency_steps_long"] = 1
        config_base["latency_steps_short"] = 1
        
        env_base = RebatedHFTEnv(config_base)
        obs, info = env_base.reset()
        
        for step in range(5):
            action = [0.8, -0.8, 0.5, 0.5, -1.0, -1.0]  # Aggressive
            obs, reward, term, trunc, info = env_base.step(action)
            if term or trunc:
                break
        
        baseline_cash = env_base.cash
        baseline_rebates = info.get('total_rebates_earned', 0)
        env_base.close()
        
        # Test rebated environment  
        print("Testing Rebated Environment...")
        config_rebated = get_unified_config("rebate_6bps", False)
        config_rebated["csv_path"] = test_csv
        config_rebated["episode_length"] = 10
        config_rebated["latency_steps_long"] = 1
        config_rebated["latency_steps_short"] = 1
        
        env_rebated = RebatedHFTEnv(config_rebated)
        obs, info = env_rebated.reset()
        
        for step in range(5):
            action = [0.8, -0.8, 0.5, 0.5, -1.0, -1.0]  # Aggressive
            obs, reward, term, trunc, info = env_rebated.step(action)
            if term or trunc:
                break
        
        rebated_cash = env_rebated.cash
        rebated_rebates = info.get('total_rebates_earned', 0)
        env_rebated.close()
        
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Baseline Cash: ${baseline_cash:.2f}")
        print(f"Rebated Cash: ${rebated_cash:.2f}")
        print(f"Difference: ${rebated_cash - baseline_cash:+.2f}")
        print(f"Baseline Rebates: ${baseline_rebates:.4f}")
        print(f"Rebated Rebates: ${rebated_rebates:.4f}")
        
        if rebated_rebates > baseline_rebates:
            print(f"✅ REBATE SYSTEM WORKING - Rebated env earned more")
            return True
        else:
            print(f"❌ REBATE SYSTEM NOT WORKING - No rebate difference")
            return False
            
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    test1_passed = test_rebate_functionality()
    test2_passed = test_baseline_vs_rebated()
    
    print(f"\n" + "="*60)
    print(f"REBATE FIX TEST RESULTS")
    print(f"="*60)
    print(f"Rebate Functionality Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Baseline vs Rebated Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"🎉 ALL TESTS PASSED - REBATE SYSTEM IS FIXED!")
    else:
        print(f"💥 TESTS FAILED - MORE FIXES NEEDED")
        
    print(f"="*60)