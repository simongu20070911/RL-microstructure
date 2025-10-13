#!/usr/bin/env python3
"""
Deep debugging of rebate calculations in unified environment
ULTRATHINK: Find the actual problems by tracing every transaction
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
    """Create deterministic test data for precise tracking."""
    data = []
    
    for i in range(50):
        row = {'datetime': i}
        base_price = 1800.0  # Fixed price for consistency
        
        for level in range(1, 11):
            bid_offset = (level - 1) * 0.01 + 0.01
            ask_offset = (level - 1) * 0.01 + 0.01
            
            row[f'bid{level}'] = base_price - bid_offset
            row[f'bidqty{level}'] = 10.0  # Fixed quantity
            row[f'ask{level}'] = base_price + ask_offset  
            row[f'askqty{level}'] = 10.0  # Fixed quantity
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def debug_rebate_calculation():
    """Deep debug of rebate calculation logic."""
    print("🔍 ULTRATHINK DEBUG: Tracking every transaction")
    
    test_csv = create_test_data()
    
    try:
        # Test rebated environment
        config = get_unified_config("rebate_4bps", False)
        config["csv_path"] = test_csv
        config["max_steps"] = 50
        config["episode_length"] = 20
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial state:")
        print(f"  Cash: ${env.cash:.6f}")
        print(f"  Transaction cost rate: {env.config['transaction_cost_long']:.6f}")
        print(f"  Rebate rate: {getattr(env, 'rebate_rate', 'NOT SET')}")
        print(f"  Is rebated: {getattr(env, 'is_rebated', 'NOT SET')}")
        
        # Check if rebate tracking variables exist and are initialized
        print(f"\nRebate tracking variables:")
        print(f"  total_rebates_earned: {getattr(env, 'total_rebates_earned', 'NOT FOUND')}")
        print(f"  maker_volume: {getattr(env, 'maker_volume', 'NOT FOUND')}")
        print(f"  taker_volume: {getattr(env, 'taker_volume', 'NOT FOUND')}")
        
        # Force a trade by placing orders
        for step in range(10):
            print(f"\n--- STEP {step} ---")
            
            pre_cash = env.cash
            pre_volume = getattr(env, 'last_executed_volume', 0.0)
            pre_rebates = getattr(env, 'total_rebates_earned', 0.0)
            
            print(f"Before action:")
            print(f"  Cash: ${pre_cash:.6f}")
            print(f"  Volume: {pre_volume:.4f}")  
            print(f"  Rebates: ${pre_rebates:.6f}")
            print(f"  BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"  Active orders: {len(env.active_orders)}")
            print(f"  Pending orders: {len(env.pending_orders)}")
            
            # Market making action - place both buy and sell orders
            action = [0.1, -0.1, 0.8, 0.8, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            post_cash = env.cash
            post_volume = getattr(env, 'last_executed_volume', 0.0)
            post_rebates = getattr(env, 'total_rebates_earned', 0.0)
            
            print(f"After action:")
            print(f"  Cash: ${post_cash:.6f} (Δ${post_cash - pre_cash:+.6f})")
            print(f"  Volume: {post_volume:.4f} (Δ{post_volume - pre_volume:+.4f})")
            print(f"  Rebates: ${post_rebates:.6f} (Δ${post_rebates - pre_rebates:+.6f})")
            print(f"  Reward: {reward:.6f}")
            print(f"  Active orders: {len(env.active_orders)}")
            
            # Check info dict for rebate information
            rebate_info_keys = [k for k in info.keys() if 'rebate' in k.lower()]
            if rebate_info_keys:
                print(f"  Rebate info in dict: {rebate_info_keys}")
                for key in rebate_info_keys:
                    print(f"    {key}: {info[key]}")
            else:
                print(f"  ❌ NO rebate info in info dict!")
                
            # Check if any orders were actually executed
            if post_volume > pre_volume:
                print(f"  ✅ EXECUTION DETECTED: {post_volume - pre_volume:.4f} volume")
                
                # Calculate expected rebate
                volume_executed = post_volume - pre_volume
                expected_rebate = volume_executed * abs(env.config['transaction_cost_long']) * env.midprice
                actual_cash_change = post_cash - pre_cash
                
                print(f"  Expected rebate: ${expected_rebate:.6f}")
                print(f"  Actual cash change: ${actual_cash_change:+.6f}")
                
                # Check if rebate was actually applied
                if actual_cash_change > 0 and abs(actual_cash_change) > 0.001:
                    print(f"  ✅ POSITIVE cash change - rebate possibly applied")
                else:
                    print(f"  ❌ NO positive cash change - rebate NOT applied")
                    
            if term or trunc:
                break
                
        print(f"\n=== FINAL ANALYSIS ===")
        print(f"Final cash: ${env.cash:.6f}")
        print(f"Final rebates tracked: ${getattr(env, 'total_rebates_earned', 0.0):.6f}")
        print(f"Final volume: {getattr(env, 'last_executed_volume', 0.0):.4f}")
        
        # Check if rebates were actually earned
        if env.cash > 100000:
            net_gain = env.cash - 100000
            print(f"✅ Net gain: ${net_gain:.6f} - rebates working through negative costs")
        else:
            net_loss = 100000 - env.cash
            print(f"❌ Net loss: ${net_loss:.6f} - rebates NOT working")
            
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def debug_maker_taker_logic():
    """Debug the maker/taker detection and application logic."""
    print("\n🔍 DEBUGGING MAKER/TAKER LOGIC")
    
    test_csv = create_test_data()
    
    try:
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = test_csv
        config["max_steps"] = 50
        config["episode_length"] = 10
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        # Place orders and track maker/taker status
        for step in range(5):
            print(f"\n--- MAKER/TAKER STEP {step} ---")
            
            # Place aggressive orders (should be takers)
            action = [0.9, -0.9, 0.5, 0.5, -1.0, -1.0]  # Aggressive pricing
            obs, reward, term, trunc, info = env.step(action)
            
            # Check active orders for taker flags
            if env.active_orders:
                print(f"Active orders with taker flags:")
                for i, order in enumerate(env.active_orders):
                    is_taker = order.get('is_taker_at_activation', 'NOT SET')
                    print(f"  Order {i}: is_taker_at_activation = {is_taker}")
                    
            if env.pending_orders:
                print(f"Pending orders (waiting for activation): {len(env.pending_orders)}")
                
            if term or trunc:
                break
                
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def debug_post_only_mode():
    """Debug post-only mode implementation."""
    print("\n🔍 DEBUGGING POST-ONLY MODE")
    
    test_csv = create_test_data()
    
    try:
        config = get_unified_config("rebate_8bps", True)  # Post-only mode
        config["csv_path"] = test_csv
        config["max_steps"] = 50
        config["episode_length"] = 10
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Post-only mode: {env.post_only_mode}")
        print(f"Allowed aggressiveness: {config['allowed_aggressiveness_ticks']}")
        print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Try to place aggressive orders (should be rejected in post-only)
        for step in range(3):
            print(f"\n--- POST-ONLY STEP {step} ---")
            
            # Extremely aggressive action
            action = [1.0, -1.0, 0.8, 0.8, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            print(f"Orders placed - Active: {len(env.active_orders)}, Pending: {len(env.pending_orders)}")
            
            # Check if any orders crossed the spread
            for order in env.active_orders + env.pending_orders:
                if order['is_buy'] and order['price'] >= env.best_ask:
                    print(f"❌ BUY order crossed spread: {order['price']:.2f} >= {env.best_ask:.2f}")
                elif not order['is_buy'] and order['price'] <= env.best_bid:
                    print(f"❌ SELL order crossed spread: {order['price']:.2f} <= {env.best_bid:.2f}")
                    
            if term or trunc:
                break
                
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_rebate_calculation()
    debug_maker_taker_logic() 
    debug_post_only_mode()