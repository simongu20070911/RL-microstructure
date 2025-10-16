#!/usr/bin/env python3
"""
Test script to reproduce the rebate calculation bug.
Expected: 0.6 bps rebate
Actual: 1582 bps rebate (263x overpayment)
"""

import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_test_data():
    """Create test data with tight spreads to force executions."""
    data = []
    
    for i in range(50):
        row = {'datetime': i}
        base_price = 100.0  # Use round numbers for easier calculation
        
        # Create tight spread to force fills
        for level in range(1, 11):
            bid_offset = level * 0.01
            ask_offset = level * 0.01
            
            row[f'bid{level}'] = base_price - bid_offset
            row[f'bidqty{level}'] = 100.0  # Large quantity
            row[f'ask{level}'] = base_price + ask_offset  
            row[f'askqty{level}'] = 100.0  # Large quantity
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_rebate_calculation():
    """Test the actual rebate calculation bug."""
    print("🔍 TESTING REBATE CALCULATION BUG")
    print("Expected: 0.6 bps rebate for 6bps structure")
    print("Checking for 263x overpayment...")
    
    test_csv = create_test_data()
    
    try:
        # Test rebate_6bps structure
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = test_csv
        config["max_steps"] = 50
        config["episode_length"] = 10
        config["min_position_size"] = 0.1  # Small position size
        config["max_position_size"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"\nEnvironment setup:")
        print(f"  Fee structure: rebate_6bps")
        print(f"  Transaction cost rate: {env.config['transaction_cost_long']:.6f}")
        print(f"  Rebate rate: {env.rebate_rate:.6f}")
        print(f"  Expected rebate rate: 0.00006 (6 bps)")
        
        # Check if rebate_rate is correctly set
        expected_rebate_rate = 0.00006
        if abs(env.rebate_rate - expected_rebate_rate) > 1e-8:
            print(f"  ❌ REBATE_RATE BUG: Expected {expected_rebate_rate:.6f}, Got {env.rebate_rate:.6f}")
        else:
            print(f"  ✅ Rebate rate correctly set")
            
        print(f"\nBest bid/ask: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Midprice: {env.midprice:.2f}")
        
        # Force execution with market order by placing order at best bid/ask
        # This should create a taker order that gets filled immediately
        print(f"\n--- FORCING EXECUTION ---")
        
        initial_cash = env.cash
        print(f"Initial cash: ${initial_cash:.6f}")
        
        # Place a very small aggressive order to force execution
        # Action format: [buy_price_offset, sell_price_offset, buy_qty, sell_qty, buy_active, sell_active]
        aggressive_action = [0.0, 0.0, 0.1, 0.1, 1.0, -1.0]  # Place at best bid/ask
        
        obs, reward, term, trunc, info = env.step(aggressive_action)
        
        final_cash = env.cash
        cash_change = final_cash - initial_cash
        
        print(f"Final cash: ${final_cash:.6f}")
        print(f"Cash change: ${cash_change:+.6f}")
        print(f"Total rebates earned: ${env.total_rebates_earned:.6f}")
        
        # Calculate expected rebate for comparison
        if hasattr(env, 'last_executed_volume') and env.last_executed_volume > 0:
            volume_executed = env.last_executed_volume
            expected_rebate = volume_executed * env.rebate_rate * env.midprice
            
            print(f"\n--- REBATE ANALYSIS ---")
            print(f"Volume executed: {volume_executed:.4f}")
            print(f"Expected rebate: ${expected_rebate:.6f}")
            print(f"Actual rebate: ${env.total_rebates_earned:.6f}")
            
            if env.total_rebates_earned > 0:
                overpayment_ratio = env.total_rebates_earned / expected_rebate
                print(f"Overpayment ratio: {overpayment_ratio:.1f}x")
                
                if overpayment_ratio > 260:
                    print(f"❌ CONFIRMED: 263x overpayment bug!")
                    print(f"   Expected: {expected_rebate:.6f} bps")
                    print(f"   Actual: {env.total_rebates_earned:.6f} bps")
                else:
                    print(f"✅ Rebate calculation appears correct")
            else:
                print(f"❌ No rebates earned despite execution")
        else:
            print(f"❌ No execution occurred")
            
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    test_rebate_calculation()