#!/usr/bin/env python3
"""
Simple test script for rebated environments with proper data handling.
"""

import pandas as pd
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_004 import HFTEnvRebated004, get_default_rebated_config_004
from env_rebated_006 import HFTEnvRebated006, get_default_rebated_config_006  
from env_rebated_008 import HFTEnvRebated008, get_default_rebated_config_008

def create_test_csv():
    """Create a properly formatted test CSV with timestamp column."""
    # Load existing data
    original_path = "/home/gaen/Documents/billions_db/orderbooks/binance/futures/ethusdc/30-Mar-2025/binance_futures_ethusdc_orderbook_30-Mar-2025.csv"
    df = pd.read_csv(original_path, nrows=1000)  # Use first 1000 rows for testing
    
    # Rename datetime to timestamp for compatibility
    df = df.rename(columns={'datetime': 'timestamp'})
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

def test_rebated_environment(env_class, config_func, rebate_bps):
    """Test a specific rebated environment."""
    print(f"\n=== Testing {rebate_bps} bps Rebated Environment ===")
    
    # Create test data
    test_csv = create_test_csv()
    
    try:
        # Get config and update with test data
        config = config_func()
        config["csv_path"] = test_csv
        config["max_steps"] = 100
        config["episode_length"] = 50
        
        # Create environment
        env = env_class(config)
        print(f"✓ Environment created successfully")
        print(f"  Rebate Rate: {env.rebate_rate:.6f} ({env.rebate_bps} bps)")
        print(f"  Transaction Cost Long: {env.config['transaction_cost_long']:.6f}")
        print(f"  Transaction Cost Short: {env.config['transaction_cost_short']:.6f}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Rebate info keys: {[k for k in info.keys() if 'rebate' in k]}")
        
        # Test steps
        total_reward = 0
        for step in range(10):
            # Market making action: place orders near spread
            action = [0.1, -0.1, 0.5, 0.5, -1.0, -1.0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step == 0:
                print(f"✓ First step completed")
                print(f"  Reward: {reward:.4f}")
                print(f"  Terminated: {terminated}, Truncated: {truncated}")
            
            if terminated or truncated:
                break
        
        print(f"✓ Test completed successfully")
        print(f"  Total steps: {step + 1}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Total rebates earned: ${info.get('total_rebates_earned', 0.0):.6f}")
        print(f"  Rebated volume: {info.get('rebated_volume', 0.0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def main():
    """Run tests for all rebated environments."""
    print("Testing Rebated HFT Environments")
    print("=" * 50)
    
    test_configs = [
        (HFTEnvRebated004, get_default_rebated_config_004, 4),
        (HFTEnvRebated006, get_default_rebated_config_006, 6),
        (HFTEnvRebated008, get_default_rebated_config_008, 8)
    ]
    
    results = []
    for env_class, config_func, rebate_bps in test_configs:
        success = test_rebated_environment(env_class, config_func, rebate_bps)
        results.append((rebate_bps, success))
    
    print(f"\n=== Test Summary ===")
    all_passed = True
    for rebate_bps, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {rebate_bps} bps environment: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All rebated environments working correctly!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)