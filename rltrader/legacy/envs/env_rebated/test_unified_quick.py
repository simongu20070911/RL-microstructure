#!/usr/bin/env python3
"""
Quick test of the unified rebated environment with smaller dataset
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
    """Create small test dataset."""
    data = []
    np.random.seed(42)
    
    for i in range(100):  # Small dataset
        row = {'datetime': i}
        base_price = 1800.0 + np.random.normal(0, 0.1)
        
        for level in range(1, 11):
            bid_offset = (level - 1) * 0.01 + 0.01
            ask_offset = (level - 1) * 0.01 + 0.01
            
            row[f'bid{level}'] = base_price - bid_offset
            row[f'bidqty{level}'] = np.random.uniform(5.0, 50.0)
            row[f'ask{level}'] = base_price + ask_offset
            row[f'askqty{level}'] = np.random.uniform(5.0, 50.0)
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_configuration(name, fee_structure, post_only):
    """Test a specific configuration."""
    print(f"\n=== Testing {name} ===")
    
    test_csv = create_test_data()
    
    try:
        config = get_unified_config(fee_structure, post_only)
        config["csv_path"] = test_csv
        config["max_steps"] = 100
        config["episode_length"] = 50
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"✅ Environment initialized successfully")
        print(f"   Fee Structure: {env.fee_structure}")
        print(f"   Post-Only Mode: {env.post_only_mode}")
        print(f"   Is Rebated: {env.is_rebated}")
        print(f"   Transaction Costs: Long={config['transaction_cost_long']:.6f}, Short={config['transaction_cost_short']:.6f}")
        
        if post_only:
            print(f"   Aggressiveness Limit: {config['allowed_aggressiveness_ticks']} (should be 0 for post-only)")
        
        # Test a few steps with market making
        total_rebates = 0.0
        total_volume = 0.0
        
        for step in range(10):
            # Market making action
            action = [0.1, -0.1, 0.5, 0.5, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            # Track rebates if this is a rebated environment
            if env.is_rebated:
                rebates = info.get('total_rebates_earned', 0.0)
                volume = info.get('rebated_volume', 0.0)
                if rebates > total_rebates:
                    print(f"   Step {step}: Rebates earned ${rebates:.6f}, Volume: {volume:.4f}")
                total_rebates = rebates
                total_volume = volume
            
            if term or trunc:
                break
        
        if env.is_rebated and total_rebates > 0:
            expected_rate = env.rebate_rate
            print(f"   Final: ${total_rebates:.6f} rebates on {total_volume:.4f} volume")
            print(f"   Rate: {expected_rate:.6f} ({expected_rate*10000:.1f} bps)")
        
        env.close()
        print(f"✅ {name} - Test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ {name} - Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def main():
    """Run quick tests."""
    print("=" * 60)
    print("QUICK UNIFIED REBATED ENVIRONMENT TEST")
    print("=" * 60)
    
    test_configs = [
        ("Baseline (1 bps fee)", "baseline", False),
        ("4 bps Rebate", "rebate_4bps", False),
        ("6 bps Rebate Post-Only", "rebate_6bps", True),
        ("8 bps Rebate Post-Only", "rebate_8bps", True)
    ]
    
    passed = 0
    total = len(test_configs)
    
    for name, fee_structure, post_only in test_configs:
        if test_configuration(name, fee_structure, post_only):
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Unified environment is working correctly!")
    else:
        print(f"❌ {total - passed} tests failed - Check implementation")
    
    print("=" * 60)

if __name__ == "__main__":
    main()