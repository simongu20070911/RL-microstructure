#!/usr/bin/env python3
"""
Final validation test that confirms the rebated environment is working correctly
"""

import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_controlled_test_data():
    """Create carefully controlled market data that guarantees proper maker/taker classification."""
    data = []
    
    for i in range(30):
        row = {'datetime': i}
        
        if i < 10:
            # Phase 1: Stable market for order placement
            base_bid = 1799.99
            base_ask = 1800.01
        elif i < 20:
            # Phase 2: Market moves to execute passive orders
            # Market gradually moves up to hit our passive buy orders at 1799.99
            base_bid = 1799.99 + (i-10) * 0.002
            base_ask = 1800.01 + (i-10) * 0.002
        else:
            # Phase 3: Market moves back down to hit passive sell orders
            base_bid = 1800.09 - (i-20) * 0.003
            base_ask = 1800.11 - (i-20) * 0.003
        
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

def test_rebated_functionality():
    """Test core rebated functionality with guaranteed maker orders."""
    print("🎯 FINAL VALIDATION: REBATED ENVIRONMENT FUNCTIONALITY")
    
    test_csv = create_controlled_test_data()
    results = []
    
    try:
        # Test key environments
        test_configs = [
            ("Baseline (1 bps fee)", "baseline", False, 0.000100),
            ("6 bps Rebate", "rebate_6bps", False, 0.000060),
            ("6 bps Post-Only", "rebate_6bps", True, 0.000060),
        ]
        
        for name, fee_structure, post_only, expected_rate in test_configs:
            print(f"\n=== TESTING {name} ===")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 25
            config["latency_steps_long"] = 1
            config["latency_steps_short"] = 1
            config["max_order_volume"] = 2.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"  Setup: {env.fee_structure}, Post-only: {env.post_only_mode}")
            if hasattr(env, 'rebate_rate'):
                print(f"  Rebate rate: {env.rebate_rate:.6f}")
            
            # Strategy: Place orders early and let market move to execute them
            for step in range(20):
                if step < 5:
                    # Place passive orders at BBO
                    action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]
                elif step < 8:
                    # Place more orders slightly inside spread
                    action = [-0.05, 0.05, 0.6, 0.6, -1.0, -1.0]
                else:
                    # Let market move to execute orders
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]
                
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            
            # Collect results
            final_cash = env.cash
            initial_cash = 100000.0
            net_change = final_cash - initial_cash
            
            maker_vol = info.get('maker_volume', 0)
            taker_vol = info.get('taker_volume', 0)
            total_rebates = info.get('total_rebates_earned', 0)
            
            result = {
                'name': name,
                'fee_structure': fee_structure,
                'post_only': post_only,
                'net_change': net_change,
                'maker_volume': maker_vol,
                'taker_volume': taker_vol,
                'total_rebates': total_rebates,
                'expected_rate': expected_rate,
                'is_rebated': getattr(env, 'is_rebated', False)
            }
            
            results.append(result)
            
            print(f"  Results:")
            print(f"    Net cash change: ${net_change:+.2f}")
            print(f"    Maker volume: {maker_vol:.2f}")
            print(f"    Taker volume: {taker_vol:.2f}")
            print(f"    Total rebates: ${total_rebates:.4f}")
            
            env.close()
        
        return results
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def validate_results(results):
    """Validate the test results."""
    print(f"\n" + "="*80)
    print(f"FINAL VALIDATION ANALYSIS")
    print(f"="*80)
    
    all_tests_passed = True
    maker_volume_detected = False
    
    for result in results:
        name = result['name']
        net_change = result['net_change']
        maker_vol = result['maker_volume']
        taker_vol = result['taker_volume']
        rebates = result['total_rebates']
        is_rebated = result['is_rebated']
        post_only = result['post_only']
        expected_rate = result['expected_rate']
        
        print(f"\n{name}:")
        print(f"  Net Change: ${net_change:+.2f}")
        print(f"  Maker Volume: {maker_vol:.2f}")
        print(f"  Taker Volume: {taker_vol:.2f}")
        print(f"  Rebates: ${rebates:.4f}")
        
        # Test validation
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Maker volume detection
        total_tests += 1
        if maker_vol > 0:
            print(f"  ✅ Maker volume detected: {maker_vol:.2f}")
            tests_passed += 1
            maker_volume_detected = True
        else:
            print(f"  ❌ No maker volume")
        
        # Test 2: Rebate functionality (for rebated environments)
        if is_rebated:
            total_tests += 1
            if maker_vol > 0 and rebates > 0:
                # Verify rebate rate
                avg_price = 1800.0  # Approximate
                trade_value = maker_vol * avg_price
                actual_rate = rebates / trade_value
                rate_diff = abs(actual_rate - expected_rate)
                
                if rate_diff < 0.000001:
                    print(f"  ✅ Rebate rate correct: {actual_rate:.6f} (expected {expected_rate:.6f})")
                    tests_passed += 1
                else:
                    print(f"  ❌ Rebate rate incorrect: {actual_rate:.6f} (expected {expected_rate:.6f})")
            elif maker_vol > 0:
                print(f"  ❌ Maker volume but no rebates")
            else:
                print(f"  ⚠️  Cannot test rebates without maker volume")
        
        # Test 3: Post-only mode (should have no takers)
        if post_only:
            total_tests += 1
            if taker_vol == 0:
                print(f"  ✅ Post-only mode working: No taker volume")
                tests_passed += 1
            else:
                print(f"  ❌ Post-only mode broken: {taker_vol:.2f} taker volume")
        
        # Test 4: Baseline should lose money (if it's baseline)
        if name.startswith("Baseline"):
            total_tests += 1
            if net_change < 0:
                print(f"  ✅ Baseline loses money to fees")
                tests_passed += 1
            else:
                print(f"  ❌ Baseline should lose money but gained ${net_change:+.2f}")
        
        # Overall result for this environment
        if tests_passed == total_tests:
            print(f"  🎉 ALL TESTS PASSED ({tests_passed}/{total_tests})")
        else:
            print(f"  ❌ SOME TESTS FAILED ({tests_passed}/{total_tests})")
            all_tests_passed = False
    
    print(f"\n" + "="*80)
    print(f"FINAL VERDICT")
    print(f"="*80)
    
    if all_tests_passed and maker_volume_detected:
        print(f"🎉 COMPLETE SUCCESS!")
        print(f"   ✓ Rebated environment is fully functional")
        print(f"   ✓ Maker order executions confirmed")
        print(f"   ✓ Rebate calculations verified as correct")
        print(f"   ✓ Post-only mode prevents taker orders")
        print(f"   ✓ Ready for production use")
        return True
    elif maker_volume_detected:
        print(f"⚠️  PARTIAL SUCCESS")
        print(f"   ✓ Maker volume detected")
        print(f"   ❌ Some validation tests failed")
        return False
    else:
        print(f"❌ VALIDATION FAILED")
        print(f"   ❌ No maker volume detected")
        print(f"   ❌ Cannot properly test rebate functionality")
        return False

def main():
    """Run the final validation test."""
    print("🚀 FINAL REBATED ENVIRONMENT VALIDATION")
    print("="*80)
    
    results = test_rebated_functionality()
    success = validate_results(results)
    
    print(f"\n" + "="*80)
    if success:
        print(f"🎉 VALIDATION COMPLETE: REBATED ENVIRONMENT IS READY!")
    else:
        print(f"💥 VALIDATION INCOMPLETE: Additional work needed")
    print(f"="*80)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)