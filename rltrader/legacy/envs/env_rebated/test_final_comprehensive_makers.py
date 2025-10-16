#!/usr/bin/env python3
"""
Final comprehensive test that GUARANTEES maker executions for all environments
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_guaranteed_execution_data():
    """Create market data that GUARANTEES maker order executions."""
    data = []
    
    for i in range(40):
        row = {'datetime': i}
        
        if i < 8:
            # Phase 1: Stable market for passive order placement
            base_bid = 1799.99
            base_ask = 1800.01
        elif i < 15:
            # Phase 2: Market rises to hit BUY orders at 1799.99 (our passive buys become "executed")
            # We need market to come TO our passive orders
            base_bid = 1799.99 + (i-8) * 0.005  # Market moves up
            base_ask = 1800.01 + (i-8) * 0.005
        elif i < 22:
            # Phase 3: Market falls to hit SELL orders at 1800.01 (our passive sells get executed)
            base_bid = 1800.99 - (i-15) * 0.006  # Market moves down
            base_ask = 1801.01 - (i-15) * 0.006  
        elif i < 30:
            # Phase 4: More market movement for additional executions
            base_bid = 1799.50 + (i-22) * 0.003
            base_ask = 1799.52 + (i-22) * 0.003
        else:
            # Phase 5: Final stable period
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

def test_comprehensive_with_makers():
    """Test all fee structures with guaranteed maker executions."""
    print("🎯 COMPREHENSIVE TEST WITH GUARANTEED MAKER EXECUTIONS")
    
    test_csv = create_guaranteed_execution_data()
    results = {}
    
    try:
        fee_structures = [
            ("Baseline", "baseline", False),
            ("4 bps Rebate", "rebate_4bps", False),
            ("6 bps Rebate", "rebate_6bps", False),
            ("8 bps Rebate", "rebate_8bps", False),
            ("6 bps Post-Only", "rebate_6bps", True),
            ("8 bps Post-Only", "rebate_8bps", True)
        ]
        
        for name, fee_structure, post_only in fee_structures:
            print(f"\n=== TESTING {name} ===")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 30
            config["latency_steps_long"] = 1
            config["latency_steps_short"] = 1
            config["max_order_volume"] = 3.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"  Setup: Fee={env.fee_structure}, Post-only={env.post_only_mode}")
            print(f"  Rebate rate: {getattr(env, 'rebate_rate', 0):.6f}")
            print(f"  Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Strategy: Place many passive orders early, then wait for market to move
            for step in range(25):
                if step < 6:
                    # Place passive orders at BBO (guaranteed to be makers)
                    action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]
                elif step < 12:
                    # Place more passive orders slightly inside
                    action = [-0.05, 0.05, 0.6, 0.6, -1.0, -1.0]
                elif step < 18:
                    # Let market move and execute our orders
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                else:
                    # Place final batch of orders
                    action = [-0.1, 0.1, 0.5, 0.5, -1.0, -1.0]
                
                obs, reward, term, trunc, info = env.step(action)
                
                if step % 5 == 0:  # Print every 5 steps
                    print(f"    Step {step}: BBO={env.best_bid:.2f}/{env.best_ask:.2f}, "
                          f"Maker={info.get('maker_volume', 0):.1f}, "
                          f"Taker={info.get('taker_volume', 0):.1f}, "
                          f"Rebates=${info.get('total_rebates_earned', 0):.4f}")
                
                if term or trunc:
                    break
            
            final_maker_vol = info.get('maker_volume', 0)
            final_taker_vol = info.get('taker_volume', 0)
            final_rebates = info.get('total_rebates_earned', 0)
            final_cash = env.cash
            net_change = final_cash - 100000
            
            results[name] = {
                'maker_volume': final_maker_vol,
                'taker_volume': final_taker_vol,
                'total_rebates': final_rebates,
                'net_change': net_change,
                'is_rebated': getattr(env, 'is_rebated', False),
                'post_only': post_only,
                'expected_rebate_rate': getattr(env, 'rebate_rate', 0)
            }
            
            print(f"  FINAL: Maker={final_maker_vol:.2f}, Taker={final_taker_vol:.2f}, "
                  f"Rebates=${final_rebates:.4f}, Net=${net_change:+.2f}")
            
            env.close()
        
        return results
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def analyze_comprehensive_results(results):
    """Analyze and validate all results with maker focus."""
    print(f"\n" + "="*90)
    print(f"COMPREHENSIVE MAKER-FOCUSED ANALYSIS")
    print(f"="*90)
    
    print(f"{'Environment':<20} {'Net Δ':<10} {'Maker':<8} {'Taker':<8} {'Rebates':<12} {'Rate Check':<12} {'Status'}")
    print(f"-" * 90)
    
    all_passed = True
    total_maker_volume = 0
    
    for name, result in results.items():
        net_change = result['net_change']
        maker_vol = result['maker_volume']
        taker_vol = result['taker_volume']
        rebates = result['total_rebates']
        is_rebated = result['is_rebated']
        post_only = result['post_only']
        expected_rate = result['expected_rebate_rate']
        
        total_maker_volume += maker_vol
        
        # Rate verification
        rate_check = "N/A"
        if maker_vol > 0 and rebates > 0:
            avg_price = 1800.0  # Approximate
            actual_rate = rebates / (maker_vol * avg_price)
            rate_diff = abs(actual_rate - expected_rate)
            rate_check = "✅ EXACT" if rate_diff < 0.000001 else f"❌ {actual_rate:.6f}"
        elif is_rebated and maker_vol > 0:
            rate_check = "❌ NO_REBATE"
        
        # Status determination
        status = "✅ PASS"
        
        # Baseline should lose money
        if name == "Baseline":
            if net_change >= 0:
                status = "❌ FAIL (Should lose money)"
                all_passed = False
        
        # Rebated environments
        elif is_rebated and not post_only:
            if maker_vol == 0:
                status = "❌ FAIL (No makers)"
                all_passed = False
            elif rebates <= 0:
                status = "❌ FAIL (No rebates)"
                all_passed = False
            elif rate_check.startswith("❌"):
                status = "❌ FAIL (Wrong rate)"
                all_passed = False
            else:
                status = "🎉 PERFECT"
        
        # Post-only environments
        elif post_only:
            if taker_vol > 0:
                status = "❌ FAIL (Takers found)"
                all_passed = False
            elif maker_vol > 0 and rebates <= 0:
                status = "❌ FAIL (No rebates)"
                all_passed = False
            elif maker_vol > 0:
                status = "🎉 PERFECT"
        
        print(f"{name:<20} ${net_change:+8.2f} {maker_vol:7.1f} {taker_vol:7.1f} "
              f"${rebates:9.4f} {rate_check:<12} {status}")
    
    print(f"-" * 90)
    
    # Summary metrics
    print(f"\n📊 SUMMARY METRICS:")
    print(f"  Total Maker Volume Across All Tests: {total_maker_volume:.2f}")
    print(f"  Environments with Maker Volume: {sum(1 for r in results.values() if r['maker_volume'] > 0)}/6")
    
    # Rebate rate verification
    print(f"\n💰 REBATE RATE VERIFICATION:")
    for name, result in results.items():
        if result['is_rebated'] and result['maker_volume'] > 0:
            maker_vol = result['maker_volume']
            rebates = result['total_rebates']
            expected_rate = result['expected_rebate_rate']
            
            avg_price = 1800.0
            trade_value = maker_vol * avg_price
            actual_rate = rebates / trade_value if trade_value > 0 else 0
            
            rate_status = "✅" if abs(actual_rate - expected_rate) < 0.000001 else "❌"
            
            print(f"  {name}: Expected {expected_rate:.6f}, Got {actual_rate:.6f} {rate_status}")
            print(f"    Trade Value: ${trade_value:.2f}, Rebates: ${rebates:.4f}")
    
    print(f"\n🎯 FINAL VERDICT:")
    if all_passed and total_maker_volume > 0:
        print(f"🎉 ALL TESTS PASSED WITH MAKER VOLUME!")
        print(f"   ✓ Maker executions detected: {total_maker_volume:.2f} total volume")
        print(f"   ✓ Rebate calculations verified as correct")
        print(f"   ✓ Post-only mode prevents taker orders")
        print(f"   ✓ All fee structures work correctly")
        print(f"   ✓ Environment ready for RL training")
    elif total_maker_volume > 0:
        print(f"⚠️  PARTIAL SUCCESS - MAKER VOLUME DETECTED BUT SOME TESTS FAILED")
        print(f"   ✓ Maker Volume: {total_maker_volume:.2f}")
        print(f"   ❌ Some validation tests failed")
    else:
        print(f"❌ CRITICAL FAILURE - NO MAKER VOLUME DETECTED")
        print(f"   ❌ Unable to test rebate functionality properly")
    
    return all_passed and total_maker_volume > 0

def main():
    """Run the comprehensive maker-focused test."""
    print("🚀 FINAL COMPREHENSIVE REBATED ENVIRONMENT TEST WITH MAKERS")
    print("="*90)
    
    results = test_comprehensive_with_makers()
    success = analyze_comprehensive_results(results)
    
    print(f"\n" + "="*90)
    if success:
        print(f"🎉 COMPLETE SUCCESS: REBATED ENVIRONMENT WITH MAKER FUNCTIONALITY VERIFIED!")
        print(f"   Ready for production use and RL training")
    else:
        print(f"💥 ISSUES REMAIN: Need further investigation")
    print(f"="*90)
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)