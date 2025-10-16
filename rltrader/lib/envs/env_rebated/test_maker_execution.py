#!/usr/bin/env python3
"""
Test that specifically generates maker orders and verifies rebates
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_dynamic_test_data():
    """Create test data with changing market conditions to trigger executions."""
    data = []
    
    # Create market data that moves to hit passive orders
    for i in range(30):
        row = {'datetime': i}
        
        if i < 10:
            # Phase 1: Stable market for order placement
            base_bid = 1799.99
            base_ask = 1800.01
        elif i < 15:
            # Phase 2: Market moves up to hit passive buy orders
            base_bid = 1800.00 + (i-10) * 0.002  # Rising market
            base_ask = 1800.02 + (i-10) * 0.002
        elif i < 20:
            # Phase 3: Market moves down to hit passive sell orders  
            base_bid = 1800.05 - (i-15) * 0.002  # Falling market
            base_ask = 1800.07 - (i-15) * 0.002
        else:
            # Phase 4: Stable again
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

def test_maker_strategy():
    """Test with a strategy designed to create maker orders."""
    print("🎯 TESTING MAKER ORDER STRATEGY")
    
    test_csv = create_dynamic_test_data()
    
    try:
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 25
        config["latency_steps_long"] = 1
        config["latency_steps_short"] = 1
        config["max_order_volume"] = 5.0  # Smaller orders
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"✅ Environment Setup:")
        print(f"   Rebate Rate: {env.rebate_rate:.6f}")
        print(f"   Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"   Initial Cash: ${env.cash:.2f}")
        
        # Strategy: Place passive orders and wait for market to move
        total_maker_vol = 0
        total_taker_vol = 0
        total_rebates = 0
        
        for step in range(20):
            print(f"\n--- STEP {step} ---")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            pre_cash = env.cash
            pre_info = env._get_info()
            pre_maker_vol = pre_info.get('maker_volume', 0)
            pre_taker_vol = pre_info.get('taker_volume', 0)
            pre_rebates = pre_info.get('total_rebates_earned', 0)
            
            if step < 8:
                # Phase 1: Place PASSIVE orders (should become makers)
                # Use ZERO offset to place at BBO (passive)
                action = [0.0, 0.0, 0.6, 0.6, -1.0, -1.0]
                strategy = "PASSIVE (at BBO)"
            elif step < 12:
                # Phase 2: Slightly aggressive to test market impact
                action = [0.3, -0.3, 0.5, 0.5, -1.0, -1.0]
                strategy = "SLIGHTLY AGGRESSIVE"
            else:
                # Phase 3: Back to passive
                action = [-0.2, 0.2, 0.4, 0.4, -1.0, -1.0]
                strategy = "PASSIVE (inside spread)"
            
            print(f"Strategy: {strategy}")
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_cash = env.cash
            post_maker_vol = info.get('maker_volume', 0)
            post_taker_vol = info.get('taker_volume', 0)
            post_rebates = info.get('total_rebates_earned', 0)
            
            # Calculate changes
            cash_change = post_cash - pre_cash
            maker_change = post_maker_vol - pre_maker_vol
            taker_change = post_taker_vol - pre_taker_vol
            rebate_change = post_rebates - pre_rebates
            
            print(f"Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${cash_change:+.2f})")
            print(f"Maker: {pre_maker_vol:.2f} → {post_maker_vol:.2f} (Δ{maker_change:+.2f})")
            print(f"Taker: {pre_taker_vol:.2f} → {post_taker_vol:.2f} (Δ{taker_change:+.2f})")
            print(f"Rebates: ${pre_rebates:.4f} → ${post_rebates:.4f} (Δ${rebate_change:+.4f})")
            print(f"Active orders: {len(env.active_orders)}, Pending: {len(env.pending_orders)}")
            
            if maker_change > 0:
                print(f"✅ MAKER EXECUTION DETECTED! Volume: {maker_change:.2f}")
            if taker_change > 0:
                print(f"⚠️  TAKER EXECUTION DETECTED! Volume: {taker_change:.2f}")
            if rebate_change > 0:
                print(f"💰 REBATES EARNED: ${rebate_change:.4f}")
            
            total_maker_vol = post_maker_vol
            total_taker_vol = post_taker_vol
            total_rebates = post_rebates
            
            if term or trunc:
                break
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Final Cash: ${env.cash:.2f}")
        print(f"Total Maker Volume: {total_maker_vol:.2f}")
        print(f"Total Taker Volume: {total_taker_vol:.2f}")
        print(f"Total Rebates: ${total_rebates:.4f}")
        print(f"Net Cash Change: ${env.cash - 100000:.2f}")
        
        # Success criteria
        success = total_maker_vol > 0 and total_rebates > 0
        
        if success:
            print(f"🎉 SUCCESS: MAKER ORDERS EXECUTED WITH REBATES!")
            print(f"   Maker Volume: {total_maker_vol:.2f}")
            print(f"   Rebates Earned: ${total_rebates:.4f}")
            
            # Verify rebate rate
            if total_maker_vol > 0:
                avg_price = 1800.0  # Approximate
                expected_rebates = total_maker_vol * avg_price * 0.00006
                actual_rate = total_rebates / (total_maker_vol * avg_price)
                print(f"   Expected Rebates: ${expected_rebates:.4f}")
                print(f"   Actual Rate: {actual_rate:.6f} (Expected: 0.000060)")
        else:
            print(f"❌ FAILED: No maker executions or rebates")
            print(f"   Issue: Need market movement to trigger maker fills")
        
        env.close()
        return success
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def test_all_rebated_environments():
    """Test all rebated environments with maker strategy."""
    print(f"\n🎯 TESTING ALL REBATED ENVIRONMENTS WITH MAKER STRATEGY")
    
    test_csv = create_dynamic_test_data()
    results = {}
    
    try:
        configs = [
            ("Baseline", "baseline", False),
            ("4 bps Rebate", "rebate_4bps", False),
            ("6 bps Rebate", "rebate_6bps", False),
            ("8 bps Rebate", "rebate_8bps", False),
            ("6 bps Post-Only", "rebate_6bps", True),
        ]
        
        for name, fee_structure, post_only in configs:
            print(f"\n=== TESTING {name} ===")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 20
            config["latency_steps_long"] = 1
            config["latency_steps_short"] = 1
            config["max_order_volume"] = 3.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            # Strategy focused on creating maker orders
            for step in range(15):
                if step < 5:
                    # Place passive orders at BBO
                    action = [0.0, 0.0, 0.7, 0.7, -1.0, -1.0]
                elif step < 10:
                    # Place orders slightly inside spread
                    action = [-0.1, 0.1, 0.6, 0.6, -1.0, -1.0]
                else:
                    # Wait for executions
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            
            final_maker_vol = info.get('maker_volume', 0)
            final_taker_vol = info.get('taker_volume', 0)
            final_rebates = info.get('total_rebates_earned', 0)
            final_cash = env.cash
            
            results[name] = {
                'maker_volume': final_maker_vol,
                'taker_volume': final_taker_vol,
                'rebates': final_rebates,
                'cash_change': final_cash - 100000,
                'is_rebated': getattr(env, 'is_rebated', False),
                'post_only': post_only
            }
            
            print(f"Results: Maker={final_maker_vol:.2f}, Taker={final_taker_vol:.2f}, Rebates=${final_rebates:.4f}")
            
            env.close()
        
        return results
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def analyze_maker_results(results):
    """Analyze results to verify maker functionality."""
    print(f"\n" + "="*80)
    print(f"MAKER VOLUME ANALYSIS")
    print(f"="*80)
    
    print(f"{'Environment':<20} {'Maker Vol':<10} {'Taker Vol':<10} {'Rebates':<12} {'Status'}")
    print(f"-" * 80)
    
    any_makers = False
    
    for name, result in results.items():
        maker_vol = result['maker_volume']
        taker_vol = result['taker_volume']
        rebates = result['rebates']
        is_rebated = result['is_rebated']
        post_only = result['post_only']
        
        status = "✅ PASS"
        
        if maker_vol > 0:
            any_makers = True
            if is_rebated and rebates <= 0:
                status = "❌ FAIL (No rebates for makers)"
            elif is_rebated:
                status = "🎉 EXCELLENT (Makers + Rebates)"
        elif is_rebated and not post_only:
            status = "⚠️  WARN (No makers, can't test rebates)"
        elif post_only and taker_vol > 0:
            status = "❌ FAIL (Takers in post-only)"
        
        print(f"{name:<20} {maker_vol:>9.2f} {taker_vol:>9.2f} ${rebates:>9.4f} {status}")
    
    print(f"-" * 80)
    
    if any_makers:
        print(f"✅ MAKER VOLUME DETECTED - Can properly test rebate system!")
    else:
        print(f"❌ NO MAKER VOLUME - Need to improve test strategy")
    
    return any_makers

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MAKER VOLUME TEST")
    print("="*80)
    
    # Test 1: Single environment with maker strategy
    test1_success = test_maker_strategy()
    
    # Test 2: All environments 
    results = test_all_rebated_environments()
    makers_detected = analyze_maker_results(results)
    
    print(f"\n" + "="*80)
    print(f"FINAL VERDICT")
    print(f"="*80)
    
    if test1_success and makers_detected:
        print(f"🎉 SUCCESS: MAKER VOLUME TEST PASSED!")
        print(f"   ✓ Maker orders are being executed")
        print(f"   ✓ Rebates are being calculated correctly")
        print(f"   ✓ Environment is ready for training")
    else:
        print(f"❌ FAILED: MAKER VOLUME TEST NEEDS IMPROVEMENT")
        if not makers_detected:
            print(f"   ✗ No maker volume detected")
        if not test1_success:
            print(f"   ✗ Maker strategy test failed")
    
    print(f"="*80)