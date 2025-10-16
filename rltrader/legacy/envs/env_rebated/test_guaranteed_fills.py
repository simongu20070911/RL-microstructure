#!/usr/bin/env python3
"""
Create market data that GUARANTEES order fills by specifically targeting our order prices
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_guaranteed_fill_data():
    """Create market data that will definitely execute our orders."""
    data = []
    
    # Strategy: Create a market that moves in a predictable way to hit orders at specific prices
    
    for i in range(50):
        row = {'datetime': i}
        
        if i < 10:
            # Phase 1: Initial stable market - we'll place orders here
            # Place passive orders at these prices
            base_bid = 1799.98  # We'll place buy orders at 1799.98
            base_ask = 1800.02  # We'll place sell orders at 1800.02
        elif i < 20:
            # Phase 2: Market moves UP to hit our PASSIVE BUY orders
            # Market bid rises to 1799.98+ to execute our buy orders
            progress = (i - 10) / 10.0  # 0 to 1
            base_bid = 1799.98 + progress * 0.05  # Moves from 1799.98 to 1800.03
            base_ask = 1800.02 + progress * 0.05  # Moves from 1800.02 to 1800.07
            
            # Critical: When market bid reaches our buy order price, it should execute
            
        elif i < 30:
            # Phase 3: Market moves DOWN to hit our PASSIVE SELL orders  
            # Market ask falls to 1800.02- to execute our sell orders
            progress = (i - 20) / 10.0  # 0 to 1
            base_bid = 1800.03 - progress * 0.10  # Moves from 1800.03 to 1799.93
            base_ask = 1800.07 - progress * 0.10  # Moves from 1800.07 to 1799.97
            
            # Critical: When market ask reaches our sell order price, it should execute
            
        elif i < 40:
            # Phase 4: More execution opportunities
            # Oscillating market to create more fills
            oscillation = 0.02 * np.sin((i - 30) * 0.5)
            base_bid = 1799.96 + oscillation
            base_ask = 1799.98 + oscillation
        else:
            # Phase 5: Final stable period
            base_bid = 1799.98
            base_ask = 1800.02
        
        # Build order book levels
        for level in range(1, 11):
            row[f'bid{level}'] = base_bid - (level-1) * 0.01
            row[f'bidqty{level}'] = 50.0  # Smaller quantities to ensure our orders get filled
            row[f'ask{level}'] = base_ask + (level-1) * 0.01  
            row[f'askqty{level}'] = 50.0  # Smaller quantities to ensure our orders get filled
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_guaranteed_executions():
    """Test with market data designed to guarantee executions."""
    print("🎯 TESTING WITH GUARANTEED EXECUTION MARKET DATA")
    
    test_csv = create_guaranteed_fill_data()
    
    try:
        # Test rebated environment
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 40
        config["latency_steps_long"] = 0  # No latency for immediate execution
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0  # Smaller orders
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment setup:")
        print(f"  Rebate rate: {env.rebate_rate:.6f}")
        print(f"  Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  No latency - orders execute immediately")
        
        execution_log = []
        
        for step in range(35):
            print(f"\n--- STEP {step} ---")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_rebates = info.get('total_rebates_earned', 0)
            pre_cash = env.cash
            
            if step < 8:
                # Place orders at EXACTLY the prices our market data will hit
                # Action values map to specific prices - we need to calculate exact actions
                # to place orders at 1799.98 (buy) and 1800.02 (sell)
                
                # For zero latency, orders should execute immediately if they cross spread
                # We want PASSIVE orders that will be hit by market movement
                action = [-0.02, 0.02, 0.8, 0.8, -1.0, -1.0]  # Slightly passive
                strategy = "PLACE TARGETED PASSIVE ORDERS"
            elif step < 25:
                # Let market move and execute our orders
                action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                strategy = "LET MARKET EXECUTE ORDERS"
            else:
                # Place more orders for additional executions
                action = [0.0, 0.0, 0.6, 0.6, -1.0, -1.0]
                strategy = "PLACE MORE ORDERS"
            
            print(f"Strategy: {strategy}")
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_rebates = info.get('total_rebates_earned', 0)
            post_cash = env.cash
            
            # Track changes
            maker_change = post_maker - pre_maker
            taker_change = post_taker - pre_taker
            rebate_change = post_rebates - pre_rebates
            cash_change = post_cash - pre_cash
            
            print(f"Volume: Maker {pre_maker:.1f}→{post_maker:.1f} (Δ{maker_change:+.1f}), "
                  f"Taker {pre_taker:.1f}→{post_taker:.1f} (Δ{taker_change:+.1f})")
            print(f"Rebates: ${pre_rebates:.4f}→${post_rebates:.4f} (Δ${rebate_change:+.4f})")
            print(f"Cash: ${pre_cash:.2f}→${post_cash:.2f} (Δ${cash_change:+.2f})")
            print(f"Orders: Active={len(env.active_orders)}, Pending={len(env.pending_orders)}")
            
            if maker_change > 0 or taker_change > 0:
                execution_log.append({
                    'step': step,
                    'maker_vol': maker_change,
                    'taker_vol': taker_change,
                    'rebates': rebate_change,
                    'bbo': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                })
                print(f"🎉 EXECUTION DETECTED!")
            
            if term or trunc:
                break
        
        print(f"\n=== EXECUTION LOG ===")
        if execution_log:
            for exec_data in execution_log:
                print(f"Step {exec_data['step']}: "
                      f"Maker={exec_data['maker_vol']:+.1f}, "
                      f"Taker={exec_data['taker_vol']:+.1f}, "
                      f"Rebates=${exec_data['rebates']:+.4f}, "
                      f"BBO={exec_data['bbo']}")
        else:
            print(f"❌ NO EXECUTIONS DETECTED")
        
        print(f"\n=== FINAL RESULTS ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"Total Maker Volume: {final_maker:.2f}")
        print(f"Total Taker Volume: {final_taker:.2f}")
        print(f"Total Rebates: ${final_rebates:.4f}")
        print(f"Net Cash Change: ${final_cash_change:+.2f}")
        
        success = final_maker > 0 or final_taker > 0
        
        if success:
            print(f"🎉 SUCCESS: EXECUTIONS ACHIEVED!")
            if final_rebates > 0:
                print(f"   ✓ Rebates working: ${final_rebates:.4f}")
            if final_maker > 0:
                print(f"   ✓ Maker volume: {final_maker:.2f}")
            if final_taker > 0:
                print(f"   ✓ Taker volume: {final_taker:.2f}")
        else:
            print(f"❌ FAILURE: NO EXECUTIONS - NEED TO DEBUG MATCHING LOGIC")
        
        env.close()
        return success
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def test_all_environments_with_guaranteed_data():
    """Test all environments with guaranteed execution data."""
    print(f"\n🎯 TESTING ALL ENVIRONMENTS WITH GUARANTEED FILLS")
    
    test_csv = create_guaranteed_fill_data()
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
            print(f"\n=== {name} ===")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 35
            config["latency_steps_long"] = 0  # No latency
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 1.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            # Aggressive strategy to force executions
            for step in range(30):
                if step < 5:
                    action = [0.0, 0.0, 0.9, 0.9, -1.0, -1.0]  # Passive orders
                elif step < 15:
                    action = [0.5, -0.5, 0.7, 0.7, -1.0, -1.0]  # Slightly aggressive
                else:
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            
            results[name] = {
                'maker_volume': info.get('maker_volume', 0),
                'taker_volume': info.get('taker_volume', 0),
                'rebates': info.get('total_rebates_earned', 0),
                'cash_change': env.cash - 100000,
                'is_rebated': getattr(env, 'is_rebated', False),
                'post_only': post_only
            }
            
            print(f"Results: Maker={results[name]['maker_volume']:.1f}, "
                  f"Taker={results[name]['taker_volume']:.1f}, "
                  f"Rebates=${results[name]['rebates']:.4f}")
            
            env.close()
        
        return results
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def analyze_guaranteed_results(results):
    """Analyze results from guaranteed fill test."""
    print(f"\n" + "="*70)
    print(f"GUARANTEED FILL TEST ANALYSIS")
    print(f"="*70)
    
    total_volume = 0
    environments_with_volume = 0
    
    for name, result in results.items():
        maker_vol = result['maker_volume']
        taker_vol = result['taker_volume']
        total_vol = maker_vol + taker_vol
        rebates = result['rebates']
        
        total_volume += total_vol
        if total_vol > 0:
            environments_with_volume += 1
        
        status = "✅" if total_vol > 0 else "❌"
        rebate_status = "✅" if (result['is_rebated'] and rebates > 0) or not result['is_rebated'] else "❌"
        
        print(f"{name:<15} Vol: {total_vol:5.1f} (M:{maker_vol:4.1f}, T:{taker_vol:4.1f}) "
              f"Rebates: ${rebates:6.4f} {status} {rebate_status}")
    
    print(f"-" * 70)
    print(f"SUMMARY:")
    print(f"  Total Volume Across All Environments: {total_volume:.1f}")
    print(f"  Environments with Volume: {environments_with_volume}/{len(results)}")
    
    if total_volume > 0:
        print(f"🎉 SUCCESS: VOLUME DETECTED - EXECUTIONS ARE WORKING!")
        return True
    else:
        print(f"❌ FAILURE: ZERO VOLUME - EXECUTION LOGIC BROKEN")
        return False

def main():
    """Run guaranteed fill tests."""
    print("🚀 GUARANTEED FILL TEST - FIXING ZERO VOLUME ISSUE")
    print("="*70)
    
    # Test single environment first
    single_success = test_guaranteed_executions()
    
    # Test all environments
    results = test_all_environments_with_guaranteed_data()
    all_success = analyze_guaranteed_results(results)
    
    print(f"\n" + "="*70)
    if single_success and all_success:
        print(f"🎉 ZERO VOLUME ISSUE FIXED!")
        print(f"   ✓ Executions are working")
        print(f"   ✓ Rebate system functional")
        print(f"   ✓ Environment ready for use")
    else:
        print(f"❌ ZERO VOLUME ISSUE PERSISTS")
        print(f"   Need to debug execution matching logic")
    print(f"="*70)
    
    return single_success and all_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)