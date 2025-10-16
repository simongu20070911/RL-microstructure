#!/usr/bin/env python3
"""
Comprehensive testing of all cases to find bugs
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_comprehensive_test_data():
    """Create comprehensive test data with various market conditions."""
    data = []
    
    for i in range(30):  # Long test
        row = {'datetime': i}
        
        if i < 5:
            # Phase 1: Stable for order placement
            base_bid, base_ask = 1800.00, 1800.10
        elif i < 10:
            # Phase 2: Market UP - execute sell orders
            base_bid, base_ask = 1800.50, 1800.60
        elif i < 15:
            # Phase 3: Market DOWN - execute buy orders  
            base_bid, base_ask = 1799.50, 1799.60
        elif i < 20:
            # Phase 4: Extreme volatility - test edge cases
            base_bid, base_ask = 1801.00, 1801.10  # Big jump
        elif i < 25:
            # Phase 5: Crash - test inventory limits
            base_bid, base_ask = 1798.00, 1798.10  # Big drop
        else:
            # Phase 6: Return to normal
            base_bid, base_ask = 1800.00, 1800.10
        
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

def comprehensive_bug_hunt():
    """Comprehensive test to find bugs."""
    print("🔍 COMPREHENSIVE BUG HUNTING TEST")
    print("=" * 80)
    
    test_csv = create_comprehensive_test_data()
    
    # Test configurations
    test_configs = [
        ("Baseline", "baseline", False),
        ("4bps Rebate", "rebate_4bps", False),
        ("6bps Rebate", "rebate_6bps", False), 
        ("8bps Rebate", "rebate_8bps", False),
        ("6bps Post-Only", "rebate_6bps", True),
    ]
    
    results = {}
    potential_bugs = []
    
    try:
        for config_name, fee_structure, post_only in test_configs:
            print(f"\n{'='*20} TESTING {config_name} {'='*20}")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 25
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 1.0  # Larger volume for stress test
            config["max_inventory"] = 10.0  # Higher limit for stress test
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Configuration:")
            print(f"  Fee structure: {fee_structure}")
            print(f"  Post-only: {post_only}")
            print(f"  Transaction costs: {config['transaction_cost_long']:.6f}")
            print(f"  Max inventory: {config['max_inventory']}")
            print(f"  Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Track metrics
            execution_log = []
            error_log = []
            inventory_violations = []
            cash_anomalies = []
            volume_inconsistencies = []
            
            # PHASE 1: Place diverse orders early
            print(f"\n📋 PLACING DIVERSE ORDERS")
            
            # Place multiple order types
            actions_to_test = [
                # (description, action)
                ("Aggressive Buy/Sell", [-1.0, 1.0, 1.0, 1.0, -1.0, 0.0]),  # Very aggressive
                ("Passive Buy/Sell", [0.0, 0.0, 0.8, 0.8, -1.0, 0.0]),      # At market
                ("Conservative", [0.5, -0.5, 0.5, 0.5, -1.0, 0.0]),         # Conservative
            ]
            
            for desc, action in actions_to_test:
                try:
                    obs, reward, term, trunc, info = env.step(action)
                    print(f"  ✅ {desc}: Placed successfully")
                except Exception as e:
                    error_log.append(f"Order placement error ({desc}): {e}")
                    print(f"  ❌ {desc}: ERROR - {e}")
            
            # Activate all orders
            for _ in range(3):  # Multiple activation steps
                obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            print(f"Orders after placement: {len(env.active_orders)} active, {len(env.pending_orders)} pending")
            
            # PHASE 2: Step through market conditions
            print(f"\n📈 STEPPING THROUGH MARKET CONDITIONS")
            
            step_count = 6  # Start after placement
            max_steps = 22
            
            while step_count < max_steps and not (term or trunc):
                step_count += 1
                
                # Track state before
                pre_cash = env.cash
                pre_inventory = env.long_position - env.short_position
                pre_maker = info.get('maker_volume', 0)
                pre_taker = info.get('taker_volume', 0)
                pre_rebates = info.get('total_rebates_earned', 0)
                pre_active = len(env.active_orders)
                
                # Determine test phase
                if step_count <= 10:
                    phase = "UP_MOVE"
                elif step_count <= 15:
                    phase = "DOWN_MOVE"
                elif step_count <= 20:
                    phase = "VOLATILITY"
                else:
                    phase = "STABILIZE"
                
                # Execute step with execution enabled
                try:
                    obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
                except Exception as e:
                    error_log.append(f"Step {step_count} execution error: {e}")
                    print(f"  ❌ Step {step_count}: ERROR - {e}")
                    break
                
                # Track state after
                post_cash = env.cash
                post_inventory = env.long_position - env.short_position
                post_maker = info.get('maker_volume', 0)
                post_taker = info.get('taker_volume', 0)
                post_rebates = info.get('total_rebates_earned', 0)
                post_active = len(env.active_orders)
                
                # Calculate changes
                cash_change = post_cash - pre_cash
                inventory_change = post_inventory - pre_inventory
                maker_change = post_maker - pre_maker
                taker_change = post_taker - pre_taker
                rebate_change = post_rebates - pre_rebates
                
                # Log significant events
                if step_count % 3 == 0 or abs(cash_change) > 0.01 or abs(inventory_change) > 0.01:
                    print(f"  Step {step_count} ({phase}): Market {env.best_bid:.2f}/{env.best_ask:.2f}")
                    print(f"    Cash: ${pre_cash:.2f}→${post_cash:.2f} (Δ${cash_change:+.2f})")
                    print(f"    Inventory: {pre_inventory:.2f}→{post_inventory:.2f} (Δ{inventory_change:+.2f})")
                    print(f"    Volume: M{pre_maker:.2f}→{post_maker:.2f} (+{maker_change:.2f}), T{pre_taker:.2f}→{post_taker:.2f} (+{taker_change:.2f})")
                    print(f"    Orders: {pre_active}→{post_active}, Rebates: ${rebate_change:+.6f}")
                
                # Check for potential bugs
                
                # 1. Inventory violations
                max_inv = config["max_inventory"]
                if abs(post_inventory) > max_inv + 0.001:
                    inventory_violations.append({
                        'step': step_count,
                        'inventory': post_inventory,
                        'limit': max_inv,
                        'violation': abs(post_inventory) - max_inv
                    })
                
                # 2. Cash anomalies
                if post_cash < -0.01:  # Should never be significantly negative
                    cash_anomalies.append({
                        'step': step_count,
                        'cash': post_cash,
                        'change': cash_change
                    })
                
                # 3. Volume consistency
                if post_only and post_taker > pre_taker:
                    volume_inconsistencies.append({
                        'step': step_count,
                        'issue': 'Post-only mode generated taker volume',
                        'taker_change': taker_change
                    })
                
                # 4. Rebate calculation checks
                if env.is_rebated and maker_change > 0:
                    expected_rebate = maker_change * env.rebate_rate * env.best_bid  # Approximate
                    rebate_error = abs(rebate_change - expected_rebate) / max(expected_rebate, 0.000001)
                    if rebate_error > 0.1:  # 10% tolerance
                        volume_inconsistencies.append({
                            'step': step_count,
                            'issue': 'Rebate calculation mismatch',
                            'expected': expected_rebate,
                            'actual': rebate_change,
                            'error_pct': rebate_error * 100
                        })
                
                # 5. Execution tracking
                if maker_change > 0 or taker_change > 0:
                    execution_log.append({
                        'step': step_count,
                        'phase': phase,
                        'maker_vol': maker_change,
                        'taker_vol': taker_change,
                        'cash_change': cash_change,
                        'rebate_change': rebate_change,
                        'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                    })
            
            # FINAL ANALYSIS
            print(f"\n📊 FINAL ANALYSIS FOR {config_name}:")
            
            final_cash = env.cash
            final_inventory = env.long_position - env.short_position
            final_maker = info.get('maker_volume', 0)
            final_taker = info.get('taker_volume', 0)
            final_rebates = info.get('total_rebates_earned', 0)
            final_cash_change = final_cash - 100000
            
            print(f"  Final cash: ${final_cash:.2f} (Δ${final_cash_change:+.2f})")
            print(f"  Final inventory: {final_inventory:.2f}")
            print(f"  Final volumes: Maker {final_maker:.2f}, Taker {final_taker:.2f}")
            print(f"  Total rebates: ${final_rebates:.6f}")
            print(f"  Active orders: {len(env.active_orders)}")
            print(f"  Executions: {len(execution_log)}")
            
            # Store results
            results[config_name] = {
                'final_cash_change': final_cash_change,
                'final_inventory': final_inventory,
                'maker_volume': final_maker,
                'taker_volume': final_taker,
                'total_rebates': final_rebates,
                'executions': len(execution_log),
                'errors': len(error_log),
                'inventory_violations': len(inventory_violations),
                'cash_anomalies': len(cash_anomalies),
                'volume_inconsistencies': len(volume_inconsistencies)
            }
            
            # Report issues
            if error_log:
                print(f"  ❌ ERRORS FOUND ({len(error_log)}):")
                for error in error_log:
                    print(f"    • {error}")
                    potential_bugs.append(f"{config_name}: {error}")
            
            if inventory_violations:
                print(f"  ⚠️  INVENTORY VIOLATIONS ({len(inventory_violations)}):")
                for violation in inventory_violations:
                    print(f"    • Step {violation['step']}: Inventory {violation['inventory']:.2f} exceeds limit {violation['limit']:.2f}")
                    potential_bugs.append(f"{config_name}: Inventory violation at step {violation['step']}")
            
            if cash_anomalies:
                print(f"  💰 CASH ANOMALIES ({len(cash_anomalies)}):")
                for anomaly in cash_anomalies:
                    print(f"    • Step {anomaly['step']}: Negative cash ${anomaly['cash']:.2f}")
                    potential_bugs.append(f"{config_name}: Negative cash at step {anomaly['step']}")
            
            if volume_inconsistencies:
                print(f"  📊 VOLUME INCONSISTENCIES ({len(volume_inconsistencies)}):")
                for inconsistency in volume_inconsistencies:
                    print(f"    • Step {inconsistency['step']}: {inconsistency['issue']}")
                    potential_bugs.append(f"{config_name}: {inconsistency['issue']}")
            
            if not (error_log or inventory_violations or cash_anomalies or volume_inconsistencies):
                print(f"  ✅ NO BUGS FOUND")
            
            env.close()
        
        # COMPARATIVE ANALYSIS
        print(f"\n" + "=" * 80)
        print(f"🔬 COMPARATIVE ANALYSIS ACROSS ALL CONFIGURATIONS")
        print(f"=" * 80)
        
        print(f"{'Configuration':<15} {'Cash Δ':<10} {'Maker Vol':<10} {'Taker Vol':<10} {'Rebates':<12} {'Executions':<10} {'Issues':<8}")
        print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")
        
        for config_name, result in results.items():
            issues = result['errors'] + result['inventory_violations'] + result['cash_anomalies'] + result['volume_inconsistencies']
            print(f"{config_name:<15} ${result['final_cash_change']:>8.2f} {result['maker_volume']:>9.2f} {result['taker_volume']:>9.2f} ${result['total_rebates']:>10.6f} {result['executions']:>9d} {issues:>7d}")
        
        # CROSS-CONFIGURATION BUG DETECTION
        print(f"\n🐛 CROSS-CONFIGURATION BUG ANALYSIS:")
        
        # Check for inconsistencies between similar configs
        baseline_maker = results.get('Baseline', {}).get('maker_volume', 0)
        rebate_4_maker = results.get('4bps Rebate', {}).get('maker_volume', 0)
        rebate_6_maker = results.get('6bps Rebate', {}).get('maker_volume', 0)
        rebate_8_maker = results.get('8bps Rebate', {}).get('maker_volume', 0)
        post_only_maker = results.get('6bps Post-Only', {}).get('maker_volume', 0)
        
        # Volume consistency check
        normal_volumes = [baseline_maker, rebate_4_maker, rebate_6_maker, rebate_8_maker]
        volume_variance = np.std(normal_volumes) if normal_volumes else 0
        
        if volume_variance > 0.1:
            potential_bugs.append(f"HIGH VOLUME VARIANCE: Normal modes show inconsistent volumes (std={volume_variance:.3f})")
            print(f"  ⚠️  Volume variance between normal modes: {volume_variance:.3f}")
        
        # Post-only specific checks
        if post_only_maker == 0:
            potential_bugs.append("Post-only mode shows 0 volume despite execution opportunities")
            print(f"  ❌ Post-only mode failed to execute trades")
        
        # Rebate progression check
        rebate_amounts = [
            results.get('4bps Rebate', {}).get('total_rebates', 0),
            results.get('6bps Rebate', {}).get('total_rebates', 0),
            results.get('8bps Rebate', {}).get('total_rebates', 0),
        ]
        
        if rebate_amounts[0] > 0 and rebate_amounts[1] > 0 and rebate_amounts[2] > 0:
            # Check if rebates increase with rate (approximately)
            if not (rebate_amounts[0] < rebate_amounts[1] < rebate_amounts[2]):
                potential_bugs.append("Rebate amounts don't increase with rebate rate as expected")
                print(f"  ⚠️  Rebate progression anomaly: {rebate_amounts}")
        
        # FINAL BUG REPORT
        print(f"\n🎯 FINAL BUG HUNT RESULTS:")
        if potential_bugs:
            print(f"❌ BUGS FOUND ({len(potential_bugs)}):")
            for i, bug in enumerate(potential_bugs, 1):
                print(f"  {i}. {bug}")
        else:
            print(f"✅ NO BUGS FOUND - All configurations working correctly!")
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    comprehensive_bug_hunt()