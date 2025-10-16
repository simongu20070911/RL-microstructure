#!/usr/bin/env python3
"""
Comprehensive stress test suite for edge cases
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_stress_test_data(scenario):
    """Create data for stress testing scenarios."""
    data = []
    
    if scenario == "locked_market":
        # Test locked market (bid = ask)
        for i in range(30):
            row = {'datetime': i}
            if i < 5:
                # Normal market
                base_bid, base_ask = 1800.00, 1800.10
            else:
                # Locked market
                base_bid, base_ask = 1800.05, 1800.05  # Same price
            
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 50.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 50.0
            data.append(row)
            
    elif scenario == "price_gaps":
        # Test price gaps and jumps
        price_phases = [
            (1800.00, 1800.10),  # Normal
            (1820.00, 1820.10),  # Jump up
            (1780.00, 1780.10),  # Jump down
            (1850.00, 1850.10),  # Big jump up
            (1800.00, 1800.10),  # Return to normal
            (1799.00, 1799.10),  # Slight down
        ]
        
        for i in range(30):
            row = {'datetime': i}
            phase_idx = min(i // 5, len(price_phases) - 1)
            base_bid, base_ask = price_phases[phase_idx]
            
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 75.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 75.0
            data.append(row)
            
    elif scenario == "thin_liquidity":
        # Test very thin liquidity
        for i in range(30):
            row = {'datetime': i}
            base_bid, base_ask = 1800.00, 1800.10
            
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                
                # Very thin liquidity
                if level <= 2:
                    row[f'bidqty{level}'] = 0.1  # Tiny quantity
                    row[f'askqty{level}'] = 0.1
                else:
                    row[f'bidqty{level}'] = 10.0
                    row[f'askqty{level}'] = 10.0
            data.append(row)
            
    elif scenario == "inventory_limits":
        # Test inventory limit stress
        for i in range(35):
            row = {'datetime': i}
            # Alternate sides to stress inventory
            if i % 4 < 2:
                base_bid, base_ask = 1805.00, 1805.10  # High - execute sells
            else:
                base_bid, base_ask = 1795.00, 1795.10  # Low - execute buys
                
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "rapid_oscillation":
        # Test rapid price oscillation
        for i in range(30):
            row = {'datetime': i}
            # Rapid oscillation between high and low
            if i % 2 == 0:
                base_bid, base_ask = 1800.50, 1800.60
            else:
                base_bid, base_ask = 1799.50, 1799.60
                
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
    
    else:  # default normal case
        for i in range(30):
            row = {'datetime': i}
            base_bid, base_ask = 1800.00 + i * 0.01, 1800.10 + i * 0.01
            
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

def stress_test_scenario(scenario_name, test_configs):
    """Stress test a specific scenario."""
    print(f"\n🔥 STRESS TESTING: {scenario_name.upper()}")
    print(f"{'='*60}")
    
    data_file = create_stress_test_data(scenario_name)
    bugs_found = []
    results = {}
    
    try:
        for config_name, fee_structure, post_only in test_configs:
            print(f"\n--- Testing {config_name} ---")
            
            error_count = 0
            execution_count = 0
            anomalies = []
            
            try:
                config = get_unified_config(fee_structure, post_only)
                config["csv_path"] = data_file
                config["episode_length"] = 25  # Larger episode length as requested
                config["latency_steps_long"] = 0
                config["latency_steps_short"] = 0
                config["max_order_volume"] = 1.0
                config["max_inventory"] = 3.0 if scenario_name == "inventory_limits" else 50.0
                
                env = RebatedHFTEnv(config)
                obs, info = env.reset()
                
                print(f"  Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
                
                # Test aggressive order placement
                if not post_only:
                    try:
                        action = [-1.0, 1.0, 1.0, 1.0, -1.0, 0.0]
                        obs, reward, term, trunc, info = env.step(action)
                        print(f"  ✅ Aggressive orders placed")
                    except Exception as e:
                        error_count += 1
                        bugs_found.append(f"{config_name} ({scenario_name}): Aggressive order error - {str(e)[:100]}")
                        print(f"  ❌ Aggressive order error: {str(e)[:50]}...")
                
                # Test passive orders
                try:
                    action = [0.3, -0.3, 0.8, 0.8, -1.0, 0.0]
                    obs, reward, term, trunc, info = env.step(action)
                    print(f"  ✅ Passive orders placed")
                except Exception as e:
                    error_count += 1
                    bugs_found.append(f"{config_name} ({scenario_name}): Passive order error - {str(e)[:100]}")
                    print(f"  ❌ Passive order error: {str(e)[:50]}...")
                
                # Run execution steps
                step_count = 2
                max_steps = 22
                
                while step_count < max_steps and not (term or trunc):
                    step_count += 1
                    
                    pre_cash = env.cash
                    pre_inventory = env.long_position - env.short_position
                    pre_maker = info.get('maker_volume', 0)
                    pre_active = len(env.active_orders)
                    
                    try:
                        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
                        
                        post_cash = env.cash
                        post_inventory = env.long_position - env.short_position
                        post_maker = info.get('maker_volume', 0)
                        post_active = len(env.active_orders)
                        
                        # Check for anomalies
                        if math.isnan(post_cash) or math.isinf(post_cash):
                            anomalies.append(f"Step {step_count}: Invalid cash {post_cash}")
                        
                        if post_cash < -10000:  # Very negative cash
                            anomalies.append(f"Step {step_count}: Extreme negative cash ${post_cash:.2f}")
                        
                        if abs(post_inventory) > config["max_inventory"] + 0.1:
                            anomalies.append(f"Step {step_count}: Inventory violation {post_inventory:.2f}")
                        
                        if post_maker > pre_maker:
                            execution_count += 1
                            
                        # Log interesting steps
                        if step_count % 3 == 0:
                            cash_change = post_cash - pre_cash
                            print(f"    Step {step_count}: Market {env.best_bid:.2f}/{env.best_ask:.2f}, "
                                  f"Cash Δ${cash_change:+.2f}, Active {pre_active}→{post_active}")
                            
                    except Exception as e:
                        error_count += 1
                        bugs_found.append(f"{config_name} ({scenario_name}): Step {step_count} error - {str(e)[:100]}")
                        print(f"  ❌ Step {step_count} error: {str(e)[:50]}...")
                        break
                
                # Final validation
                final_cash = env.cash
                final_inventory = env.long_position - env.short_position
                final_maker = info.get('maker_volume', 0)
                final_taker = info.get('taker_volume', 0)
                final_rebates = info.get('total_rebates_earned', 0)
                
                print(f"  📊 Final: Cash=${final_cash:.2f}, Inv={final_inventory:.2f}, "
                      f"M={final_maker:.2f}, T={final_taker:.2f}, R=${final_rebates:.4f}")
                print(f"  📈 Executions: {execution_count}, Errors: {error_count}, Anomalies: {len(anomalies)}")
                
                # Scenario-specific validation
                if scenario_name == "inventory_limits":
                    if abs(final_inventory) < 2.0 and execution_count > 0:
                        print(f"  ⚠️  Expected more inventory stress (final: {final_inventory:.2f})")
                
                if scenario_name == "rapid_oscillation" and execution_count == 0:
                    anomalies.append("No executions despite rapid price changes")
                
                if anomalies:
                    print(f"  ⚠️  Anomalies detected:")
                    for anomaly in anomalies[:3]:  # Show first 3
                        print(f"    • {anomaly}")
                        bugs_found.append(f"{config_name} ({scenario_name}): {anomaly}")
                
                results[config_name] = {
                    'final_cash': final_cash,
                    'final_inventory': final_inventory,
                    'maker_volume': final_maker,
                    'taker_volume': final_taker,
                    'rebates': final_rebates,
                    'executions': execution_count,
                    'errors': error_count,
                    'anomalies': len(anomalies)
                }
                
                env.close()
                
            except Exception as e:
                error_count += 1
                print(f"  💥 FATAL ERROR: {str(e)[:100]}...")
                bugs_found.append(f"{config_name} ({scenario_name}): Fatal error - {str(e)[:100]}")
                
                # Still record partial results
                results[config_name] = {
                    'final_cash': 0,
                    'final_inventory': 0,
                    'maker_volume': 0,
                    'taker_volume': 0,
                    'rebates': 0,
                    'executions': 0,
                    'errors': error_count,
                    'anomalies': 0
                }
    
    finally:
        if os.path.exists(data_file):
            os.unlink(data_file)
    
    return bugs_found, results

def run_stress_tests():
    """Run comprehensive stress tests."""
    print("🔥 COMPREHENSIVE STRESS TEST SUITE")
    print("=" * 80)
    
    scenarios = [
        "locked_market",
        "price_gaps", 
        "thin_liquidity",
        "inventory_limits",
        "rapid_oscillation"
    ]
    
    test_configs = [
        ("Baseline", "baseline", False),
        ("4bps Rebate", "rebate_4bps", False),
        ("8bps Rebate", "rebate_8bps", False),
        ("6bps Post-Only", "rebate_6bps", True),
    ]
    
    all_bugs = []
    all_results = {}
    
    for scenario in scenarios:
        bugs, results = stress_test_scenario(scenario, test_configs)
        all_bugs.extend(bugs)
        all_results[scenario] = results
    
    # COMPREHENSIVE ANALYSIS
    print(f"\n" + "=" * 80)
    print(f"🎯 STRESS TEST ANALYSIS")
    print(f"=" * 80)
    
    # Bug summary
    if all_bugs:
        print(f"\n❌ BUGS/ISSUES FOUND ({len(all_bugs)}):")
        for i, bug in enumerate(all_bugs[:10], 1):  # Show first 10
            print(f"  {i}. {bug}")
        if len(all_bugs) > 10:
            print(f"  ... and {len(all_bugs) - 10} more")
    else:
        print(f"\n✅ NO BUGS FOUND - ROBUST UNDER STRESS!")
    
    # Performance matrix
    print(f"\n📊 STRESS TEST PERFORMANCE MATRIX:")
    print(f"{'Scenario':<18} {'Config':<12} {'Exec':<5} {'Err':<4} {'M.Vol':<6} {'T.Vol':<6} {'Rebates':<8}")
    print(f"{'-'*18} {'-'*12} {'-'*5} {'-'*4} {'-'*6} {'-'*6} {'-'*8}")
    
    for scenario, configs in all_results.items():
        for config_name, result in configs.items():
            print(f"{scenario:<18} {config_name:<12} {result['executions']:>4d} {result['errors']:>3d} "
                  f"{result['maker_volume']:>5.1f} {result['taker_volume']:>5.1f} ${result['rebates']:>6.3f}")
    
    # Cross-scenario analysis
    print(f"\n🔍 CROSS-SCENARIO ANOMALY DETECTION:")
    
    # Check for scenarios with high error rates
    high_error_scenarios = []
    for scenario, configs in all_results.items():
        total_errors = sum(config['errors'] for config in configs.values())
        if total_errors > 5:
            high_error_scenarios.append((scenario, total_errors))
    
    if high_error_scenarios:
        print(f"  ⚠️  High error rate scenarios:")
        for scenario, errors in high_error_scenarios:
            print(f"    • {scenario}: {errors} errors")
            all_bugs.append(f"High error rate in {scenario}: {errors} errors")
    
    # Check for configurations that consistently fail
    config_errors = {}
    for scenario, configs in all_results.items():
        for config_name, result in configs.items():
            if config_name not in config_errors:
                config_errors[config_name] = 0
            config_errors[config_name] += result['errors']
    
    problematic_configs = [(config, errors) for config, errors in config_errors.items() if errors > 3]
    if problematic_configs:
        print(f"  ⚠️  Problematic configurations:")
        for config, errors in problematic_configs:
            print(f"    • {config}: {errors} total errors")
    
    # Success metrics
    total_executions = sum(sum(config['executions'] for config in configs.values()) 
                          for configs in all_results.values())
    total_errors = sum(sum(config['errors'] for config in configs.values()) 
                      for configs in all_results.values())
    
    print(f"\n🏆 STRESS TEST SUMMARY:")
    print(f"  Scenarios tested: {len(scenarios)}")
    print(f"  Configurations tested: {len(test_configs)}")
    print(f"  Total executions: {total_executions}")
    print(f"  Total errors: {total_errors}")
    print(f"  Success rate: {(1 - total_errors / max(total_executions + total_errors, 1)) * 100:.1f}%")
    
    if len(all_bugs) == 0:
        print(f"  🎉 EXCELLENT: Environment handles all stress scenarios robustly!")
    elif len(all_bugs) < 5:
        print(f"  ✅ GOOD: Minor issues found, mostly edge case handling")
    else:
        print(f"  ⚠️  ATTENTION: Multiple issues found, recommend investigation")
    
    return all_bugs

if __name__ == "__main__":
    run_stress_tests()