#!/usr/bin/env python3
"""
Advanced edge case testing to find bugs in corner scenarios
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

def create_edge_case_data(scenario):
    """Create data for specific edge case scenarios."""
    data = []
    
    if scenario == "zero_spread":
        # Test minimal spread / tight market
        for i in range(10):
            row = {'datetime': i}
            base_bid, base_ask = 1800.00, 1800.01  # Minimal spread
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "crossed_market":
        # Test crossed market (bid > ask)
        for i in range(10):
            row = {'datetime': i}
            if i < 5:
                # Normal market
                base_bid, base_ask = 1800.00, 1800.10
            else:
                # Crossed market
                base_bid, base_ask = 1800.10, 1800.00  # Bid > Ask!
            
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "extreme_volatility":
        # Test extreme price movements
        for i in range(15):
            row = {'datetime': i}
            if i < 3:
                base_bid, base_ask = 1800.00, 1800.10
            elif i < 6:
                base_bid, base_ask = 1850.00, 1850.10  # +50 tick jump
            elif i < 9:
                base_bid, base_ask = 1750.00, 1750.10  # -100 tick crash
            elif i < 12:
                base_bid, base_ask = 1900.00, 1900.10  # +150 tick spike
            else:
                base_bid, base_ask = 1800.00, 1800.10  # Return to normal
                
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "zero_liquidity":
        # Test zero liquidity scenarios
        for i in range(10):
            row = {'datetime': i}
            base_bid, base_ask = 1800.00, 1800.10
            
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                
                # Zero liquidity in middle period
                if 3 <= i <= 6:
                    row[f'bidqty{level}'] = 0.0
                    row[f'askqty{level}'] = 0.0
                else:
                    row[f'bidqty{level}'] = 100.0
                    row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "inventory_stress":
        # Test inventory limit stress
        for i in range(20):
            row = {'datetime': i}
            # Continuously favor one side to stress inventory
            if i < 10:
                base_bid, base_ask = 1801.00, 1801.10  # High - execute sells
            else:
                base_bid, base_ask = 1799.00, 1799.10  # Low - execute buys
                
            for level in range(1, 11):
                row[f'bid{level}'] = base_bid - (level-1) * 0.01
                row[f'bidqty{level}'] = 100.0
                row[f'ask{level}'] = base_ask + (level-1) * 0.01
                row[f'askqty{level}'] = 100.0
            data.append(row)
            
    elif scenario == "latency_test":
        # Test various latency scenarios  
        for i in range(15):
            row = {'datetime': i}
            # Market moves every step to test latency effects
            base_bid = 1800.00 + (i % 3 - 1) * 0.20  # Oscillating market
            base_ask = base_bid + 0.10
            
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

def test_edge_case(scenario_name, data_file, test_configs):
    """Test a specific edge case scenario."""
    print(f"\n{'='*20} TESTING {scenario_name.upper()} {'='*20}")
    
    bugs_found = []
    results = {}
    
    for config_name, fee_structure, post_only in test_configs:
        print(f"\n--- {config_name} ---")
        
        try:
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = data_file
            config["episode_length"] = 12
            config["latency_steps_long"] = 1 if scenario_name == "latency_test" else 0
            config["latency_steps_short"] = 1 if scenario_name == "latency_test" else 0
            config["max_order_volume"] = 2.0
            config["max_inventory"] = 5.0 if scenario_name == "inventory_stress" else 50.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Initial: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Test different order patterns
            execution_count = 0
            error_count = 0
            anomalies = []
            
            # Pattern 1: Aggressive orders (if allowed)
            if not post_only:
                try:
                    action = [-1.0, 1.0, 1.0, 1.0, -1.0, 0.0]  # Very aggressive
                    obs, reward, term, trunc, info = env.step(action)
                    print(f"  Aggressive orders: ✅")
                except Exception as e:
                    error_count += 1
                    print(f"  Aggressive orders: ❌ {e}")
                    bugs_found.append(f"{config_name} ({scenario_name}): Aggressive order error - {e}")
            
            # Pattern 2: Passive orders
            try:
                action = [0.5, -0.5, 0.8, 0.8, -1.0, 0.0]  # Passive
                obs, reward, term, trunc, info = env.step(action)
                print(f"  Passive orders: ✅")
            except Exception as e:
                error_count += 1
                print(f"  Passive orders: ❌ {e}")
                bugs_found.append(f"{config_name} ({scenario_name}): Passive order error - {e}")
            
            # Pattern 3: Activate and execute
            step_count = 2
            while step_count < 10 and not (term or trunc):
                step_count += 1
                
                pre_cash = env.cash
                pre_inventory = env.long_position - env.short_position
                pre_maker = info.get('maker_volume', 0)
                
                try:
                    obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
                    
                    post_cash = env.cash
                    post_inventory = env.long_position - env.short_position
                    post_maker = info.get('maker_volume', 0)
                    
                    # Check for anomalies
                    if post_cash < -1000:  # Very negative cash
                        anomalies.append(f"Step {step_count}: Extreme negative cash ${post_cash:.2f}")
                    
                    if abs(post_inventory) > config["max_inventory"] + 0.1:
                        anomalies.append(f"Step {step_count}: Inventory violation {post_inventory:.2f} > {config['max_inventory']}")
                    
                    if math.isnan(post_cash) or math.isinf(post_cash):
                        anomalies.append(f"Step {step_count}: Invalid cash value {post_cash}")
                    
                    if post_maker > pre_maker:
                        execution_count += 1
                        
                except Exception as e:
                    error_count += 1
                    print(f"  Step {step_count}: ❌ {e}")
                    bugs_found.append(f"{config_name} ({scenario_name}): Step {step_count} error - {e}")
                    break
            
            # Final validation
            final_cash = env.cash
            final_inventory = env.long_position - env.short_position
            final_maker = info.get('maker_volume', 0)
            final_taker = info.get('taker_volume', 0)
            
            print(f"  Final state: Cash=${final_cash:.2f}, Inv={final_inventory:.2f}, M={final_maker:.2f}, T={final_taker:.2f}")
            print(f"  Executions: {execution_count}, Errors: {error_count}, Anomalies: {len(anomalies)}")
            
            if anomalies:
                print(f"  ⚠️  Anomalies found:")
                for anomaly in anomalies:
                    print(f"    • {anomaly}")
                    bugs_found.append(f"{config_name} ({scenario_name}): {anomaly}")
            
            # Scenario-specific checks
            if scenario_name == "zero_spread" and final_maker == 0 and final_taker == 0:
                bugs_found.append(f"{config_name} ({scenario_name}): No execution despite zero spread")
            
            if scenario_name == "inventory_stress" and abs(final_inventory) < 3.0:
                print(f"  ⚠️  Expected inventory stress but final inventory only {final_inventory:.2f}")
            
            if post_only and final_taker > 0:
                # This is expected behavior now (market moving into orders)
                print(f"  ℹ️  Post-only generated {final_taker:.2f} taker volume (market moved into orders)")
            
            results[config_name] = {
                'final_cash': final_cash,
                'final_inventory': final_inventory,
                'maker_volume': final_maker,
                'taker_volume': final_taker,
                'executions': execution_count,
                'errors': error_count,
                'anomalies': len(anomalies)
            }
            
            env.close()
            
        except Exception as e:
            if 'error_count' not in locals():
                error_count = 0
            error_count += 1
            print(f"  FATAL ERROR: {e}")
            bugs_found.append(f"{config_name} ({scenario_name}): Fatal error - {e}")
    
    return bugs_found, results

def run_advanced_tests():
    """Run comprehensive advanced tests."""
    print("🧪 ADVANCED EDGE CASE TESTING")
    print("=" * 80)
    
    # Test scenarios
    scenarios = [
        "zero_spread",
        "crossed_market", 
        "extreme_volatility",
        "zero_liquidity",
        "inventory_stress",
        "latency_test"
    ]
    
    # Test configurations
    test_configs = [
        ("Baseline", "baseline", False),
        ("6bps Rebate", "rebate_6bps", False),
        ("6bps Post-Only", "rebate_6bps", True),
    ]
    
    all_bugs = []
    all_results = {}
    
    try:
        for scenario in scenarios:
            print(f"\n🔬 CREATING {scenario.upper()} TEST DATA")
            data_file = create_edge_case_data(scenario)
            
            try:
                bugs, results = test_edge_case(scenario, data_file, test_configs)
                all_bugs.extend(bugs)
                all_results[scenario] = results
                
            finally:
                if os.path.exists(data_file):
                    os.unlink(data_file)
        
        # COMPREHENSIVE ANALYSIS
        print(f"\n" + "=" * 80)
        print(f"🔍 COMPREHENSIVE EDGE CASE ANALYSIS")
        print(f"=" * 80)
        
        # Bug summary
        if all_bugs:
            print(f"\n❌ BUGS FOUND ({len(all_bugs)}):")
            for i, bug in enumerate(all_bugs, 1):
                print(f"  {i}. {bug}")
        else:
            print(f"\n✅ NO BUGS FOUND IN EDGE CASES!")
        
        # Scenario analysis
        print(f"\n📊 SCENARIO PERFORMANCE SUMMARY:")
        print(f"{'Scenario':<18} {'Config':<15} {'Cash Δ':<10} {'Maker':<8} {'Taker':<8} {'Errors':<7}")
        print(f"{'-'*18} {'-'*15} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")
        
        for scenario, configs in all_results.items():
            for config_name, result in configs.items():
                cash_change = result['final_cash'] - 100000
                print(f"{scenario:<18} {config_name:<15} ${cash_change:>8.2f} {result['maker_volume']:>7.2f} {result['taker_volume']:>7.2f} {result['errors']:>6d}")
        
        # Cross-scenario bug detection
        print(f"\n🐛 CROSS-SCENARIO BUG ANALYSIS:")
        
        # Check for configuration inconsistencies across scenarios
        baseline_results = {}
        rebate_results = {}
        postonly_results = {}
        
        for scenario, configs in all_results.items():
            if 'Baseline' in configs:
                baseline_results[scenario] = configs['Baseline']['maker_volume']
            if '6bps Rebate' in configs:
                rebate_results[scenario] = configs['6bps Rebate']['maker_volume']
            if '6bps Post-Only' in configs:
                postonly_results[scenario] = configs['6bps Post-Only']['maker_volume']
        
        # Check for scenarios where configs behave very differently
        for scenario in scenarios:
            if scenario in baseline_results and scenario in rebate_results:
                baseline_vol = baseline_results[scenario]
                rebate_vol = rebate_results[scenario]
                
                if baseline_vol > 0 and rebate_vol > 0:
                    vol_diff = abs(baseline_vol - rebate_vol) / max(baseline_vol, rebate_vol)
                    if vol_diff > 0.5:  # 50% difference
                        all_bugs.append(f"Large volume difference in {scenario}: Baseline {baseline_vol:.2f} vs Rebate {rebate_vol:.2f}")
        
        # Final summary
        print(f"\n🎯 FINAL EDGE CASE TEST RESULTS:")
        if all_bugs:
            print(f"❌ Found {len(all_bugs)} potential bugs/issues")
            print(f"🔧 Recommend investigating scenarios with errors or anomalies")
        else:
            print(f"✅ All edge cases handled correctly!")
            print(f"🏆 Environment shows robust behavior under stress conditions")
        
        # Stress test summary
        total_executions = sum(sum(configs[config]['executions'] for config in configs.values()) 
                              for configs in all_results.values())
        total_errors = sum(sum(configs[config]['errors'] for config in configs.values()) 
                          for configs in all_results.values())
        
        print(f"\n📈 STRESS TEST METRICS:")
        print(f"  Total executions across all tests: {total_executions}")
        print(f"  Total errors encountered: {total_errors}")
        print(f"  Error rate: {total_errors / max(total_executions, 1) * 100:.2f}%")
        
    except Exception as e:
        print(f"FATAL TEST ERROR: {e}")
        all_bugs.append(f"Test framework error: {e}")

if __name__ == "__main__":
    run_advanced_tests()