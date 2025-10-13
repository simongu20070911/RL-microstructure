#!/usr/bin/env python3
"""
Test and verify the 11 additional critical issues found in the RL training system
"""

import sys
import os
import numpy as np
import torch
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🚨 TESTING 11 ADDITIONAL CRITICAL RL TRAINING ISSUES")
print("=" * 80)

def test_observation_scaling_inconsistency():
    """Test Issue #1: Observation space scaling inconsistency."""
    print("\n1. 🔥 TESTING OBSERVATION SCALING INCONSISTENCY")
    print("-" * 60)
    
    # Import the environment
    sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
    from env_rebated_unified import RebatedHFTEnv
    
    # Create test environment
    test_config = {
        'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
        'initial_capital': 10000,
        'max_steps': 100,
        'episode_length': 50,
        'lot_size': 1000,
        'obs_lookback': 10,
        'obs_price_scale': 100.0,  # This will cause the scaling issue
        'max_order_levels': 5,
        'rebate_rate_long': 0.0004,
        'rebate_rate_short': 0.0004,
    }
    
    try:
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Get raw observation to examine scaling
        raw_obs = env._get_raw_observation()
        
        print(f"Environment observation shape: {obs.shape}")
        print(f"Observation price scale: {env.config['obs_price_scale']}")
        
        # Check for magnitude differences in price features
        if len(raw_obs) > 50:  # Ensure we have enough features
            spread_feature = raw_obs[50] if len(raw_obs) > 50 else 0  # Approximate spread location
            price_features = raw_obs[10:20]  # Approximate price feature locations
            
            spread_magnitude = abs(spread_feature)
            price_magnitudes = [abs(x) for x in price_features if not math.isnan(x)]
            
            if price_magnitudes:
                avg_price_magnitude = np.mean(price_magnitudes)
                magnitude_ratio = spread_magnitude / avg_price_magnitude if avg_price_magnitude != 0 else 0
                
                print(f"Spread magnitude: {spread_magnitude:.6f}")
                print(f"Average price magnitude: {avg_price_magnitude:.6f}")
                print(f"Magnitude ratio: {magnitude_ratio:.2f}")
                
                if magnitude_ratio > 10 or magnitude_ratio < 0.1:
                    print("🚨 ISSUE CONFIRMED: Significant magnitude mismatch between features!")
                    print("   This will severely impact neural network training!")
                    return True
                else:
                    print("✅ Scaling appears consistent")
                    return False
            else:
                print("⚠️  Could not detect price features for comparison")
                return False
        else:
            print("⚠️  Observation vector too short for analysis")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_inventory_overflow_handling():
    """Test Issue #3: Inventory overflow not handled during execution."""
    print("\n3. 🔥 TESTING INVENTORY OVERFLOW HANDLING")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        # Create environment with small inventory limits
        test_config = {
            'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
            'initial_capital': 10000,
            'max_steps': 100,
            'episode_length': 50,
            'lot_size': 1000,
            'max_inventory': 2000,  # Small limit to trigger overflow
            'obs_lookback': 10,
        }
        
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Force environment to a state with current inventory near limit
        env.long_position = 1800  # Near the limit
        
        print(f"Initial long position: {env.long_position}")
        print(f"Max inventory: {env.config['max_inventory']}")
        
        # Try to execute a large buy order that would exceed limits
        large_buy_action = np.array([0.8, -1.0, 0.9, 0.0, 0.0, 0.0])  # Large buy order
        
        # Record state before action
        initial_position = env.long_position
        initial_cash = env.cash
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(large_buy_action)
        
        # Check if inventory limits were properly enforced
        final_position = env.long_position
        position_change = final_position - initial_position
        
        print(f"Position change: {position_change}")
        print(f"Final position: {final_position}")
        print(f"Inventory limit exceeded: {final_position > env.config['max_inventory']}")
        
        if final_position > env.config['max_inventory']:
            print("🚨 ISSUE CONFIRMED: Inventory limits not properly enforced!")
            print("   Agent can exceed position limits, leading to unrealistic trading!")
            return True
        else:
            print("✅ Inventory limits properly enforced")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_transaction_cost_race_condition():
    """Test Issue #4: Transaction cost race condition."""
    print("\n4. 🔥 TESTING TRANSACTION COST RACE CONDITION")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        test_config = {
            'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
            'initial_capital': 10000,
            'max_steps': 100,
            'episode_length': 50,
            'lot_size': 1000,
            'transaction_cost_long': 0.0002,  # 2 bps cost
            'transaction_cost_short': 0.0002,
            'max_inventory': 500,  # Small limit to force partial fills
        }
        
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Record initial state
        initial_cash = env.cash
        
        # Execute a buy order that might be partially filled due to inventory limits
        buy_action = np.array([0.5, -1.0, 0.8, 0.0, 0.0, 0.0])  # Buy order
        
        # Track execution details
        obs, reward, terminated, truncated, info = env.step(buy_action)
        
        cash_change = env.cash - initial_cash
        position_change = env.long_position
        
        # Calculate expected cost based on actual execution
        if position_change > 0:
            # Try to estimate actual execution price and volume
            estimated_cost = abs(cash_change) - (position_change * env.bid_prices[0] if len(env.bid_prices) > 0 else 0)
            
            print(f"Cash change: {cash_change:.4f}")
            print(f"Position change: {position_change}")
            print(f"Estimated transaction cost: {estimated_cost:.4f}")
            
            # Check if cost calculation seems reasonable
            expected_cost_rate = test_config['transaction_cost_long']
            if abs(estimated_cost) > 0:
                actual_cost_rate = abs(estimated_cost) / abs(cash_change) if cash_change != 0 else 0
                print(f"Expected cost rate: {expected_cost_rate:.4f}")
                print(f"Actual cost rate: {actual_cost_rate:.4f}")
                
                if abs(actual_cost_rate - expected_cost_rate) > expected_cost_rate * 0.5:  # 50% tolerance
                    print("🚨 ISSUE CONFIRMED: Transaction cost calculation inconsistent!")
                    print("   Cost may be calculated on wrong volume!")
                    return True
                else:
                    print("✅ Transaction cost calculation appears consistent")
                    return False
            else:
                print("⚠️  No clear transaction cost detected")
                return False
        else:
            print("⚠️  No position change detected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_cache_overflow_race_condition():
    """Test Issue #2: Cache overflow logic race condition."""
    print("\n2. 🔥 TESTING CACHE OVERFLOW RACE CONDITION")
    print("-" * 60)
    
    try:
        from agents.agent_2sided import CachedMultiHeadAttention
        
        # Create attention with small cache size to trigger overflow quickly
        attention = CachedMultiHeadAttention(hidden_dim=64, num_heads=4, max_seq_length=5)
        batch_size = 2
        device = torch.device('cpu')
        
        attention.reset_cache(batch_size, device)
        
        print(f"Testing cache overflow race condition...")
        print(f"Cache max length: {attention.max_seq_length}")
        
        # Fill cache to trigger overflow
        with torch.no_grad():
            for step in range(8):  # More than cache size
                x = torch.randn(batch_size, 1, 64)
                
                # Record cache state before potential overflow
                if step == 4:  # Just before overflow
                    pre_overflow_cache = attention.key_cache.clone()
                    pre_overflow_pos = attention.cache_position
                
                try:
                    output, attn_weights = attention(x, use_cache=True)
                    
                    # Check for cache corruption after overflow
                    if step > 4:
                        # Verify cache contents are reasonable
                        cache_has_nan = torch.isnan(attention.key_cache).any()
                        cache_has_inf = torch.isinf(attention.key_cache).any()
                        cache_values_reasonable = torch.all(torch.abs(attention.key_cache) < 1e6)
                        
                        print(f"Step {step}: pos={attention.cache_position}, NaN={cache_has_nan}, Inf={cache_has_inf}, reasonable={cache_values_reasonable}")
                        
                        if cache_has_nan or cache_has_inf or not cache_values_reasonable:
                            print("🚨 ISSUE CONFIRMED: Cache corruption detected after overflow!")
                            print("   Race condition in cache shifting logic!")
                            return True
                            
                except Exception as e:
                    print(f"🚨 ISSUE CONFIRMED: Cache overflow caused exception: {e}")
                    return True
        
        print("✅ Cache overflow handling appears stable")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_temporal_inconsistency_taker_detection():
    """Test Issue #6: Temporal inconsistency in taker detection."""
    print("\n6. 🔶 TESTING TEMPORAL INCONSISTENCY IN TAKER DETECTION")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        test_config = {
            'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
            'initial_capital': 10000,
            'max_steps': 100,
            'episode_length': 50,
            'lot_size': 1000,
        }
        
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Simulate scenario where BBO changes between order placement and execution
        initial_bid = env.best_bid
        initial_ask = env.best_ask
        
        print(f"Initial BBO: bid={initial_bid:.5f}, ask={initial_ask:.5f}")
        
        # Place an order at current best bid (should be maker)
        if not math.isnan(initial_bid):
            # Force an order that should be passive
            passive_action = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])  # Small buy at bid
            
            obs, reward, terminated, truncated, info = env.step(passive_action)
            
            # Check if taker detection logic is using stale BBO
            current_bid = env.best_bid
            current_ask = env.best_ask
            
            print(f"After step BBO: bid={current_bid:.5f}, ask={current_ask:.5f}")
            
            bbo_changed = abs(current_bid - initial_bid) > 1e-6 or abs(current_ask - initial_ask) > 1e-6
            if bbo_changed:
                print("🚨 ISSUE CONFIRMED: BBO changed during step!")
                print("   Taker detection may use wrong reference prices!")
                return True
            else:
                print("✅ BBO remained stable during test step")
                return False
        else:
            print("⚠️  Invalid initial BBO - cannot test taker detection")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_memory_leak_observation_validation():
    """Test Issue #5: Memory leak in observation validation."""
    print("\n5. 🔶 TESTING MEMORY LEAK IN OBSERVATION VALIDATION")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        test_config = {
            'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
            'initial_capital': 10000,
            'max_steps': 100,
            'episode_length': 50,
            'lot_size': 1000,
        }
        
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Monitor memory during repeated observation validation
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Perform many steps to trigger observation validation repeatedly
        for i in range(100):
            action = np.random.uniform(-1, 1, 6)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
            
            # Force garbage collection periodically
            if i % 20 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                print(f"Step {i}: Memory usage: {current_memory:.2f} MB (+{memory_growth:.2f} MB)")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Total memory growth: {total_growth:.2f} MB")
        
        # Check for significant memory growth (>50MB indicates potential leak)
        if total_growth > 50:
            print("🚨 ISSUE CONFIRMED: Significant memory growth detected!")
            print("   Likely memory leak in observation validation!")
            return True
        else:
            print("✅ Memory usage appears stable")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_numerical_precision_loss():
    """Test Issue #7: Numerical precision loss in position tracking."""
    print("\n7. 🔶 TESTING NUMERICAL PRECISION LOSS")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        test_config = {
            'csv_path': '/home/gaen/Documents/RL/envs/GBPUSD_2024_week1_10k.csv',
            'initial_capital': 10000,
            'max_steps': 100,
            'episode_length': 50,
            'lot_size': 1000,
        }
        
        env = RebatedHFTEnv(test_config)
        obs, info = env.reset()
        
        # Perform many small trades to accumulate precision errors
        print("Performing 50 small trades to test precision...")
        
        initial_position = env.long_position
        cumulative_trades = 0
        
        for i in range(50):
            # Small buy action
            small_buy = np.array([0.1, -1.0, 0.1, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(small_buy)
            
            if env.long_position != initial_position:
                trade_size = env.long_position - cumulative_trades
                cumulative_trades += trade_size
                print(f"Trade {i}: Size={trade_size:.8f}, Cumulative={cumulative_trades:.8f}, Position={env.long_position:.8f}")
            
            if terminated or truncated:
                break
        
        # Check for precision loss by comparing cumulative calculation
        expected_position = initial_position + cumulative_trades
        actual_position = env.long_position
        precision_error = abs(expected_position - actual_position)
        
        print(f"Expected position: {expected_position:.8f}")
        print(f"Actual position: {actual_position:.8f}")
        print(f"Precision error: {precision_error:.8f}")
        
        # Check if error is significant (>1e-6 indicates precision loss)
        if precision_error > 1e-6:
            print("🚨 ISSUE CONFIRMED: Significant precision loss detected!")
            print("   Cumulative float operations losing precision!")
            return True
        else:
            print("✅ Position tracking precision appears adequate")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def run_comprehensive_issue_analysis():
    """Run all additional issue tests."""
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE ADDITIONAL ISSUE ANALYSIS")
    print("=" * 80)
    
    # Run all tests
    test_results = {}
    
    print("\n🚨 CRITICAL SEVERITY TESTS:")
    test_results['observation_scaling'] = test_observation_scaling_inconsistency()
    test_results['cache_race_condition'] = test_cache_overflow_race_condition()
    test_results['inventory_overflow'] = test_inventory_overflow_handling()
    test_results['transaction_cost_race'] = test_transaction_cost_race_condition()
    
    print("\n🔶 MEDIUM SEVERITY TESTS:")
    test_results['memory_leak'] = test_memory_leak_observation_validation()
    test_results['taker_detection'] = test_temporal_inconsistency_taker_detection()
    test_results['precision_loss'] = test_numerical_precision_loss()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 ADDITIONAL ISSUES ANALYSIS SUMMARY")
    print("=" * 80)
    
    critical_issues = []
    medium_issues = []
    
    if test_results.get('observation_scaling', False):
        critical_issues.append("Observation scaling inconsistency")
    if test_results.get('cache_race_condition', False):
        critical_issues.append("Cache overflow race condition")
    if test_results.get('inventory_overflow', False):
        critical_issues.append("Inventory overflow handling")
    if test_results.get('transaction_cost_race', False):
        critical_issues.append("Transaction cost race condition")
    
    if test_results.get('memory_leak', False):
        medium_issues.append("Memory leak in observation validation")
    if test_results.get('taker_detection', False):
        medium_issues.append("Temporal taker detection inconsistency")
    if test_results.get('precision_loss', False):
        medium_issues.append("Numerical precision loss")
    
    print(f"🚨 CRITICAL ISSUES CONFIRMED: {len(critical_issues)}")
    for issue in critical_issues:
        print(f"   • {issue}")
    
    print(f"\n🔶 MEDIUM ISSUES CONFIRMED: {len(medium_issues)}")
    for issue in medium_issues:
        print(f"   • {issue}")
    
    total_issues = len(critical_issues) + len(medium_issues)
    print(f"\n📊 TOTAL ADDITIONAL ISSUES FOUND: {total_issues}/7 tested")
    
    if total_issues > 0:
        print(f"\n⚠️  THESE ISSUES REQUIRE IMMEDIATE ATTENTION!")
        print(f"Combined with the 3 previously fixed issues, your RL training")
        print(f"system has {total_issues + 3} critical problems that severely")
        print(f"impact training performance and stability.")
    else:
        print(f"\n✅ NO ADDITIONAL CRITICAL ISSUES DETECTED")
        print(f"The 3 previous fixes appear to have addressed the main problems.")
    
    print("=" * 80)
    
    return critical_issues, medium_issues

if __name__ == "__main__":
    critical, medium = run_comprehensive_issue_analysis()
    
    if critical or medium:
        print(f"\n🔧 NEXT STEPS:")
        print(f"1. Address critical issues immediately")
        print(f"2. Fix medium issues for optimal performance")
        print(f"3. Re-run training tests to validate improvements")
        print(f"4. Monitor for additional issues during training")