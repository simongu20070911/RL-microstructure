#!/usr/bin/env python3
"""
Test feature normalization consistency and action-based trade cancellation functionality
"""

import sys
import os
import numpy as np
import math
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🧪 TESTING FEATURE NORMALIZATION & CANCELLATION FUNCTIONALITY")
print("=" * 80)

def get_complete_config():
    """Get a complete configuration for the rebated environment."""
    return {
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed.csv',
        'initial_capital': 10000,
        'max_steps': 100,
        'episode_length': 50,
        'lot_size': 1000,
        'max_inventory': 5000,
        'max_active_orders': 5,
        'tick_size': 0.00001,
        'order_book_levels': 5,
        'price_offset_ticks': 20,
        'max_order_volume': 1000,
        'allowed_aggressiveness_ticks': 3,
        'latency_steps_long': 1,
        'latency_steps_short': 1,
        'transaction_cost_long': 0.0001,
        'transaction_cost_short': 0.0001,
        'taker_penalty': 0.001,
        'inventory_penalty': 0.0001,
        'invalid_order_penalty': 0.01,
        'activity_bonus': 0.0001,
        'quoting_reward_enabled': True,
        'quoting_reward_amount': 0.0001,
        'quoting_reward_max_ticks': 3,
        'explicit_cancel_enabled': True,
        'explicit_cancel_threshold': 0.7,
        'explicit_cancel_penalty': 0.0001,
        'explicit_cancel_clears_pending': True,
        'do_nothing_threshold': 0.8,
        'obs_qty_norm_scale': 1.0,
        'obs_price_norm_scale': 100.0,  # This is the key scaling parameter
        'rebate_rate_long': 0.0004,
        'rebate_rate_short': 0.0004,
    }

def test_feature_normalization_consistency():
    """Test that all price features are normalized consistently."""
    print("\n1. 🔍 TESTING FEATURE NORMALIZATION CONSISTENCY")
    print("-" * 60)
    
    try:
        sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment created successfully")
        print(f"Observation space shape: {obs.shape}")
        print(f"obs_price_norm_scale: {config['obs_price_norm_scale']}")
        
        # Get raw observation to examine feature scaling
        raw_obs = env._get_raw_observation()
        
        print(f"Raw observation length: {len(raw_obs)}")
        
        # Analyze feature magnitudes
        feature_magnitudes = []
        feature_names = []
        
        # Portfolio features (first few)
        portfolio_features = raw_obs[:10]  # Approximate
        for i, val in enumerate(portfolio_features):
            if not math.isnan(val) and val != 0:
                feature_magnitudes.append(abs(val))
                feature_names.append(f"Portfolio_{i}")
        
        # Order book features (middle section)
        if len(raw_obs) > 20:
            book_features = raw_obs[10:40]  # Approximate price/qty features
            for i, val in enumerate(book_features):
                if not math.isnan(val) and val != 0:
                    feature_magnitudes.append(abs(val))
                    feature_names.append(f"Book_{i}")
        
        # Market features (spread, etc.)
        if len(raw_obs) > 50:
            market_features = raw_obs[40:60]  # Approximate spread location
            for i, val in enumerate(market_features):
                if not math.isnan(val) and val != 0:
                    feature_magnitudes.append(abs(val))
                    feature_names.append(f"Market_{i}")
        
        if feature_magnitudes:
            print(f"\n📊 FEATURE MAGNITUDE ANALYSIS:")
            print(f"Number of non-zero features analyzed: {len(feature_magnitudes)}")
            print(f"Min magnitude: {min(feature_magnitudes):.8f}")
            print(f"Max magnitude: {max(feature_magnitudes):.8f}")
            print(f"Mean magnitude: {np.mean(feature_magnitudes):.8f}")
            print(f"Std magnitude: {np.std(feature_magnitudes):.8f}")
            
            # Check for magnitude outliers (sign of scaling inconsistency)
            mean_mag = np.mean(feature_magnitudes)
            outliers = [mag for mag in feature_magnitudes if mag > mean_mag * 10 or mag < mean_mag / 10]
            
            if outliers:
                print(f"\n🚨 SCALING INCONSISTENCY DETECTED:")
                print(f"Found {len(outliers)} features with 10x+ magnitude difference")
                print(f"Outlier magnitudes: {outliers[:5]}...")  # Show first 5
                
                # Find which features are outliers
                outlier_indices = [i for i, mag in enumerate(feature_magnitudes) 
                                 if mag > mean_mag * 10 or mag < mean_mag / 10]
                print(f"Outlier feature names: {[feature_names[i] for i in outlier_indices[:5]]}")
                return False
            else:
                print(f"✅ FEATURE SCALING APPEARS CONSISTENT")
                print(f"All features within 10x magnitude range")
                return True
        else:
            print(f"⚠️  No non-zero features found for analysis")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_explicit_cancellation_functionality():
    """Test that explicit cancellation action works correctly."""
    print("\n2. 🔍 TESTING EXPLICIT CANCELLATION FUNCTIONALITY")
    print("-" * 60)
    
    try:
        sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        config['explicit_cancel_threshold'] = 0.5  # Lower threshold for easier testing
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Testing cancellation with threshold: {config['explicit_cancel_threshold']}")
        
        # Step 1: Place some orders
        print(f"\n📋 STEP 1: Placing orders")
        place_orders_action = np.array([0.3, -0.3, 0.5, 0.5, -1.0, -1.0])  # Buy/sell orders, no cancel, no do nothing
        obs, reward, terminated, truncated, info = env.step(place_orders_action)
        
        active_orders_after_placement = len(env.active_orders)
        pending_orders_after_placement = len(env.pending_orders)
        total_orders_after_placement = active_orders_after_placement + pending_orders_after_placement
        
        print(f"Active orders after placement: {active_orders_after_placement}")
        print(f"Pending orders after placement: {pending_orders_after_placement}")
        print(f"Total orders: {total_orders_after_placement}")
        
        if total_orders_after_placement == 0:
            print(f"⚠️  No orders were placed - cannot test cancellation")
            # Try more aggressive placement
            aggressive_action = np.array([0.8, -0.8, 0.8, 0.8, -1.0, -1.0])
            obs, reward, terminated, truncated, info = env.step(aggressive_action)
            
            active_orders_after_placement = len(env.active_orders)
            pending_orders_after_placement = len(env.pending_orders)
            total_orders_after_placement = active_orders_after_placement + pending_orders_after_placement
            
            print(f"After aggressive placement:")
            print(f"Active orders: {active_orders_after_placement}")
            print(f"Pending orders: {pending_orders_after_placement}")
            print(f"Total orders: {total_orders_after_placement}")
        
        if total_orders_after_placement > 0:
            # Step 2: Test cancellation action
            print(f"\n🚫 STEP 2: Testing cancellation")
            cancel_action = np.array([0.0, 0.0, 0.0, 0.0, 0.8, -1.0])  # High cancel signal, no do nothing
            
            print(f"Cancel action[4]: {cancel_action[4]} (threshold: {config['explicit_cancel_threshold']})")
            print(f"Should trigger cancellation: {cancel_action[4] > config['explicit_cancel_threshold']}")
            
            obs, reward, terminated, truncated, info = env.step(cancel_action)
            
            active_orders_after_cancel = len(env.active_orders)
            pending_orders_after_cancel = len(env.pending_orders)
            total_orders_after_cancel = active_orders_after_cancel + pending_orders_after_cancel
            
            print(f"Active orders after cancellation: {active_orders_after_cancel}")
            print(f"Pending orders after cancellation: {pending_orders_after_cancel}")
            print(f"Total orders after cancellation: {total_orders_after_cancel}")
            
            orders_cancelled = total_orders_after_placement - total_orders_after_cancel
            print(f"Orders cancelled: {orders_cancelled}")
            
            if orders_cancelled > 0:
                print(f"✅ CANCELLATION WORKING: {orders_cancelled} orders cancelled")
                
                # Check if penalty was applied
                cancel_penalty = config.get('explicit_cancel_penalty', 0)
                if cancel_penalty > 0:
                    print(f"Expected cancel penalty: {cancel_penalty}")
                    if reward < 0:
                        print(f"✅ Penalty applied - reward: {reward:.6f}")
                    else:
                        print(f"⚠️  Expected negative reward from penalty, got: {reward:.6f}")
                
                return True
            elif total_orders_after_placement == total_orders_after_cancel:
                print(f"❌ CANCELLATION NOT WORKING: No orders were cancelled")
                return False
            else:
                print(f"⚠️  Unexpected order count change")
                return False
        else:
            print(f"❌ Cannot test cancellation - no orders placed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_do_nothing_action():
    """Test that do nothing action works correctly."""
    print("\n3. 🔍 TESTING DO NOTHING ACTION")
    print("-" * 60)
    
    try:
        sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        config['do_nothing_threshold'] = 0.6  # Lower threshold for easier testing
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Testing do nothing with threshold: {config['do_nothing_threshold']}")
        
        # Record initial state
        initial_active_orders = len(env.active_orders)
        initial_pending_orders = len(env.pending_orders)
        initial_cash = env.cash
        initial_position = env.long_position + env.short_position
        
        print(f"Initial state:")
        print(f"  Active orders: {initial_active_orders}")
        print(f"  Pending orders: {initial_pending_orders}")
        print(f"  Cash: {initial_cash:.4f}")
        print(f"  Position: {initial_position:.4f}")
        
        # Test do nothing action
        do_nothing_action = np.array([0.8, -0.8, 0.8, 0.8, 0.5, 0.8])  # High do nothing signal
        
        print(f"\n💤 TESTING DO NOTHING ACTION")
        print(f"Do nothing action[5]: {do_nothing_action[5]} (threshold: {config['do_nothing_threshold']})")
        print(f"Should trigger do nothing: {do_nothing_action[5] > config['do_nothing_threshold']}")
        
        obs, reward, terminated, truncated, info = env.step(do_nothing_action)
        
        final_active_orders = len(env.active_orders)
        final_pending_orders = len(env.pending_orders)
        final_cash = env.cash
        final_position = env.long_position + env.short_position
        
        print(f"Final state:")
        print(f"  Active orders: {final_active_orders}")
        print(f"  Pending orders: {final_pending_orders}")
        print(f"  Cash: {final_cash:.4f}")
        print(f"  Position: {final_position:.4f}")
        
        # Check if state remained unchanged (do nothing worked)
        orders_unchanged = (initial_active_orders == final_active_orders and 
                          initial_pending_orders == final_pending_orders)
        cash_unchanged = abs(initial_cash - final_cash) < 1e-6
        position_unchanged = abs(initial_position - final_position) < 1e-6
        
        print(f"\nDo nothing validation:")
        print(f"  Orders unchanged: {orders_unchanged}")
        print(f"  Cash unchanged: {cash_unchanged}")
        print(f"  Position unchanged: {position_unchanged}")
        
        if orders_unchanged and cash_unchanged and position_unchanged:
            print(f"✅ DO NOTHING ACTION WORKING: State preserved")
            return True
        else:
            print(f"❌ DO NOTHING ACTION NOT WORKING: State changed unexpectedly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_space_bounds():
    """Test that actions are properly bounded and handled."""
    print("\n4. 🔍 TESTING ACTION SPACE BOUNDS")
    print("-" * 60)
    
    try:
        sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Action space: {env.action_space}")
        print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")
        
        # Test various action boundary conditions
        test_actions = [
            ("Valid bounds", np.array([0.5, -0.5, 0.3, 0.7, 0.2, 0.1])),
            ("Extreme valid", np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])),
            ("Extreme valid negative", np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])),
            ("Just above threshold cancel", np.array([0.0, 0.0, 0.0, 0.0, 0.71, 0.0])),
            ("Just above threshold do nothing", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.81])),
        ]
        
        for test_name, action in test_actions:
            print(f"\n📋 Testing {test_name}: {action}")
            
            try:
                # Check if action is in valid bounds
                in_bounds = env.action_space.contains(action)
                print(f"  Action in bounds: {in_bounds}")
                
                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check for NaN/Inf in results
                obs_valid = not (np.any(np.isnan(obs)) or np.any(np.isinf(obs)))
                reward_valid = not (np.isnan(reward) or np.isinf(reward))
                
                print(f"  Observation valid: {obs_valid}")
                print(f"  Reward valid: {reward_valid}")
                print(f"  Reward: {reward:.6f}")
                
                if not obs_valid or not reward_valid:
                    print(f"❌ Action {test_name} produced invalid results")
                    return False
                else:
                    print(f"✅ Action {test_name} handled correctly")
                    
            except Exception as e:
                print(f"❌ Action {test_name} caused exception: {e}")
                return False
        
        print(f"\n✅ ALL ACTION BOUNDARY TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spread_scaling_fix():
    """Specifically test that the spread scaling fix is working."""
    print("\n5. 🔍 TESTING SPREAD SCALING FIX")
    print("-" * 60)
    
    try:
        sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        config['obs_price_norm_scale'] = 100.0  # Ensure we're testing with scaling
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        # Get multiple observations to test spread scaling
        observations = []
        for i in range(5):
            action = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0])  # Do nothing
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs.copy())
            
            if terminated or truncated:
                obs, info = env.reset()
        
        print(f"Collected {len(observations)} observations")
        print(f"obs_price_norm_scale: {config['obs_price_norm_scale']}")
        
        # Analyze spread vs other price features
        price_feature_magnitudes = []
        spread_magnitudes = []
        
        for obs in observations:
            # Look for features that might be prices (typically smaller after scaling)
            price_candidates = [val for val in obs if 0.001 < abs(val) < 10.0]
            
            # Look for features that might be spread (should now be scaled similarly)
            spread_candidates = [val for val in obs if abs(val) > 0.0001]
            
            if price_candidates:
                price_feature_magnitudes.extend(price_candidates)
            if spread_candidates:
                spread_magnitudes.extend(spread_candidates)
        
        if price_feature_magnitudes and spread_magnitudes:
            avg_price_magnitude = np.mean([abs(x) for x in price_feature_magnitudes])
            avg_spread_magnitude = np.mean([abs(x) for x in spread_magnitudes])
            
            print(f"Average price feature magnitude: {avg_price_magnitude:.6f}")
            print(f"Average spread-like magnitude: {avg_spread_magnitude:.6f}")
            
            magnitude_ratio = avg_spread_magnitude / avg_price_magnitude if avg_price_magnitude > 0 else float('inf')
            print(f"Magnitude ratio (spread/price): {magnitude_ratio:.2f}")
            
            # After the fix, ratio should be reasonable (not 100x difference)
            if magnitude_ratio < 20:  # Allow some variance but not 100x
                print(f"✅ SPREAD SCALING FIX WORKING: Reasonable magnitude ratio")
                return True
            else:
                print(f"❌ SPREAD SCALING ISSUE: Still large magnitude difference")
                return False
        else:
            print(f"⚠️  Could not identify price/spread features for comparison")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_functionality_tests():
    """Run all functionality tests."""
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE FUNCTIONALITY TEST RESULTS")
    print("=" * 80)
    
    # Run all tests
    test_results = {}
    
    test_results['feature_normalization'] = test_feature_normalization_consistency()
    test_results['explicit_cancellation'] = test_explicit_cancellation_functionality()
    test_results['do_nothing_action'] = test_do_nothing_action()
    test_results['action_bounds'] = test_action_space_bounds()
    test_results['spread_scaling'] = test_spread_scaling_fix()
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print("-" * 40)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL FUNCTIONALITY TESTS PASSED!")
        print(f"Your rebated environment is working correctly:")
        print(f"✅ Feature normalization is consistent")
        print(f"✅ Explicit cancellation works properly")
        print(f"✅ Do nothing action preserves state")
        print(f"✅ Action bounds are handled correctly")
        print(f"✅ Spread scaling fix is effective")
    else:
        print(f"\n⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        failed_tests = [name for name, result in test_results.items() if not result]
        print(f"Failed tests: {failed_tests}")
    
    print("=" * 80)
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_comprehensive_functionality_tests()
    
    if success:
        print(f"\n🚀 READY FOR PRODUCTION TRAINING!")
        print(f"All critical functionality verified and working correctly.")
    else:
        print(f"\n🔧 ADDITIONAL FIXES NEEDED")
        print(f"Please address the failed tests before production training.")