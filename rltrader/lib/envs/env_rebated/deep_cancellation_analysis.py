#!/usr/bin/env python3
"""
Deep analysis of why orders aren't being placed and cancellation functionality
"""

import sys
import os
import numpy as np
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')

print("🔍 DEEP CANCELLATION & ORDER PLACEMENT ANALYSIS")
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
        'latency_steps_long': 0,  # No latency for immediate testing
        'latency_steps_short': 0,
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
        'explicit_cancel_threshold': 0.3,  # Lower threshold
        'explicit_cancel_penalty': 0.0001,
        'explicit_cancel_clears_pending': True,
        'do_nothing_threshold': 0.9,  # Higher threshold
        'obs_qty_norm_scale': 1.0,
        'obs_price_norm_scale': 100.0,
        'rebate_rate_long': 0.0004,
        'rebate_rate_short': 0.0004,
    }

def analyze_order_placement_failure():
    """Analyze why orders aren't being placed."""
    print("\n1. 📋 ANALYZING ORDER PLACEMENT FAILURE")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment initialized successfully")
        print(f"Current step: {env.current_step}")
        print(f"Best bid: {env.best_bid:.5f}, Best ask: {env.best_ask:.5f}")
        print(f"Midprice: {(env.best_bid + env.best_ask) / 2:.5f}")
        print(f"Spread: {env.best_ask - env.best_bid:.5f}")
        
        # Test different order placement strategies
        test_actions = [
            ("Conservative", np.array([0.1, -0.1, 0.3, 0.3, -1.0, -1.0])),
            ("Moderate", np.array([0.3, -0.3, 0.5, 0.5, -1.0, -1.0])),
            ("Aggressive", np.array([0.7, -0.7, 0.8, 0.8, -1.0, -1.0])),
            ("Maximum", np.array([0.9, -0.9, 0.9, 0.9, -1.0, -1.0])),
        ]
        
        for test_name, action in test_actions:
            print(f"\n🎯 Testing {test_name} order placement: {action}")
            
            # Reset environment for clean test
            obs, info = env.reset()
            
            # Record pre-action state
            pre_active = len(env.active_orders)
            pre_pending = len(env.pending_orders)
            
            print(f"Before action - Active: {pre_active}, Pending: {pre_pending}")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record post-action state
            post_active = len(env.active_orders)
            post_pending = len(env.pending_orders)
            
            print(f"After action - Active: {post_active}, Pending: {post_pending}")
            print(f"Total orders created: {post_active + post_pending}")
            print(f"Reward: {reward:.6f}")
            
            if post_active + post_pending > 0:
                print(f"✅ SUCCESS: Orders were placed!")
                
                # Examine the orders
                all_orders = env.active_orders + env.pending_orders
                for i, order in enumerate(all_orders):
                    print(f"  Order {i}: Price={order['price']:.5f}, Volume={order['volume']}, Buy={order['is_buy']}")
                
                return True
            else:
                print(f"❌ FAILED: No orders placed")
        
        print(f"\n❌ ALL ORDER PLACEMENT ATTEMPTS FAILED")
        
        # Debug order placement logic
        print(f"\n🔍 DEBUGGING ORDER PLACEMENT LOGIC:")
        print(f"Configuration parameters:")
        print(f"  price_offset_ticks: {config['price_offset_ticks']}")
        print(f"  max_order_volume: {config['max_order_volume']}")
        print(f"  tick_size: {config['tick_size']}")
        print(f"  allowed_aggressiveness_ticks: {config['allowed_aggressiveness_ticks']}")
        print(f"  max_active_orders: {config['max_active_orders']}")
        print(f"  latency_steps: {config['latency_steps_long']}/{config['latency_steps_short']}")
        
        return False
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_order_placement():
    """Manually test order placement by examining the internal logic."""
    print("\n2. 🛠️ MANUAL ORDER PLACEMENT TEST")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Manual order placement analysis...")
        print(f"BBO: {env.best_bid:.5f} / {env.best_ask:.5f}")
        
        # Calculate order prices manually
        action = np.array([0.5, -0.5, 0.7, 0.7, -1.0, -1.0])
        
        buy_offset_signal = action[0]  # 0.5
        sell_offset_signal = action[1]  # -0.5
        buy_size_signal = action[2]  # 0.7
        sell_size_signal = action[3]  # 0.7
        
        print(f"Action signals: buy_offset={buy_offset_signal}, sell_offset={sell_offset_signal}")
        print(f"Size signals: buy_size={buy_size_signal}, sell_size={sell_size_signal}")
        
        # Calculate offsets (based on env logic)
        max_offset_ticks = config['price_offset_ticks']  # 20
        buy_offset_ticks = buy_offset_signal * max_offset_ticks  # 0.5 * 20 = 10
        sell_offset_ticks = sell_offset_signal * max_offset_ticks  # -0.5 * 20 = -10
        
        print(f"Calculated offsets: buy={buy_offset_ticks} ticks, sell={sell_offset_ticks} ticks")
        
        # Calculate target prices
        tick_size = config['tick_size']
        buy_price = env.best_bid + (buy_offset_ticks * tick_size)
        sell_price = env.best_ask + (sell_offset_ticks * tick_size)
        
        print(f"Target prices: buy={buy_price:.5f}, sell={sell_price:.5f}")
        
        # Check aggressiveness
        max_aggressive_ticks = config['allowed_aggressiveness_ticks']
        buy_aggressive_ticks = (buy_price - env.best_ask) / tick_size
        sell_aggressive_ticks = (env.best_bid - sell_price) / tick_size
        
        print(f"Aggressiveness check:")
        print(f"  Buy aggressive ticks: {buy_aggressive_ticks:.2f} (max: {max_aggressive_ticks})")
        print(f"  Sell aggressive ticks: {sell_aggressive_ticks:.2f} (max: {max_aggressive_ticks})")
        
        buy_too_aggressive = buy_aggressive_ticks > max_aggressive_ticks
        sell_too_aggressive = sell_aggressive_ticks > max_aggressive_ticks
        
        print(f"  Buy too aggressive: {buy_too_aggressive}")
        print(f"  Sell too aggressive: {sell_too_aggressive}")
        
        # Calculate volumes
        max_volume = config['max_order_volume']
        buy_volume = max(0, buy_size_signal) * max_volume  # 0.7 * 1000 = 700
        sell_volume = max(0, sell_size_signal) * max_volume  # 0.7 * 1000 = 700
        
        print(f"Calculated volumes: buy={buy_volume}, sell={sell_volume}")
        
        # Check if orders should be placed
        should_place_buy = buy_volume > 0 and not buy_too_aggressive
        should_place_sell = sell_volume > 0 and not sell_too_aggressive
        
        print(f"Should place orders: buy={should_place_buy}, sell={should_place_sell}")
        
        if should_place_buy or should_place_sell:
            print(f"✅ Orders should be placed according to logic")
        else:
            print(f"❌ No orders should be placed according to logic")
            print(f"Possible issues:")
            if buy_volume <= 0:
                print(f"  - Buy volume too small: {buy_volume}")
            if sell_volume <= 0:
                print(f"  - Sell volume too small: {sell_volume}")
            if buy_too_aggressive:
                print(f"  - Buy order too aggressive")
            if sell_too_aggressive:
                print(f"  - Sell order too aggressive")
        
        # Now execute the actual action and compare
        print(f"\n🚀 Executing actual action...")
        obs, reward, terminated, truncated, info = env.step(action)
        
        actual_active = len(env.active_orders)
        actual_pending = len(env.pending_orders)
        
        print(f"Actual result: Active={actual_active}, Pending={actual_pending}")
        
        if (should_place_buy or should_place_sell) and (actual_active + actual_pending == 0):
            print(f"🚨 DISCREPANCY: Expected orders but none were placed!")
            return False
        elif not (should_place_buy or should_place_sell) and (actual_active + actual_pending > 0):
            print(f"🚨 DISCREPANCY: Did not expect orders but some were placed!")
            return False
        else:
            print(f"✅ LOGIC CONSISTENT: Prediction matches actual result")
            return True
            
    except Exception as e:
        print(f"❌ Manual test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cancellation_with_forced_orders():
    """Force order placement and then test cancellation."""
    print("\n3. 🚫 FORCED ORDER PLACEMENT + CANCELLATION TEST")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        config['allowed_aggressiveness_ticks'] = 50  # Very permissive
        config['latency_steps_long'] = 2  # Add latency to keep orders pending
        config['latency_steps_short'] = 2
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Using permissive configuration for order placement...")
        print(f"allowed_aggressiveness_ticks: {config['allowed_aggressiveness_ticks']}")
        print(f"latency_steps: {config['latency_steps_long']}")
        
        # Force aggressive order placement
        force_action = np.array([0.9, -0.9, 0.9, 0.9, -1.0, -1.0])
        print(f"Forcing order placement with action: {force_action}")
        
        obs, reward, terminated, truncated, info = env.step(force_action)
        
        orders_after_placement = len(env.active_orders) + len(env.pending_orders)
        print(f"Orders after forced placement: Active={len(env.active_orders)}, Pending={len(env.pending_orders)}")
        
        if orders_after_placement > 0:
            print(f"✅ SUCCESS: Forced order placement worked!")
            
            # Now test cancellation
            cancel_action = np.array([0.0, 0.0, 0.0, 0.0, 0.8, -1.0])  # High cancel signal
            print(f"Testing cancellation with action: {cancel_action}")
            print(f"Cancel threshold: {config['explicit_cancel_threshold']}")
            
            obs, reward, terminated, truncated, info = env.step(cancel_action)
            
            orders_after_cancel = len(env.active_orders) + len(env.pending_orders)
            print(f"Orders after cancellation: Active={len(env.active_orders)}, Pending={len(env.pending_orders)}")
            
            orders_cancelled = orders_after_placement - orders_after_cancel
            print(f"Orders cancelled: {orders_cancelled}")
            
            if orders_cancelled > 0:
                print(f"✅ CANCELLATION WORKING: {orders_cancelled} orders cancelled")
                return True
            else:
                print(f"❌ CANCELLATION NOT WORKING: No orders cancelled")
                return False
        else:
            print(f"❌ FAILED: Could not force order placement even with permissive settings")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_feature_normalization_details():
    """Analyze feature normalization in detail."""
    print("\n4. 📊 DETAILED FEATURE NORMALIZATION ANALYSIS")
    print("-" * 60)
    
    try:
        from env_rebated_unified import RebatedHFTEnv
        
        config = get_complete_config()
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Analyzing feature normalization with obs_price_norm_scale={config['obs_price_norm_scale']}")
        
        # Get raw observation components
        raw_obs = env._get_raw_observation()
        
        print(f"Raw observation breakdown:")
        print(f"  Total features: {len(raw_obs)}")
        
        # Categorize features based on expected structure
        # Based on the env code, typical structure is:
        # - Portfolio features (positions, cash, etc.)
        # - Order book features (prices, quantities)
        # - Market features (spread, etc.)
        
        feature_categories = {}
        
        # Portfolio features (typically first ~10)
        portfolio_start = 0
        portfolio_end = min(10, len(raw_obs))
        portfolio_features = raw_obs[portfolio_start:portfolio_end]
        feature_categories['Portfolio'] = portfolio_features
        
        # Order book features (middle section)
        book_start = portfolio_end
        book_end = min(book_start + 20, len(raw_obs))
        book_features = raw_obs[book_start:book_end]
        feature_categories['OrderBook'] = book_features
        
        # Market features (remaining)
        market_start = book_end
        market_features = raw_obs[market_start:]
        feature_categories['Market'] = market_features
        
        for category, features in feature_categories.items():
            if len(features) > 0:
                non_zero_features = [f for f in features if not math.isnan(f) and f != 0]
                if non_zero_features:
                    magnitudes = [abs(f) for f in non_zero_features]
                    print(f"\n{category} features ({len(features)} total, {len(non_zero_features)} non-zero):")
                    print(f"  Min magnitude: {min(magnitudes):.8f}")
                    print(f"  Max magnitude: {max(magnitudes):.8f}")
                    print(f"  Mean magnitude: {np.mean(magnitudes):.8f}")
                    print(f"  Range ratio: {max(magnitudes)/min(magnitudes):.2f}x")
                    
                    # Show some example values
                    print(f"  Sample values: {non_zero_features[:5]}")
        
        # Check overall consistency
        all_non_zero = [f for f in raw_obs if not math.isnan(f) and f != 0]
        if all_non_zero:
            all_magnitudes = [abs(f) for f in all_non_zero]
            overall_range = max(all_magnitudes) / min(all_magnitudes)
            
            print(f"\n📈 OVERALL NORMALIZATION ASSESSMENT:")
            print(f"  Total non-zero features: {len(all_non_zero)}")
            print(f"  Overall magnitude range: {overall_range:.2f}x")
            
            if overall_range < 1000:  # Reasonable for neural networks
                print(f"✅ Feature scaling appears reasonable for neural networks")
                return True
            else:
                print(f"⚠️  Feature scaling range may be too wide for optimal neural network training")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_deep_analysis():
    """Run all deep analysis tests."""
    print("\n" + "=" * 80)
    print("🎯 DEEP ANALYSIS RESULTS")
    print("=" * 80)
    
    results = {}
    
    results['order_placement'] = analyze_order_placement_failure()
    results['manual_placement'] = test_manual_order_placement()
    results['forced_cancellation'] = test_cancellation_with_forced_orders()
    results['feature_normalization'] = analyze_feature_normalization_details()
    
    print(f"\n📊 DEEP ANALYSIS SUMMARY:")
    print("-" * 40)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ NEEDS REVIEW"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n🎉 ALL FUNCTIONALITY WORKING CORRECTLY!")
    else:
        print(f"\n🔧 SOME AREAS NEED ATTENTION")
    
    return results

if __name__ == "__main__":
    results = run_deep_analysis()
    
    print(f"\n🚀 RECOMMENDATIONS:")
    if not results.get('order_placement', False):
        print(f"• Check order placement logic and parameter settings")
    if not results.get('forced_cancellation', False):  
        print(f"• Verify cancellation threshold and logic")
    if not results.get('feature_normalization', False):
        print(f"• Consider additional feature scaling adjustments")
    
    print(f"\n" + "=" * 80)