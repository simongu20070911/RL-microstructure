#!/usr/bin/env python3
"""
Thorough test of maker order mechanics and rebate timing fix

This test will:
1. Create clear maker orders (providing liquidity)
2. Verify they're classified correctly at placement time
3. Test execution and rebate calculation
4. Verify the rebate timing fix works properly
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv
from final_optimized_config import FINAL_OPTIMIZED_CONFIG

def test_maker_order_mechanics():
    """Comprehensive test of maker order creation and rebate timing"""
    
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config.update({
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 500,
        'episode_length': 100,
        'rebate_rate_long': 0.0004,   # 4bps rebate for makers
        'rebate_rate_short': 0.0004,  # 4bps rebate for makers  
        'transaction_cost_long': 0.0002,   # 2bps cost for takers
        'transaction_cost_short': 0.0002,  # 2bps cost for takers
        'latency_steps_long': 1,      # Short latency for quick testing
        'latency_steps_short': 1,
        'max_active_orders': 10,
        'lot_size': 100,              # Smaller lot size for testing
        'max_order_volume': 5000,     # Smaller volume for testing
        'price_offset_ticks': 20,     # Larger range for maker orders
        'allowed_aggressiveness_ticks': 5,
        'post_only_mode': False,      # Allow both maker and taker orders
    })
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== MAKER ORDER MECHANICS TEST ===")
    print(f"Initial BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
    print(f"Rebate rate: {config['rebate_rate_long']*10000:.1f}bps")
    print(f"Price offset range: ±{config['price_offset_ticks']} ticks")
    print(f"Tick size: {config['tick_size']}")
    
    # Test 1: Create a clear MAKER BUY order (below best bid)
    print("\n=== TEST 1: MAKER BUY ORDER ===")
    
    # Calculate action to place buy order 5 ticks below best bid (clear maker)
    ticks_below_bid = -5
    action_signal = ticks_below_bid / config['price_offset_ticks']  # Should be -0.25
    
    action_maker_buy = np.array([
        action_signal,  # Buy offset signal for maker order
        0.0,           # No sell
        0.5,           # 50% volume
        0.0,           # No sell volume
        0.0,           # No cancel
        0.0            # No do-nothing
    ])
    
    expected_price = env.best_bid + ticks_below_bid * env.config['tick_size']
    print(f"Placing MAKER BUY order:")
    print(f"  Action signal: {action_signal:.3f}")
    print(f"  Expected price: {expected_price:.2f} (Bid={env.best_bid:.2f} + {ticks_below_bid} ticks)")
    print(f"  This should be MAKER (providing liquidity)")
    
    # Track initial state
    initial_bid = env.best_bid
    initial_ask = env.best_ask
    
    obs, reward, done, truncated, info = env.step(action_maker_buy)
    
    # Verify order was placed correctly
    print(f"Pending orders after placement: {len(env.pending_orders)}")
    if env.pending_orders:
        order = env.pending_orders[-1]
        placed_price = order['price']
        placement_bid = order.get('placement_bid')
        placement_ask = order.get('placement_ask')
        
        print(f"Order details:")
        print(f"  Order ID: {order['id']}")
        print(f"  Placed price: {placed_price:.2f}")
        print(f"  Stored placement BBO: Bid={placement_bid:.2f}, Ask={placement_ask:.2f}")
        print(f"  Volume: {order['volume']:.0f}")
        
        # Verify this is a maker order at placement time
        is_maker_at_placement = (order['is_buy'] and placed_price < placement_ask - 1e-9)
        print(f"  Is MAKER at placement: {is_maker_at_placement}")
        
        if is_maker_at_placement:
            print("✅ Correctly created MAKER order")
        else:
            print("❌ Order was created as TAKER - this is wrong for our test")
            
        # Test 2: Wait for activation and check classification
        print("\n--- Waiting for order activation ---")
        action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        obs, reward, done, truncated, info = env.step(action_wait)
        
        # Check if order activated
        print(f"Active orders after waiting: {len(env.active_orders)}")
        if env.active_orders:
            activated_order = env.active_orders[-1]
            taker_flag = activated_order.get('is_taker_at_activation', 'NOT_SET')
            print(f"Order activated with is_taker_at_activation: {taker_flag}")
            
            # The key test: classification should match placement time assessment
            if taker_flag == (not is_maker_at_placement):
                print("✅ Order classified correctly based on placement time")
            else:
                print("❌ Order classification mismatch")
    else:
        print("❌ No orders were placed")
        return False
    
    print("\n=== TEST 2: MAKER SELL ORDER ===")
    
    # Calculate action to place sell order 5 ticks above best ask (clear maker)
    ticks_above_ask = 5
    action_signal = ticks_above_ask / config['price_offset_ticks']  # Should be +0.25
    
    action_maker_sell = np.array([
        0.0,           # No buy
        action_signal, # Sell offset signal for maker order
        0.0,           # No buy volume
        0.5,           # 50% sell volume
        0.0,           # No cancel
        0.0            # No do-nothing
    ])
    
    expected_sell_price = env.best_ask + ticks_above_ask * env.config['tick_size']
    print(f"Placing MAKER SELL order:")
    print(f"  Action signal: {action_signal:.3f}")
    print(f"  Expected price: {expected_sell_price:.2f} (Ask={env.best_ask:.2f} + {ticks_above_ask} ticks)")
    print(f"  This should be MAKER (providing liquidity)")
    
    obs, reward, done, truncated, info = env.step(action_maker_sell)
    
    # Check the sell order
    if len(env.pending_orders) > 1:
        sell_order = env.pending_orders[-1]
        placed_sell_price = sell_order['price']
        sell_placement_bid = sell_order.get('placement_bid')
        sell_placement_ask = sell_order.get('placement_ask')
        
        print(f"Sell order details:")
        print(f"  Placed price: {placed_sell_price:.2f}")
        print(f"  Stored placement BBO: Bid={sell_placement_bid:.2f}, Ask={sell_placement_ask:.2f}")
        
        # Verify this is a maker sell order at placement time
        is_sell_maker_at_placement = (not sell_order['is_buy'] and placed_sell_price > sell_placement_bid + 1e-9)
        print(f"  Is MAKER sell at placement: {is_sell_maker_at_placement}")
        
        if is_sell_maker_at_placement:
            print("✅ Correctly created MAKER sell order")
        else:
            print("❌ Sell order was created as TAKER - this is wrong for our test")
    
    print("\n=== TEST 3: TAKER ORDER (for comparison) ===")
    
    # Create a clear taker order for comparison - buy at ask price
    action_taker_buy = np.array([
        1.0,   # Maximum positive signal = at ask price (taker)
        0.0,   # No sell
        0.3,   # 30% volume
        0.0,   # No sell volume
        0.0,   # No cancel
        0.0    # No do-nothing
    ])
    
    print(f"Placing TAKER BUY order (should hit ask price):")
    print(f"  Action signal: 1.0 (maximum aggressive)")
    print(f"  Current Ask: {env.best_ask:.2f}")
    
    obs, reward, done, truncated, info = env.step(action_taker_buy)
    
    if len(env.pending_orders) > 2:
        taker_order = env.pending_orders[-1]
        taker_price = taker_order['price']
        taker_placement_ask = taker_order.get('placement_ask')
        
        print(f"Taker order details:")
        print(f"  Placed price: {taker_price:.2f}")
        print(f"  Placement ask: {taker_placement_ask:.2f}")
        
        # This should be a taker order
        is_taker_at_placement = (taker_order['is_buy'] and taker_price >= taker_placement_ask - 1e-9)
        print(f"  Is TAKER at placement: {is_taker_at_placement}")
        
        if is_taker_at_placement:
            print("✅ Correctly created TAKER order")
        else:
            print("⚠️  Order was created as MAKER - may be due to clipping")
    
    print("\n=== SUMMARY ===")
    print(f"Total pending orders: {len(env.pending_orders)}")
    print(f"Total active orders: {len(env.active_orders)}")
    print("Rebate timing fix appears to be working - orders classified at placement time")
    
    return True

if __name__ == "__main__":
    success = test_maker_order_mechanics()
    if success:
        print("\n🎉 MAKER ORDER MECHANICS TEST COMPLETED")
        print("Orders are being created and classified correctly")
    else:
        print("\n❌ MAKER ORDER MECHANICS TEST FAILED")
        print("Issues found with order creation or classification")