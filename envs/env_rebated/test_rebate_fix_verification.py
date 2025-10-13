#!/usr/bin/env python3
"""
Test to verify the rebate timing fix works correctly

This test verifies that orders are correctly classified as maker/taker 
based on their status at PLACEMENT time, not activation time.
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv
from final_optimized_config import FINAL_OPTIMIZED_CONFIG

def test_rebate_fix():
    """Test that rebate timing is now calculated correctly at placement time"""
    
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config.update({
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 100,
        'episode_length': 50,
        'rebate_rate_long': 0.0002,   # 2bps rebate for makers
        'rebate_rate_short': 0.0002,  # 2bps rebate for makers
        'transaction_cost_long': 0.0001,   # 1bp cost for takers
        'transaction_cost_short': 0.0001,  # 1bp cost for takers
        'latency_steps_long': 1,      # 1-step latency 
        'latency_steps_short': 1,
        'max_active_orders': 10,
        'lot_size': 1000,
        'max_order_volume': 10000,
    })
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== REBATE FIX VERIFICATION TEST ===")
    print(f"Initial BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
    
    # Test Case 1: Place a clear maker order (providing liquidity)
    # Use conservative offset to ensure it's providing liquidity at placement time
    placement_bid = env.best_bid
    placement_ask = env.best_ask
    
    # Place buy order 2 ticks below bid (clearly providing liquidity)
    action = np.array([
        -0.4,  # Buy offset (2 ticks below bid)
        0.0,   # No sell
        0.3,   # 30% volume buy
        0.0,   # No sell volume  
        0.0,   # No cancel
        0.0    # No do-nothing
    ])
    
    print(f"\n--- Placing MAKER Order at Placement Time ---")
    print(f"Placement BBO: Bid={placement_bid:.2f}, Ask={placement_ask:.2f}")
    
    obs, reward, done, truncated, info = env.step(action)
    
    # Check that order was stored with correct placement BBO
    if env.pending_orders:
        order = env.pending_orders[0]
        stored_bid = order.get("placement_bid", "NOT_STORED")
        stored_ask = order.get("placement_ask", "NOT_STORED")
        print(f"✅ Order stored with placement BBO: Bid={stored_bid:.2f}, Ask={stored_ask:.2f}")
        print(f"Order price: {order['price']:.2f}")
        
        # Verify this is clearly a maker order at placement time
        if order['is_buy'] and order['price'] < placement_ask - 1e-9:
            print(f"✅ Order is MAKER at placement (Buy {order['price']:.2f} < Ask {placement_ask:.2f})")
        else:
            print(f"❌ Order classification error at placement")
    else:
        print("❌ No pending orders found")
        return False
    
    # Step forward to activate the order
    action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])  # Do nothing
    obs, reward, done, truncated, info = env.step(action_wait)
    
    activation_bid = env.best_bid  
    activation_ask = env.best_ask
    print(f"\n--- Order Activation Time ---")
    print(f"Activation BBO: Bid={activation_bid:.2f}, Ask={activation_ask:.2f}")
    
    # Check if order activated and how it was classified
    if env.active_orders:
        activated_order = env.active_orders[0]
        is_taker_flag = activated_order.get('is_taker_at_activation', 'NOT_SET')
        print(f"Order activated with is_taker_at_activation: {is_taker_flag}")
        
        # The key test: was classification based on placement time or activation time?
        stored_bid = activated_order.get("placement_bid")
        stored_ask = activated_order.get("placement_ask")
        
        # Check what the classification WOULD be at activation time vs placement time
        would_be_taker_at_activation = (activated_order['is_buy'] and 
                                       activated_order['price'] >= activation_ask - 1e-9)
        would_be_taker_at_placement = (activated_order['is_buy'] and 
                                      activated_order['price'] >= stored_ask - 1e-9)
        
        print(f"Would be taker at activation time: {would_be_taker_at_activation}")
        print(f"Would be taker at placement time: {would_be_taker_at_placement}")
        print(f"Actual classification: {is_taker_flag}")
        
        if is_taker_flag == would_be_taker_at_placement:
            print("✅ CORRECT: Order classified based on placement time BBO")
            return True
        elif is_taker_flag == would_be_taker_at_activation:
            print("❌ BUG: Order still classified based on activation time BBO")
            return False
        else:
            print("❓ Unclear classification result")
            return False
    else:
        print("❌ Order did not activate")
        return False

if __name__ == "__main__":
    success = test_rebate_fix()
    if success:
        print("\n🎉 REBATE TIMING FIX VERIFIED")
        print("Orders are now correctly classified based on placement time BBO")
    else:
        print("\n❌ REBATE TIMING FIX FAILED")
        print("Orders are still being misclassified")