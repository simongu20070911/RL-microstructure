#!/usr/bin/env python3
"""
Test for Rebate Timing Bug

This test demonstrates the critical bug where rebates are calculated based on 
taker/maker status at ACTIVATION time instead of PLACEMENT time.

Scenario:
1. Post a limit order that provides liquidity (should be maker)
2. Market moves against the order during latency period  
3. Order activates and gets executed immediately (looks like taker at activation)
4. BUG: Order gets charged taker fee instead of maker rebate
5. CORRECT: Should get maker rebate because it provided liquidity at placement
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv

def test_rebate_timing_bug():
    """Test the rebate timing classification bug"""
    
    # Load complete configuration and modify for test
    from final_optimized_config import FINAL_OPTIMIZED_CONFIG
    config = FINAL_OPTIMIZED_CONFIG.copy()
    
    # Override specific settings for rebate timing test
    config.update({
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 100,
        'episode_length': 50,
        'rebate_rate_long': 0.0002,   # 2bps rebate for makers
        'rebate_rate_short': 0.0002,  # 2bps rebate for makers
        'transaction_cost_long': 0.0001,   # 1bp cost for takers
        'transaction_cost_short': 0.0001,  # 1bp cost for takers
        'latency_steps_long': 2,      # 2-step latency to allow market movement
        'latency_steps_short': 2,
        'max_active_orders': 5,
        'lot_size': 1000,
        'max_order_volume': 10000,
    })
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== REBATE TIMING BUG TEST ===")
    print(f"Initial BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
    print(f"Rebate rate: {config['rebate_rate_long']*10000:.1f}bps")
    print(f"Transaction cost: {config['transaction_cost_long']*10000:.1f}bps")
    
    # Step 1: Place a limit BUY order that provides liquidity (maker at placement)
    # Use conservative offset to ensure it's clearly providing liquidity
    buy_offset_ticks = -3  # 3 ticks below bid (very conservative maker order)
    buy_price = env.best_bid + buy_offset_ticks * env.config['tick_size']
    
    action = np.array([
        -0.6,  # Buy offset (3 ticks below bid) 
        0.0,   # No sell
        0.5,   # 50% volume buy
        0.0,   # No sell volume
        0.0,   # No cancel
        0.0    # No do-nothing
    ])
    
    print(f"\n--- Step {env.current_step}: Place MAKER BUY Order ---")
    print(f"Placing BUY @ {buy_price:.2f} (Bid={env.best_bid:.2f}, offset={buy_offset_ticks} ticks)")
    print(f"This is clearly a MAKER order (providing liquidity)")
    
    obs, reward, done, truncated, info = env.step(action)
    print(f"Pending orders: {len(env.pending_orders)}")
    print(f"Active orders: {len(env.active_orders)}")
    
    # Step 2: Let market move upward so our buy order becomes aggressive at activation
    # We need to simulate market movement during latency period
    
    # Force market movement by stepping forward
    for step in range(2):  # Wait for latency
        action_do_nothing = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])  # Do nothing
        obs, reward, done, truncated, info = env.step(action_do_nothing)
        
        print(f"\n--- Step {env.current_step}: Market Movement ---")
        print(f"BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
        print(f"Pending orders: {len(env.pending_orders)}")
        print(f"Active orders: {len(env.active_orders)}")
        
        # Check if order activated and what classification it got
        if len(env.active_orders) > 0:
            for order in env.active_orders:
                is_taker_flag = order.get('is_taker_at_activation', 'N/A')
                print(f"Order activated: ID={order['id']}, Price={order['price']:.2f}, "
                      f"is_taker_at_activation={is_taker_flag}")
                
                if order['price'] >= env.best_ask - 1e-9:
                    print(f"❌ BUG DETECTED: Order {order['id']} classified as TAKER at activation")
                    print(f"   Original order was MAKER at placement (provided liquidity)")
                    print(f"   But now Price {order['price']:.2f} >= Ask {env.best_ask:.2f}")
                    print(f"   This order should still get MAKER rebate!")
                else:
                    print(f"✅ Order correctly remains MAKER classification")
    
    # Step 3: Let execution happen and check rebate/penalty
    initial_cash = env.cash
    initial_rebates = getattr(env, 'total_rebates_earned', 0.0)
    
    action_do_nothing = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
    obs, reward, done, truncated, info = env.step(action_do_nothing)
    
    final_cash = env.cash
    final_rebates = getattr(env, 'total_rebates_earned', 0.0)
    cash_change = final_cash - initial_cash
    rebate_earned = final_rebates - initial_rebates
    
    print(f"\n--- Step {env.current_step}: Execution Results ---")
    print(f"Cash change: {cash_change:.6f}")
    print(f"Rebates earned this step: {rebate_earned:.6f}")
    print(f"Executed volume: {env.last_executed_volume:.4f}")
    print(f"Active orders remaining: {len(env.active_orders)}")
    
    if rebate_earned > 0:
        print("✅ CORRECT: Order received maker rebate")
    elif rebate_earned == 0 and cash_change < 0:
        print("❌ BUG CONFIRMED: Order was charged taker fee instead of maker rebate")
        print("   This is the rebate timing classification bug!")
    
    env.close()
    return rebate_earned > 0

if __name__ == "__main__":
    success = test_rebate_timing_bug()
    if not success:
        print("\n🚨 REBATE TIMING BUG CONFIRMED")
        print("Orders are being classified at activation time instead of placement time")
        print("This causes incorrect rebate/penalty calculation")
    else:
        print("\n✅ Rebate timing appears correct")