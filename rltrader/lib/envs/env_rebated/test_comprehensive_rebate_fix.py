#!/usr/bin/env python3
"""
Comprehensive test for rebate timing fix

This test creates a scenario where:
1. Order is placed as a maker (providing liquidity)
2. Market moves during latency period
3. Order activates and executes immediately
4. Verify it gets maker rebate (not taker penalty)
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv
from final_optimized_config import FINAL_OPTIMIZED_CONFIG

def test_comprehensive_rebate_fix():
    """Comprehensive test of rebate timing fix"""
    
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config.update({
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 1000,
        'episode_length': 200,
        'rebate_rate_long': 0.0004,   # 4bps rebate for makers
        'rebate_rate_short': 0.0004,  # 4bps rebate for makers  
        'transaction_cost_long': 0.0002,   # 2bps cost for takers
        'transaction_cost_short': 0.0002,  # 2bps cost for takers
        'latency_steps_long': 2,      # 2-step latency
        'latency_steps_short': 2,
        'max_active_orders': 10,
        'lot_size': 1000,
        'max_order_volume': 50000,
        'taker_penalty': 0.001,  # Extra penalty for takers
    })
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== COMPREHENSIVE REBATE FIX TEST ===")
    print(f"Rebate rate: {config['rebate_rate_long']*10000:.1f}bps")
    print(f"Taker cost: {config['transaction_cost_long']*10000:.1f}bps") 
    print(f"Taker penalty: {config['taker_penalty']*10000:.1f}bps")
    
    # Try multiple scenarios to find one that executes
    for attempt in range(20):
        initial_cash = env.cash
        initial_rebates = getattr(env, 'total_rebates_earned', 0.0)
        
        print(f"\n--- Attempt {attempt+1} ---")
        print(f"BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
        
        # Place a buy order that's clearly providing liquidity at placement time
        # Use mid-market pricing to ensure it's a maker
        spread = env.best_ask - env.best_bid
        mid_price = (env.best_bid + env.best_ask) / 2
        maker_buy_price = mid_price - spread * 0.25  # Quarter spread below mid
        
        # Calculate the action needed for this price
        tick_offset = (maker_buy_price - env.best_bid) / env.config['tick_size']
        action_signal = max(-1.0, min(1.0, tick_offset / env.config['price_offset_ticks']))
        
        action = np.array([
            action_signal,  # Buy offset to get desired price
            0.0,    # No sell
            0.8,    # Large volume for better chance of execution
            0.0,    # No sell volume
            0.0,    # No cancel
            0.0     # No do-nothing
        ])
        
        print(f"Placing MAKER buy @ {maker_buy_price:.2f} (mid={mid_price:.2f})")
        print(f"Action signal: {action_signal:.3f}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Track the order through its lifecycle
        pending_count = len(env.pending_orders)
        print(f"Pending orders: {pending_count}")
        
        if pending_count > 0:
            order = env.pending_orders[-1]  # Get the latest order
            placement_bid = order.get("placement_bid")
            placement_ask = order.get("placement_ask") 
            print(f"Order stored: Price={order['price']:.2f}, PlacementBid={placement_bid:.2f}, PlacementAsk={placement_ask:.2f}")
            
            # Verify it's a maker at placement time
            is_maker_at_placement = (order['is_buy'] and order['price'] < placement_ask - 1e-9)
            print(f"Is maker at placement: {is_maker_at_placement}")
            
            # Wait for activation and potential execution
            for step in range(5):
                action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
                obs, reward, done, truncated, info = env.step(action_wait)
                
                final_cash = env.cash
                final_rebates = getattr(env, 'total_rebates_earned', 0.0)
                cash_change = final_cash - initial_cash
                rebate_earned = final_rebates - initial_rebates
                executed_vol = env.last_executed_volume
                
                print(f"  Step {step+1}: Cash Δ={cash_change:.6f}, Rebates Δ={rebate_earned:.6f}, ExecVol={executed_vol:.4f}")
                
                if executed_vol > 0:
                    print(f"🎯 EXECUTION DETECTED!")
                    print(f"Cash change: {cash_change:.6f}")
                    print(f"Rebates earned: {rebate_earned:.6f}")
                    print(f"Executed volume: {executed_vol:.4f}")
                    
                    # Check active orders for classification
                    if env.active_orders:
                        for active_order in env.active_orders:
                            taker_flag = active_order.get('is_taker_at_activation', 'N/A')
                            print(f"Active order taker flag: {taker_flag}")
                    
                    # Analyze the result
                    if rebate_earned > 0:
                        print("✅ SUCCESS: Order received MAKER REBATE")
                        print("✅ Rebate timing fix is working correctly!")
                        return True
                    elif rebate_earned == 0 and cash_change < 0:
                        print("❌ FAILURE: Order was charged TAKER FEE")
                        print("❌ Rebate timing bug still exists!")
                        return False
                    else:
                        print("❓ Unclear result - no clear rebate or penalty")
                
                if done or truncated:
                    break
        else:
            print("No orders placed, trying next step...")
            
        # Step forward to try different market conditions
        action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
        obs, reward, done, truncated, info = env.step(action_wait)
        
        if done or truncated:
            break
    
    print("❓ Could not create execution scenario to test rebate timing")
    return None

if __name__ == "__main__":
    result = test_comprehensive_rebate_fix()
    if result is True:
        print("\n🎉 REBATE TIMING FIX VERIFIED!")
        print("Maker orders correctly receive rebates based on placement time")
    elif result is False:
        print("\n❌ REBATE TIMING BUG PERSISTS!")
        print("Orders still misclassified - needs further investigation")
    else:
        print("\n❓ TEST INCONCLUSIVE")
        print("Could not create execution scenario - may need manual verification")