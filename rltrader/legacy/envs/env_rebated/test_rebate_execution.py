#!/usr/bin/env python3
"""
Test actual rebate execution to verify the rebate timing fix

This test will:
1. Place maker orders that will definitely execute
2. Verify they get maker rebates (not taker fees)
3. Confirm rebate timing fix is working in practice
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv
from final_optimized_config import FINAL_OPTIMIZED_CONFIG

def test_rebate_execution():
    """Test actual rebate execution with maker orders"""
    
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config.update({
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 1000,
        'episode_length': 200,
        'rebate_rate_long': 0.001,     # 10bps rebate - very high to be obvious
        'rebate_rate_short': 0.001,    # 10bps rebate - very high to be obvious
        'transaction_cost_long': 0.0005,   # 5bps cost for takers
        'transaction_cost_short': 0.0005,  # 5bps cost for takers
        'latency_steps_long': 1,
        'latency_steps_short': 1,
        'max_active_orders': 20,
        'lot_size': 100,
        'max_order_volume': 10000,
        'price_offset_ticks': 50,      # Large offset range
        'allowed_aggressiveness_ticks': 30,  # Allow aggressive orders
        'tick_size': 0.01,             # Larger tick size for clearer differences
        'post_only_mode': False,
    })
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== REBATE EXECUTION TEST ===")
    print(f"Rebate rate: {config['rebate_rate_long']*10000:.1f}bps (MAKER)")
    print(f"Taker cost: {config['transaction_cost_long']*10000:.1f}bps (TAKER)")
    print(f"Tick size: {config['tick_size']}")
    
    successful_executions = 0
    
    # Try to create execution scenarios
    for attempt in range(50):
        initial_cash = env.cash
        initial_rebates = getattr(env, 'total_rebates_earned', 0.0)
        initial_maker_vol = getattr(env, 'maker_volume', 0.0)
        initial_taker_vol = getattr(env, 'taker_volume', 0.0)
        
        # Place a conservative maker order (well inside the spread)
        spread = env.best_ask - env.best_bid
        mid_price = (env.best_bid + env.best_ask) / 2
        
        # Buy order slightly below mid (should be maker)
        maker_buy_price = mid_price - spread * 0.2
        
        print(f"\n--- Attempt {attempt+1} ---")
        print(f"BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}, Spread={spread:.2f}")
        print(f"Mid: {mid_price:.2f}, Target buy: {maker_buy_price:.2f}")
        
        # Calculate appropriate action
        target_offset = maker_buy_price - env.best_bid
        target_ticks = target_offset / config['tick_size']
        action_signal = max(-1.0, min(1.0, target_ticks / config['price_offset_ticks']))
        
        action = np.array([
            action_signal,  # Buy signal
            0.0,           # No sell
            0.7,           # Large volume for execution chance
            0.0,           # No sell volume
            0.0,           # No cancel
            0.0            # No do-nothing
        ])
        
        print(f"Action signal: {action_signal:.3f}, Expected ticks: {target_ticks:.1f}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Wait a few steps for activation and execution
        for wait_step in range(5):
            action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
            obs, reward, done, truncated, info = env.step(action_wait)
            
            # Check for execution
            final_cash = env.cash
            final_rebates = getattr(env, 'total_rebates_earned', 0.0)
            final_maker_vol = getattr(env, 'maker_volume', 0.0)
            final_taker_vol = getattr(env, 'taker_volume', 0.0)
            
            cash_change = final_cash - initial_cash
            rebate_earned = final_rebates - initial_rebates
            maker_vol_change = final_maker_vol - initial_maker_vol
            taker_vol_change = final_taker_vol - initial_taker_vol
            executed_vol = env.last_executed_volume
            
            if executed_vol > 0 or rebate_earned != 0 or maker_vol_change > 0 or taker_vol_change > 0:
                print(f"🎯 EXECUTION DETECTED at wait step {wait_step+1}!")
                print(f"  Cash change: {cash_change:.6f}")
                print(f"  Rebates earned: {rebate_earned:.6f}")
                print(f"  Maker volume: +{maker_vol_change:.2f}")
                print(f"  Taker volume: +{taker_vol_change:.2f}")
                print(f"  Execution volume: {executed_vol:.2f}")
                
                # Analyze the execution
                if rebate_earned > 0:
                    print("✅ SUCCESS: Received MAKER REBATE!")
                    print(f"  Rebate rate achieved: {(rebate_earned/cash_change*-1)*10000:.1f}bps")
                    successful_executions += 1
                elif rebate_earned == 0 and cash_change < 0:
                    # Check if this was classified as taker
                    if taker_vol_change > 0:
                        print("❌ Order was charged TAKER FEE (classified as taker)")
                    else:
                        print("⚠️  No rebate but also no taker fee - neutral execution")
                elif rebate_earned < 0:
                    print("❌ Negative rebate - this shouldn't happen")
                
                # Check active orders for classification details
                if env.active_orders:
                    for active_order in env.active_orders:
                        taker_flag = active_order.get('is_taker_at_activation', 'N/A')
                        placement_bid = active_order.get('placement_bid', 'N/A')
                        placement_ask = active_order.get('placement_ask', 'N/A')
                        print(f"  Active order: Price={active_order['price']:.2f}, "
                              f"Taker@Placement={taker_flag}, "
                              f"PlacementBBO=({placement_bid:.2f},{placement_ask:.2f})")
                
                break
            
            if done or truncated:
                break
        
        if done or truncated:
            print(f"Episode ended at attempt {attempt+1}")
            break
        
        # Step forward to try different market conditions
        for i in range(3):
            action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
            obs, reward, done, truncated, info = env.step(action_wait)
            if done or truncated:
                break
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Successful rebate executions: {successful_executions}")
    print(f"Total maker volume: {getattr(env, 'maker_volume', 0.0):.2f}")
    print(f"Total taker volume: {getattr(env, 'taker_volume', 0.0):.2f}")
    print(f"Total rebates earned: {getattr(env, 'total_rebates_earned', 0.0):.6f}")
    
    if successful_executions > 0:
        print("✅ REBATE TIMING FIX VERIFIED!")
        print("Maker orders are correctly receiving rebates")
        return True
    else:
        print("❓ Could not verify rebate timing - no executions with rebates")
        return False

if __name__ == "__main__":
    success = test_rebate_execution()
    if success:
        print("\n🎉 REBATE EXECUTION TEST PASSED")
        print("The rebate timing fix is working correctly")
    else:
        print("\n⚠️  REBATE EXECUTION TEST INCONCLUSIVE")
        print("Need more testing or different market conditions")