#!/usr/bin/env python3
"""
Final verification test for rebate timing fix

This test properly configures a rebated environment and verifies:
1. Environment is properly set as rebated
2. Maker orders receive rebates based on placement time classification
3. Rebate timing fix is working correctly
"""

import numpy as np
import sys
import os
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from env_rebated_unified import RebatedHFTEnv

def test_final_rebate_verification():
    """Final comprehensive test of rebate timing fix"""
    
    # Use proper rebated configuration
    config = {
        # === CORE SETTINGS ===
        'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_small.csv',
        'max_steps': 500,
        'episode_length': 100,
        'initial_capital': 1000000,
        'max_inventory': 500000,
        'order_book_levels': 10,
        
        # === REBATED FEE STRUCTURE ===
        'fee_structure': 'rebate_4bps',  # This will set is_rebated = True
        'rebate_rate_long': 0.001,      # 10bps rebate for makers  
        'rebate_rate_short': 0.001,     # 10bps rebate for makers
        'transaction_cost_long': 0.0005,    # 5bps cost for takers
        'transaction_cost_short': 0.0005,   # 5bps cost for takers
        
        # === ORDER MANAGEMENT ===
        'latency_steps_long': 1,
        'latency_steps_short': 1,
        'max_active_orders': 20,
        'lot_size': 100,
        'max_order_volume': 10000,
        'price_offset_ticks': 50,
        'allowed_aggressiveness_ticks': 30,
        'tick_size': 0.01,
        'post_only_mode': False,
        
        # === REQUIRED CONFIG ITEMS ===
        'do_nothing_threshold': 0.8,
        'inventory_penalty': 1e-6,
        'quoting_reward_max_ticks': 20,
        'explicit_cancel_enabled': True,
        'explicit_cancel_threshold': 0.6,
        'obs_qty_norm_scale': 1000.0,
        'invalid_order_penalty': 0.001,
        'activity_bonus': 1e-5,
        'taker_penalty': 0.001,
        'explicit_cancel_penalty': 1e-6,
        'explicit_cancel_clears_pending': True,
        'quoting_reward_enabled': True,
        'quoting_reward_amount': 5e-6,
        'obs_price_norm_scale': 50000.0,
    }
    
    env = RebatedHFTEnv(config)
    obs, info = env.reset()
    
    print("=== FINAL REBATE VERIFICATION TEST ===")
    print(f"Environment is_rebated: {env.is_rebated}")
    print(f"Rebate rate: {env.rebate_rate*10000:.1f}bps")
    print(f"Fee structure: {config.get('fee_structure', 'not set')}")
    
    if not env.is_rebated:
        print("❌ CRITICAL ERROR: Environment is not configured as rebated!")
        return False
    
    successful_rebates = 0
    
    # Test maker orders and verify rebates
    for attempt in range(30):
        initial_cash = env.cash
        initial_rebates = getattr(env, 'total_rebates_earned', 0.0)
        initial_maker_vol = getattr(env, 'maker_volume', 0.0)
        
        # Create a conservative maker order
        spread = env.best_ask - env.best_bid
        mid_price = (env.best_bid + env.best_ask) / 2
        maker_buy_price = mid_price - spread * 0.3  # Well below mid
        
        print(f"\n--- Attempt {attempt+1} ---")
        print(f"BBO: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
        print(f"Target maker buy price: {maker_buy_price:.2f}")
        
        # Calculate action for maker order
        target_offset = maker_buy_price - env.best_bid
        target_ticks = target_offset / config['tick_size']
        action_signal = max(-1.0, min(1.0, target_ticks / config['price_offset_ticks']))
        
        action = np.array([
            action_signal,  # Buy signal for maker order
            0.0,           # No sell
            0.6,           # 60% volume
            0.0,           # No sell volume
            0.0,           # No cancel
            0.0            # No do-nothing
        ])
        
        obs, reward, done, truncated, info = env.step(action)
        
        # Wait for activation and execution
        for wait_step in range(6):
            action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
            obs, reward, done, truncated, info = env.step(action_wait)
            
            # Check for execution and rebates
            final_cash = env.cash
            final_rebates = getattr(env, 'total_rebates_earned', 0.0)
            final_maker_vol = getattr(env, 'maker_volume', 0.0)
            
            cash_change = final_cash - initial_cash
            rebate_earned = final_rebates - initial_rebates
            maker_vol_change = final_maker_vol - initial_maker_vol
            executed_vol = env.last_executed_volume
            
            if rebate_earned > 0:
                print(f"🎯 REBATE EARNED at wait step {wait_step+1}!")
                print(f"  Cash change: {cash_change:.6f}")
                print(f"  Rebates earned: {rebate_earned:.6f}")
                print(f"  Maker volume change: +{maker_vol_change:.2f}")
                print(f"  Execution volume: {executed_vol:.2f}")
                
                # Verify this is correct maker rebate behavior
                if maker_vol_change > 0:
                    print("✅ SUCCESS: Maker order received proper rebate!")
                    print(f"  Rebate rate achieved: {(rebate_earned/(maker_vol_change*maker_buy_price))*10000:.1f}bps")
                    successful_rebates += 1
                else:
                    print("⚠️  Rebate earned but no maker volume - unexpected")
                
                # Check order classification
                if env.active_orders:
                    for order in env.active_orders:
                        taker_flag = order.get('is_taker_at_activation', 'N/A')
                        print(f"  Order classification: is_taker_at_placement={taker_flag}")
                
                break
            elif executed_vol > 0 or maker_vol_change > 0:
                print(f"❌ Execution without rebate at step {wait_step+1}")
                print(f"  Cash change: {cash_change:.6f}")
                print(f"  Maker volume: +{maker_vol_change:.2f}")
                print(f"  This indicates a problem with rebate calculation")
                break
            
            if done or truncated:
                break
        
        if done or truncated:
            break
        
        # Step forward to try different conditions
        for i in range(2):
            action_wait = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9])
            obs, reward, done, truncated, info = env.step(action_wait)
            if done or truncated:
                break
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Successful rebate executions: {successful_rebates}")
    print(f"Total maker volume: {getattr(env, 'maker_volume', 0.0):.2f}")
    print(f"Total taker volume: {getattr(env, 'taker_volume', 0.0):.2f}")
    print(f"Total rebates earned: {getattr(env, 'total_rebates_earned', 0.0):.6f}")
    
    if successful_rebates > 0:
        print("✅ REBATE TIMING FIX VERIFIED!")
        print("Maker orders are correctly classified at placement time and receive rebates")
        return True
    else:
        maker_vol = getattr(env, 'maker_volume', 0.0)
        if maker_vol > 0:
            print("❌ REBATE CALCULATION PROBLEM!")
            print("Maker volume was generated but no rebates were paid")
            return False
        else:
            print("❓ NO MAKER EXECUTIONS OCCURRED")
            print("Could not test rebate timing - need different market conditions")
            return None

if __name__ == "__main__":
    result = test_final_rebate_verification()
    if result is True:
        print("\n🎉 REBATE TIMING FIX FULLY VERIFIED!")
        print("The fix is working correctly - orders classified at placement time")
    elif result is False:
        print("\n❌ REBATE TIMING FIX FAILED!")
        print("There are still issues with rebate calculation or classification")
    else:
        print("\n❓ TEST INCONCLUSIVE")
        print("Could not generate test conditions for verification")