#!/usr/bin/env python3
"""
Diagnose the exact source of execution inconsistency by testing step-by-step
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_simple_execution_data():
    """Create the simplest possible data that MUST guarantee execution."""
    data = []
    
    for i in range(15):
        row = {'datetime': i}
        
        if i < 5:
            # Phase 1: Stable market
            base_bid = 1799.95
            base_ask = 1800.05
        elif i < 10:
            # Phase 2: Market jumps UP - should hit sell orders
            base_bid = 1800.20  # WAY above original ask
            base_ask = 1800.30
        else:
            # Phase 3: Market jumps DOWN - should hit buy orders  
            base_bid = 1799.70  # WAY below original bid
            base_ask = 1799.80
        
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

def detailed_execution_test():
    """Test execution with extreme detail to find the bug."""
    print("🔍 DETAILED EXECUTION DIAGNOSIS")
    
    test_csv = create_simple_execution_data()
    
    try:
        # Test both baseline and rebated
        for env_type in ["baseline", "rebate_6bps"]:
            print(f"\n{'='*60}")
            print(f"TESTING {env_type.upper()}")
            print(f"{'='*60}")
            
            config = get_unified_config(env_type, False)
            config["csv_path"] = test_csv
            config["episode_length"] = 12
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 0.5
            config["max_active_orders"] = 5  # Fewer orders to reduce complexity
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Environment: {env.fee_structure}")
            print(f"Is rebated: {getattr(env, 'is_rebated', False)}")
            print(f"Transaction costs: {config['transaction_cost_long']:.6f}")
            print(f"Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Step 1: Place orders at initial market
            print(f"\n--- STEP 1: PLACE ORDERS ---")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Simple order placement
            action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]  # One buy, one sell
            obs, reward, term, trunc, info = env.step(action)
            
            print(f"After placement:")
            print(f"  Active orders: {len(env.active_orders)}")
            print(f"  Pending orders: {len(env.pending_orders)}")
            
            # Show exact orders
            for i, order in enumerate(env.active_orders):
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"  Active {i}: {side} {order['volume']:.2f} @ {order['price']:.2f}")
            
            for i, order in enumerate(env.pending_orders):
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"  Pending {i}: {side} {order['volume']:.2f} @ {order['price']:.2f} (target step {order['target_step']})")
            
            # Step 2: Activate pending orders
            if len(env.pending_orders) > 0:
                print(f"\n--- STEP 2: ACTIVATE PENDING ---")
                obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Do nothing
                
                print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
                print(f"Active orders after activation: {len(env.active_orders)}")
                
                for i, order in enumerate(env.active_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    is_taker = order.get('is_taker_at_activation', 'N/A')
                    print(f"  Active {i}: {side} {order['volume']:.2f} @ {order['price']:.2f} (taker_flag: {is_taker})")
            
            # Steps 3-8: Market movement that should execute orders
            for step in range(3, 9):
                print(f"\n--- STEP {step}: MARKET MOVEMENT ---")
                
                pre_cash = env.cash
                pre_inventory = env.inventory
                pre_active = len(env.active_orders)
                
                obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
                
                post_cash = env.cash
                post_inventory = env.inventory
                post_active = len(env.active_orders)
                
                print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
                print(f"Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
                print(f"Inventory: {pre_inventory:.3f} → {post_inventory:.3f} (Δ{post_inventory-pre_inventory:+.3f})")
                print(f"Active orders: {pre_active} → {post_active} (Δ{post_active-pre_active:+d})")
                
                # Check for execution
                if abs(post_cash - pre_cash) > 0.1 or abs(post_inventory - pre_inventory) > 0.001:
                    print(f"🎉 EXECUTION DETECTED!")
                    
                    if hasattr(env, 'is_rebated') and env.is_rebated:
                        maker_vol = info.get('maker_volume', 0)
                        rebates = info.get('total_rebates_earned', 0)
                        print(f"  Maker volume: {maker_vol:.3f}")
                        print(f"  Rebates: ${rebates:.4f}")
                else:
                    print(f"❌ No execution")
                
                # Show remaining orders
                if len(env.active_orders) > 0:
                    print(f"Remaining active orders:")
                    for i, order in enumerate(env.active_orders):
                        side = "BUY" if order['is_buy'] else "SELL"
                        
                        # Check if this order should execute at current market
                        should_execute = False
                        if order['is_buy'] and order['price'] >= env.best_ask:
                            should_execute = True
                        elif not order['is_buy'] and order['price'] <= env.best_bid:
                            should_execute = True
                        
                        status = "SHOULD EXECUTE" if should_execute else "passive"
                        print(f"    {side} {order['volume']:.2f} @ {order['price']:.2f} - {status}")
                
                if term or trunc:
                    print(f"Episode ended")
                    break
            
            # Final summary
            final_cash = env.cash
            net_change = final_cash - 100000
            final_inventory = env.inventory
            
            print(f"\n--- FINAL RESULTS ---")
            print(f"Net cash change: ${net_change:+.2f}")
            print(f"Final inventory: {final_inventory:.3f}")
            
            if hasattr(env, 'is_rebated') and env.is_rebated:
                maker_vol = info.get('maker_volume', 0)
                taker_vol = info.get('taker_volume', 0)
                rebates = info.get('total_rebates_earned', 0)
                print(f"Maker volume: {maker_vol:.3f}")
                print(f"Taker volume: {taker_vol:.3f}")
                print(f"Total rebates: ${rebates:.4f}")
            
            execution_happened = abs(net_change) > 0.1 or abs(final_inventory) > 0.001
            print(f"Execution result: {'✅ SUCCESS' if execution_happened else '❌ FAILED'}")
            
            env.close()
            
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    detailed_execution_test()