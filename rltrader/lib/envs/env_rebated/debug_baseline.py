#!/usr/bin/env python3
"""
Debug why baseline environment shows zero volume
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_execution_data():
    """Create the same market data that worked for rebated environments."""
    data = []
    
    for i in range(50):
        row = {'datetime': i}
        
        if i < 10:
            base_bid = 1799.98
            base_ask = 1800.02
        elif i < 20:
            progress = (i - 10) / 10.0
            base_bid = 1799.98 + progress * 0.05
            base_ask = 1800.02 + progress * 0.05
        elif i < 30:
            progress = (i - 20) / 10.0
            base_bid = 1800.03 - progress * 0.10
            base_ask = 1800.07 - progress * 0.10
        elif i < 40:
            oscillation = 0.02 * np.sin((i - 30) * 0.5)
            base_bid = 1799.96 + oscillation
            base_ask = 1799.98 + oscillation
        else:
            base_bid = 1799.98
            base_ask = 1800.02
        
        for level in range(1, 11):
            row[f'bid{level}'] = base_bid - (level-1) * 0.01
            row[f'bidqty{level}'] = 50.0
            row[f'ask{level}'] = base_ask + (level-1) * 0.01  
            row[f'askqty{level}'] = 50.0
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def debug_baseline_vs_rebated():
    """Compare baseline vs rebated environment step by step."""
    print("🔍 DEBUGGING BASELINE VS REBATED")
    
    test_csv = create_execution_data()
    
    try:
        environments = [
            ("Baseline", "baseline", False),
            ("6 bps Rebate", "rebate_6bps", False)
        ]
        
        for name, fee_structure, post_only in environments:
            print(f"\n{'='*60}")
            print(f"TESTING {name}")
            print(f"{'='*60}")
            
            config = get_unified_config(fee_structure, post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 35
            config["latency_steps_long"] = 0  # No latency
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 1.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Environment Configuration:")
            print(f"  Fee structure: {env.fee_structure}")
            print(f"  Is rebated: {getattr(env, 'is_rebated', False)}")
            print(f"  Transaction costs: Long={config['transaction_cost_long']:.6f}, Short={config['transaction_cost_short']:.6f}")
            print(f"  Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            execution_detected = False
            
            for step in range(30):
                if step % 10 == 0:  # Print every 10 steps
                    print(f"\n--- STEP {step} ---")
                    print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
                
                pre_cash = env.cash
                pre_inventory = env.inventory
                
                if step < 5:
                    action = [0.0, 0.0, 0.9, 0.9, -1.0, -1.0]  # Passive orders
                elif step < 15:
                    action = [0.3, -0.3, 0.7, 0.7, -1.0, -1.0]  # Slightly aggressive
                else:
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                
                obs, reward, term, trunc, info = env.step(action)
                
                post_cash = env.cash
                post_inventory = env.inventory
                
                # Check for executions
                cash_change = abs(post_cash - pre_cash)
                inventory_change = abs(post_inventory - pre_inventory)
                
                if cash_change > 1.0 or inventory_change > 0.001:
                    execution_detected = True
                    print(f"  EXECUTION at step {step}:")
                    print(f"    Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
                    print(f"    Inventory: {pre_inventory:.3f} → {post_inventory:.3f} (Δ{post_inventory-pre_inventory:+.3f})")
                    print(f"    Active orders: {len(env.active_orders)}")
                    
                    # Check if this is a rebated environment
                    if hasattr(env, 'is_rebated'):
                        maker_vol = info.get('maker_volume', 0)
                        taker_vol = info.get('taker_volume', 0)
                        rebates = info.get('total_rebates_earned', 0)
                        print(f"    Maker volume: {maker_vol:.3f}")
                        print(f"    Taker volume: {taker_vol:.3f}")
                        print(f"    Rebates: ${rebates:.4f}")
                
                if term or trunc:
                    break
            
            # Final results
            final_cash = env.cash
            final_inventory = env.inventory
            net_change = final_cash - 100000
            
            print(f"\n=== FINAL RESULTS for {name} ===")
            print(f"Final Cash: ${final_cash:.2f}")
            print(f"Net Change: ${net_change:+.2f}")
            print(f"Final Inventory: {final_inventory:.3f}")
            print(f"Execution Detected: {execution_detected}")
            
            if hasattr(env, 'is_rebated'):
                maker_vol = info.get('maker_volume', 0)
                taker_vol = info.get('taker_volume', 0)
                total_vol = maker_vol + taker_vol
                rebates = info.get('total_rebates_earned', 0)
                print(f"Total Volume: {total_vol:.3f} (Maker: {maker_vol:.3f}, Taker: {taker_vol:.3f})")
                print(f"Total Rebates: ${rebates:.4f}")
            
            # Diagnosis
            if not execution_detected:
                print(f"❌ PROBLEM: No executions detected in {name}")
                print(f"   Possible issues:")
                print(f"   - Order placement not working")
                print(f"   - Order execution logic broken")
                print(f"   - Market data not triggering fills")
            else:
                print(f"✅ SUCCESS: Executions working in {name}")
            
            env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def debug_baseline_order_placement():
    """Debug specifically why baseline isn't placing/executing orders."""
    print(f"\n🔍 DEBUGGING BASELINE ORDER PLACEMENT")
    
    test_csv = create_execution_data()
    
    try:
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 20
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Baseline Environment Details:")
        print(f"  Fee structure: {env.fee_structure}")
        print(f"  Transaction costs: {config['transaction_cost_long']:.6f}")
        print(f"  Is rebated: {getattr(env, 'is_rebated', False)}")
        print(f"  Max order volume: {config['max_order_volume']}")
        
        for step in range(15):
            print(f"\n--- STEP {step} ---")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            pre_active = len(env.active_orders)
            pre_pending = len(env.pending_orders)
            
            # Place orders
            action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            post_active = len(env.active_orders)
            post_pending = len(env.pending_orders)
            
            print(f"Orders: Active {pre_active}→{post_active}, Pending {pre_pending}→{post_pending}")
            
            # Check what happened to orders
            if post_active > pre_active:
                print(f"  ✅ New active orders placed")
                for order in env.active_orders[-2:]:  # Check last 2 orders
                    side = "BUY" if order['is_buy'] else "SELL"
                    print(f"    {side} @ {order['price']:.2f}, Volume: {order['volume']:.3f}")
            elif post_pending > pre_pending:
                print(f"  ✅ New pending orders placed")
                for order in env.pending_orders[-2:]:  # Check last 2 orders
                    side = "BUY" if order['is_buy'] else "SELL"
                    print(f"    {side} @ {order['price']:.2f}, Volume: {order['volume']:.3f}")
            else:
                print(f"  ❌ No new orders placed")
            
            if term or trunc:
                break
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_baseline_vs_rebated()
    debug_baseline_order_placement()