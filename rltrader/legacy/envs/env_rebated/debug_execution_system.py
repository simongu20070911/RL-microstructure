#!/usr/bin/env python3
"""
Comprehensive debugging of the execution system to find why it's fragile and inconsistent
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_minimal_execution_test():
    """Create the simplest possible market data that should guarantee execution."""
    data = []
    
    # Phase 1: Stable market for order placement (steps 0-5)
    for i in range(6):
        row = {'datetime': i}
        for level in range(1, 11):
            row[f'bid{level}'] = 1799.99 - (level-1) * 0.01
            row[f'bidqty{level}'] = 10.0  # Small quantity to ensure fills
            row[f'ask{level}'] = 1800.01 + (level-1) * 0.01  
            row[f'askqty{level}'] = 10.0  # Small quantity to ensure fills
        data.append(row)
    
    # Phase 2: Market jumps to execute ALL orders (steps 6-10)
    for i in range(6, 11):
        row = {'datetime': i}
        for level in range(1, 11):
            # Market bid jumps above 1800.01 to hit all sell orders
            # Market ask drops below 1799.99 to hit all buy orders
            row[f'bid{level}'] = 1800.10 - (level-1) * 0.01  # Way above previous ask
            row[f'bidqty{level}'] = 10.0
            row[f'ask{level}'] = 1799.90 + (level-1) * 0.01  # Way below previous bid
            row[f'askqty{level}'] = 10.0
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def debug_order_lifecycle_detailed():
    """Debug every step of the order lifecycle in extreme detail."""
    print("🔍 DETAILED ORDER LIFECYCLE DEBUG")
    
    test_csv = create_minimal_execution_test()
    
    try:
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 12
        config["latency_steps_long"] = 0  # No latency for immediate testing
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        config["max_active_orders"] = 20  # Allow many orders
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment Configuration:")
        print(f"  Max order volume: {config['max_order_volume']}")
        print(f"  Max active orders: {config['max_active_orders']}")
        print(f"  Latency: {config['latency_steps_long']}/{config['latency_steps_short']}")
        print(f"  Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        order_placements = []
        executions = []
        
        for step in range(10):
            print(f"\n{'='*60}")
            print(f"STEP {step}")
            print(f"{'='*60}")
            print(f"Market BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"Market Levels:")
            print(f"  Bid1: {env.bids[0,0]:.2f} x {env.bids[0,1]:.1f}")
            print(f"  Ask1: {env.asks[0,0]:.2f} x {env.asks[0,1]:.1f}")
            
            pre_active = len(env.active_orders)
            pre_pending = len(env.pending_orders)
            pre_cash = env.cash
            pre_inventory = env.inventory
            
            print(f"\nBEFORE ACTION:")
            print(f"  Active orders: {pre_active}")
            print(f"  Pending orders: {pre_pending}")
            print(f"  Cash: ${pre_cash:.2f}")
            print(f"  Inventory: {pre_inventory:.3f}")
            
            if step < 3:
                # Place orders that should be passive at current market
                action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]  # At BBO
                strategy = "PLACE PASSIVE ORDERS AT BBO"
            elif step < 6:
                # Place more orders
                action = [-0.1, 0.1, 0.6, 0.6, -1.0, -1.0]  # Slightly inside
                strategy = "PLACE MORE PASSIVE ORDERS"
            else:
                # Market should now execute our orders
                action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Do nothing
                strategy = "LET MARKET EXECUTE ORDERS"
            
            print(f"\nACTION: {strategy}")
            print(f"  Action vector: {action}")
            
            # Step the environment
            obs, reward, term, trunc, info = env.step(action)
            
            post_active = len(env.active_orders)
            post_pending = len(env.pending_orders)
            post_cash = env.cash
            post_inventory = env.inventory
            
            print(f"\nAFTER ACTION:")
            print(f"  Active orders: {pre_active} → {post_active} (Δ{post_active-pre_active:+d})")
            print(f"  Pending orders: {pre_pending} → {post_pending} (Δ{post_pending-pre_pending:+d})")
            print(f"  Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
            print(f"  Inventory: {pre_inventory:.3f} → {post_inventory:.3f} (Δ{post_inventory-pre_inventory:+.3f})")
            
            # Detailed order analysis
            print(f"\nORDER ANALYSIS:")
            if env.active_orders:
                print(f"  Active Orders ({len(env.active_orders)}):")
                for i, order in enumerate(env.active_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    price = order['price']
                    volume = order['volume']
                    
                    # Check if order should execute
                    if order['is_buy']:
                        can_execute = price >= env.best_ask
                        execution_check = f"Buy {price:.2f} vs Ask {env.best_ask:.2f} = {'SHOULD EXECUTE' if can_execute else 'PASSIVE'}"
                    else:
                        can_execute = price <= env.best_bid
                        execution_check = f"Sell {price:.2f} vs Bid {env.best_bid:.2f} = {'SHOULD EXECUTE' if can_execute else 'PASSIVE'}"
                    
                    print(f"    {i}: {side} {volume:.2f} @ {price:.2f} - {execution_check}")
                    
                    if can_execute:
                        print(f"         ⚠️  ORDER SHOULD EXECUTE BUT DIDN'T!")
            
            if env.pending_orders:
                print(f"  Pending Orders ({len(env.pending_orders)}):")
                for i, order in enumerate(env.pending_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    target_step = order['target_step']
                    will_activate = step >= target_step
                    print(f"    {i}: {side} {order['volume']:.2f} @ {order['price']:.2f} - {'ACTIVATES NOW' if will_activate else f'ACTIVATES STEP {target_step}'}")
            
            # Check for executions
            cash_change = abs(post_cash - pre_cash)
            inventory_change = abs(post_inventory - pre_inventory)
            orders_disappeared = pre_active > post_active
            
            if cash_change > 0.01 or inventory_change > 0.001 or orders_disappeared:
                print(f"\n🎉 EXECUTION DETECTED!")
                execution_data = {
                    'step': step,
                    'cash_change': post_cash - pre_cash,
                    'inventory_change': post_inventory - pre_inventory,
                    'orders_filled': pre_active - post_active,
                    'bbo': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                }
                executions.append(execution_data)
                print(f"  Details: {execution_data}")
            else:
                print(f"\n❌ NO EXECUTION")
            
            # Record order placements
            if post_active > pre_active or post_pending > pre_pending:
                placement_data = {
                    'step': step,
                    'new_active': post_active - pre_active,
                    'new_pending': post_pending - pre_pending,
                    'bbo_at_placement': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                }
                order_placements.append(placement_data)
            
            if term or trunc:
                print(f"\nEpisode ended: terminated={term}, truncated={trunc}")
                break
        
        # Summary analysis
        print(f"\n{'='*60}")
        print(f"EXECUTION SYSTEM ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\nOrder Placements ({len(order_placements)}):")
        for placement in order_placements:
            print(f"  Step {placement['step']}: +{placement['new_active']} active, +{placement['new_pending']} pending @ {placement['bbo_at_placement']}")
        
        print(f"\nExecutions ({len(executions)}):")
        if executions:
            for execution in executions:
                print(f"  Step {execution['step']}: Cash Δ${execution['cash_change']:+.2f}, Inventory Δ{execution['inventory_change']:+.3f}, Orders filled: {execution['orders_filled']} @ {execution['bbo']}")
        else:
            print(f"  ❌ NO EXECUTIONS DETECTED - SYSTEM IS BROKEN")
        
        print(f"\nFinal State:")
        print(f"  Cash: ${env.cash:.2f} (Δ${env.cash - 100000:+.2f})")
        print(f"  Inventory: {env.inventory:.3f}")
        print(f"  Active orders: {len(env.active_orders)}")
        print(f"  Pending orders: {len(env.pending_orders)}")
        
        # Diagnosis
        orders_placed = len(order_placements) > 0
        executions_happened = len(executions) > 0
        
        print(f"\nDIAGNOSIS:")
        if not orders_placed:
            print(f"  🔴 CRITICAL: Order placement system is broken")
        elif not executions_happened:
            print(f"  🔴 CRITICAL: Order execution system is broken")
            print(f"  📋 Orders were placed but never executed despite market movement")
        else:
            print(f"  ✅ Execution system is working")
        
        env.close()
        return executions_happened
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def debug_specific_execution_logic():
    """Debug the specific execution matching logic by manually calling it."""
    print(f"\n🔍 DEBUGGING EXECUTION MATCHING LOGIC")
    
    test_csv = create_minimal_execution_test()
    
    try:
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 8
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place some orders manually
        print(f"\n--- PLACING ORDERS MANUALLY ---")
        
        # Place a buy order that should be passive
        buy_order = {
            "id": 1,
            "price": 1799.99,  # At bid
            "volume": 0.5,
            "initial_volume": 0.5,
            "is_buy": True,
            "timestamp_placed": 0,
            "target_step": 0,
            "is_taker_at_activation": False
        }
        env.active_orders.append(buy_order)
        print(f"Added BUY order: 0.5 @ 1799.99")
        
        # Place a sell order that should be passive
        sell_order = {
            "id": 2,
            "price": 1800.01,  # At ask
            "volume": 0.5,
            "initial_volume": 0.5,
            "is_buy": False,
            "timestamp_placed": 0,
            "target_step": 0,
            "is_taker_at_activation": False
        }
        env.active_orders.append(sell_order)
        print(f"Added SELL order: 0.5 @ 1800.01")
        
        print(f"Orders placed. Active: {len(env.active_orders)}")
        
        # Step forward to market that should execute orders
        for step in range(1, 8):
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Do nothing
            
            print(f"\nStep {step}: Market {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"  Active orders: {len(env.active_orders)}")
            
            if len(env.active_orders) > 0:
                for order in env.active_orders:
                    side = "BUY" if order['is_buy'] else "SELL"
                    price = order['price']
                    
                    if order['is_buy'] and env.best_bid >= price:
                        print(f"    {side} @ {price:.2f} - Market bid {env.best_bid:.2f} SHOULD FILL THIS")
                    elif not order['is_buy'] and env.best_ask <= price:
                        print(f"    {side} @ {price:.2f} - Market ask {env.best_ask:.2f} SHOULD FILL THIS")
                    else:
                        print(f"    {side} @ {price:.2f} - Still passive")
            
            # Manually test execution logic
            if step == 6:  # When market should have moved
                print(f"\n  --- MANUAL EXECUTION TEST ---")
                print(f"  Market: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}")
                print(f"  Order book top: Bid={env.bids[0,0]:.2f} x {env.bids[0,1]}, Ask={env.asks[0,0]:.2f} x {env.asks[0,1]}")
                
                pre_cash = env.cash
                pre_inventory = env.inventory
                
                # Call execution function directly
                realized_pnl = env._execute_orders()
                
                post_cash = env.cash
                post_inventory = env.inventory
                
                print(f"  Execution result:")
                print(f"    Realized PnL: {realized_pnl:.6f}")
                print(f"    Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
                print(f"    Inventory: {pre_inventory:.3f} → {post_inventory:.3f} (Δ{post_inventory-pre_inventory:+.3f})")
                print(f"    Orders remaining: {len(env.active_orders)}")
                
                if abs(post_cash - pre_cash) < 0.01 and abs(post_inventory - pre_inventory) < 0.001:
                    print(f"    ❌ EXECUTION LOGIC FAILED - No changes detected")
                else:
                    print(f"    ✅ EXECUTION LOGIC WORKED")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def test_different_execution_scenarios():
    """Test various execution scenarios to find patterns in failures."""
    print(f"\n🔍 TESTING DIFFERENT EXECUTION SCENARIOS")
    
    scenarios = [
        ("Zero Latency", 0, 0),
        ("Short Latency", 1, 1),
        ("Medium Latency", 2, 2),
        ("Long Latency", 5, 5),
    ]
    
    results = []
    
    for name, long_lat, short_lat in scenarios:
        print(f"\n--- Testing {name} (Latency: {long_lat}/{short_lat}) ---")
        
        test_csv = create_minimal_execution_test()
        
        try:
            config = get_unified_config("baseline", False)
            config["csv_path"] = test_csv
            config["episode_length"] = 12
            config["latency_steps_long"] = long_lat
            config["latency_steps_short"] = short_lat
            config["max_order_volume"] = 1.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            execution_detected = False
            
            # Simple test strategy
            for step in range(10):
                if step < 4:
                    action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]  # Place orders
                else:
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]  # Wait for execution
                
                pre_cash = env.cash
                pre_inventory = env.inventory
                
                obs, reward, term, trunc, info = env.step(action)
                
                post_cash = env.cash
                post_inventory = env.inventory
                
                # Check for execution
                if abs(post_cash - pre_cash) > 0.01 or abs(post_inventory - pre_inventory) > 0.001:
                    execution_detected = True
                    break
            
            results.append({
                'scenario': name,
                'latency': f"{long_lat}/{short_lat}",
                'execution_detected': execution_detected,
                'final_cash': env.cash,
                'final_inventory': env.inventory,
                'orders_remaining': len(env.active_orders)
            })
            
            status = "✅" if execution_detected else "❌"
            print(f"  Result: {status} Execution={'DETECTED' if execution_detected else 'NOT DETECTED'}")
            
            env.close()
            
        finally:
            if os.path.exists(test_csv):
                os.unlink(test_csv)
    
    print(f"\n--- SCENARIO COMPARISON ---")
    print(f"{'Scenario':<15} {'Latency':<10} {'Execution':<12} {'Cash Δ':<10} {'Inventory':<10}")
    print(f"-" * 65)
    
    for result in results:
        cash_delta = result['final_cash'] - 100000
        status = "✅" if result['execution_detected'] else "❌"
        
        print(f"{result['scenario']:<15} {result['latency']:<10} {status:<12} ${cash_delta:+7.2f} {result['final_inventory']:7.3f}")
    
    working_scenarios = [r for r in results if r['execution_detected']]
    broken_scenarios = [r for r in results if not r['execution_detected']]
    
    print(f"\nANALYSIS:")
    print(f"  Working scenarios: {len(working_scenarios)}/{len(results)}")
    print(f"  Broken scenarios: {len(broken_scenarios)}/{len(results)}")
    
    if len(working_scenarios) > 0:
        print(f"  Pattern: Execution works with latencies: {[r['latency'] for r in working_scenarios]}")
    
    if len(broken_scenarios) > 0:
        print(f"  Pattern: Execution fails with latencies: {[r['latency'] for r in broken_scenarios]}")
    
    return len(working_scenarios) > 0

def main():
    """Run comprehensive execution system debugging."""
    print("🚀 COMPREHENSIVE EXECUTION SYSTEM DEBUG")
    print("="*70)
    
    print("\n1. DETAILED ORDER LIFECYCLE DEBUG")
    lifecycle_works = debug_order_lifecycle_detailed()
    
    print("\n2. SPECIFIC EXECUTION LOGIC DEBUG")
    debug_specific_execution_logic()
    
    print("\n3. DIFFERENT EXECUTION SCENARIOS")
    scenarios_work = test_different_execution_scenarios()
    
    print(f"\n" + "="*70)
    print(f"EXECUTION SYSTEM DIAGNOSIS")
    print(f"="*70)
    
    if lifecycle_works and scenarios_work:
        print(f"✅ Execution system is working but may be inconsistent")
        print(f"   Investigation needed into specific conditions that cause failures")
    elif scenarios_work:
        print(f"⚠️  Execution system works in some scenarios but not others")
        print(f"   Pattern analysis needed to identify root cause")
    else:
        print(f"❌ Execution system is fundamentally broken")
        print(f"   Critical bugs in order matching/execution logic")
    
    print(f"="*70)

if __name__ == "__main__":
    main()