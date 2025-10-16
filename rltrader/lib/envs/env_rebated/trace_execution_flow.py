#!/usr/bin/env python3
"""
Trace the exact execution flow to find where it's failing
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def trace_execution_flow():
    """Trace every step of the execution flow."""
    print("🔬 TRACING EXECUTION FLOW STEP BY STEP")
    
    # Create simple test data
    data = []
    for i in range(5):
        row = {'datetime': i}
        if i < 2:
            base_bid, base_ask = 1800.00, 1800.10  # For placement
        else:
            base_bid, base_ask = 1800.50, 1800.60  # For execution
        
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
    
    try:
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 5
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place order 
        action = [-1.0, 0.0, 0.0, 0.8, -1.0, -1.0]  # Sell order
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        print(f"\nAfter placement and activation:")
        for order in env.active_orders:
            print(f"  Order: SELL {order['volume']:.2f} @ {order['price']:.2f}")
            print(f"  Taker status: {order.get('is_taker_at_activation', 'N/A')}")
        
        # Now trace the execution step
        print(f"\n🔬 TRACING EXECUTION STEP:")
        print(f"Current market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Monkey patch the execution method to add detailed logging
        original_execute = env._execute_orders
        
        def traced_execute_orders():
            print(f"\n--- _execute_orders() START ---")
            print(f"Active orders count: {len(env.active_orders)}")
            
            if not env.active_orders:
                print("❌ NO ACTIVE ORDERS TO EXECUTE")
                return 0.0
            
            # Initialize tracking variables
            realized_pnl_from_fills = 0.0
            total_transaction_costs = 0.0
            taker_penalty_applied_total = 0.0
            env.last_executed_volume = 0.0
            orders_to_match = []
            
            print(f"LOB state:")
            print(f"  Bids: {env.bids.shape if env.bids is not None else 'None'}")
            print(f"  Asks: {env.asks.shape if env.asks is not None else 'None'}")
            
            if env.bids is not None and env.bids.size > 0:
                print(f"  Best bid: {env.bids[0, 0]:.2f} x {env.bids[0, 1]:.1f}")
            if env.asks is not None and env.asks.size > 0:
                print(f"  Best ask: {env.asks[0, 0]:.2f} x {env.asks[0, 1]:.1f}")
            
            # Process each order
            for i, order in enumerate(env.active_orders):
                print(f"\nProcessing order {i}:")
                print(f"  Side: {'BUY' if order['is_buy'] else 'SELL'}")
                print(f"  Price: {order['price']:.2f}")
                print(f"  Volume: {order['volume']:.2f}")
                print(f"  Taker at activation: {order.get('is_taker_at_activation', 'N/A')}")
                
                # Apply taker penalty if needed
                is_taker_activation = order.get("is_taker_at_activation", False)
                if is_taker_activation:
                    penalty = env.config.get("taker_penalty", 0.0) * order["volume"]
                    if penalty > 0:
                        env.cash -= penalty
                        taker_penalty_applied_total += penalty
                        print(f"  Applied taker penalty: ${penalty:.6f}")
                
                orders_to_match.append(order)
            
            print(f"\nOrders ready for matching: {len(orders_to_match)}")
            
            # Prepare LOB copies
            current_bids = env.bids.copy() if env.bids is not None and env.bids.size > 0 else np.array([])
            current_asks = env.asks.copy() if env.asks is not None and env.asks.size > 0 else np.array([])
            
            print(f"LOB copies prepared:")
            print(f"  Current bids shape: {current_bids.shape}")
            print(f"  Current asks shape: {current_asks.shape}")
            
            # Sort orders for matching
            orders_to_match.sort(key=lambda x: (-x['price'] if x['is_buy'] else x['price']))
            print(f"Orders sorted for matching")
            
            final_active_orders = []
            
            # Match each order
            for order_idx, order in enumerate(orders_to_match):
                print(f"\n--- MATCHING ORDER {order_idx} ---")
                order_id = order['id']
                is_buy = order["is_buy"]
                limit_price = order["price"]
                remaining_volume = order["volume"]
                
                print(f"Order details: ID={order_id}, {'BUY' if is_buy else 'SELL'}, Price={limit_price:.2f}, Volume={remaining_volume:.2f}")
                
                if remaining_volume <= 1e-9:
                    print(f"  Skipping: volume too small")
                    continue
                
                # Select LOB side
                levels_to_match = current_asks if is_buy else current_bids
                match_side = 'Asks' if is_buy else 'Bids'
                
                print(f"  Matching against {match_side}, levels available: {levels_to_match.shape[0]}")
                
                if levels_to_match.shape[0] == 0:
                    print(f"  ❌ No {match_side} liquidity available")
                    final_active_orders.append(order)
                    continue
                
                # Show available levels
                print(f"  Available {match_side} levels:")
                for i in range(min(3, levels_to_match.shape[0])):
                    price, qty = levels_to_match[i]
                    print(f"    Level {i}: {price:.2f} x {qty:.1f}")
                
                volume_filled_this_order = 0.0
                order_fully_filled = False
                
                # Try to match against each level
                for level_idx in range(levels_to_match.shape[0]):
                    if remaining_volume <= 1e-9:
                        order_fully_filled = True
                        break
                    
                    level_price, level_qty = levels_to_match[level_idx]
                    
                    if np.isnan(level_price) or np.isnan(level_qty) or level_qty < 1e-9:
                        print(f"    Skipping level {level_idx}: invalid price/qty")
                        continue
                    
                    print(f"    Checking level {level_idx}: {level_price:.2f} x {level_qty:.1f}")
                    
                    # Check if order can fill at this level
                    can_fill_at_level = False
                    if is_buy:
                        can_fill_at_level = limit_price >= level_price - 1e-9
                        print(f"      BUY check: {limit_price:.2f} >= {level_price:.2f} - 1e-9 = {can_fill_at_level}")
                    else:
                        can_fill_at_level = limit_price <= level_price + 1e-9
                        print(f"      SELL check: {limit_price:.2f} <= {level_price:.2f} + 1e-9 = {can_fill_at_level}")
                    
                    if not can_fill_at_level:
                        print(f"      ❌ Cannot fill at this level")
                        break  # Price condition not met
                    
                    print(f"      ✅ CAN FILL at this level!")
                    
                    # Calculate fill quantity
                    fill_qty_possible = min(remaining_volume, level_qty)
                    print(f"      Fill quantity possible: min({remaining_volume:.2f}, {level_qty:.1f}) = {fill_qty_possible:.2f}")
                    
                    # Check inventory limits
                    current_net_inventory = env.long_position - env.short_position
                    inventory_change = fill_qty_possible if is_buy else -fill_qty_possible
                    potential_new_inventory = current_net_inventory + inventory_change
                    max_inv = env.config["max_inventory"]
                    
                    print(f"      Inventory check:")
                    print(f"        Current: {current_net_inventory:.2f}")
                    print(f"        Change: {inventory_change:.2f}")
                    print(f"        Potential: {potential_new_inventory:.2f}")
                    print(f"        Max allowed: ±{max_inv:.2f}")
                    
                    fill_qty = fill_qty_possible
                    inventory_limit_hit = False
                    
                    if is_buy and potential_new_inventory > max_inv + 1e-9:
                        fill_qty = max(0.0, max_inv - current_net_inventory)
                        inventory_limit_hit = True
                    elif not is_buy and potential_new_inventory < -max_inv - 1e-9:
                        fill_qty = max(0.0, current_net_inventory - (-max_inv))
                        inventory_limit_hit = True
                    
                    if inventory_limit_hit:
                        print(f"        ⚠️  Inventory limit hit, adjusted fill: {fill_qty:.2f}")
                    
                    if fill_qty <= 1e-9:
                        print(f"        ❌ Fill blocked by inventory limit")
                        break
                    
                    print(f"      🎉 EXECUTING FILL: {fill_qty:.2f} @ {level_price:.2f}")
                    
                    # Execute the fill
                    executed_value = fill_qty * level_price
                    is_maker = not order.get("is_taker_at_activation", True)
                    
                    print(f"        Executed value: ${executed_value:.2f}")
                    print(f"        Is maker: {is_maker}")
                    
                    # Apply transaction costs/rebates
                    if env.is_rebated and is_maker:
                        rebate = env.rebate_rate * executed_value
                        env.cash += rebate
                        env.total_rebates_earned += rebate
                        env.maker_volume += fill_qty
                        transaction_cost = -rebate
                        print(f"        Rebate applied: +${rebate:.6f}")
                    else:
                        cost_rate = env.config["transaction_cost_long"] if is_buy else env.config["transaction_cost_short"]
                        transaction_cost = cost_rate * executed_value
                        env.cash -= transaction_cost
                        if is_maker:
                            env.maker_volume += fill_qty
                        else:
                            env.taker_volume += fill_qty
                        print(f"        Transaction cost: -${transaction_cost:.6f}")
                    
                    total_transaction_costs += transaction_cost
                    
                    # Update cash for trade
                    if is_buy:
                        env.cash -= executed_value
                        print(f"        Cash -= ${executed_value:.2f} (buying)")
                    else:
                        env.cash += executed_value
                        print(f"        Cash += ${executed_value:.2f} (selling)")
                    
                    # Update positions
                    if is_buy:
                        env.long_position += fill_qty
                        print(f"        Long position += {fill_qty:.2f} → {env.long_position:.2f}")
                    else:
                        env.short_position += fill_qty  
                        print(f"        Short position += {fill_qty:.2f} → {env.short_position:.2f}")
                    
                    # Update tracking
                    remaining_volume -= fill_qty
                    volume_filled_this_order += fill_qty
                    env.last_executed_volume += fill_qty
                    
                    # Update LOB
                    levels_to_match[level_idx, 1] = max(0.0, levels_to_match[level_idx, 1] - fill_qty)
                    print(f"        LOB level quantity reduced by {fill_qty:.2f}")
                    
                    print(f"        Order remaining volume: {remaining_volume:.2f}")
                
                # Keep order if not fully filled
                if remaining_volume > 1e-9:
                    order['volume'] = remaining_volume
                    final_active_orders.append(order)
                    print(f"  Order partially filled, keeping with volume {remaining_volume:.2f}")
                else:
                    print(f"  Order fully filled and removed")
            
            # Update active orders list
            env.active_orders = final_active_orders
            
            print(f"\n--- _execute_orders() COMPLETE ---")
            print(f"Total executed volume: {env.last_executed_volume:.2f}")
            print(f"Final active orders: {len(env.active_orders)}")
            print(f"Realized PnL: {realized_pnl_from_fills:.6f}")
            
            return realized_pnl_from_fills
        
        # Replace the method
        env._execute_orders = traced_execute_orders
        
        # Execute the step with full tracing
        print(f"\n🚀 EXECUTING STEP WITH FULL TRACING:")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Market after: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Active orders: {len(env.active_orders)}")
        print(f"Cash: {env.cash}")
        print(f"Maker volume: {getattr(env, 'maker_volume', 0)}")
        print(f"Episode status: Term={term}, Trunc={trunc}")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    trace_execution_flow()