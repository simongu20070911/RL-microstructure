#!/usr/bin/env python3
"""
Debug whether taker execution is using limit order pricing vs market pricing
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def debug_taker_limit_vs_market():
    """Debug if taker executions use limit order price vs market price."""
    print("🔍 DEBUGGING: TAKER EXECUTION - LIMIT ORDER PRICE vs MARKET PRICE")
    print("=" * 80)
    
    # Create scenario where we can clearly see the difference
    data = []
    for i in range(4):
        row = {'datetime': i}
        if i < 2:
            # Initial market for order placement
            base_bid, base_ask = 1800.00, 1800.10
        else:
            # Market moves significantly to trigger taker execution
            base_bid, base_ask = 1820.00, 1820.10  # Big jump up
            
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
        config = get_unified_config("baseline", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 4
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place a sell order well below the market (will become profitable when market jumps)
        action = [-1.0, 0.5, 0.0, 1.0, -1.0, 0.0]  # Sell order at mid-market
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        print(f"\nOrder placed:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.4f} @ {order['price']:.2f}")
            
        if not env.active_orders:
            print("  No orders found - trying different approach")
            return
            
        order_price = env.active_orders[0]['price'] if env.active_orders else 0
        order_volume = env.active_orders[0]['volume'] if env.active_orders else 0
        is_sell = not env.active_orders[0]['is_buy'] if env.active_orders else False
        
        # Step to next market state where execution will happen
        print(f"\n📈 MARKET JUMPS TO TRIGGER TAKER EXECUTION:")
        
        pre_cash = env.cash
        pre_inventory = env.long_position - env.short_position
        pre_taker = info.get('taker_volume', 0)
        pre_maker = info.get('maker_volume', 0)
        
        print(f"\nBEFORE EXECUTION:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Cash: ${pre_cash:.2f}")
        print(f"  Inventory: {pre_inventory:.4f}")
        print(f"  Order: {'SELL' if is_sell else 'BUY'} {order_volume:.4f} @ {order_price:.2f}")
        
        # Analysis of what SHOULD happen
        if is_sell and env.best_bid >= order_price:
            print(f"  🎯 SELL order should execute as TAKER")
            print(f"     Order price: ${order_price:.2f}")
            print(f"     Current bid: ${env.best_bid:.2f}")
            print(f"     Expected execution price:")
            print(f"       - If LIMIT ORDER pricing: ${order_price:.2f}")
            print(f"       - If MARKET pricing: ${env.best_bid:.2f}")
        elif not is_sell and env.best_ask <= order_price:
            print(f"  🎯 BUY order should execute as TAKER")
            print(f"     Order price: ${order_price:.2f}")
            print(f"     Current ask: ${env.best_ask:.2f}")
            print(f"     Expected execution price:")
            print(f"       - If LIMIT ORDER pricing: ${order_price:.2f}")
            print(f"       - If MARKET pricing: ${env.best_ask:.2f}")
        
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        post_cash = env.cash
        post_inventory = env.long_position - env.short_position
        post_taker = info.get('taker_volume', 0)
        post_maker = info.get('maker_volume', 0)
        
        cash_change = post_cash - pre_cash
        inventory_change = post_inventory - pre_inventory
        taker_change = post_taker - pre_taker
        maker_change = post_maker - pre_maker
        
        print(f"\nAFTER EXECUTION:")
        print(f"  Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"  Cash: ${post_cash:.2f} (Δ${cash_change:+.2f})")
        print(f"  Inventory: {post_inventory:.4f} (Δ{inventory_change:+.4f})")
        print(f"  Taker volume: {post_taker:.4f} (+{taker_change:.4f})")
        print(f"  Maker volume: {post_maker:.4f} (+{maker_change:.4f})")
        
        if abs(inventory_change) > 0.0001:
            print(f"\n🔬 EXECUTION ANALYSIS:")
            
            # Calculate the actual execution price
            if is_sell:
                # For sell: positive cash change, negative inventory change
                gross_price = abs(cash_change) / abs(inventory_change)
            else:
                # For buy: negative cash change, positive inventory change
                gross_price = abs(cash_change) / abs(inventory_change)
            
            print(f"  Volume executed: {abs(inventory_change):.4f}")
            print(f"  Cash change: ${cash_change:+.2f}")
            print(f"  Gross execution price: ${gross_price:.2f}")
            
            # Account for transaction costs to get net price
            transaction_cost_rate = config.get('transaction_cost_long', 0.0001)
            if is_sell:
                # For sell: gross = net * (1 - cost)
                net_price = gross_price / (1 - transaction_cost_rate)
            else:
                # For buy: gross = net * (1 + cost)
                net_price = gross_price / (1 + transaction_cost_rate)
            
            print(f"  Transaction cost: {transaction_cost_rate:.6f}")
            print(f"  Net execution price: ${net_price:.2f}")
            
            print(f"\n  📊 PRICE COMPARISON:")
            print(f"    Order price: ${order_price:.2f}")
            print(f"    Market bid: ${env.best_bid:.2f}")
            print(f"    Market ask: ${env.best_ask:.2f}")
            print(f"    Actual net execution: ${net_price:.2f}")
            
            # Determine which pricing mechanism was used
            limit_diff = abs(net_price - order_price)
            market_bid_diff = abs(net_price - env.best_bid)
            market_ask_diff = abs(net_price - env.best_ask)
            
            print(f"\n  🎯 PRICING MECHANISM DETERMINATION:")
            print(f"    Difference from order price: {limit_diff:.4f}")
            print(f"    Difference from market bid: {market_bid_diff:.4f}")
            print(f"    Difference from market ask: {market_ask_diff:.4f}")
            
            if limit_diff < 0.01:
                print(f"    ✅ USES LIMIT ORDER PRICING (order fills at order price)")
                print(f"    📝 This means it's a LIMIT ORDER execution, not market execution")
            elif is_sell and market_bid_diff < 0.01:
                print(f"    ✅ USES MARKET PRICING (sell at bid)")
                print(f"    📝 This means it's true MARKET ORDER execution")
            elif not is_sell and market_ask_diff < 0.01:
                print(f"    ✅ USES MARKET PRICING (buy at ask)")
                print(f"    📝 This means it's true MARKET ORDER execution")
            else:
                print(f"    ⚠️ UNCLEAR PRICING MECHANISM")
                print(f"    📝 Price doesn't clearly match limit or market pricing")
                
            # Determine if this is maker or taker
            if taker_change > 0:
                print(f"\n  📈 EXECUTION TYPE: TAKER")
                if limit_diff < 0.01:
                    print(f"     💡 INSIGHT: Taker execution at LIMIT PRICE")
                    print(f"        This is correct - limit orders execute at their limit price")
                    print(f"        even when they become takers due to market movement")
            elif maker_change > 0:
                print(f"\n  📈 EXECUTION TYPE: MAKER")
                print(f"     💡 This would be normal limit order at limit price")
        
        # Test with multiple scenarios
        print(f"\n" + "=" * 60)
        print(f"🔬 TESTING DIFFERENT PRICE LEVELS")
        print(f"=" * 60)
        
        test_cases = [
            ("Below market", 0.3),   # Order below current market
            ("At market", 0.5),      # Order at current market  
            ("Above market", 0.7),   # Order above current market
        ]
        
        for desc, offset in test_cases:
            print(f"\n--- {desc} order ---")
            
            obs, info = env.reset()
            
            action = [-1.0, offset, 0.0, 1.0, -1.0, 0.0]
            obs, reward, term, trunc, info = env.step(action)
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            if env.active_orders:
                test_order_price = env.active_orders[0]['price']
                test_order_volume = env.active_orders[0]['volume']
                print(f"  Order: SELL {test_order_volume:.4f} @ ${test_order_price:.2f}")
                
                pre_cash = env.cash
                pre_inventory = env.long_position - env.short_position
                
                obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
                
                post_cash = env.cash
                post_inventory = env.long_position - env.short_position
                
                cash_change = post_cash - pre_cash
                inventory_change = post_inventory - pre_inventory
                
                if abs(inventory_change) > 0.0001:
                    execution_price = abs(cash_change) / abs(inventory_change)
                    net_price = execution_price / (1 - 0.0001)  # Adjust for costs
                    print(f"  Executed at: ${net_price:.2f}")
                    print(f"  Market bid: ${env.best_bid:.2f}")
                    
                    if abs(net_price - test_order_price) < 0.01:
                        print(f"  ✅ Executed at ORDER price (limit order behavior)")
                    elif abs(net_price - env.best_bid) < 0.01:
                        print(f"  ✅ Executed at MARKET price (market order behavior)")
                    else:
                        print(f"  ❓ Executed at unexpected price")
                else:
                    print(f"  No execution occurred")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    debug_taker_limit_vs_market()