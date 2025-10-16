#!/usr/bin/env python3
"""
Debug rapid oscillation scenario and taker execution pricing
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def debug_rapid_oscillation():
    """Debug rapid oscillation and taker execution pricing."""
    print("🔍 DEBUGGING RAPID OSCILLATION AND TAKER EXECUTION PRICING")
    print("=" * 80)
    
    # Create rapid oscillation data with detailed logging
    data = []
    print("\n📊 CREATING RAPID OSCILLATION DATA:")
    
    for i in range(12):
        row = {'datetime': i}
        # Rapid oscillation between high and low
        if i % 2 == 0:
            base_bid, base_ask = 1800.50, 1800.60
        else:
            base_bid, base_ask = 1799.50, 1799.60
            
        print(f"  Step {i}: Market {base_bid:.2f}/{base_ask:.2f}")
            
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
        # Test baseline configuration first
        config = get_unified_config("baseline", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 12
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"\n🏗️  INITIAL SETUP:")
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place orders that should execute during oscillation
        print(f"\n📋 PLACING ORDERS FOR OSCILLATION TESTING:")
        
        # Place buy order at 1800.00 (should execute when market drops to 1799.50/1799.60)
        # Place sell order at 1800.00 (should execute when market rises to 1800.50/1800.60)
        action = [0.0, 0.0, 0.8, 0.8, -1.0, 0.0]  # At market orders
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"Orders placed:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Activate orders
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        print(f"\nActive orders after activation:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Now step through the oscillation with detailed execution tracking
        print(f"\n🔄 STEPPING THROUGH RAPID OSCILLATION:")
        print(f"{'Step':<4} {'Market':<13} {'Should Execute':<25} {'Maker':<8} {'Taker':<8} {'Exec Price':<12} {'Cash Δ':<10}")
        print(f"{'-'*4} {'-'*13} {'-'*25} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")
        
        step_count = 2
        execution_log = []
        
        while step_count < 11 and not (term or trunc):
            step_count += 1
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            
            # Check execution potential BEFORE step
            should_execute = []
            execution_prices = []
            
            for order in env.active_orders:
                if order['is_buy'] and env.best_ask <= order['price']:
                    should_execute.append(f"BUY@{order['price']:.2f}")
                    execution_prices.append(f"Ask={env.best_ask:.2f}")
                elif not order['is_buy'] and env.best_bid >= order['price']:
                    should_execute.append(f"SELL@{order['price']:.2f}")
                    execution_prices.append(f"Bid={env.best_bid:.2f}")
            
            should_execute_str = ", ".join(should_execute) if should_execute else "None"
            
            # Execute step
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            
            maker_change = post_maker - pre_maker
            taker_change = post_taker - pre_taker
            cash_change = post_cash - pre_cash
            
            # Determine execution type and price
            exec_type = ""
            exec_price = ""
            
            if maker_change > 0:
                exec_type = "MAKER"
                # For maker, we're providing liquidity at our order price
                exec_price = "Order Price"
            elif taker_change > 0:
                exec_type = "TAKER"
                # For taker, we're taking liquidity at market price
                exec_price = ", ".join(execution_prices) if execution_prices else "Market"
            
            execution_summary = f"{exec_type} {exec_price}" if exec_type else "No Exec"
            
            print(f"{step_count:<4} {env.best_bid:.2f}/{env.best_ask:.2f} {should_execute_str:<25} "
                  f"{maker_change:>7.2f} {taker_change:>7.2f} {execution_summary:<12} ${cash_change:>8.2f}")
            
            if maker_change > 0 or taker_change > 0:
                execution_log.append({
                    'step': step_count,
                    'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}",
                    'maker_vol': maker_change,
                    'taker_vol': taker_change,
                    'cash_change': cash_change,
                    'exec_type': exec_type,
                    'should_execute': should_execute_str
                })
        
        # Final analysis
        print(f"\n📊 RAPID OSCILLATION ANALYSIS:")
        print(f"Total executions: {len(execution_log)}")
        print(f"Final maker volume: {info.get('maker_volume', 0):.2f}")
        print(f"Final taker volume: {info.get('taker_volume', 0):.2f}")
        print(f"Final cash change: ${env.cash - 100000:+.2f}")
        
        if execution_log:
            print(f"\n✅ EXECUTIONS FOUND:")
            for exec_data in execution_log:
                print(f"  Step {exec_data['step']}: {exec_data['exec_type']} - "
                      f"M+{exec_data['maker_vol']:.2f}, T+{exec_data['taker_vol']:.2f}, "
                      f"Cash ${exec_data['cash_change']:+.2f}")
                print(f"    Market: {exec_data['market']}, Expected: {exec_data['should_execute']}")
        else:
            print(f"\n❌ NO EXECUTIONS - INVESTIGATING WHY:")
            print(f"   Possible reasons:")
            print(f"   1. Orders not placed at optimal levels for oscillation")
            print(f"   2. Market moves don't cross order prices")
            print(f"   3. Timing/latency issues")
            print(f"   4. Order placement strategy needs adjustment")
        
        # Test with different order placement strategy
        print(f"\n🔧 TESTING WITH AGGRESSIVE ORDER PLACEMENT:")
        
        # Reset and try aggressive orders
        obs, info = env.reset()
        
        # Place aggressive orders that should definitely execute
        action = [-0.5, 0.5, 1.0, 1.0, -1.0, 0.0]  # More aggressive
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        print(f"Aggressive orders placed:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Step through again
        aggressive_executions = 0
        for step in range(3, 8):
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            
            if post_maker > pre_maker or post_taker > pre_taker:
                aggressive_executions += 1
                print(f"  Step {step}: Execution! Market {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        print(f"Aggressive strategy executions: {aggressive_executions}")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def debug_taker_execution_pricing():
    """Debug exactly what price taker executions use."""
    print(f"\n" + "=" * 80)
    print("🎯 DEBUGGING TAKER EXECUTION PRICING")
    print("=" * 80)
    
    # Create simple test scenario where we know taker execution will happen
    data = []
    for i in range(6):
        row = {'datetime': i}
        if i < 2:
            # Initial market for order placement
            base_bid, base_ask = 1800.00, 1800.10
        else:
            # Market moves to trigger taker execution
            base_bid, base_ask = 1800.25, 1800.35  # Market moves up
            
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
        config["episode_length"] = 6
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place sell order at 1800.05 (below initial ask, should become taker when market moves up)
        action = [-1.0, 0.25, 0.0, 0.8, -1.0, 0.0]  # Sell order slightly below market
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
        
        print(f"\nOrder placed:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Now step to trigger taker execution
        print(f"\n📈 STEPPING TO TRIGGER TAKER EXECUTION:")
        
        for step in range(2, 5):
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            
            print(f"\nStep {step}:")
            print(f"  Before: Market {env.best_bid:.2f}/{env.best_ask:.2f}, Cash ${pre_cash:.2f}")
            
            # Check what should execute
            for order in env.active_orders:
                if not order['is_buy'] and env.best_bid >= order['price']:
                    print(f"  🎯 SELL order @{order['price']:.2f} should execute as TAKER at bid {env.best_bid:.2f}")
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.0])
            
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            
            cash_change = post_cash - pre_cash
            taker_change = post_taker - pre_taker
            
            print(f"  After:  Market {env.best_bid:.2f}/{env.best_ask:.2f}, Cash ${post_cash:.2f}")
            print(f"  Changes: Taker +{taker_change:.2f}, Cash ${cash_change:+.2f}")
            
            if taker_change > 0:
                # Calculate what price was used
                implied_price = abs(cash_change) / taker_change if taker_change > 0 else 0
                print(f"  💰 TAKER EXECUTION DETECTED!")
                print(f"     Taker volume: {taker_change:.2f}")
                print(f"     Cash change: ${cash_change:+.2f}")
                print(f"     Implied execution price: ${implied_price:.2f}")
                print(f"     Market bid when executed: ${env.best_bid:.2f}")
                print(f"     Market ask when executed: ${env.best_ask:.2f}")
                
                # Determine if execution was at bid or ask
                if abs(implied_price - env.best_bid) < 0.01:
                    print(f"     ✅ Executed at BID price (selling into bid)")
                elif abs(implied_price - env.best_ask) < 0.01:
                    print(f"     ✅ Executed at ASK price (buying from ask)")
                else:
                    print(f"     ⚠️  Execution price doesn't match current bid/ask")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    debug_rapid_oscillation()
    debug_taker_execution_pricing()