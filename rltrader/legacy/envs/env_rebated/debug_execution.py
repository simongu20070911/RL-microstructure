#!/usr/bin/env python3
"""
Debug why orders aren't executing in the unified environment
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_simple_data():
    """Create very simple data for execution testing."""
    data = []
    
    for i in range(10):
        row = {'datetime': i}
        # Simple, stable order book with proper spread
        base_bid = 1799.99
        base_ask = 1800.01
        
        for level in range(1, 11):
            row[f'bid{level}'] = base_bid - (level-1) * 0.01
            row[f'bidqty{level}'] = 100.0  # Large quantity
            row[f'ask{level}'] = base_ask + (level-1) * 0.01  
            row[f'askqty{level}'] = 100.0  # Large quantity
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def debug_order_lifecycle():
    """Debug the complete order lifecycle from placement to execution."""
    print("🔍 DEBUGGING ORDER LIFECYCLE")
    
    test_csv = create_simple_data()
    
    try:
        config = get_unified_config("rebate_4bps", False)
        config["csv_path"] = test_csv
        config["max_steps"] = 10
        config["episode_length"] = 8
        config["latency_steps_long"] = 1  # Shorter latency
        config["latency_steps_short"] = 1
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        print(f"Initial spread: {env.spread:.4f}")
        print(f"Latency: Long={config['latency_steps_long']}, Short={config['latency_steps_short']}")
        
        for step in range(8):
            print(f"\n=== STEP {step} ===")
            print(f"Current step in data: {env.current_step}")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Place very aggressive orders to force execution
            if step < 3:
                # Place market-taking orders
                action = [0.8, -0.8, 0.5, 0.5, -1.0, -1.0]  # Aggressive
                print(f"Action: AGGRESSIVE (should cross spread)")
            else:
                # Place passive orders
                action = [0.1, -0.1, 0.5, 0.5, -1.0, -1.0]  # Passive
                print(f"Action: PASSIVE (should not cross spread)")
            
            # Before step
            pre_pending = len(env.pending_orders)
            pre_active = len(env.active_orders)
            pre_cash = env.cash
            
            obs, reward, term, trunc, info = env.step(action)
            
            # After step
            post_pending = len(env.pending_orders)
            post_active = len(env.active_orders)
            post_cash = env.cash
            
            print(f"Orders: Pending {pre_pending}->{post_pending}, Active {pre_active}->{post_active}")
            print(f"Cash: ${pre_cash:.2f} -> ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
            print(f"Volume executed: {env.last_executed_volume:.4f}")
            
            # Examine active orders in detail
            if env.active_orders:
                print(f"Active orders details:")
                for i, order in enumerate(env.active_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    price = order['price']
                    volume = order['volume']
                    is_taker = order.get('is_taker_at_activation', 'N/A')
                    
                    # Check if order should execute
                    if order['is_buy'] and price >= env.best_ask:
                        should_execute = f"YES (buy {price:.2f} >= ask {env.best_ask:.2f})"
                    elif not order['is_buy'] and price <= env.best_bid:
                        should_execute = f"YES (sell {price:.2f} <= bid {env.best_bid:.2f})"
                    else:
                        should_execute = "NO (passive)"
                        
                    print(f"  {i}: {side} {volume:.3f} @ {price:.2f}, taker={is_taker}, should_exec={should_execute}")
            
            # Examine pending orders
            if env.pending_orders:
                print(f"Pending orders details:")
                for i, order in enumerate(env.pending_orders):
                    side = "BUY" if order['is_buy'] else "SELL"
                    target_step = order['target_step']
                    current_step = env.current_step
                    will_activate = "YES" if current_step >= target_step else f"NO (need step {target_step})"
                    print(f"  {i}: {side} {order['volume']:.3f} @ {order['price']:.2f}, activates={will_activate}")
            
            if term or trunc:
                print(f"Episode ended: term={term}, trunc={trunc}")
                break
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def debug_order_book_matching():
    """Debug the order book matching logic specifically."""
    print("\n🔍 DEBUGGING ORDER BOOK MATCHING")
    
    test_csv = create_simple_data()
    
    try:
        config = get_unified_config("baseline", False)  # Use baseline to focus on execution
        config["csv_path"] = test_csv
        config["max_steps"] = 10
        config["episode_length"] = 5
        config["latency_steps_long"] = 0  # No latency
        config["latency_steps_short"] = 0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Order book levels:")
        print(f"Bids: {env.bids[:3, :]}")  # Top 3 levels
        print(f"Asks: {env.asks[:3, :]}")  # Top 3 levels
        
        # Place a single very aggressive buy order
        print(f"\nPlacing aggressive buy order...")
        # Manually place order to bypass latency
        order = {
            "id": 999,
            "price": env.best_ask + 0.10,  # Way above ask
            "volume": 1.0,
            "initial_volume": 1.0,
            "is_buy": True,
            "timestamp_placed": env.current_step,
            "target_step": env.current_step,
            "is_taker_at_activation": True
        }
        
        env.active_orders.append(order)
        print(f"Added order: BUY 1.0 @ {order['price']:.2f}")
        print(f"Best ask: {env.best_ask:.2f}")
        print(f"Should execute: {order['price'] >= env.best_ask}")
        
        # Manually call execution
        print(f"\nCalling _execute_orders()...")
        pre_cash = env.cash
        realized_pnl = env._execute_orders()
        post_cash = env.cash
        
        print(f"Execution result:")
        print(f"  Realized PnL: {realized_pnl:.6f}")
        print(f"  Cash change: ${post_cash - pre_cash:+.6f}")
        print(f"  Volume executed: {env.last_executed_volume:.4f}")
        print(f"  Remaining active orders: {len(env.active_orders)}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_order_lifecycle()
    debug_order_book_matching()