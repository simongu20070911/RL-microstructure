#!/usr/bin/env python3
"""
Debug why post-only mode shows 0.0 volume
The user questioned: "if it decides to post it should still be able to trade, but only negative pnl right?"
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_maker_friendly_data():
    """Create data that should allow maker orders to execute."""
    data = []
    
    for i in range(10):
        row = {'datetime': i}
        
        if i < 5:
            # Phase 1: Stable market for order placement
            base_bid = 1799.98
            base_ask = 1800.02
        else:
            # Phase 2: Tighter market - spread narrows so passive orders might get hit
            base_bid = 1799.99  # Bid moves up
            base_ask = 1800.01  # Ask moves down - should hit passive sell orders
        
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

def debug_post_only():
    """Debug post-only mode behavior."""
    print("🔍 DEBUGGING POST-ONLY MODE TRADING")
    
    test_csv = create_maker_friendly_data()
    
    try:
        # Test post-only mode with rebated structure
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 10
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment configuration:")
        print(f"  Fee structure: {env.fee_structure}")
        print(f"  Post-only mode: {env.post_only_mode}")
        print(f"  Is rebated: {env.is_rebated}")
        print(f"  Allowed aggressiveness ticks: {env.config['allowed_aggressiveness_ticks']}")
        print(f"  Price offset ticks: {env.config['price_offset_ticks']}")
        print(f"  Transaction cost: {env.config['transaction_cost_long']:.6f}")
        
        print(f"\nInitial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Try to place passive orders that should become makers
        print(f"\n=== ATTEMPTING TO PLACE PASSIVE ORDERS ===")
        
        # Place a passive sell order (should be above best ask)
        # Action: sell_offset_action = 1.0 (maximum positive offset = most passive)
        #         sell_size_signal = 0.8 (large volume)
        action = [-1.0, 1.0, 0.0, 0.8, -1.0, -1.0]  # Only passive sell order
        
        print(f"Action: sell_offset_action=1.0 (most passive), sell_size=0.8")
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"After placement:")
        print(f"  Pending orders: {len(env.pending_orders)}")
        print(f"  Active orders: {len(env.active_orders)}")
        if env.pending_orders:
            for order in env.pending_orders:
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"    Pending: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Activate the order
        print(f"\n=== ACTIVATING ORDERS ===")
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Do nothing
        
        print(f"After activation:")
        print(f"  Pending orders: {len(env.pending_orders)}")
        print(f"  Active orders: {len(env.active_orders)}")
        if env.active_orders:
            for order in env.active_orders:
                side = "BUY" if order['is_buy'] else "SELL"
                print(f"    Active: {side} {order['volume']:.2f} @ {order['price']:.2f}")
        
        # Now step through market data to see if orders can execute
        print(f"\n=== STEPPING THROUGH MARKET DATA ===")
        
        step_count = 0
        while step_count < 8 and not (term or trunc):
            step_count += 1
            
            pre_maker = info.get('maker_volume', 0)
            pre_cash = env.cash
            
            print(f"\nStep {step_count}:")
            print(f"  Before: BBO {env.best_bid:.2f}/{env.best_ask:.2f}, Cash ${pre_cash:.2f}, Maker vol {pre_maker:.2f}")
            
            # Show active orders vs market
            if env.active_orders:
                for order in env.active_orders:
                    side = "BUY" if order['is_buy'] else "SELL"
                    if order['is_buy']:
                        executable = order['price'] >= env.best_ask
                        status = "EXECUTABLE" if executable else "PASSIVE"
                    else:
                        executable = order['price'] <= env.best_bid
                        status = "EXECUTABLE" if executable else "PASSIVE"
                    print(f"    {side} order @ {order['price']:.2f} vs market: {status}")
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Do nothing
            
            post_maker = info.get('maker_volume', 0)
            post_cash = env.cash
            
            print(f"  After:  BBO {env.best_bid:.2f}/{env.best_ask:.2f}, Cash ${post_cash:.2f}, Maker vol {post_maker:.2f}")
            
            if post_maker > pre_maker:
                print(f"  🎉 MAKER EXECUTION! Volume increased by {post_maker - pre_maker:.2f}")
            
            if abs(post_cash - pre_cash) > 0.01:
                print(f"  💰 CASH CHANGE! ${post_cash - pre_cash:+.2f}")
        
        print(f"\n=== FINAL RESULTS ===")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"Final Maker Volume: {final_maker:.2f}")
        print(f"Final Taker Volume: {final_taker:.2f}")
        print(f"Total Rebates Earned: ${final_rebates:.4f}")
        print(f"Cash Change: ${final_cash_change:+.2f}")
        print(f"Active orders remaining: {len(env.active_orders)}")
        
        if final_maker == 0:
            print(f"\n❌ ANALYSIS: Post-only mode shows 0 volume")
            print(f"Possible reasons:")
            print(f"1. Orders are too passive and never get hit by market movement")
            print(f"2. Post-only mode prevents aggressive orders completely")
            print(f"3. Market data doesn't provide execution opportunities")
            print(f"4. Order placement logic has issues in post-only mode")
        else:
            print(f"\n✅ ANALYSIS: Post-only mode DOES allow trading with volume {final_maker:.2f}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_post_only()