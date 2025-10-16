#!/usr/bin/env python3
"""
Debug the exact taker status assignment and execution logic
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_clear_execution_data():
    """Create data with very clear execution opportunity."""
    data = []
    
    for i in range(6):
        row = {'datetime': i}
        
        if i < 3:
            # Phase 1: Order placement market
            base_bid = 1799.90
            base_ask = 1800.10  # Wide spread for clear passive orders
        else:
            # Phase 2: Market jumps WAY up - should definitely execute sell orders
            base_bid = 1800.50  # WAY above any sell order we'll place
            base_ask = 1800.60
        
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

def debug_taker_status():
    """Debug taker status assignment and execution."""
    print("🔍 DEBUGGING TAKER STATUS AND EXECUTION")
    
    test_csv = create_clear_execution_data()
    
    try:
        # Test both normal and post-only modes
        for post_only in [False, True]:
            mode_name = "POST-ONLY" if post_only else "NORMAL"
            print(f"\n{'='*20} {mode_name} MODE {'='*20}")
            
            config = get_unified_config("rebate_6bps", post_only)
            config["csv_path"] = test_csv
            config["episode_length"] = 6
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 0.5
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            print(f"Environment: Post-only={env.post_only_mode}, Aggressiveness ticks={env.config['allowed_aggressiveness_ticks']}")
            print(f"Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            # Place a sell order that should definitely be passive
            print(f"\n--- PLACING PASSIVE SELL ORDER ---")
            # sell_offset_action = 1.0 means most passive (highest price)
            action = [-1.0, 1.0, 0.0, 0.8, -1.0, -1.0]  # Only sell order, most passive
            
            obs, reward, term, trunc, info = env.step(action)
            
            if env.pending_orders:
                sell_order = env.pending_orders[0]
                print(f"Pending sell order: {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
                print(f"Current ask: {env.best_ask:.2f}")
                print(f"Order price vs ask: {sell_order['price']:.2f} vs {env.best_ask:.2f}")
                print(f"Order should be passive: {sell_order['price'] > env.best_ask}")
            
            # Activate the order
            print(f"\n--- ACTIVATING ORDER ---")
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            if env.active_orders:
                sell_order = env.active_orders[0]
                is_taker = sell_order.get('is_taker_at_activation', 'N/A')
                print(f"Active sell order: {sell_order['volume']:.2f} @ {sell_order['price']:.2f}")
                print(f"Taker at activation: {is_taker}")
                print(f"Market at activation: {env.best_bid:.2f}/{env.best_ask:.2f}")
                
                # Manual taker check
                manual_taker = sell_order['price'] <= env.best_bid + 1e-9
                print(f"Manual taker check: sell_price ({sell_order['price']:.2f}) <= bid ({env.best_bid:.2f}) = {manual_taker}")
            
            # Step to execution market
            print(f"\n--- STEPPING TO EXECUTION MARKET ---")
            pre_cash = env.cash
            pre_maker = info.get('maker_volume', 0)
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            post_cash = env.cash
            post_maker = info.get('maker_volume', 0)
            
            print(f"Market after step: {env.best_bid:.2f}/{env.best_ask:.2f}")
            print(f"Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
            print(f"Maker volume: {pre_maker:.2f} → {post_maker:.2f} (Δ{post_maker-pre_maker:+.2f})")
            print(f"Active orders remaining: {len(env.active_orders)}")
            
            if env.active_orders:
                for order in env.active_orders:
                    print(f"  Remaining order: {order['volume']:.2f} @ {order['price']:.2f}")
                    print(f"  Should execute: sell_price ({order['price']:.2f}) <= bid ({env.best_bid:.2f}) = {order['price'] <= env.best_bid}")
            
            if post_maker > pre_maker:
                print(f"✅ EXECUTION SUCCESS! Volume increased by {post_maker - pre_maker:.2f}")
            else:
                print(f"❌ EXECUTION FAILED! No volume increase")
            
            env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    debug_taker_status()