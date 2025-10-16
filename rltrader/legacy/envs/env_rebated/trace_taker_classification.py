#!/usr/bin/env python3
"""
Trace exactly how orders get classified as taker vs maker
"""

import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_moving_market_data():
    """Create market data that moves to trigger executions."""
    data = []
    
    for i in range(25):
        row = {'datetime': i}
        
        if i < 8:
            # Stable market
            base_bid = 1799.99
            base_ask = 1800.01
        elif i < 15:
            # Market moves up to hit passive buy orders
            base_bid = 1799.99 + (i-8) * 0.01
            base_ask = 1800.01 + (i-8) * 0.01
        else:
            # Market stabilizes higher
            base_bid = 1800.06
            base_ask = 1800.08
        
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

def trace_order_classification():
    """Trace order classification in detail."""
    print("🔍 TRACING ORDER CLASSIFICATION")
    
    test_csv = create_moving_market_data()
    
    try:
        config = get_unified_config("rebate_6bps", True)  # Post-only
        config["csv_path"] = test_csv
        config["episode_length"] = 20
        config["latency_steps_long"] = 1
        config["latency_steps_short"] = 1
        config["max_order_volume"] = 2.0
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment: Post-only={env.post_only_mode}, Aggressiveness={env.config['allowed_aggressiveness_ticks']}")
        print(f"Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        order_history = []
        
        for step in range(15):
            print(f"\n=== STEP {step} ===")
            print(f"BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            
            if step < 6:
                # Place passive orders
                action = [0.0, 0.0, 0.7, 0.7, -1.0, -1.0]
                strategy = "PASSIVE AT BBO"
            else:
                # Do nothing, let market move
                action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]
                strategy = "DO NOTHING"
            
            print(f"Strategy: {strategy}")
            
            # Track orders before step
            active_before = [(o['id'], o['price'], o['is_buy'], o.get('is_taker_at_activation', 'N/A')) for o in env.active_orders]
            pending_before = [(o['id'], o['price'], o['is_buy']) for o in env.pending_orders]
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            
            # Track orders after step
            active_after = [(o['id'], o['price'], o['is_buy'], o.get('is_taker_at_activation', 'N/A')) for o in env.active_orders]
            pending_after = [(o['id'], o['price'], o['is_buy']) for o in env.pending_orders]
            
            print(f"Orders Before: Active={len(active_before)}, Pending={len(pending_before)}")
            print(f"Orders After:  Active={len(active_after)}, Pending={len(pending_after)}")
            
            # Check for volume changes
            maker_change = post_maker - pre_maker
            taker_change = post_taker - pre_taker
            
            if maker_change > 0:
                print(f"✅ MAKER EXECUTION: +{maker_change:.2f} volume")
            if taker_change > 0:
                print(f"❌ TAKER EXECUTION: +{taker_change:.2f} volume (SHOULD NOT HAPPEN IN POST-ONLY)")
            
            # Track activated orders
            new_actives = [o for o in active_after if o not in active_before]
            for order_id, price, is_buy, taker_flag in new_actives:
                side = "BUY" if is_buy else "SELL"
                print(f"  ACTIVATED: Order {order_id} {side} @ {price:.2f}, Taker@Activation: {taker_flag}")
                
                # Check if this should be taker in post-only (it shouldn't)
                if taker_flag and env.post_only_mode:
                    print(f"    ❌ ERROR: Order marked as taker in post-only mode!")
            
            # Track executed orders (disappeared from active)
            executed_orders = [o for o in active_before if o not in active_after]
            for order_id, price, is_buy, taker_flag in executed_orders:
                side = "BUY" if is_buy else "SELL"
                classification = "TAKER" if taker_flag else "MAKER"
                print(f"  EXECUTED: Order {order_id} {side} @ {price:.2f} as {classification}")
                
                # This tells us exactly which orders are being classified as takers
                if taker_flag:
                    print(f"    ❌ This order contributed to TAKER volume!")
                    order_history.append({
                        'step': step,
                        'order_id': order_id,
                        'price': price,
                        'is_buy': is_buy,
                        'classification': 'TAKER',
                        'bbo_at_execution': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                    })
                else:
                    print(f"    ✅ This order contributed to MAKER volume")
                    order_history.append({
                        'step': step,
                        'order_id': order_id,
                        'price': price,
                        'is_buy': is_buy,
                        'classification': 'MAKER',
                        'bbo_at_execution': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                    })
            
            if term or trunc:
                break
        
        print(f"\n=== EXECUTION HISTORY ===")
        for execution in order_history:
            print(f"Step {execution['step']}: Order {execution['order_id']} "
                  f"{'BUY' if execution['is_buy'] else 'SELL'} @ {execution['price']:.2f} "
                  f"classified as {execution['classification']} (BBO: {execution['bbo_at_execution']})")
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Total Maker Volume: {info.get('maker_volume', 0):.2f}")
        print(f"Total Taker Volume: {info.get('taker_volume', 0):.2f}")
        
        taker_count = sum(1 for e in order_history if e['classification'] == 'TAKER')
        if taker_count > 0:
            print(f"❌ PROBLEM: {taker_count} orders classified as TAKER in post-only mode")
        else:
            print(f"✅ SUCCESS: No taker orders in post-only mode")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    trace_order_classification()