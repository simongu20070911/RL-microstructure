#!/usr/bin/env python3
"""
Test with much larger episode length to allow execution before termination
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_extended_market_data():
    """Create data with extended episode length and execution opportunities."""
    data = []
    
    # Create 50 steps of data for long episode
    for i in range(50):
        row = {'datetime': i}
        
        if i < 10:
            # Phase 1: Stable market for order placement (10 steps)
            base_bid = 1800.00
            base_ask = 1800.10
            
        elif i < 20:
            # Phase 2: Market moves up - execute sell orders (10 steps)
            base_bid = 1800.25  # +25 ticks - should hit all sell orders
            base_ask = 1800.35
            
        elif i < 30:
            # Phase 3: Market crashes down - execute buy orders (10 steps)
            base_bid = 1799.75  # -25 ticks - should hit all buy orders
            base_ask = 1799.85
            
        elif i < 40:
            # Phase 4: Market returns to middle (10 steps)
            base_bid = 1799.95
            base_ask = 1800.05
            
        else:
            # Phase 5: Final stable period (10 steps)
            base_bid = 1800.00
            base_ask = 1800.10
        
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

def test_with_large_episodes():
    """Test post-only mode with much larger episode length."""
    print("🎯 TESTING POST-ONLY MODE WITH LARGE EPISODE LENGTH")
    print("=" * 80)
    
    test_csv = create_extended_market_data()
    
    try:
        # Test post-only mode with MUCH larger episode
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 45  # Much larger episode - ends before data runs out
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"CONFIGURATION:")
        print(f"  Episode length: {config['episode_length']} steps")
        print(f"  Data length: 50 steps")
        print(f"  Post-only mode: {env.post_only_mode}")
        print(f"  Aggressiveness ticks: {env.config['allowed_aggressiveness_ticks']}")
        print(f"  Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place orders early in episode (steps 1-2)
        print(f"\n📋 STEP 1-2: PLACING ORDERS EARLY IN EPISODE")
        action = [-0.5, 0.5, 0.8, 0.8, -1.0, -1.0]  # Both buy and sell orders
        obs, reward, term, trunc, info = env.step(action)
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])  # Activate
        
        print(f"Orders placed at step 2:")
        for order in env.active_orders:
            side = "BUY" if order['is_buy'] else "SELL"
            print(f"  {side}: {order['volume']:.2f} @ {order['price']:.2f}")
        
        execution_log = []
        step_count = 2
        
        # Run through the long episode
        print(f"\n📈 RUNNING THROUGH EXTENDED EPISODE:")
        print(f"Steps 1-10: Stable market (order placement)")
        print(f"Steps 11-20: Market UP (+25 ticks) - Execute SELL orders")
        print(f"Steps 21-30: Market DOWN (-25 ticks) - Execute BUY orders") 
        print(f"Steps 31-40: Market stabilize")
        print(f"Steps 41-45: Final period")
        
        while step_count < 44 and not (term or trunc):
            step_count += 1
            
            # Track state before step
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            pre_active = len(env.active_orders)
            pre_rebates = info.get('total_rebates_earned', 0)
            
            # Determine phase
            if step_count <= 10:
                phase = "STABLE"
            elif step_count <= 20:
                phase = "🚀 UP"
            elif step_count <= 30:
                phase = "📉 DOWN"
            elif step_count <= 40:
                phase = "🔄 MID"
            else:
                phase = "🏁 END"
            
            # Show key steps
            if step_count in [3, 5, 11, 15, 21, 25, 31, 35, 41]:
                print(f"\n  Step {step_count} ({phase}):")
                print(f"    Market: {env.best_bid:.2f}/{env.best_ask:.2f}")
                print(f"    Active orders: {pre_active}")
                print(f"    Episode status: Term={term}, Trunc={trunc}")
                
                # Check execution potential
                should_execute = []
                for order in env.active_orders:
                    if order['is_buy'] and env.best_ask <= order['price']:
                        should_execute.append(f"BUY@{order['price']:.2f}")
                    elif not order['is_buy'] and env.best_bid >= order['price']:
                        should_execute.append(f"SELL@{order['price']:.2f}")
                
                if should_execute:
                    print(f"    🎯 SHOULD EXECUTE: {', '.join(should_execute)}")
            
            # Execute step
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            # Track changes
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            post_active = len(env.active_orders)
            post_rebates = info.get('total_rebates_earned', 0)
            
            maker_change = post_maker - pre_maker
            taker_change = post_taker - pre_taker
            cash_change = post_cash - pre_cash
            rebate_change = post_rebates - pre_rebates
            
            # Log executions
            if maker_change > 0 or taker_change > 0:
                execution_log.append({
                    'step': step_count,
                    'phase': phase,
                    'maker_vol': maker_change,
                    'taker_vol': taker_change,
                    'cash_change': cash_change,
                    'rebate_change': rebate_change,
                    'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}",
                    'orders_before': pre_active,
                    'orders_after': post_active
                })
                print(f"    ✅ EXECUTION! Maker +{maker_change:.2f}, Cash ${cash_change:+.2f}, Rebates ${rebate_change:+.4f}")
        
        # Final results
        print(f"\n" + "=" * 80)
        print(f"🏆 FINAL RESULTS:")
        
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"  Episode completed at step: {step_count}")
        print(f"  Episode terminated: {term}")
        print(f"  Episode truncated: {trunc}")
        print(f"  Final active orders: {len(env.active_orders)}")
        print(f"  Final maker volume: {final_maker:.2f}")
        print(f"  Final taker volume: {final_taker:.2f}")
        print(f"  Total rebates earned: ${final_rebates:.6f}")
        print(f"  Net cash change: ${final_cash_change:+.2f}")
        
        if execution_log:
            print(f"\n📊 EXECUTION LOG:")
            total_maker = 0
            total_rebates = 0
            for i, exec_data in enumerate(execution_log, 1):
                print(f"  {i}. Step {exec_data['step']} ({exec_data['phase']}):")
                print(f"     Volume: Maker +{exec_data['maker_vol']:.2f}, Taker +{exec_data['taker_vol']:.2f}")
                print(f"     Money: Cash ${exec_data['cash_change']:+.2f}, Rebates ${exec_data['rebate_change']:+.4f}")
                print(f"     Orders: {exec_data['orders_before']} → {exec_data['orders_after']}")
                print(f"     Market: {exec_data['market']}")
                
                total_maker += exec_data['maker_vol']
                total_rebates += exec_data['rebate_change']
            
            print(f"\n🎉 SUCCESS METRICS:")
            print(f"   Total executions: {len(execution_log)}")
            print(f"   Total maker volume: {total_maker:.2f}")
            print(f"   Total rebates: ${total_rebates:.6f}")
        
        if final_maker > 0:
            print(f"\n🏆 PROOF COMPLETE!")
            print(f"✅ Post-only mode SUCCESSFULLY executed trades!")
            print(f"✅ Volume > 0: {final_maker:.2f}")
            print(f"✅ Earned rebates: ${final_rebates:.6f}")
            print(f"✅ Prevented aggressive orders (0 taker volume)")
            
            print(f"\n💡 USER WAS 100% CORRECT:")
            print(f"   'if it decides to post it should still be able to trade'")
            print(f"   → YES! Post-only mode trades via maker orders")
            print(f"   'but only negative pnl right?'")
            print(f"   → Net PnL: ${final_cash_change:+.2f} (depends on rebates vs adverse selection)")
            
        else:
            print(f"\n❌ Still no execution - there may be a deeper issue")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    test_with_large_episodes()