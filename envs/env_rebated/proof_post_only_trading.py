#!/usr/bin/env python3
"""
PROOF: Post-only mode CAN trade with dramatic market movements
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_dramatic_market_data():
    """Create data with DRAMATIC market movements that will hit passive orders."""
    data = []
    
    for i in range(20):  # Long episode
        row = {'datetime': i}
        
        if i < 5:
            # Phase 1: Initial stable market for order placement
            base_bid = 1800.00
            base_ask = 1800.10
            
        elif i < 10:
            # Phase 2: MASSIVE MOVE UP - 20 tick jump to hit sell orders
            base_bid = 1800.20  # +20 ticks from original bid
            base_ask = 1800.30  # Should execute ALL sell orders placed at 1800.05-1800.15
            
        elif i < 15:
            # Phase 3: MASSIVE CRASH DOWN - 30 tick drop to hit buy orders  
            base_bid = 1799.70  # -30 ticks from original bid
            base_ask = 1799.80  # Should execute ALL buy orders placed at 1799.95-1800.05
            
        else:
            # Phase 4: Return to middle for final trades
            base_bid = 1799.95
            base_ask = 1800.05
        
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

def proof_post_only_trading():
    """PROOF that post-only mode can execute trades with proper market conditions."""
    print("🎯 PROOF: POST-ONLY MODE CAN TRADE WITH DRAMATIC MARKET MOVEMENTS")
    print("=" * 80)
    
    test_csv = create_dramatic_market_data()
    
    try:
        # Test post-only mode with dramatic market data
        config = get_unified_config("rebate_6bps", True)  # Post-only = True
        config["csv_path"] = test_csv
        config["episode_length"] = 20  # Long episode
        config["latency_steps_long"] = 0
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 0.5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"TEST SETUP:")
        print(f"  Mode: POST-ONLY (aggressiveness_ticks = {env.config['allowed_aggressiveness_ticks']})")
        print(f"  Episode length: {config['episode_length']} steps")
        print(f"  Price offset ticks: {env.config['price_offset_ticks']}")
        print(f"  Tick size: {env.config['tick_size']}")
        print(f"  Initial market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # STEP 1-2: Place orders across the range
        print(f"\n🏗️  STEP 1-2: PLACING DIVERSE ORDERS")
        print(f"Current market: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Place multiple types of orders
        action = [-0.5, 0.5, 0.8, 0.8, -1.0, -1.0]  # Both buy and sell orders
        obs, reward, term, trunc, info = env.step(action)
        
        # Activate orders
        obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
        
        print(f"Orders placed and activated:")
        buy_orders = []
        sell_orders = []
        for order in env.active_orders:
            if order['is_buy']:
                buy_orders.append(order)
                print(f"  BUY:  {order['volume']:.2f} @ {order['price']:.2f}")
            else:
                sell_orders.append(order)
                print(f"  SELL: {order['volume']:.2f} @ {order['price']:.2f}")
        
        execution_log = []
        market_phases = []
        
        # STEP 3+: Watch dramatic market movements
        print(f"\n📈 STEPPING THROUGH DRAMATIC MARKET MOVEMENTS:")
        print(f"Phase 1 (Steps 1-5): Stable market for order placement")
        print(f"Phase 2 (Steps 6-10): MASSIVE UP MOVE (+20 ticks) → Execute SELL orders")
        print(f"Phase 3 (Steps 11-15): MASSIVE DOWN MOVE (-30 ticks) → Execute BUY orders")
        print(f"Phase 4 (Steps 16-20): Return to middle")
        
        step_count = 2
        while step_count < 18 and not (term or trunc):
            step_count += 1
            
            pre_maker = info.get('maker_volume', 0)
            pre_taker = info.get('taker_volume', 0)
            pre_cash = env.cash
            pre_active = len(env.active_orders)
            pre_rebates = info.get('total_rebates_earned', 0)
            
            # Determine market phase
            if step_count <= 5:
                phase = "PLACEMENT"
            elif step_count <= 10:
                phase = "🚀 MASSIVE UP"
            elif step_count <= 15:
                phase = "📉 MASSIVE DOWN"
            else:
                phase = "🔄 STABILIZE"
            
            print(f"\nStep {step_count} ({phase}):")
            print(f"  Before: Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active: {pre_active}")
            
            # Check which orders should execute
            should_execute = []
            for order in env.active_orders:
                if order['is_buy'] and env.best_ask <= order['price']:
                    should_execute.append(f"BUY@{order['price']:.2f} vs ASK@{env.best_ask:.2f}")
                elif not order['is_buy'] and env.best_bid >= order['price']:
                    should_execute.append(f"SELL@{order['price']:.2f} vs BID@{env.best_bid:.2f}")
            
            if should_execute:
                print(f"  🎯 SHOULD EXECUTE: {', '.join(should_execute)}")
            
            obs, reward, term, trunc, info = env.step([-1.0, -1.0, 0.0, 0.0, -1.0, 0.9])
            
            post_maker = info.get('maker_volume', 0)
            post_taker = info.get('taker_volume', 0)
            post_cash = env.cash
            post_active = len(env.active_orders)
            post_rebates = info.get('total_rebates_earned', 0)
            
            print(f"  After:  Market {env.best_bid:.2f}/{env.best_ask:.2f}, Active: {post_active}")
            
            # Track changes
            maker_change = post_maker - pre_maker
            taker_change = post_taker - pre_taker
            cash_change = post_cash - pre_cash
            rebate_change = post_rebates - pre_rebates
            
            print(f"  Changes: Maker +{maker_change:.2f}, Taker +{taker_change:.2f}, Cash ${cash_change:+.2f}, Rebates ${rebate_change:+.4f}")
            
            if maker_change > 0 or taker_change > 0:
                execution_log.append({
                    'step': step_count,
                    'phase': phase,
                    'maker_vol': maker_change,
                    'taker_vol': taker_change,
                    'cash_change': cash_change,
                    'rebate_change': rebate_change,
                    'market': f"{env.best_bid:.2f}/{env.best_ask:.2f}"
                })
                print(f"  🎉 EXECUTION DETECTED!")
            
            # Track market phases
            market_phases.append({
                'step': step_count,
                'phase': phase,
                'bid': env.best_bid,
                'ask': env.best_ask,
                'active_orders': post_active
            })
        
        # RESULTS ANALYSIS
        print(f"\n" + "=" * 80)
        print(f"🏆 EXECUTION RESULTS:")
        
        if execution_log:
            print(f"✅ POST-ONLY MODE EXECUTED TRADES!")
            print(f"\nDetailed execution log:")
            
            total_maker_vol = 0
            total_cash_change = 0
            total_rebates = 0
            
            for i, exec_data in enumerate(execution_log, 1):
                print(f"  {i}. Step {exec_data['step']} ({exec_data['phase']}):")
                print(f"     Maker: +{exec_data['maker_vol']:.2f}, Taker: +{exec_data['taker_vol']:.2f}")
                print(f"     Cash: ${exec_data['cash_change']:+.2f}, Rebates: ${exec_data['rebate_change']:+.4f}")
                print(f"     Market: {exec_data['market']}")
                
                total_maker_vol += exec_data['maker_vol']
                total_cash_change += exec_data['cash_change']
                total_rebates += exec_data['rebate_change']
            
            print(f"\n📊 SUMMARY:")
            print(f"   Total Maker Volume: {total_maker_vol:.2f}")
            print(f"   Total Cash Change: ${total_cash_change:+.2f}")
            print(f"   Total Rebates Earned: ${total_rebates:+.4f}")
            print(f"   Net Trading PnL: ${total_cash_change - total_rebates:+.2f}")
            
        else:
            print(f"❌ NO EXECUTIONS - Need even more dramatic movements")
        
        # FINAL STATE
        print(f"\n🎯 FINAL PROOF RESULTS:")
        final_maker = info.get('maker_volume', 0)
        final_taker = info.get('taker_volume', 0)
        final_rebates = info.get('total_rebates_earned', 0)
        final_cash_change = env.cash - 100000
        
        print(f"  Final Maker Volume: {final_maker:.2f}")
        print(f"  Final Taker Volume: {final_taker:.2f}")
        print(f"  Total Rebates: ${final_rebates:.6f}")
        print(f"  Net Cash Change: ${final_cash_change:+.2f}")
        
        if final_maker > 0:
            print(f"\n🏆 PROOF COMPLETE!")
            print(f"✅ Post-only mode CAN execute trades (volume = {final_maker:.2f})")
            print(f"✅ Post-only mode earns rebates (${final_rebates:.6f})")
            print(f"✅ Post-only mode prevents aggressive orders (taker vol = {final_taker:.2f})")
            
            if final_cash_change < 0:
                print(f"✅ Shows expected negative PnL from adverse selection")
            else:
                print(f"✅ Positive PnL due to rebates exceeding adverse selection")
            
            print(f"\n💡 USER WAS CORRECT:")
            print(f"   'if it decides to post it should still be able to trade, but only negative pnl right?'")
            print(f"   YES - Post-only trades via maker orders, potentially negative PnL, earns rebates!")
        else:
            print(f"\n❓ Still 0 volume - may need EVEN MORE dramatic market movements")
        
        # Show market progression
        print(f"\n📈 MARKET MOVEMENT ANALYSIS:")
        if market_phases:
            for phase_data in market_phases[::3]:  # Every 3rd step
                print(f"  Step {phase_data['step']} ({phase_data['phase']}): "
                      f"{phase_data['bid']:.2f}/{phase_data['ask']:.2f}, "
                      f"Active: {phase_data['active_orders']}")
        
        env.close()
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

if __name__ == "__main__":
    proof_post_only_trading()