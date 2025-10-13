#!/usr/bin/env python3
"""
Fix baseline test using the EXACT same approach that worked for rebated environments
"""

import sys
import os
import tempfile
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def create_exact_same_data():
    """Create the EXACT same data that produced 1.7 volume for rebated environments."""
    data = []
    
    for i in range(50):
        row = {'datetime': i}
        
        if i < 10:
            # Phase 1: Initial stable market
            base_bid = 1799.98  
            base_ask = 1800.02  
        elif i < 20:
            # Phase 2: Market moves UP to hit passive BUY orders
            progress = (i - 10) / 10.0  
            base_bid = 1799.98 + progress * 0.05  
            base_ask = 1800.02 + progress * 0.05  
        elif i < 30:
            # Phase 3: Market moves DOWN to hit passive SELL orders  
            progress = (i - 20) / 10.0  
            base_bid = 1800.03 - progress * 0.10  
            base_ask = 1800.07 - progress * 0.10  
        elif i < 40:
            # Phase 4: Oscillating market
            oscillation = 0.02 * np.sin((i - 30) * 0.5)
            base_bid = 1799.96 + oscillation
            base_ask = 1799.98 + oscillation
        else:
            # Phase 5: Final stable period
            base_bid = 1799.98
            base_ask = 1800.02
        
        for level in range(1, 11):
            row[f'bid{level}'] = base_bid - (level-1) * 0.01
            row[f'bidqty{level}'] = 50.0  
            row[f'ask{level}'] = base_ask + (level-1) * 0.01  
            row[f'askqty{level}'] = 50.0  
            
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    return temp_file.name

def test_baseline_with_working_approach():
    """Use the EXACT same approach that worked for rebated environments."""
    print("🎯 TESTING BASELINE WITH EXACT WORKING APPROACH")
    
    test_csv = create_exact_same_data()
    
    try:
        # Test baseline with EXACT same config as working rebated test
        config = get_unified_config("baseline", False)
        config["csv_path"] = test_csv
        config["episode_length"] = 35
        config["latency_steps_long"] = 0  # Same as working test
        config["latency_steps_short"] = 0
        config["max_order_volume"] = 1.0  # Same as working test
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"Environment setup:")
        print(f"  Fee structure: {env.fee_structure}")
        print(f"  Transaction costs: {config['transaction_cost_long']:.6f}")
        print(f"  Initial BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
        
        # Use EXACT same action strategy that produced 1.7 volume
        for step in range(30):
            if step < 5:
                # Same passive orders
                action = [0.0, 0.0, 0.9, 0.9, -1.0, -1.0]
                strategy = "PASSIVE"
            elif step < 15:
                # Same slightly aggressive orders  
                action = [0.3, -0.3, 0.7, 0.7, -1.0, -1.0]
                strategy = "SLIGHTLY AGGRESSIVE"
            else:
                # Same do nothing
                action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]
                strategy = "DO NOTHING"
            
            if step % 5 == 0:
                print(f"\nStep {step}: {strategy}, BBO: {env.best_bid:.2f}/{env.best_ask:.2f}")
            
            pre_cash = env.cash
            pre_inventory = env.inventory
            
            obs, reward, term, trunc, info = env.step(action)
            
            post_cash = env.cash
            post_inventory = env.inventory
            
            # Check for executions (same logic as working test)
            cash_change = abs(post_cash - pre_cash)
            inventory_change = abs(post_inventory - pre_inventory)
            
            if cash_change > 1.0 or inventory_change > 0.001:
                print(f"  🎉 EXECUTION at step {step}:")
                print(f"    Cash: ${pre_cash:.2f} → ${post_cash:.2f} (Δ${post_cash-pre_cash:+.2f})")
                print(f"    Inventory: {pre_inventory:.3f} → {post_inventory:.3f}")
            
            if term or trunc:
                break
        
        # Final results
        final_cash = env.cash
        net_change = final_cash - 100000
        
        print(f"\n=== BASELINE RESULTS ===")
        print(f"Final Cash: ${final_cash:.2f}")
        print(f"Net Change: ${net_change:+.2f}")
        print(f"Final Inventory: {env.inventory:.3f}")
        
        # Check volume tracking for baseline
        maker_vol = getattr(env, 'maker_volume', 0)
        taker_vol = getattr(env, 'taker_volume', 0)
        total_vol = maker_vol + taker_vol
        
        print(f"Maker Volume: {maker_vol:.3f}")
        print(f"Taker Volume: {taker_vol:.3f}")
        print(f"Total Volume: {total_vol:.3f}")
        
        if total_vol > 0:
            print(f"✅ SUCCESS: Baseline shows {total_vol:.3f} volume")
            if net_change < 0:
                print(f"✅ CORRECT: Baseline lost money to fees (${net_change:+.2f})")
            else:
                print(f"⚠️  UNEXPECTED: Baseline gained money (${net_change:+.2f})")
        else:
            print(f"❌ PROBLEM: Baseline still shows zero volume")
        
        env.close()
        return total_vol > 0
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def compare_baseline_vs_rebated_final():
    """Final comparison using working approach for both."""
    print(f"\n🎯 FINAL COMPARISON: BASELINE VS REBATED")
    
    test_csv = create_exact_same_data()
    
    try:
        results = {}
        
        for name, fee_structure in [("Baseline", "baseline"), ("6 bps Rebate", "rebate_6bps")]:
            print(f"\n--- Testing {name} ---")
            
            config = get_unified_config(fee_structure, False)
            config["csv_path"] = test_csv
            config["episode_length"] = 35
            config["latency_steps_long"] = 0
            config["latency_steps_short"] = 0
            config["max_order_volume"] = 1.0
            
            env = RebatedHFTEnv(config)
            obs, info = env.reset()
            
            # EXACT same strategy for both
            for step in range(30):
                if step < 5:
                    action = [0.0, 0.0, 0.9, 0.9, -1.0, -1.0]
                elif step < 15:
                    action = [0.3, -0.3, 0.7, 0.7, -1.0, -1.0]
                else:
                    action = [-1.0, -1.0, 0.0, 0.0, -1.0, 0.9]
                
                obs, reward, term, trunc, info = env.step(action)
                if term or trunc:
                    break
            
            # Collect results
            final_cash = env.cash
            net_change = final_cash - 100000
            
            # Volume tracking
            maker_vol = info.get('maker_volume', 0)
            taker_vol = info.get('taker_volume', 0)
            total_vol = maker_vol + taker_vol
            rebates = info.get('total_rebates_earned', 0)
            
            results[name] = {
                'net_change': net_change,
                'total_volume': total_vol,
                'maker_volume': maker_vol,
                'taker_volume': taker_vol,
                'rebates': rebates
            }
            
            print(f"  Net Change: ${net_change:+.2f}")
            print(f"  Total Volume: {total_vol:.3f}")
            print(f"  Rebates: ${rebates:.4f}")
            
            env.close()
        
        print(f"\n" + "="*50)
        print(f"FINAL COMPARISON RESULTS")
        print(f"="*50)
        
        for name, result in results.items():
            net = result['net_change']
            vol = result['total_volume']
            rebates = result['rebates']
            
            status = "✅" if vol > 0 else "❌"
            print(f"{name:<15} Net: ${net:+7.2f}  Vol: {vol:5.1f}  Rebates: ${rebates:6.4f} {status}")
        
        baseline_vol = results['Baseline']['total_volume']
        rebated_vol = results['6 bps Rebate']['total_volume']
        
        if baseline_vol > 0 and rebated_vol > 0:
            print(f"\n🎉 SUCCESS: Both environments show trading activity!")
            return True
        else:
            print(f"\n❌ ISSUE: Some environments still show zero volume")
            return False
        
    finally:
        if os.path.exists(test_csv):
            os.unlink(test_csv)

def main():
    """Run the baseline fix test."""
    print("🚀 FIXING BASELINE ZERO VOLUME ISSUE")
    print("="*50)
    
    baseline_success = test_baseline_with_working_approach()
    comparison_success = compare_baseline_vs_rebated_final()
    
    print(f"\n" + "="*50)
    if baseline_success and comparison_success:
        print(f"🎉 BASELINE FIXED - ALL ENVIRONMENTS WORKING!")
    else:
        print(f"❌ BASELINE ISSUE PERSISTS")
    print(f"="*50)

if __name__ == "__main__":
    main()