#!/usr/bin/env python3
"""
Debug script to test with actual trading actions instead of "do nothing"
"""

from pathlib import Path

import numpy as np

from rltrader.envs import TwoSidedMarketEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Simple test configuration
config = {
    "csv_path": str((DATA_DIR / "orderbook_trimmed_large.csv").resolve()),
    "initial_capital": 1000000.0,
    "order_book_levels": 5,
    "max_order_volume": 1.0,
    "episode_length": 100,
    "max_steps": 1000,
    "latency_steps_long": 0,
    "latency_steps_short": 0,
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_active_orders": 5,
    "max_inventory": 10.0,
    "inventory_penalty": 0.0,
    "transaction_cost_long": 0.0,
    "transaction_cost_short": 0.0,
    "invalid_order_penalty": 0.0,
    "activity_bonus": 0.0,
    "taker_penalty": 0.0,
    "price_offset_ticks": 10,
    "allowed_aggressiveness_ticks": 2,
    "quoting_reward_enabled": False,
    "quoting_reward_amount": 0.0,
    "quoting_reward_max_ticks": 0,
    "explicit_cancel_enabled": False,
    "explicit_cancel_threshold": 0.9,
    "explicit_cancel_penalty": 0.0,
    "explicit_cancel_clears_pending": False,
    "do_nothing_threshold": 0.9,  # High threshold so normal actions work
    "obs_price_norm_scale": 1.0,
    "obs_qty_norm_scale": 1.0,
}

def test_trading_actions():
    """Test with actual trading actions instead of do nothing"""
    print("=== TESTING WITH TRADING ACTIONS ===")
    
    try:
    env = TwoSidedMarketEnv(config)
        print(f"Environment created successfully")
        
        # Test multiple episodes with trading actions
        episode_results = []
        
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            
            # Reset environment
            obs, info = env.reset()
            print(f"Reset - current_step: {env.current_step}")
            print(f"Reset - midprice: {env.midprice:.4f}")
            print(f"Reset - spread: {env.spread:.4f}")
            
            # Run episode with trading actions
            episode_rewards = []
            episode_pnls = []
            midprices = []
            step_count = 0
            
            done = False
            while not done and step_count < 20:  # Limit to first 20 steps
                # Trading action: buy and sell orders
                # action[0]: buy offset, action[1]: sell offset
                # action[2]: buy size, action[3]: sell size  
                # action[4]: cancel signal, action[5]: do nothing signal
                action = np.array([0.1, -0.1, 0.5, 0.5, 0.0, 0.0])  # Place buy/sell orders
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_rewards.append(reward)
                episode_pnls.append(info.get('episode_pnl', 0.0))
                midprices.append(env.midprice)
                
                if step_count < 10:  # Log first 10 steps
                    print(f"  Step {step_count}: current_step={env.current_step}, "
                          f"reward={reward:.6f}, episode_pnl={info.get('episode_pnl', 0.0):.6f}, "
                          f"midprice={env.midprice:.4f}, active_orders={len(env.active_orders)}, "
                          f"pending_orders={len(env.pending_orders)}")
                
                step_count += 1
            
            final_info = {
                'episode': episode + 1,
                'steps_taken': step_count,
                'final_current_step': env.current_step,
                'final_reward_sum': sum(episode_rewards),
                'final_episode_pnl': episode_pnls[-1] if episode_pnls else 0.0,
                'first_10_rewards': episode_rewards[:10],
                'first_10_pnls': episode_pnls[:10],
                'first_10_midprices': midprices[:10],
                'final_cash': env.cash,
                'final_inventory': env.inventory,
                'final_active_orders': len(env.active_orders),
                'final_pending_orders': len(env.pending_orders),
            }
            
            episode_results.append(final_info)
            print(f"Episode {episode + 1} completed:")
            print(f"  Final current_step: {final_info['final_current_step']}")
            print(f"  Final episode_pnl: {final_info['final_episode_pnl']:.6f}")
            print(f"  Final cash: {final_info['final_cash']:.2f}")
            print(f"  Final inventory: {final_info['final_inventory']:.6f}")
            print(f"  Final active orders: {final_info['final_active_orders']}")
            print(f"  Final pending orders: {final_info['final_pending_orders']}")
        
        # Compare episodes
        print(f"\n=== EPISODE COMPARISON ===")
        for i, result in enumerate(episode_results):
            print(f"Episode {i+1}:")
            print(f"  Final current_step: {result['final_current_step']}")
            print(f"  Final episode_pnl: {result['final_episode_pnl']:.6f}")
            print(f"  Reward sum: {result['final_reward_sum']:.6f}")
            print(f"  First 5 rewards: {result['first_10_rewards'][:5]}")
            print(f"  First 5 PnLs: {result['first_10_pnls'][:5]}")
            print(f"  First 5 midprices: {result['first_10_midprices'][:5]}")
        
        # Check if results are identical
        if len(episode_results) >= 2:
            identical_pnls = all(
                abs(episode_results[0]['final_episode_pnl'] - result['final_episode_pnl']) < 1e-10
                for result in episode_results[1:]
            )
            
            identical_rewards = all(
                np.allclose(episode_results[0]['first_10_rewards'], result['first_10_rewards'], atol=1e-10)
                for result in episode_results[1:]
            )
            
            identical_midprices = all(
                np.allclose(episode_results[0]['first_10_midprices'], result['first_10_midprices'], atol=1e-10)
                for result in episode_results[1:]
            )
            
            print(f"\n=== DIAGNOSIS ===")
            print(f"Identical final PnLs: {identical_pnls}")
            print(f"Identical reward sequences: {identical_rewards}")
            print(f"Identical midprice sequences: {identical_midprices}")
            
            if identical_pnls and identical_rewards and identical_midprices:
                print("🚨 BUG CONFIRMED: Episodes with trading actions produce identical results!")
                print("🔍 This confirms the bug is NOT in the 'do nothing' logic")
                print("🔍 Bug likely in: environment state not properly reset between episodes")
            else:
                print("✅ Episodes with trading actions produce different results")
                print("🔍 Bug might be specific to 'do nothing' logic")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_trading_actions()
