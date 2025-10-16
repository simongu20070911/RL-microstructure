#!/usr/bin/env python3
"""
Comprehensive statistical analysis of training progress.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = PROJECT_ROOT / "runs" / "logs"

def analyze_training_progress():
    """Analyze training progress with statistical tests."""
    
    print("📊 TRAINING PROGRESS STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Load training data
    try:
        progress_files = sorted(
            LOG_ROOT.glob("extended_rebated_*/tensorboard/progress.csv"),
            reverse=True,
        )
        if not progress_files:
            raise FileNotFoundError("No progress.csv files found under runs/logs/extended_rebated_*")
        progress_path = progress_files[0]
        print(f"Loading progress data from: {progress_path}")
        df = pd.read_csv(progress_path)
        print(f"✓ Loaded {len(df)} validation checkpoints")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n📈 BASIC TRAINING METRICS")
    print("-" * 40)
    print(f"Total timesteps: {df['time/total_timesteps'].iloc[-1]:,}")
    print(f"Total episodes: {df['time/episodes'].iloc[-1]:,}")
    print(f"Training completion: {df['progress/completion_pct'].iloc[-1]:.1f}%")
    print(f"Average episode length: {df['rollout/ep_len_mean'].iloc[-1]:.1f}")
    print(f"Training time: {df['time/time_elapsed'].iloc[-1]/60:.1f} minutes")
    
    # Extract key metrics
    timesteps = df['time/total_timesteps'].values
    episode_rewards = df['rollout/ep_rew_mean'].values
    eval_pnl = df['eval/mean_pnl'].values
    eval_rebates = df['eval/mean_rebates'].values
    eval_rewards = df['eval/mean_reward'].values
    
    print("\n🎯 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Episode rewards: {episode_rewards}")
    print(f"Evaluation PnL: {eval_pnl}")
    print(f"Evaluation rebates: {eval_rebates}")
    print(f"Evaluation rewards: {eval_rewards}")
    
    # Learning Progress Analysis
    print("\n📊 LEARNING PROGRESS ANALYSIS")
    print("-" * 40)
    
    # Test 1: Episode Reward Trend
    if len(episode_rewards) >= 3:
        # Split into early vs late training
        mid_point = len(episode_rewards) // 2
        early_rewards = episode_rewards[:mid_point] if mid_point > 0 else episode_rewards[:1]
        late_rewards = episode_rewards[mid_point:]
        
        print(f"Early training rewards (first {len(early_rewards)} checkpoints): {early_rewards}")
        print(f"Late training rewards (last {len(late_rewards)} checkpoints): {late_rewards}")
        
        # T-test for improvement
        if len(early_rewards) > 1 and len(late_rewards) > 1:
            t_stat, p_value = stats.ttest_ind(late_rewards, early_rewards)
            print(f"T-test for reward improvement:")
            print(f"  Early mean: {np.mean(early_rewards):.2f} ± {np.std(early_rewards):.2f}")
            print(f"  Late mean: {np.mean(late_rewards):.2f} ± {np.std(late_rewards):.2f}")
            print(f"  T-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                if t_stat > 0:
                    print("  ✅ SIGNIFICANT IMPROVEMENT detected!")
                else:
                    print("  ⚠️  SIGNIFICANT DEGRADATION detected!")
            else:
                print("  ➖ No statistically significant change")
    
    # Test 2: Evaluation PnL Analysis
    print("\n💰 EVALUATION PnL ANALYSIS")
    print("-" * 40)
    
    non_zero_pnl = eval_pnl[eval_pnl != 0]
    if len(non_zero_pnl) > 0:
        print(f"Non-zero PnL episodes: {len(non_zero_pnl)}/{len(eval_pnl)}")
        print(f"PnL statistics: Mean={np.mean(non_zero_pnl):.2f}, Std={np.std(non_zero_pnl):.2f}")
        print(f"PnL range: [{np.min(non_zero_pnl):.2f}, {np.max(non_zero_pnl):.2f}]")
        
        # Test if PnL is significantly different from zero
        t_stat, p_value = stats.ttest_1samp(non_zero_pnl, 0)
        print(f"One-sample t-test (H0: PnL = 0):")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            if t_stat > 0:
                print("  ✅ Agent making SIGNIFICANT POSITIVE PnL!")
            else:
                print("  ❌ Agent making SIGNIFICANT NEGATIVE PnL!")
        else:
            print("  ➖ PnL not significantly different from zero")
    else:
        print("❌ No profitable episodes detected")
    
    # Test 3: Rebate Earning Analysis
    print("\n🎁 REBATE EARNING ANALYSIS")
    print("-" * 40)
    
    non_zero_rebates = eval_rebates[eval_rebates != 0]
    if len(non_zero_rebates) > 0:
        print(f"Rebate-earning episodes: {len(non_zero_rebates)}/{len(eval_rebates)}")
        print(f"Average rebates when earned: ${np.mean(non_zero_rebates):.2f}")
        print(f"Total rebates earned: ${np.sum(non_zero_rebates):.2f}")
        
        # Check if rebate earning is improving over time
        rebate_trend = np.polyfit(range(len(eval_rebates)), eval_rebates, 1)
        print(f"Rebate trend slope: {rebate_trend[0]:.4f} (positive = improving)")
        
        if rebate_trend[0] > 0.01:
            print("  ✅ Rebate earning IMPROVING over time")
        elif rebate_trend[0] < -0.01:
            print("  ⚠️  Rebate earning DECLINING over time")
        else:
            print("  ➖ Rebate earning stable")
    else:
        print("❌ No rebates earned - agent may not be trading")
    
    # Test 4: Trading Activity Analysis
    print("\n📈 TRADING ACTIVITY ANALYSIS")
    print("-" * 40)
    
    # Look at episode lengths and rewards to infer activity
    ep_lengths = df['rollout/ep_len_mean'].values
    print(f"Episode length trend: {ep_lengths}")
    
    # Check if episode lengths are consistent (indicates stable execution)
    ep_length_stability = np.std(ep_lengths) / np.mean(ep_lengths)
    print(f"Episode length stability (CV): {ep_length_stability:.3f}")
    
    if ep_length_stability < 0.05:
        print("  ✅ Episode lengths STABLE - consistent execution")
    else:
        print("  ⚠️  Episode lengths VARIABLE - possible instability")
    
    # Test 5: Learning Convergence
    print("\n🧠 LEARNING CONVERGENCE ANALYSIS")
    print("-" * 40)
    
    # Analyze critic loss trend
    critic_losses = df['train/critic_loss'].values
    print(f"Critic loss progression: {critic_losses}")
    
    # Check if loss is decreasing (learning)
    if len(critic_losses) >= 3:
        loss_trend = np.polyfit(range(len(critic_losses)), critic_losses, 1)
        print(f"Critic loss trend slope: {loss_trend[0]:.2e}")
        
        if loss_trend[0] < -1000:  # Significant decrease
            print("  ✅ Critic loss DECREASING - model is learning!")
        elif loss_trend[0] > 1000:   # Significant increase
            print("  ❌ Critic loss INCREASING - possible overfitting")
        else:
            print("  ➖ Critic loss stable - learning may have plateaued")
    
    # Overall Assessment
    print("\n🎯 OVERALL LEARNING ASSESSMENT")
    print("=" * 60)
    
    # Score different aspects
    scores = []
    
    # Score 1: PnL Performance
    if len(non_zero_pnl) > 0 and np.mean(non_zero_pnl) > 0:
        scores.append("PnL: Positive")
    elif len(non_zero_pnl) > 0:
        scores.append("PnL: Negative")
    else:
        scores.append("PnL: None")
    
    # Score 2: Trading Activity
    if len(non_zero_rebates) > 0:
        scores.append("Activity: Trading")
    else:
        scores.append("Activity: Inactive")
    
    # Score 3: Learning Progress
    if len(critic_losses) >= 2 and critic_losses[-1] < critic_losses[0]:
        scores.append("Learning: Improving")
    else:
        scores.append("Learning: Stable/Degrading")
    
    print("Assessment scores:")
    for score in scores:
        print(f"  • {score}")
    
    # Final recommendation
    if "Positive" in scores[0] and "Trading" in scores[1]:
        print("\n✅ CONCLUSION: Model is LEARNING and generating profits!")
    elif "Trading" in scores[1]:
        print("\n⚠️  CONCLUSION: Model is trading but needs optimization")
    else:
        print("\n❌ CONCLUSION: Model learned to do nothing - needs intervention")
    
    return df

if __name__ == "__main__":
    df = analyze_training_progress()
