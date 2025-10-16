#!/usr/bin/env python3
"""
Statistical tests to prove if the agent is converging/improving.
"""

import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def extract_episode_pnls_from_log():
    """Extract all episode PnLs from the training log."""
    
    print("📊 EXTRACTING EPISODE DATA FROM TRAINING LOG")
    print("=" * 60)
    
    # Read the training log
    with open('training_with_memory.log', 'r') as f:
        log_content = f.read()
    
    # Extract all episode PnL entries
    pnl_pattern = r'Final Episode PnL \(including liquidation\): \$(-?\d+\.?\d*)'
    matches = re.findall(pnl_pattern, log_content)
    
    episode_pnls = [float(match) for match in matches]
    
    print(f"✓ Extracted {len(episode_pnls)} completed episodes")
    print(f"PnL range: ${min(episode_pnls):.2f} to ${max(episode_pnls):.2f}")
    print(f"Mean PnL: ${np.mean(episode_pnls):.2f}")
    print(f"Std PnL: ${np.std(episode_pnls):.2f}")
    
    return episode_pnls

def test_improvement_trend(pnls, window_size=50):
    """Test if there's a statistically significant improvement trend."""
    
    print(f"\n📈 IMPROVEMENT TREND ANALYSIS (window size: {window_size})")
    print("-" * 60)
    
    if len(pnls) < window_size * 2:
        print(f"❌ Need at least {window_size * 2} episodes for trend analysis")
        return False, None, None
    
    # Split into early vs late periods
    n_episodes = len(pnls)
    early_episodes = pnls[:window_size]
    late_episodes = pnls[-window_size:]
    
    print(f"Early {window_size} episodes: Mean=${np.mean(early_episodes):.2f}, Std=${np.std(early_episodes):.2f}")
    print(f"Late {window_size} episodes: Mean=${np.mean(late_episodes):.2f}, Std=${np.std(late_episodes):.2f}")
    
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(late_episodes, early_episodes, equal_var=False)
    
    print(f"\nWelch's t-test results:")
    print(f"  H0: Late episodes = Early episodes")
    print(f"  H1: Late episodes > Early episodes")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value (two-tailed): {p_value:.6f}")
    print(f"  P-value (one-tailed): {p_value/2:.6f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(early_episodes)-1)*np.var(early_episodes) + 
                         (len(late_episodes)-1)*np.var(late_episodes)) / 
                        (len(early_episodes) + len(late_episodes) - 2))
    cohens_d = (np.mean(late_episodes) - np.mean(early_episodes)) / pooled_std
    
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Interpretation
    is_improving = t_stat > 0 and (p_value/2) < 0.05
    
    if is_improving:
        print(f"  ✅ SIGNIFICANT IMPROVEMENT detected! (α=0.05)")
        if abs(cohens_d) > 0.8:
            print(f"  📈 Large effect size - substantial improvement")
        elif abs(cohens_d) > 0.5:
            print(f"  📈 Medium effect size - moderate improvement")
        else:
            print(f"  📈 Small effect size - minor improvement")
    else:
        print(f"  ❌ No significant improvement detected")
    
    return is_improving, t_stat, p_value

def test_linear_trend(pnls):
    """Test for linear trend using regression analysis."""
    
    print(f"\n📊 LINEAR TREND ANALYSIS")
    print("-" * 60)
    
    x = np.arange(len(pnls))
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, pnls)
    
    print(f"Linear regression results:")
    print(f"  Slope: ${slope:.6f} per episode")
    print(f"  R-squared: {r_value**2:.6f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Standard error: {std_err:.6f}")
    
    # Projected improvement over training
    total_improvement = slope * len(pnls)
    print(f"  Total projected improvement: ${total_improvement:.2f}")
    
    # Significance test
    is_trending = p_value < 0.05 and slope > 0
    
    if is_trending:
        print(f"  ✅ SIGNIFICANT POSITIVE TREND detected!")
        print(f"  📈 Agent improving by ${slope:.4f} per episode on average")
    else:
        if slope > 0:
            print(f"  ➖ Positive trend but not statistically significant")
        else:
            print(f"  ❌ No positive trend detected")
    
    return is_trending, slope, p_value

def test_convergence_stability(pnls, recent_window=100):
    """Test if recent performance is becoming more stable (converging)."""
    
    print(f"\n🎯 CONVERGENCE STABILITY ANALYSIS")
    print("-" * 60)
    
    if len(pnls) < recent_window * 2:
        print(f"❌ Need at least {recent_window * 2} episodes for stability analysis")
        return False, None
    
    # Compare variance of early vs recent episodes
    early_episodes = pnls[:recent_window]
    recent_episodes = pnls[-recent_window:]
    
    early_var = np.var(early_episodes)
    recent_var = np.var(recent_episodes)
    
    print(f"Early {recent_window} episodes variance: {early_var:.2f}")
    print(f"Recent {recent_window} episodes variance: {recent_var:.2f}")
    
    # F-test for equal variances
    f_stat = early_var / recent_var if early_var > recent_var else recent_var / early_var
    df1 = recent_window - 1
    df2 = recent_window - 1
    p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))  # Two-tailed
    
    print(f"F-test for equal variances:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    # Check if variance decreased (more stable)
    variance_ratio = recent_var / early_var
    print(f"  Variance ratio (recent/early): {variance_ratio:.4f}")
    
    is_converging = variance_ratio < 0.8 and p_value < 0.05
    
    if is_converging:
        print(f"  ✅ CONVERGENCE detected - performance becoming more stable!")
        print(f"  📉 Variance reduced by {(1-variance_ratio)*100:.1f}%")
    else:
        if variance_ratio < 1.0:
            print(f"  ➖ Some stabilization but not statistically significant")
        else:
            print(f"  ❌ No convergence - performance still highly variable")
    
    return is_converging, variance_ratio

def test_positive_episodes_trend(pnls):
    """Test if the proportion of positive episodes is increasing."""
    
    print(f"\n💰 POSITIVE EPISODES TREND ANALYSIS")
    print("-" * 60)
    
    # Split into chunks and calculate success rate for each chunk
    chunk_size = 20
    chunks = [pnls[i:i+chunk_size] for i in range(0, len(pnls), chunk_size)]
    
    if len(chunks) < 3:
        print(f"❌ Need at least 3 chunks of {chunk_size} episodes")
        return False
    
    success_rates = []
    for i, chunk in enumerate(chunks):
        if len(chunk) >= chunk_size:  # Only use full chunks
            positive_count = sum(1 for pnl in chunk if pnl > 0)
            success_rate = positive_count / len(chunk)
            success_rates.append(success_rate)
            print(f"  Chunk {i+1}: {positive_count}/{len(chunk)} positive ({success_rate:.2%})")
    
    if len(success_rates) < 3:
        print(f"❌ Need at least 3 full chunks")
        return False
    
    # Test for trend in success rates
    x = np.arange(len(success_rates))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, success_rates)
    
    print(f"\nSuccess rate trend analysis:")
    print(f"  Slope: {slope:.4f} per chunk")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.6f}")
    
    improvement_trend = slope > 0 and p_value < 0.05
    
    if improvement_trend:
        print(f"  ✅ SUCCESS RATE IMPROVING over time!")
        print(f"  📈 Success rate increasing by {slope:.1%} per chunk")
    else:
        print(f"  ❌ No significant improvement in success rate")
    
    return improvement_trend

def main():
    """Run all convergence tests."""
    
    # Extract data
    pnls = extract_episode_pnls_from_log()
    
    if len(pnls) < 50:
        print(f"❌ Need at least 50 episodes for statistical tests. Found: {len(pnls)}")
        return
    
    print(f"\n🧪 RUNNING STATISTICAL CONVERGENCE TESTS")
    print(f"Total episodes: {len(pnls)}")
    print("=" * 60)
    
    # Test 1: Improvement trend
    improving, t_stat, p_val = test_improvement_trend(pnls)
    
    # Test 2: Linear trend
    trending, slope, p_trend = test_linear_trend(pnls)
    
    # Test 3: Convergence stability
    converging, var_ratio = test_convergence_stability(pnls)
    
    # Test 4: Success rate trend
    success_improving = test_positive_episodes_trend(pnls)
    
    # Overall assessment
    print(f"\n🎯 OVERALL STATISTICAL ASSESSMENT")
    print("=" * 60)
    
    evidence_count = sum([improving, trending, converging, success_improving])
    
    print(f"Statistical evidence summary:")
    print(f"  • Performance improvement: {'✅' if improving else '❌'}")
    print(f"  • Linear upward trend: {'✅' if trending else '❌'}")
    print(f"  • Convergence/stability: {'✅' if converging else '❌'}")
    print(f"  • Success rate improving: {'✅' if success_improving else '❌'}")
    
    print(f"\nEvidence score: {evidence_count}/4")
    
    if evidence_count >= 3:
        print(f"✅ STRONG STATISTICAL EVIDENCE of learning and convergence!")
    elif evidence_count >= 2:
        print(f"⚠️  MODERATE EVIDENCE of improvement - agent is learning but slowly")
    elif evidence_count >= 1:
        print(f"➖ WEAK EVIDENCE - some improvement but not conclusive")
    else:
        print(f"❌ NO STATISTICAL EVIDENCE of improvement")
    
    # Final recommendation
    if improving and trending:
        print(f"\n🚀 RECOMMENDATION: Continue training - agent is actively learning!")
    elif evidence_count >= 2:
        print(f"\n⏳ RECOMMENDATION: Continue training - progress is slow but measurable")
    else:
        print(f"\n🛑 RECOMMENDATION: Consider adjusting hyperparameters or strategy")

if __name__ == "__main__":
    main()
