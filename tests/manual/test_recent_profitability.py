#!/usr/bin/env python3
"""
T-test to prove if the last 20 episodes are significantly greater than zero.
"""

from pathlib import Path
import re
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "runs" / "logs" / "training_with_memory.log"

def test_recent_profitability():
    """Test if the most recent episodes are significantly profitable."""
    
    print("🎯 TESTING RECENT PROFITABILITY (Last 20 Episodes)")
    print("=" * 60)
    
    if not LOG_PATH.exists():
        print(f"❌ Training log not found at {LOG_PATH}. Run a training session first.")
        return {
            'mean': None,
            'std': None,
            't_stat': None,
            'p_value': None,
            'significant': False,
            'episodes': [],
            'total_pnl': None,
        }

    # Extract episode PnLs from training log
    with LOG_PATH.open('r') as f:
        log_content = f.read()
    
    pnl_pattern = r'Final Episode PnL \(including liquidation\): \$(-?\d+\.?\d*)'
    matches = re.findall(pnl_pattern, log_content)
    episode_pnls = [float(match) for match in matches]
    
    print(f"✓ Total episodes extracted: {len(episode_pnls)}")
    
    # Get last 20 episodes
    recent_episodes = episode_pnls[-20:]
    
    print(f"\n📊 LAST 20 EPISODES DATA:")
    print("-" * 40)
    for i, pnl in enumerate(recent_episodes, 1):
        print(f"Episode {len(episode_pnls)-20+i}: ${pnl:.2f}")
    
    # Calculate statistics
    mean_pnl = np.mean(recent_episodes)
    std_pnl = np.std(recent_episodes, ddof=1)  # Sample standard deviation
    n = len(recent_episodes)
    
    print(f"\n📈 DESCRIPTIVE STATISTICS:")
    print("-" * 40)
    print(f"Sample size (n): {n}")
    print(f"Mean PnL: ${mean_pnl:.2f}")
    print(f"Standard deviation: ${std_pnl:.2f}")
    print(f"Standard error: ${std_pnl/np.sqrt(n):.2f}")
    print(f"Min PnL: ${np.min(recent_episodes):.2f}")
    print(f"Max PnL: ${np.max(recent_episodes):.2f}")
    
    # Count positive vs negative episodes
    positive_count = sum(1 for pnl in recent_episodes if pnl > 0)
    zero_count = sum(1 for pnl in recent_episodes if pnl == 0)
    negative_count = sum(1 for pnl in recent_episodes if pnl < 0)
    
    print(f"\n📊 EPISODE BREAKDOWN:")
    print("-" * 40)
    print(f"Positive episodes: {positive_count}/{n} ({positive_count/n:.1%})")
    print(f"Zero episodes: {zero_count}/{n} ({zero_count/n:.1%})")
    print(f"Negative episodes: {negative_count}/{n} ({negative_count/n:.1%})")
    
    # One-sample t-test against zero
    print(f"\n🧪 ONE-SAMPLE T-TEST:")
    print("-" * 40)
    print(f"H₀: μ = 0 (mean PnL equals zero)")
    print(f"H₁: μ > 0 (mean PnL greater than zero)")
    
    t_statistic, p_value_two_tailed = stats.ttest_1samp(recent_episodes, 0)
    p_value_one_tailed = p_value_two_tailed / 2 if t_statistic > 0 else 1 - (p_value_two_tailed / 2)
    
    print(f"\nTest results:")
    print(f"  T-statistic: {t_statistic:.4f}")
    print(f"  Degrees of freedom: {n-1}")
    print(f"  P-value (two-tailed): {p_value_two_tailed:.6f}")
    print(f"  P-value (one-tailed): {p_value_one_tailed:.6f}")
    
    # Calculate confidence interval
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, n-1)
    margin_error = t_critical * (std_pnl / np.sqrt(n))
    ci_lower = mean_pnl - margin_error
    ci_upper = mean_pnl + margin_error
    
    print(f"  95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")
    
    # Effect size (Cohen's d)
    cohens_d = mean_pnl / std_pnl
    print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
    
    # Statistical significance test
    print(f"\n🎯 SIGNIFICANCE TEST RESULTS:")
    print("-" * 40)
    
    if p_value_one_tailed < 0.05:
        print(f"✅ STATISTICALLY SIGNIFICANT RESULT!")
        print(f"   The last 20 episodes are significantly greater than zero")
        print(f"   (p = {p_value_one_tailed:.6f} < 0.05)")
        
        if cohens_d > 0.8:
            print(f"   📈 LARGE effect size - strong profitability")
        elif cohens_d > 0.5:
            print(f"   📈 MEDIUM effect size - moderate profitability")
        elif cohens_d > 0.2:
            print(f"   📈 SMALL effect size - weak but significant profitability")
        else:
            print(f"   📈 VERY SMALL effect size - minimal profitability")
            
    elif p_value_one_tailed < 0.10:
        print(f"⚠️  MARGINALLY SIGNIFICANT (p = {p_value_one_tailed:.6f})")
        print(f"   Weak evidence that episodes are greater than zero")
        
    else:
        print(f"❌ NOT STATISTICALLY SIGNIFICANT")
        print(f"   Cannot conclude that episodes are greater than zero")
        print(f"   (p = {p_value_one_tailed:.6f} > 0.05)")
    
    # Additional insights
    print(f"\n💡 ADDITIONAL INSIGHTS:")
    print("-" * 40)
    
    if ci_lower > 0:
        print(f"✅ Lower confidence bound (${ci_lower:.2f}) > 0")
        print(f"   Strong evidence of consistent profitability")
    elif ci_upper > 0:
        print(f"⚠️  Upper confidence bound (${ci_upper:.2f}) > 0")
        print(f"   But lower bound (${ci_lower:.2f}) ≤ 0 - mixed evidence")
    else:
        print(f"❌ Entire confidence interval ≤ 0")
        print(f"   Evidence suggests non-profitability")
    
    # Practical significance
    total_recent_pnl = np.sum(recent_episodes)
    print(f"\n💰 PRACTICAL SIGNIFICANCE:")
    print("-" * 40)
    print(f"Total PnL over last 20 episodes: ${total_recent_pnl:.2f}")
    print(f"Average per episode: ${mean_pnl:.2f}")
    
    if total_recent_pnl > 100:
        print(f"✅ PRACTICALLY SIGNIFICANT - substantial profits")
    elif total_recent_pnl > 20:
        print(f"⚠️  MODEST PROFITS - economically meaningful")
    elif total_recent_pnl > 0:
        print(f"➖ SMALL PROFITS - barely positive")
    else:
        print(f"❌ LOSSES - not profitable")
    
    return {
        'mean': mean_pnl,
        'std': std_pnl,
        't_stat': t_statistic,
        'p_value': p_value_one_tailed,
        'significant': p_value_one_tailed < 0.05,
        'episodes': recent_episodes,
        'total_pnl': total_recent_pnl
    }

if __name__ == "__main__":
    results = test_recent_profitability()
