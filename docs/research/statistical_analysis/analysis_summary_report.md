# RL Trading Experiments - Statistical Analysis Report
Generated: 2025-07-07 05:16:03

## Summary Statistics
- Total experiments: 157
- Unique environment types: 5
- Training completion rate: 51.6%
- Interruption rate: 0.6%

## Environment Type Breakdown
- unknown: 89 experiments
- 2sided: 57 experiments
- post_only: 6 experiments
- taker_only: 3 experiments
- 2sided_nocheat: 2 experiments

## Final Reward Performance
- Mean: -8.2153
- Std: 60.6829
- Median: -0.3831
- Range: [-333.2783, 299.2943]
- Positive results: 34/112 (30.4%)

## Top Performing Experiments
- 20250323-095001 (unknown): PnL = 71006.5000
- 20250323-043549 (unknown): PnL = 26830.1094
- 20250323-040711 (unknown): PnL = 17043.0078
- 20250327-124133 (unknown): PnL = 7029.6084
- 20250323-060948 (unknown): PnL = 276.6996

## Significant Environment Differences
### final_reward_mean
- 2sided_vs_taker_only: p=0.0003, effect_size=1.6671
- post_only_vs_taker_only: p=0.0193, effect_size=1.6169
- anova: p=0.0002
### best_validation_reward
- post_only_vs_taker_only: p=0.0185, effect_size=1.6327
