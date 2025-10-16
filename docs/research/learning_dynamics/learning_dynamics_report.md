# Learning Dynamics Analysis Report
Generated: 2025-07-07 05:20:00
Total experiments analyzed: 157

## Key Learning Dynamics Insights
- Most efficient learning environment: unknown (efficiency: 54.4%)
- Most stable training configuration: episode_length_400.0 (stability score: 0.283)
- Excellent duration-performance configurations: 1 found

## Convergence Patterns by Environment
| Environment | Completion Rate | Learning Efficiency | Improvement Ratio | Stability |
|-------------|-----------------|--------------------|--------------------|-----------|
| post_only | 100.0% | 16.7% | 0.486 | 0.29 |
| unknown | 76.4% | 54.4% | 0.059 | 70.40 |

## Training Stability by Episode Length
| Episode Length | Completion Rate | Stability Score | Duration Effect | PnL CV |
|----------------|-----------------|-----------------|-----------------|---------|
| episode_length_400.0 | 47.1% | 0.283 | excellent | 5.47 |

## Loss Convergence Analysis
| Environment | Actor Loss | Actor Quality | Critic Loss | Critic Quality |
|-------------|------------|---------------|-------------|----------------|
| 2sided | -5.797 | poor_convergence_with_poor_stability | 0.454 | good_convergence_with_poor_stability |
| post_only | 6.200 | poor_convergence_with_poor_stability | 0.035 | acceptable_convergence_with_excellent_stability |
| taker_only | 6.007 | poor_convergence_with_poor_stability | 0.190 | acceptable_convergence_with_excellent_stability |
| unknown | 62516529336675104926466048.000 | poor_convergence_with_poor_stability | inf | poor_convergence_with_poor_stability |

## Learning Optimization Recommendations
Based on the learning dynamics analysis:
1. **Environment Selection**: Choose environments with high learning efficiency (>50%)
2. **Episode Length**: Medium-length episodes (400 steps) show best stability-performance balance
3. **Training Monitoring**: Monitor actor/critic loss convergence for early stopping decisions
4. **Hyperparameter Tuning**: Focus on parameters that show consistent learning patterns
5. **Stability Optimization**: Prioritize configurations with stability scores >0.7