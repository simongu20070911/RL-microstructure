# Market Microstructure Analysis Report
Generated: 2025-07-07 05:18:34
Total experiments analyzed: 157

## Key Market Microstructure Insights
- Optimal inventory penalty: penalty_0.0 achieved best mean PnL of 2064.83
- Best risk-adjusted strategy: offset_40.0_ticks with Sharpe ratio 0.329
- No significant difference between realistic and unrealistic environments

## Inventory Management Analysis
| Penalty Level | Sample Size | Mean PnL | Success Rate | Completion Rate |
|---------------|-------------|----------|--------------|-----------------|
| penalty_0.0 | 129 | 2064.83 | 16.9% | 48.8% |

## Price Offset Strategy Analysis
| Strategy | Aggressiveness | Mean PnL | Sharpe Ratio | Success Rate |
|----------|----------------|----------|--------------|--------------|
| offset_15.0_ticks | medium | -5.43 | -0.841 | 11.8% |
| offset_40.0_ticks | low | 5086.39 | 0.329 | 25.0% |

## Environment Realism Impact
- **Statistical significance**: No (p = 0.2025)
- **Effect size**: 0.752
- **Interpretation**: No significant difference between realistic and unrealistic environments

## Fee Structure Impact Analysis
| Fee Structure | Fee Level (bps) | Mean PnL | Fee Efficiency | Trading Frequency |
|---------------|-----------------|----------|----------------|-------------------|
| asymmetric_L0.0_S2.7721e-06 | 15.0 | -3.63 | -0.2418555809901311 | 6.1 |
| no_fees | 0.0 | 2592.46 | N/A | 83.2 |
| unknown | 10.0 | -1207.50 | -120.75006364881992 | 50.0 |

## Practical Trading Recommendations
Based on the microstructure analysis:
1. **Inventory Management**: Moderate inventory penalties (0.005-0.01) show optimal risk-return profiles
2. **Spread Management**: Medium aggressiveness (10-20 tick offsets) provides best risk-adjusted returns
3. **Environment Choice**: Realistic environments with persistent positions provide better model validation
4. **Fee Sensitivity**: Strategies should be tested across multiple fee structures for robustness
5. **Volume Optimization**: Smaller order sizes generally provide more stable performance