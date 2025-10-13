# Data Quality Validation Report
Generated: 2025-07-07 05:24:04
Total experiments analyzed: 157

## Executive Summary
- **Total Data Quality Issues Found**: 246
- **Reliable Experiments**: 14/157 (8.9%)
- **Critical Issue**: 4 experiments with unrealistic profits (>1000 PnL)
- **Missing Costs**: 79 experiments with missing transaction cost data

## 🚨 Critical Issues Requiring Immediate Attention

### Experiments with Unrealistic Profits (>1000 PnL)
| Experiment ID | PnL | Source | Environment | Long Cost | Short Cost |
|---------------|-----|--------|-------------|-----------|------------|
| 20250323-095001 | 71006.5 | prev_loggged | unknown | NaN | NaN |
| 20250323-043549 | 26830.1 | prev_loggged | unknown | NaN | NaN |
| 20250323-040711 | 17043.0 | prev_loggged | unknown | NaN | NaN |
| 20250327-124133 | 7029.6 | prev_loggged | unknown | NaN | NaN |

**Analysis**: These experiments likely have missing or inadequate transaction cost implementation.

## Data Quality by Source

| Source | Experiments | Completion Rate | Data Quality Score | Reliability | Issues |
|--------|-------------|-----------------|--------------------|-----------| -------|
| current_logs | 1 | 100.0% | 0.45 | no_data | 1 |
| prev_loggged | 156 | 51.3% | 0.36 | unreliable | 235 |

## Detailed Problem Breakdown

### Missing Transaction Costs (79 experiments)

### Zero Transaction Costs (64 experiments)

### Unrealistic Profits (4 experiments)
- PnL range: 7029.6 to 71006.5
- By source: {'prev_loggged': 4}

### Missing Environment Info (89 experiments)

### Validation Issues (10 experiments)

## Recommendations

### Immediate Actions Required:
1. **Exclude unrealistic experiments**: Remove experiments with PnL > 1000 from analysis
2. **Fix transaction cost implementation**: Ensure all environments have realistic trading costs
3. **Validate prev_loggged data**: Review all experiments from legacy source for data quality
4. **Re-run problematic experiments**: Use corrected environments with proper cost implementation

### For Future Experiments:
1. **Mandatory transaction costs**: All environments must implement realistic transaction costs (1-10 bps)
2. **Data validation pipeline**: Implement automatic validation checks before experiment storage
3. **Performance sanity checks**: Flag experiments with unrealistic profits for review
4. **Environment categorization**: Ensure proper environment type classification

## Clean Dataset Summary
- **Reliable experiments**: 14
- **Flagged experiments**: 143
- **Recommended for analysis**: Use only reliable experiments