# Rebated HFT Environments

This package contains High-Frequency Trading (HFT) environments with market maker rebate structures. These environments simulate realistic rebate programs offered by major cryptocurrency exchanges to high-volume market makers.

## Overview

Market maker rebates are negative transaction fees that exchanges pay to traders who provide liquidity to the order book. These rebated environments model three common VIP tier rebate structures:

- **HFTEnvRebated004**: 0.004% rebate (4 basis points)
- **HFTEnvRebated006**: 0.006% rebate (6 basis points)  
- **HFTEnvRebated008**: 0.008% rebate (8 basis points)

## Rebate Structure Comparison

| Exchange | VIP Level | Monthly Volume (USD) | Maker Rebate |
|----------|-----------|---------------------|--------------|
| Binance  | VIP 5     | $1M - $10M         | 0.004%       |
| Binance  | VIP 7     | $100M - $500M      | 0.006%       |
| Binance  | VIP 9     | $1B+               | 0.008%       |

*Note: Actual exchange rebate structures may vary. These are representative values for research purposes.*

## Key Features

### Rebate Implementation
- **Negative Transaction Costs**: Rebates are modeled as negative transaction costs
- **Real-time Tracking**: Environments track total rebates earned and rebated volume
- **Market Making Incentives**: Rebates incentivize providing liquidity rather than taking it

### Environment Enhancements
- **Rebate Statistics**: Each environment tracks cumulative rebates and trading volume
- **Info Dictionary**: Step and reset methods return rebate information
- **Configuration Isolation**: Original configurations are preserved via deep copying

## Usage

### Basic Usage

```python
from env_rebated import HFTEnvRebated004, get_default_rebated_config_004

# Create environment with 4 bps rebate
config = get_default_rebated_config_004()
config["csv_path"] = "path/to/your/orderbook/data.csv"

env = HFTEnvRebated004(config)

# Reset environment
obs, info = env.reset()
print(f"Rebate rate: {info['rebate_bps']} bps")

# Take trading actions
for step in range(100):
    action = env.action_space.sample()  # Replace with your trading strategy
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {step}: Rebates earned: ${info['total_rebates_earned']:.4f}")
    
    if terminated or truncated:
        break
```

### Advanced Usage

```python
from env_rebated import HFTEnvRebated006, HFTEnvRebated008
import numpy as np

# Compare different rebate structures
configs = [
    get_default_rebated_config_004(),
    get_default_rebated_config_006(), 
    get_default_rebated_config_008()
]

environments = [
    HFTEnvRebated004(configs[0]),
    HFTEnvRebated006(configs[1]),
    HFTEnvRebated008(configs[2])
]

# Run comparison
for i, env in enumerate(environments):
    obs, info = env.reset(seed=42)  # Same seed for fair comparison
    
    total_reward = 0
    total_rebates = 0
    
    for step in range(1000):
        # Market making strategy: place orders near mid
        action = np.array([0.1, -0.1, 0.8, 0.8, -1.0, -1.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        total_rebates = info['total_rebates_earned']
        
        if terminated or truncated:
            break
    
    print(f"Environment {i} ({info['rebate_bps']} bps):")
    print(f"  Total Reward: ${total_reward:.2f}")
    print(f"  Total Rebates: ${total_rebates:.4f}")
    print(f"  Rebated Volume: {info['rebated_volume']:.2f}")
    print()
```

## Configuration

### Required Configuration Parameters

All rebated environments use the same configuration structure as the base HFT environment, with the following key differences:

```python
config = {
    # Data and Environment
    "csv_path": "path/to/orderbook/data.csv",
    "initial_capital": 100000.0,
    "order_book_levels": 10,
    "max_steps": 50000,
    "episode_length": 10000,
    
    # Market Microstructure
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_order_volume": 10.0,
    "max_inventory": 50.0,
    
    # Rebated Transaction Costs (automatically set by environment)
    "transaction_cost_long": -0.00004,   # Negative = rebate
    "transaction_cost_short": -0.00004,  # Negative = rebate
    
    # Risk Management
    "inventory_penalty": 0.001,
    "invalid_order_penalty": -1.0,
    "taker_penalty": -0.01,
    
    # Action Controls
    "price_offset_ticks": 5,
    "allowed_aggressiveness_ticks": 3,
    
    # Rewards
    "activity_bonus": 0.001,
    "quoting_reward_enabled": True,
    "quoting_reward_amount": 0.0001,
    "quoting_reward_max_ticks": 5,
    
    # Other parameters...
}
```

### Data Requirements

The environments require order book data in CSV format with the following columns:

- `timestamp`: Time index
- `bid1`, `bidqty1`, `ask1`, `askqty1`: Best bid/ask prices and quantities
- `bid2`, `bidqty2`, `ask2`, `askqty2`: Second level bid/ask
- ... (up to `order_book_levels` levels)

Example data structure:
```csv
timestamp,bid1,bidqty1,ask1,askqty1,bid2,bidqty2,ask2,askqty2,...
0,1799.98,10.5,1800.02,8.2,1799.96,15.1,1800.04,12.3,...
1,1799.99,9.8,1800.01,7.9,1799.97,14.8,1800.03,11.8,...
...
```

## Information Dictionary

Each environment provides detailed information about rebates in the `info` dictionary returned by `step()` and `reset()`:

```python
info = {
    'rebate_rate': 0.00004,           # Rebate rate (decimal)
    'rebate_bps': 4,                  # Rebate rate in basis points
    'total_rebates_earned': 12.34,    # Cumulative rebates earned ($)
    'rebated_volume': 156.7,          # Total volume that earned rebates
    'mtm': 100234.56,                 # Mark-to-market value
    'episode_pnl': 234.56,            # Episode P&L
    # ... other standard HFT environment info
}
```

## Testing

Run the comprehensive test suite to validate environment functionality:

```bash
cd /home/gaen/Documents/RL/envs/env_rebated/
python test_rebated_envs.py
```

The test suite includes:
- **Initialization Tests**: Verify correct rebate rates and configuration
- **Functionality Tests**: Validate step/reset behavior and rebate tracking
- **Calculation Tests**: Ensure accurate rebate calculations
- **Comparison Tests**: Verify different rebate rates produce different outcomes
- **Performance Tests**: Check memory usage and execution speed

## Research Applications

### Academic Paper Enhancement

These rebated environments are designed to enhance academic research by:

1. **Realistic Market Conditions**: Model actual exchange rebate structures
2. **Economic Viability**: Show when HFT strategies become profitable with rebates
3. **Comparative Analysis**: Enable studies across different fee structures
4. **Market Making Incentives**: Demonstrate the impact of liquidity provision rewards

### Experimental Design

Recommended experimental setup for academic studies:

```python
# Define experimental conditions
rebate_conditions = [
    ("No Rebate", 0.0001),      # Standard 1 bps fee
    ("Low Rebate", -0.00004),   # 4 bps rebate
    ("Medium Rebate", -0.00006), # 6 bps rebate  
    ("High Rebate", -0.00008)   # 8 bps rebate
]

# Run experiments across conditions
results = {}
for condition_name, transaction_cost in rebate_conditions:
    # Configure environment
    config = get_default_rebated_config_004()
    config["transaction_cost_long"] = transaction_cost
    config["transaction_cost_short"] = transaction_cost
    
    # Run multiple seeds for statistical significance
    condition_results = []
    for seed in range(10):
        # Train agent and evaluate performance
        performance = train_and_evaluate(config, seed)
        condition_results.append(performance)
    
    results[condition_name] = condition_results

# Analyze results with statistical tests
analyze_rebate_impact(results)
```

## Implementation Details

### Rebate Calculation

Rebates are calculated as:
```
rebate_earned = executed_volume × rebate_rate × execution_price
```

Where:
- `executed_volume`: Quantity of assets traded
- `rebate_rate`: Negative transaction cost (e.g., -0.00004 for 4 bps)
- `execution_price`: Price at which the trade was executed

### Transaction Cost Override

The rebated environments override transaction costs in the configuration:

```python
# Original config
original_config = {
    "transaction_cost_long": 0.0001,   # 1 bps fee
    "transaction_cost_short": 0.0001   # 1 bps fee
}

# Rebated config (4 bps rebate)
rebated_config = copy.deepcopy(original_config)
rebated_config["transaction_cost_long"] = -0.00004   # 4 bps rebate
rebated_config["transaction_cost_short"] = -0.00004  # 4 bps rebate
```

### Cash Flow Impact

With rebates, the cash flow equation becomes:
```
cash_change = -executed_value + rebate_earned
cash_change = -executed_value + (executed_volume × rebate_rate × price)
cash_change = -executed_value × (1 - rebate_rate)
```

For a 4 bps rebate:
```
cash_change = -executed_value × (1 - 0.00004) = -executed_value × 0.99996
```

This means the trader pays 99.996% of the trade value instead of 100%, effectively receiving 0.004% back.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure parent directory is in path
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

2. **Data Loading Issues**
   ```python
   # Verify CSV file path and format
   import pandas as pd
   df = pd.read_csv(config["csv_path"])
   print(df.columns.tolist())  # Check column names
   print(df.head())            # Check data format
   ```

3. **Configuration Errors**
   ```python
   # Validate configuration
   env = HFTEnvRebated004(config)  # Will raise detailed error messages
   ```

### Performance Optimization

For large-scale experiments:

1. **Reduce Episode Length**: Use shorter episodes for faster training
2. **Limit Order Book Levels**: Use fewer levels (e.g., 5 instead of 10)
3. **Batch Processing**: Process multiple seeds in parallel
4. **Memory Management**: Delete environments after use to free memory

## License

This code is part of the HFT research framework and follows the same licensing terms as the parent project.

## Citation

If you use these rebated environments in your research, please cite:

```bibtex
@article{rebated_hft_envs,
    title={Market Maker Rebates in High-Frequency Trading: A Reinforcement Learning Analysis},
    author={[Your Name]},
    journal={[Journal Name]},
    year={2025}
}
```

## Contact

For questions or issues related to the rebated environments, please contact [contact information] or open an issue in the project repository.