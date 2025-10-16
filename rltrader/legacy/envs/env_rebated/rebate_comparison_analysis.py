#!/usr/bin/env python3
"""
Rebate Impact Analysis for Academic Paper

This script demonstrates the economic impact of different rebate structures
on HFT market making strategies. Used to generate illustrations and data
for academic research on market maker incentives.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import sys
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_004 import HFTEnvRebated004, get_default_rebated_config_004
from env_rebated_006 import HFTEnvRebated006, get_default_rebated_config_006  
from env_rebated_008 import HFTEnvRebated008, get_default_rebated_config_008

# Also test baseline (no rebate) environment
from env_2sided import HFTEnv

# Set style for academic plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RebateAnalysis:
    """Analyze the impact of different rebate structures on trading performance."""
    
    def __init__(self, output_dir="rebate_analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test data
        self.test_csv = self._create_test_data()
        
        # Define test configurations
        self.configurations = [
            ("Baseline (1 bps fee)", HFTEnv, self._get_baseline_config, 0.0001, "red"),
            ("4 bps Rebate", HFTEnvRebated004, get_default_rebated_config_004, -0.00004, "orange"),
            ("6 bps Rebate", HFTEnvRebated006, get_default_rebated_config_006, -0.00006, "green"),
            ("8 bps Rebate", HFTEnvRebated008, get_default_rebated_config_008, -0.00008, "blue")
        ]
        
    def _create_test_data(self) -> str:
        """Create standardized test data for fair comparison."""
        # Load real market data
        original_path = "/home/gaen/Documents/billions_db/orderbooks/binance/futures/ethusdc/30-Mar-2025/binance_futures_ethusdc_orderbook_30-Mar-2025.csv"
        df = pd.read_csv(original_path, nrows=5000)  # Use more data for statistical significance
        
        # Rename for compatibility
        df = df.rename(columns={'datetime': 'timestamp'})
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
        
    def _get_baseline_config(self):
        """Get baseline configuration with standard transaction costs."""
        config = get_default_rebated_config_004()  # Use same base config
        config["transaction_cost_long"] = 0.0001   # 1 bps fee
        config["transaction_cost_short"] = 0.0001  # 1 bps fee
        return config
        
    def run_episode(self, env_class, config_func, num_episodes=3, episode_length=1000) -> Dict:
        """Run multiple episodes and collect performance metrics."""
        config = config_func()
        config["csv_path"] = self.test_csv
        config["episode_length"] = episode_length
        config["max_steps"] = 5000
        
        episode_results = []
        
        for episode in range(num_episodes):
            env = env_class(config)
            obs, info = env.reset(seed=42 + episode)  # Different seeds for variation
            
            episode_data = {
                'step': [],
                'reward': [],
                'mtm': [],
                'inventory': [],
                'cash': [],
                'total_rebates': [],
                'rebated_volume': []
            }
            
            # Market making strategy with some randomness
            np.random.seed(42 + episode)
            
            for step in range(episode_length):
                # Adaptive market making strategy
                # Place orders closer to mid when spread is wide
                spread_factor = min(abs(env.best_ask - env.best_bid) / env.best_bid * 10000, 5.0)
                
                # Random variation in strategy
                noise = np.random.normal(0, 0.1, 2)
                buy_offset = 0.2 + noise[0] + spread_factor * 0.05
                sell_offset = -0.2 + noise[1] - spread_factor * 0.05
                
                # Volume based on inventory (inventory management)
                inv_penalty = abs(env.inventory) / env.config["max_inventory"]
                volume_adj = max(0.3, 1.0 - inv_penalty)
                
                action = [
                    np.clip(buy_offset, -1, 1),   # Buy offset
                    np.clip(sell_offset, -1, 1),  # Sell offset  
                    np.clip(0.6 * volume_adj, 0, 1),  # Buy size
                    np.clip(0.6 * volume_adj, 0, 1),  # Sell size
                    -1.0,  # No cancel
                    -1.0   # No do nothing
                ]
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Record data
                episode_data['step'].append(step)
                episode_data['reward'].append(reward)
                episode_data['mtm'].append(info.get('mtm', env.cash))
                episode_data['inventory'].append(env.inventory)
                episode_data['cash'].append(env.cash)
                episode_data['total_rebates'].append(info.get('total_rebates_earned', 0.0))
                episode_data['rebated_volume'].append(info.get('rebated_volume', 0.0))
                
                if terminated or truncated:
                    break
            
            episode_results.append(episode_data)
        
        # Aggregate results
        return self._aggregate_episode_results(episode_results)
        
    def _aggregate_episode_results(self, episode_results: List[Dict]) -> Dict:
        """Aggregate results across multiple episodes."""
        aggregated = {}
        
        # Calculate means and std across episodes
        for key in episode_results[0].keys():
            if key == 'step':
                aggregated[key] = episode_results[0][key]  # Steps are the same
            else:
                # Stack episode data and calculate statistics
                stacked = np.array([ep[key] for ep in episode_results])
                aggregated[f'{key}_mean'] = np.mean(stacked, axis=0)
                aggregated[f'{key}_std'] = np.std(stacked, axis=0)
                aggregated[f'{key}_final'] = stacked[:, -1]  # Final values
        
        return aggregated
        
    def run_full_analysis(self) -> Dict:
        """Run complete analysis across all rebate configurations."""
        print("Running comprehensive rebate analysis...")
        
        results = {}
        
        for name, env_class, config_func, transaction_cost, color in self.configurations:
            print(f"\nTesting {name}...")
            
            try:
                result = self.run_episode(env_class, config_func)
                result['name'] = name
                result['transaction_cost'] = transaction_cost
                result['color'] = color
                results[name] = result
                
                # Print summary
                final_pnl = np.mean(result['mtm_final']) - 100000  # Subtract initial capital
                final_rebates = np.mean(result['total_rebates_final']) if 'total_rebates_final' in result else 0
                final_volume = np.mean(result['rebated_volume_final']) if 'rebated_volume_final' in result else 0
                
                print(f"  Final P&L: ${final_pnl:.2f}")
                print(f"  Total Rebates: ${final_rebates:.4f}")
                print(f"  Trading Volume: {final_volume:.2f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        return results
        
    def create_visualizations(self, results: Dict):
        """Create comprehensive visualizations for academic paper."""
        
        # 1. P&L Comparison Over Time
        plt.figure(figsize=(12, 8))
        
        for name, result in results.items():
            if 'mtm_mean' in result:
                pnl = result['mtm_mean'] - 100000  # Convert to P&L
                steps = result['step']
                plt.plot(steps, pnl, label=name, color=result['color'], linewidth=2)
                
                # Add confidence intervals
                pnl_std = result['mtm_std']
                plt.fill_between(steps, pnl - pnl_std, pnl + pnl_std, 
                               color=result['color'], alpha=0.2)
        
        plt.xlabel('Trading Steps')
        plt.ylabel('Cumulative P&L ($)')
        plt.title('Market Making Performance by Rebate Structure')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pnl_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Economic Viability Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final P&L comparison
        names = []
        final_pnls = []
        rebate_rates = []
        colors = []
        
        for name, result in results.items():
            if 'mtm_final' in result:
                names.append(name)
                final_pnl = np.mean(result['mtm_final']) - 100000
                final_pnls.append(final_pnl)
                rebate_rates.append(result['transaction_cost'] * -10000)  # Convert to bps
                colors.append(result['color'])
        
        bars1 = ax1.bar(names, final_pnls, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Final P&L ($)')
        ax1.set_title('Final P&L by Rebate Structure')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, pnl in zip(bars1, final_pnls):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -15),
                    f'${pnl:.0f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Economic viability threshold
        rebate_points = [-1, 0, 4, 6, 8]
        pnl_points = [final_pnls[i] if i < len(final_pnls) else 0 for i in range(len(rebate_points))]
        
        ax2.plot(rebate_points, pnl_points, 'o-', linewidth=2, markersize=8)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax2.set_xlabel('Rebate Rate (bps)')
        ax2.set_ylabel('Final P&L ($)')
        ax2.set_title('Economic Viability vs Rebate Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/economic_viability.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Trading Activity Analysis
        plt.figure(figsize=(12, 6))
        
        for name, result in results.items():
            if 'rebated_volume_mean' in result:
                volume = result['rebated_volume_mean']
                steps = result['step']
                plt.plot(steps, volume, label=name, color=result['color'], linewidth=2)
        
        plt.xlabel('Trading Steps')
        plt.ylabel('Cumulative Trading Volume')
        plt.title('Trading Volume by Rebate Structure')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/trading_volume.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_statistics_table(self, results: Dict) -> pd.DataFrame:
        """Generate detailed statistics table for academic paper."""
        stats_data = []
        
        for name, result in results.items():
            if 'mtm_final' in result:
                final_mtm = result['mtm_final']
                final_pnl = final_mtm - 100000
                
                row = {
                    'Configuration': name,
                    'Transaction Cost (bps)': result['transaction_cost'] * 10000,
                    'Mean Final P&L ($)': np.mean(final_pnl),
                    'Std Final P&L ($)': np.std(final_pnl),
                    'Min Final P&L ($)': np.min(final_pnl),
                    'Max Final P&L ($)': np.max(final_pnl),
                    'Profitable Episodes': np.sum(final_pnl > 0),
                    'Total Episodes': len(final_pnl)
                }
                
                # Add rebate-specific metrics
                if 'total_rebates_final' in result:
                    rebates = result['total_rebates_final']
                    volume = result['rebated_volume_final']
                    row['Mean Rebates Earned ($)'] = np.mean(rebates)
                    row['Mean Trading Volume'] = np.mean(volume)
                else:
                    row['Mean Rebates Earned ($)'] = 0.0
                    row['Mean Trading Volume'] = 0.0
                
                stats_data.append(row)
        
        df = pd.DataFrame(stats_data)
        
        # Save to CSV
        df.to_csv(f'{self.output_dir}/rebate_analysis_statistics.csv', index=False)
        
        # Save formatted table for LaTeX
        latex_table = df.to_latex(index=False, float_format='{:.2f}'.format)
        with open(f'{self.output_dir}/rebate_analysis_table.tex', 'w') as f:
            f.write(latex_table)
        
        return df
        
    def cleanup(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_csv):
            os.unlink(self.test_csv)

def main():
    """Main analysis function."""
    print("Rebate Impact Analysis for Academic Paper")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = RebateAnalysis()
    
    try:
        # Run comprehensive analysis
        results = analyzer.run_full_analysis()
        
        if not results:
            print("No results generated. Check configurations.")
            return
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        analyzer.create_visualizations(results)
        
        # Generate statistics table
        print("\nGenerating statistics table...")
        stats_df = analyzer.generate_statistics_table(results)
        print("\nStatistics Summary:")
        print(stats_df.to_string(index=False))
        
        # Print summary insights
        print(f"\n=== Key Insights ===")
        baseline_pnl = None
        for name, result in results.items():
            if 'Baseline' in name and 'mtm_final' in result:
                baseline_pnl = np.mean(result['mtm_final']) - 100000
                break
        
        if baseline_pnl is not None:
            print(f"Baseline (1 bps fee) P&L: ${baseline_pnl:.2f}")
            
            for name, result in results.items():
                if 'Rebate' in name and 'mtm_final' in result:
                    rebate_pnl = np.mean(result['mtm_final']) - 100000
                    improvement = rebate_pnl - baseline_pnl
                    print(f"{name} P&L: ${rebate_pnl:.2f} (Δ${improvement:+.2f})")
        
        print(f"\nResults saved to: {analyzer.output_dir}/")
        print("Files generated:")
        print("  - pnl_comparison.png: P&L comparison chart")
        print("  - economic_viability.png: Economic viability analysis")
        print("  - trading_volume.png: Trading volume comparison")
        print("  - rebate_analysis_statistics.csv: Detailed statistics")
        print("  - rebate_analysis_table.tex: LaTeX formatted table")
        
    finally:
        # Cleanup
        analyzer.cleanup()

if __name__ == "__main__":
    main()