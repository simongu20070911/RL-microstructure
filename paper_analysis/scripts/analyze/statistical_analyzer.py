#!/usr/bin/env python3
"""
Statistical Analyzer for RL Trading Research Paper
Performs comprehensive statistical analysis across experimental configurations.
"""

import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingExperimentAnalyzer:
    def __init__(self, data_path="/home/gaen/Documents/RL/paper_analysis"):
        self.data_path = data_path
        self.raw_results_path = os.path.join(data_path, "raw_results")
        self.output_path = os.path.join(data_path, "statistical_analysis")
        os.makedirs(self.output_path, exist_ok=True)
        
    def load_experiment_data(self):
        """Load all experiment data from extracted JSON files."""
        experiment_files = glob(os.path.join(self.raw_results_path, "*.json"))
        experiments = []
        
        for file_path in experiment_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    experiments.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(experiments)} experiments")
        return experiments
    
    def create_performance_dataframe(self, experiments):
        """Create a structured DataFrame with performance metrics."""
        rows = []
        
        for exp in experiments:
            row = {
                'experiment_id': exp['experiment_id'],
                'environment_type': exp['category']['environment_type'],
                'fee_structure': exp['category']['fee_structure'],
                'source': exp['source'],
                'timestamp': exp['timestamp']
            }
            
            # Add final metrics
            for metric, value in exp['final_metrics'].items():
                row[metric] = value
            
            # Add configuration parameters
            config = exp.get('config', {})
            row.update({
                'initial_capital': config.get('initial_capital', None),
                'max_order_volume': config.get('max_order_volume', None),
                'episode_length': config.get('episode_length', None),
                'inventory_penalty': config.get('inventory_penalty', None),
                'transaction_cost_long': config.get('transaction_cost_long', None),
                'transaction_cost_short': config.get('transaction_cost_short', None),
                'price_offset_ticks': config.get('price_offset_ticks', None),
                'quoting_reward_enabled': config.get('quoting_reward_enabled', False),
                'explicit_cancel_enabled': config.get('explicit_cancel_enabled', False)
            })
            
            # Model completion status
            model_files = exp.get('model_files', {})
            row['training_completed'] = model_files.get('final_model', False) or model_files.get('best_model', False)
            row['was_interrupted'] = model_files.get('interrupted_model', False)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        logger.info(f"Created performance DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def analyze_environment_performance(self, df):
        """Compare performance across different environment types."""
        analysis = {}
        
        # Group by environment type
        env_groups = df.groupby('environment_type')
        
        for env_type, group in env_groups:
            if len(group) < 2:
                continue
                
            env_analysis = {
                'count': len(group),
                'completion_rate': group['training_completed'].mean(),
                'interruption_rate': group['was_interrupted'].mean()
            }
            
            # Performance metrics analysis
            for metric in ['final_reward_mean', 'final_validation_pnl', 'best_validation_reward']:
                if metric in group.columns and group[metric].notna().sum() > 0:
                    values = group[metric].dropna()
                    env_analysis[metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median(),
                        'min': values.min(),
                        'max': values.max(),
                        'count': len(values)
                    }
            
            analysis[env_type] = env_analysis
        
        return analysis
    
    def compare_environment_types(self, df):
        """Statistical comparison between environment types."""
        comparisons = {}
        
        env_types = df['environment_type'].unique()
        env_types = [e for e in env_types if e != 'unknown']
        
        for metric in ['final_reward_mean', 'final_validation_pnl', 'best_validation_reward']:
            if metric not in df.columns:
                continue
                
            metric_comparisons = {}
            
            # Pairwise t-tests
            for i, env1 in enumerate(env_types):
                for env2 in env_types[i+1:]:
                    group1 = df[df['environment_type'] == env1][metric].dropna()
                    group2 = df[df['environment_type'] == env2][metric].dropna()
                    
                    if len(group1) >= 3 and len(group2) >= 3:
                        t_stat, p_value = stats.ttest_ind(group1, group2)
                        effect_size = (group1.mean() - group2.mean()) / np.sqrt(((group1.std()**2 + group2.std()**2) / 2))
                        
                        metric_comparisons[f"{env1}_vs_{env2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'effect_size': effect_size,
                            'group1_mean': group1.mean(),
                            'group2_mean': group2.mean(),
                            'group1_n': len(group1),
                            'group2_n': len(group2)
                        }
            
            # ANOVA if more than 2 groups
            if len(env_types) > 2:
                groups = [df[df['environment_type'] == env][metric].dropna() for env in env_types]
                groups = [g for g in groups if len(g) >= 3]
                
                if len(groups) > 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    metric_comparisons['anova'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            comparisons[metric] = metric_comparisons
        
        return comparisons
    
    def analyze_fee_impact(self, df):
        """Analyze the impact of different fee structures."""
        analysis = {}
        
        # Group by fee structure
        fee_groups = df.groupby('fee_structure')
        
        for fee_type, group in fee_groups:
            if len(group) < 2:
                continue
                
            fee_analysis = {
                'count': len(group),
                'environments': group['environment_type'].unique().tolist()
            }
            
            # Performance impact
            for metric in ['final_reward_mean', 'final_validation_pnl']:
                if metric in group.columns and group[metric].notna().sum() > 0:
                    values = group[metric].dropna()
                    fee_analysis[metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'median': values.median()
                    }
            
            analysis[fee_type] = fee_analysis
        
        return analysis
    
    def analyze_hyperparameter_sensitivity(self, df):
        """Analyze sensitivity to key hyperparameters."""
        sensitivity_analysis = {}
        
        # Key hyperparameters to analyze
        hyperparams = [
            'inventory_penalty',
            'price_offset_ticks',
            'max_order_volume',
            'episode_length'
        ]
        
        target_metrics = ['final_reward_mean', 'final_validation_pnl', 'best_validation_reward']
        
        for hyperparam in hyperparams:
            if hyperparam not in df.columns:
                continue
                
            param_analysis = {}
            
            # Remove NaN values
            param_df = df[[hyperparam] + target_metrics].dropna()
            
            if len(param_df) < 10:
                continue
                
            for metric in target_metrics:
                if metric in param_df.columns:
                    correlation, p_value = stats.pearsonr(param_df[hyperparam], param_df[metric])
                    
                    param_analysis[metric] = {
                        'correlation': correlation,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'sample_size': len(param_df)
                    }
            
            sensitivity_analysis[hyperparam] = param_analysis
        
        return sensitivity_analysis
    
    def generate_summary_statistics(self, df):
        """Generate comprehensive summary statistics."""
        summary = {
            'total_experiments': len(df),
            'unique_environments': df['environment_type'].nunique(),
            'completion_rate': df['training_completed'].mean(),
            'interruption_rate': df['was_interrupted'].mean(),
            'date_range': {
                'earliest': df['timestamp'].min(),
                'latest': df['timestamp'].max()
            }
        }
        
        # Performance metrics summary
        performance_metrics = ['final_reward_mean', 'final_validation_pnl', 'best_validation_reward']
        
        for metric in performance_metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary[metric] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'count': int(len(values)),
                        'positive_results': int((values > 0).sum()),
                        'negative_results': int((values < 0).sum())
                    }
        
        return summary
    
    def run_comprehensive_analysis(self):
        """Run all analyses and save results."""
        logger.info("Starting comprehensive statistical analysis...")
        
        # Load data
        experiments = self.load_experiment_data()
        df = self.create_performance_dataframe(experiments)
        
        # Save the main DataFrame
        df_path = os.path.join(self.output_path, "experiments_dataframe.csv")
        df.to_csv(df_path, index=False)
        logger.info(f"Saved experiments DataFrame to {df_path}")
        
        # Run analyses
        analyses = {
            'summary_statistics': self.generate_summary_statistics(df),
            'environment_performance': self.analyze_environment_performance(df),
            'environment_comparisons': self.compare_environment_types(df),
            'fee_impact_analysis': self.analyze_fee_impact(df),
            'hyperparameter_sensitivity': self.analyze_hyperparameter_sensitivity(df),
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_experiments_analyzed': len(df),
                'analysis_version': '1.0'
            }
        }
        
        # Save comprehensive analysis
        analysis_path = os.path.join(self.output_path, "comprehensive_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        logger.info(f"Saved comprehensive analysis to {analysis_path}")
        
        # Generate summary report
        self.generate_summary_report(analyses, df)
        
        return analyses, df
    
    def generate_summary_report(self, analyses, df):
        """Generate a human-readable summary report."""
        report_lines = []
        
        report_lines.append("# RL Trading Experiments - Statistical Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary statistics
        summary = analyses['summary_statistics']
        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Total experiments: {summary['total_experiments']}")
        report_lines.append(f"- Unique environment types: {summary['unique_environments']}")
        report_lines.append(f"- Training completion rate: {summary['completion_rate']:.1%}")
        report_lines.append(f"- Interruption rate: {summary['interruption_rate']:.1%}")
        report_lines.append("")
        
        # Environment breakdown
        env_counts = df['environment_type'].value_counts()
        report_lines.append("## Environment Type Breakdown")
        for env_type, count in env_counts.items():
            report_lines.append(f"- {env_type}: {count} experiments")
        report_lines.append("")
        
        # Performance metrics
        if 'final_reward_mean' in summary:
            reward_stats = summary['final_reward_mean']
            report_lines.append("## Final Reward Performance")
            report_lines.append(f"- Mean: {reward_stats['mean']:.4f}")
            report_lines.append(f"- Std: {reward_stats['std']:.4f}")
            report_lines.append(f"- Median: {reward_stats['median']:.4f}")
            report_lines.append(f"- Range: [{reward_stats['min']:.4f}, {reward_stats['max']:.4f}]")
            report_lines.append(f"- Positive results: {reward_stats['positive_results']}/{reward_stats['count']} ({reward_stats['positive_results']/reward_stats['count']:.1%})")
            report_lines.append("")
        
        # Best performing experiments
        report_lines.append("## Top Performing Experiments")
        if 'final_validation_pnl' in df.columns:
            top_pnl = df.nlargest(5, 'final_validation_pnl')[['experiment_id', 'environment_type', 'final_validation_pnl']]
            for _, row in top_pnl.iterrows():
                report_lines.append(f"- {row['experiment_id']} ({row['environment_type']}): PnL = {row['final_validation_pnl']:.4f}")
        report_lines.append("")
        
        # Statistical significance findings
        comparisons = analyses.get('environment_comparisons', {})
        if comparisons:
            report_lines.append("## Significant Environment Differences")
            for metric, comparisons_data in comparisons.items():
                significant_pairs = [k for k, v in comparisons_data.items() 
                                   if isinstance(v, dict) and v.get('significant', False)]
                if significant_pairs:
                    report_lines.append(f"### {metric}")
                    for pair in significant_pairs:
                        comp_data = comparisons_data[pair]
                        effect_size_text = f", effect_size={comp_data['effect_size']:.4f}" if 'effect_size' in comp_data else ""
                        report_lines.append(f"- {pair}: p={comp_data['p_value']:.4f}{effect_size_text}")
            report_lines.append("")
        
        # Save report
        report_path = os.path.join(self.output_path, "analysis_summary_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved summary report to {report_path}")


def main():
    """Main execution function."""
    analyzer = TradingExperimentAnalyzer()
    analyses, df = analyzer.run_comprehensive_analysis()
    
    print(f"\n=== STATISTICAL ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(df)} experiments")
    print(f"Environment types: {', '.join(df['environment_type'].unique())}")
    print(f"Results saved to: /home/gaen/Documents/RL/paper_analysis/statistical_analysis/")


if __name__ == "__main__":
    main()