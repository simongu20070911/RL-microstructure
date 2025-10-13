#!/usr/bin/env python3
"""
Data Cleaner and Validator for RL Trading Research
Identifies and cleans problematic experiments, validates data quality, and flags unrealistic results.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingDataCleanerValidator:
    def __init__(self, data_path="/home/gaen/Documents/RL/paper_analysis"):
        self.data_path = data_path
        self.output_path = os.path.join(data_path, "data_validation")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load the raw experiment data
        self.df = self.load_experiment_data()
        
    def load_experiment_data(self):
        """Load the comprehensive experiment dataset."""
        df_path = os.path.join(self.data_path, "statistical_analysis", "experiments_dataframe.csv")
        if os.path.exists(df_path):
            return pd.read_csv(df_path)
        else:
            logger.error("Experiment DataFrame not found")
            return pd.DataFrame()
    
    def identify_problematic_experiments(self):
        """Identify experiments with data quality issues."""
        problems = {
            'missing_transaction_costs': [],
            'zero_transaction_costs': [],
            'unrealistic_profits': [],
            'missing_environment_info': [],
            'validation_issues': [],
            'configuration_problems': []
        }
        
        for idx, row in self.df.iterrows():
            exp_id = row['experiment_id']
            
            # Check for missing transaction costs
            if pd.isna(row['transaction_cost_long']) or pd.isna(row['transaction_cost_short']):
                problems['missing_transaction_costs'].append({
                    'id': exp_id,
                    'pnl': row['final_validation_pnl'],
                    'source': row['source'],
                    'issue': 'NaN transaction costs'
                })
            
            # Check for zero transaction costs
            elif (row['transaction_cost_long'] == 0.0 and row['transaction_cost_short'] == 0.0):
                problems['zero_transaction_costs'].append({
                    'id': exp_id,
                    'pnl': row['final_validation_pnl'],
                    'source': row['source'],
                    'issue': 'Zero transaction costs'
                })
            
            # Check for unrealistic profits (likely indicating missing costs)
            if row['final_validation_pnl'] > 1000:
                problems['unrealistic_profits'].append({
                    'id': exp_id,
                    'pnl': row['final_validation_pnl'],
                    'source': row['source'],
                    'env_type': row['environment_type'],
                    'transaction_cost_long': row['transaction_cost_long'],
                    'transaction_cost_short': row['transaction_cost_short']
                })
            
            # Check for missing environment information
            if row['environment_type'] == 'unknown':
                problems['missing_environment_info'].append({
                    'id': exp_id,
                    'pnl': row['final_validation_pnl'],
                    'source': row['source'],
                    'issue': 'Environment type unknown'
                })
            
            # Check for validation issues
            if pd.isna(row['final_validation_pnl']) and row['training_completed']:
                problems['validation_issues'].append({
                    'id': exp_id,
                    'issue': 'Training completed but no validation PnL',
                    'source': row['source']
                })
        
        return problems
    
    def analyze_data_quality_by_source(self):
        """Analyze data quality issues grouped by source."""
        analysis = {}
        
        source_groups = self.df.groupby('source')
        
        for source, group in source_groups:
            # Basic stats
            total_experiments = len(group)
            completed_experiments = group['training_completed'].sum()
            
            # Data quality issues
            missing_costs = (pd.isna(group['transaction_cost_long']) | pd.isna(group['transaction_cost_short'])).sum()
            zero_costs = ((group['transaction_cost_long'] == 0.0) & (group['transaction_cost_short'] == 0.0)).sum()
            unknown_envs = (group['environment_type'] == 'unknown').sum()
            unrealistic_pnl = (group['final_validation_pnl'] > 1000).sum()
            
            # Performance stats
            pnl_values = group['final_validation_pnl'].dropna()
            
            analysis[source] = {
                'total_experiments': total_experiments,
                'completed_experiments': int(completed_experiments),
                'completion_rate': completed_experiments / total_experiments,
                'data_quality_issues': {
                    'missing_transaction_costs': int(missing_costs),
                    'zero_transaction_costs': int(zero_costs),
                    'unknown_environments': int(unknown_envs),
                    'unrealistic_pnl_count': int(unrealistic_pnl)
                },
                'data_quality_score': self.calculate_data_quality_score(group),
                'performance_stats': {
                    'mean_pnl': float(pnl_values.mean()) if len(pnl_values) > 0 else None,
                    'max_pnl': float(pnl_values.max()) if len(pnl_values) > 0 else None,
                    'min_pnl': float(pnl_values.min()) if len(pnl_values) > 0 else None,
                    'std_pnl': float(pnl_values.std()) if len(pnl_values) > 0 else None
                },
                'reliability_assessment': self.assess_source_reliability(group)
            }
        
        return analysis
    
    def calculate_data_quality_score(self, group):
        """Calculate a data quality score (0-1) for a group of experiments."""
        total_experiments = len(group)
        if total_experiments == 0:
            return 0.0
        
        # Quality factors (higher is better)
        has_transaction_costs = (~pd.isna(group['transaction_cost_long']) & ~pd.isna(group['transaction_cost_short'])).sum()
        non_zero_costs = ((group['transaction_cost_long'] > 0) | (group['transaction_cost_short'] > 0)).sum()
        known_environments = (group['environment_type'] != 'unknown').sum()
        realistic_pnl = (group['final_validation_pnl'] <= 1000).sum()
        has_validation_data = (~pd.isna(group['final_validation_pnl'])).sum()
        
        # Weighted scoring
        score = (
            0.25 * (has_transaction_costs / total_experiments) +
            0.25 * (non_zero_costs / total_experiments) +
            0.20 * (known_environments / total_experiments) +
            0.20 * (realistic_pnl / total_experiments) +
            0.10 * (has_validation_data / total_experiments)
        )
        
        return float(score)
    
    def assess_source_reliability(self, group):
        """Assess the reliability of experiments from a particular source."""
        pnl_values = group['final_validation_pnl'].dropna()
        
        if len(pnl_values) == 0:
            return "no_data"
        
        # Check for multiple red flags
        red_flags = 0
        
        # Flag 1: Unrealistic profits
        if (pnl_values > 1000).any():
            red_flags += 1
        
        # Flag 2: No transaction costs
        if ((group['transaction_cost_long'] == 0.0) & (group['transaction_cost_short'] == 0.0)).any():
            red_flags += 1
        
        # Flag 3: Missing environment info
        if (group['environment_type'] == 'unknown').any():
            red_flags += 1
        
        # Flag 4: Extremely high volatility
        if len(pnl_values) > 1 and pnl_values.std() > pnl_values.mean() * 3:
            red_flags += 1
        
        # Assess reliability
        if red_flags == 0:
            return "high_reliability"
        elif red_flags == 1:
            return "medium_reliability"
        elif red_flags == 2:
            return "low_reliability"
        else:
            return "unreliable"
    
    def create_cleaned_dataset(self):
        """Create a cleaned dataset by removing/flagging problematic experiments."""
        df_clean = self.df.copy()
        
        # Add data quality flags
        df_clean['data_quality_flags'] = ''
        df_clean['is_reliable'] = True
        df_clean['exclusion_reason'] = ''
        
        for idx, row in df_clean.iterrows():
            flags = []
            reliable = True
            exclusion_reasons = []
            
            # Flag missing transaction costs
            if pd.isna(row['transaction_cost_long']) or pd.isna(row['transaction_cost_short']):
                flags.append('missing_costs')
                reliable = False
                exclusion_reasons.append('Missing transaction cost data')
            
            # Flag zero transaction costs
            elif row['transaction_cost_long'] == 0.0 and row['transaction_cost_short'] == 0.0:
                flags.append('zero_costs')
                reliable = False
                exclusion_reasons.append('Zero transaction costs - unrealistic')
            
            # Flag unrealistic profits
            if row['final_validation_pnl'] > 1000:
                flags.append('unrealistic_profits')
                reliable = False
                exclusion_reasons.append(f'Unrealistic PnL: {row["final_validation_pnl"]:.1f}')
            
            # Flag unknown environments
            if row['environment_type'] == 'unknown':
                flags.append('unknown_env')
                # Don't mark as unreliable just for this - might still have valid data
            
            # Flag prev_loggged source as potentially problematic
            if row['source'] == 'prev_loggged':
                flags.append('legacy_source')
                # Only mark unreliable if it also has other issues
            
            df_clean.loc[idx, 'data_quality_flags'] = ','.join(flags)
            df_clean.loc[idx, 'is_reliable'] = reliable
            df_clean.loc[idx, 'exclusion_reason'] = '; '.join(exclusion_reasons)
        
        return df_clean
    
    def extract_realistic_experiments_only(self, df_clean):
        """Extract only the realistic, reliable experiments."""
        realistic_experiments = df_clean[
            (df_clean['is_reliable'] == True) &
            (df_clean['final_validation_pnl'] <= 1000) &
            (df_clean['transaction_cost_long'] > 0) &
            (df_clean['transaction_cost_short'] > 0) &
            (~pd.isna(df_clean['final_validation_pnl']))
        ].copy()
        
        logger.info(f"Extracted {len(realistic_experiments)} realistic experiments from {len(df_clean)} total")
        return realistic_experiments
    
    def generate_data_validation_report(self, problems, source_analysis, df_clean):
        """Generate a comprehensive data validation report."""
        report_lines = []
        
        report_lines.append("# Data Quality Validation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total experiments analyzed: {len(self.df)}")
        report_lines.append("")
        
        # Executive Summary
        total_problems = sum(len(problem_list) for problem_list in problems.values())
        reliable_experiments = len(df_clean[df_clean['is_reliable'] == True])
        
        report_lines.append("## Executive Summary")
        report_lines.append(f"- **Total Data Quality Issues Found**: {total_problems}")
        report_lines.append(f"- **Reliable Experiments**: {reliable_experiments}/{len(df_clean)} ({reliable_experiments/len(df_clean):.1%})")
        report_lines.append(f"- **Critical Issue**: {len(problems['unrealistic_profits'])} experiments with unrealistic profits (>1000 PnL)")
        report_lines.append(f"- **Missing Costs**: {len(problems['missing_transaction_costs'])} experiments with missing transaction cost data")
        report_lines.append("")
        
        # Critical Issues Section
        report_lines.append("## 🚨 Critical Issues Requiring Immediate Attention")
        report_lines.append("")
        
        # Unrealistic Profits
        if problems['unrealistic_profits']:
            report_lines.append("### Experiments with Unrealistic Profits (>1000 PnL)")
            report_lines.append("| Experiment ID | PnL | Source | Environment | Long Cost | Short Cost |")
            report_lines.append("|---------------|-----|--------|-------------|-----------|------------|")
            
            for exp in sorted(problems['unrealistic_profits'], key=lambda x: x['pnl'], reverse=True):
                long_cost = exp['transaction_cost_long'] if not pd.isna(exp['transaction_cost_long']) else 'NaN'
                short_cost = exp['transaction_cost_short'] if not pd.isna(exp['transaction_cost_short']) else 'NaN'
                report_lines.append(f"| {exp['id']} | {exp['pnl']:.1f} | {exp['source']} | {exp['env_type']} | {long_cost} | {short_cost} |")
            
            report_lines.append("")
            report_lines.append("**Analysis**: These experiments likely have missing or inadequate transaction cost implementation.")
            report_lines.append("")
        
        # Source Quality Analysis
        report_lines.append("## Data Quality by Source")
        report_lines.append("")
        report_lines.append("| Source | Experiments | Completion Rate | Data Quality Score | Reliability | Issues |")
        report_lines.append("|--------|-------------|-----------------|--------------------|-----------| -------|")
        
        for source, analysis in source_analysis.items():
            quality_score = analysis['data_quality_score']
            reliability = analysis['reliability_assessment']
            issues = analysis['data_quality_issues']
            issue_count = sum(issues.values())
            
            report_lines.append(f"| {source} | {analysis['total_experiments']} | {analysis['completion_rate']:.1%} | {quality_score:.2f} | {reliability} | {issue_count} |")
        
        report_lines.append("")
        
        # Detailed Problem Breakdown
        report_lines.append("## Detailed Problem Breakdown")
        report_lines.append("")
        
        for problem_type, problem_list in problems.items():
            if problem_list:
                report_lines.append(f"### {problem_type.replace('_', ' ').title()} ({len(problem_list)} experiments)")
                
                if problem_type == 'unrealistic_profits':
                    max_pnl = max(exp['pnl'] for exp in problem_list)
                    min_pnl = min(exp['pnl'] for exp in problem_list)
                    report_lines.append(f"- PnL range: {min_pnl:.1f} to {max_pnl:.1f}")
                    
                    sources = {}
                    for exp in problem_list:
                        sources[exp['source']] = sources.get(exp['source'], 0) + 1
                    report_lines.append(f"- By source: {dict(sources)}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        report_lines.append("### Immediate Actions Required:")
        report_lines.append("1. **Exclude unrealistic experiments**: Remove experiments with PnL > 1000 from analysis")
        report_lines.append("2. **Fix transaction cost implementation**: Ensure all environments have realistic trading costs")
        report_lines.append("3. **Validate prev_loggged data**: Review all experiments from legacy source for data quality")
        report_lines.append("4. **Re-run problematic experiments**: Use corrected environments with proper cost implementation")
        report_lines.append("")
        
        report_lines.append("### For Future Experiments:")
        report_lines.append("1. **Mandatory transaction costs**: All environments must implement realistic transaction costs (1-10 bps)")
        report_lines.append("2. **Data validation pipeline**: Implement automatic validation checks before experiment storage")
        report_lines.append("3. **Performance sanity checks**: Flag experiments with unrealistic profits for review")
        report_lines.append("4. **Environment categorization**: Ensure proper environment type classification")
        report_lines.append("")
        
        # Clean Dataset Summary
        reliable_count = len(df_clean[df_clean['is_reliable'] == True])
        report_lines.append("## Clean Dataset Summary")
        report_lines.append(f"- **Reliable experiments**: {reliable_count}")
        report_lines.append(f"- **Flagged experiments**: {len(df_clean) - reliable_count}")
        report_lines.append(f"- **Recommended for analysis**: Use only reliable experiments")
        
        return '\n'.join(report_lines)
    
    def run_comprehensive_validation(self):
        """Run comprehensive data validation and cleaning."""
        logger.info("Starting comprehensive data validation...")
        
        # Identify problems
        problems = self.identify_problematic_experiments()
        
        # Analyze by source
        source_analysis = self.analyze_data_quality_by_source()
        
        # Create cleaned dataset
        df_clean = self.create_cleaned_dataset()
        
        # Extract realistic experiments
        df_realistic = self.extract_realistic_experiments_only(df_clean)
        
        # Save results
        results = {
            'problems_identified': problems,
            'source_analysis': source_analysis,
            'validation_metadata': {
                'total_experiments': len(self.df),
                'reliable_experiments': len(df_clean[df_clean['is_reliable'] == True]),
                'realistic_experiments': len(df_realistic),
                'validation_date': datetime.now().isoformat()
            }
        }
        
        # Save comprehensive results
        results_path = os.path.join(self.output_path, "data_validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save cleaned datasets
        clean_path = os.path.join(self.output_path, "experiments_cleaned.csv")
        df_clean.to_csv(clean_path, index=False)
        
        realistic_path = os.path.join(self.output_path, "experiments_realistic_only.csv")
        df_realistic.to_csv(realistic_path, index=False)
        
        # Generate validation report
        validation_report = self.generate_data_validation_report(problems, source_analysis, df_clean)
        report_path = os.path.join(self.output_path, "data_validation_report.md")
        with open(report_path, 'w') as f:
            f.write(validation_report)
        
        logger.info(f"Data validation complete. Results saved to {self.output_path}")
        
        # Print critical summary
        unrealistic_count = len(problems['unrealistic_profits'])
        reliable_count = len(df_clean[df_clean['is_reliable'] == True])
        
        print(f"\n🚨 CRITICAL FINDINGS:")
        print(f"   - {unrealistic_count} experiments with unrealistic profits (>1000 PnL)")
        print(f"   - {len(self.df) - reliable_count} experiments flagged as unreliable")
        print(f"   - Only {reliable_count}/{len(self.df)} experiments are reliable for analysis")
        print(f"   - {len(df_realistic)} experiments meet realistic trading criteria")
        
        return results, df_clean, df_realistic


def main():
    """Main execution function."""
    validator = TradingDataCleanerValidator()
    results, df_clean, df_realistic = validator.run_comprehensive_validation()
    
    print(f"\n=== DATA VALIDATION COMPLETE ===")
    print(f"Results saved to: /home/gaen/Documents/RL/paper_analysis/data_validation/")
    print(f"Clean dataset: experiments_cleaned.csv")
    print(f"Realistic only: experiments_realistic_only.csv")
    print(f"Validation report: data_validation_report.md")


if __name__ == "__main__":
    main()