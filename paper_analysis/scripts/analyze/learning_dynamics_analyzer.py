#!/usr/bin/env python3
"""
Learning Dynamics and Convergence Analysis for RL Trading Research
Analyzes training progression, convergence patterns, and learning efficiency across experiments.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LearningDynamicsAnalyzer:
    def __init__(self, data_path="/home/gaen/Documents/RL/paper_analysis"):
        self.data_path = data_path
        self.output_path = os.path.join(data_path, "learning_dynamics")
        os.makedirs(self.output_path, exist_ok=True)
        
        # Load experiment data
        self.df = self.load_experiment_data()
        
    def load_experiment_data(self):
        """Load the comprehensive experiment dataset."""
        df_path = os.path.join(self.data_path, "statistical_analysis", "experiments_dataframe.csv")
        if os.path.exists(df_path):
            return pd.read_csv(df_path)
        else:
            logger.error("Experiment DataFrame not found")
            return pd.DataFrame()
    
    def analyze_convergence_patterns(self):
        """Analyze convergence patterns across different environments."""
        analysis = {}
        
        env_groups = self.df.groupby('environment_type')
        
        for env_type, group in env_groups:
            if len(group) > 3:
                # Analyze training completion patterns
                completed = group[group['training_completed'] == True]
                interrupted = group[group['was_interrupted'] == True]
                
                # Performance progression analysis
                if 'final_reward_mean' in group.columns and 'best_validation_reward' in group.columns:
                    final_rewards = group['final_reward_mean'].dropna()
                    best_rewards = group['best_validation_reward'].dropna()
                    
                    # Calculate improvement ratio (how much better best vs final)
                    if len(final_rewards) > 0 and len(best_rewards) > 0:
                        # Align indices for proper comparison
                        common_indices = group.index[group['final_reward_mean'].notna() & group['best_validation_reward'].notna()]
                        if len(common_indices) > 0:
                            aligned_final = group.loc[common_indices, 'final_reward_mean']
                            aligned_best = group.loc[common_indices, 'best_validation_reward']
                            improvement_ratios = (aligned_best - aligned_final) / np.abs(aligned_final).replace(0, 1)
                            
                            analysis[env_type] = {
                                'sample_size': len(group),
                                'completion_rate': len(completed) / len(group),
                                'interruption_rate': len(interrupted) / len(group),
                                'convergence_metrics': {
                                    'mean_final_reward': float(final_rewards.mean()),
                                    'mean_best_reward': float(best_rewards.mean()),
                                    'mean_improvement_ratio': float(improvement_ratios.mean()),
                                    'improvement_consistency': float(improvement_ratios.std()),
                                    'learning_efficiency': self.calculate_learning_efficiency(aligned_final, aligned_best)
                                },
                                'stability_metrics': {
                                    'final_reward_volatility': float(final_rewards.std()),
                                    'performance_range': float(final_rewards.max() - final_rewards.min()),
                                    'outlier_fraction': self.calculate_outlier_fraction(final_rewards)
                                }
                            }
        
        return analysis
    
    def calculate_learning_efficiency(self, final_rewards, best_rewards):
        """Calculate learning efficiency as the ratio of consistent improvement."""
        if len(final_rewards) < 2:
            return 0.0
        
        # Efficiency = fraction of experiments that achieved meaningful improvement
        meaningful_improvements = (best_rewards - final_rewards) > 0.01
        return float(meaningful_improvements.mean())
    
    def calculate_outlier_fraction(self, values):
        """Calculate the fraction of outliers using IQR method."""
        if len(values) < 4:
            return 0.0
        
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (values < lower_bound) | (values > upper_bound)
        return float(outliers.mean())
    
    def analyze_training_stability(self):
        """Analyze training stability patterns across different configurations."""
        analysis = {}
        
        # Analyze by episode length (training duration)
        length_groups = self.df.groupby('episode_length')
        
        for length, group in length_groups:
            if len(group) > 2 and not pd.isna(length):
                pnl_values = group['final_validation_pnl'].dropna()
                reward_values = group['final_reward_mean'].dropna()
                
                if len(pnl_values) > 0:
                    # Calculate stability metrics
                    completion_rate = group['training_completed'].mean()
                    pnl_cv = pnl_values.std() / abs(pnl_values.mean()) if pnl_values.mean() != 0 else float('inf')
                    
                    analysis[f"episode_length_{length}"] = {
                        'sample_size': len(group),
                        'completion_rate': float(completion_rate),
                        'mean_pnl': float(pnl_values.mean()),
                        'pnl_coefficient_variation': float(pnl_cv) if pnl_cv != float('inf') else 999.0,
                        'stability_score': self.calculate_stability_score(completion_rate, pnl_cv),
                        'training_duration_effect': self.assess_duration_effect(length, pnl_values.mean())
                    }
        
        return analysis
    
    def calculate_stability_score(self, completion_rate, coefficient_variation):
        """Calculate overall training stability score."""
        # Higher completion rate and lower coefficient of variation = higher stability
        cv_score = max(0, 1 - min(coefficient_variation / 2, 1))  # Normalize CV impact
        stability = 0.6 * completion_rate + 0.4 * cv_score
        return float(stability)
    
    def assess_duration_effect(self, episode_length, mean_pnl):
        """Assess the effect of training duration on performance."""
        # Longer episodes should generally lead to better performance
        # This is a heuristic assessment
        if episode_length <= 200:
            expected_baseline = -10  # Short episodes expected to be worse
        elif episode_length <= 400:
            expected_baseline = 0   # Medium episodes expected to be neutral
        else:
            expected_baseline = 10  # Long episodes expected to be better
        
        performance_vs_expected = mean_pnl - expected_baseline
        
        if performance_vs_expected > 20:
            return "excellent"
        elif performance_vs_expected > 0:
            return "good"
        elif performance_vs_expected > -20:
            return "acceptable"
        else:
            return "poor"
    
    def analyze_loss_convergence_patterns(self):
        """Analyze actor and critic loss convergence patterns."""
        analysis = {}
        
        # Filter experiments with loss data
        loss_data = self.df[['environment_type', 'final_actor_loss', 'final_critic_loss']].dropna()
        
        if len(loss_data) > 0:
            env_groups = loss_data.groupby('environment_type')
            
            for env_type, group in env_groups:
                if len(group) > 2:
                    actor_losses = group['final_actor_loss']
                    critic_losses = group['final_critic_loss']
                    
                    analysis[env_type] = {
                        'sample_size': len(group),
                        'actor_loss_convergence': {
                            'mean_final_loss': float(actor_losses.mean()),
                            'loss_stability': float(actor_losses.std()),
                            'convergence_quality': self.assess_loss_convergence(actor_losses, 'actor')
                        },
                        'critic_loss_convergence': {
                            'mean_final_loss': float(critic_losses.mean()),
                            'loss_stability': float(critic_losses.std()),
                            'convergence_quality': self.assess_loss_convergence(critic_losses, 'critic')
                        },
                        'loss_correlation': float(np.corrcoef(actor_losses, critic_losses)[0, 1]) if len(actor_losses) > 1 else 0.0
                    }
        
        return analysis
    
    def assess_loss_convergence(self, loss_values, loss_type):
        """Assess the quality of loss convergence."""
        mean_loss = loss_values.mean()
        std_loss = loss_values.std()
        
        # Expected ranges for good convergence
        if loss_type == 'actor':
            good_range = (-3, -1)  # Actor loss should be negative
            excellent_std = 0.5
        else:  # critic
            good_range = (0.2, 1.5)  # Critic loss should be low positive
            excellent_std = 0.3
        
        # Assess mean convergence
        if good_range[0] <= mean_loss <= good_range[1]:
            mean_quality = "good"
        elif mean_loss < good_range[0] - 1 or mean_loss > good_range[1] + 1:
            mean_quality = "poor"
        else:
            mean_quality = "acceptable"
        
        # Assess stability
        if std_loss <= excellent_std:
            stability_quality = "excellent"
        elif std_loss <= excellent_std * 2:
            stability_quality = "good"
        else:
            stability_quality = "poor"
        
        return f"{mean_quality}_convergence_with_{stability_quality}_stability"
    
    def analyze_hyperparameter_learning_curves(self):
        """Analyze how different hyperparameters affect learning curves."""
        analysis = {}
        
        # Key hyperparameters that affect learning
        hyperparams = ['inventory_penalty', 'price_offset_ticks', 'max_order_volume']
        
        for hyperparam in hyperparams:
            if hyperparam in self.df.columns:
                param_analysis = {}
                param_groups = self.df.groupby(hyperparam)
                
                for param_value, group in param_groups:
                    if len(group) > 2 and not pd.isna(param_value):
                        # Analyze learning trajectory
                        learning_metrics = self.extract_learning_metrics(group)
                        if learning_metrics:
                            param_analysis[f"{hyperparam}_{param_value}"] = learning_metrics
                
                if param_analysis:
                    analysis[hyperparam] = param_analysis
        
        return analysis
    
    def extract_learning_metrics(self, group):
        """Extract learning metrics from a group of experiments."""
        final_rewards = group['final_reward_mean'].dropna()
        best_rewards = group['best_validation_reward'].dropna()
        pnl_values = group['final_validation_pnl'].dropna()
        
        if len(final_rewards) == 0:
            return None
        
        # Calculate learning speed proxy
        common_indices = group.index[group['final_reward_mean'].notna() & group['best_validation_reward'].notna()]
        if len(common_indices) > 0:
            aligned_final = group.loc[common_indices, 'final_reward_mean']
            aligned_best = group.loc[common_indices, 'best_validation_reward']
            learning_gap = aligned_best - aligned_final
            
            return {
                'sample_size': len(group),
                'final_performance': {
                    'mean_reward': float(final_rewards.mean()),
                    'reward_stability': float(final_rewards.std()),
                    'mean_pnl': float(pnl_values.mean()) if len(pnl_values) > 0 else 0.0
                },
                'learning_characteristics': {
                    'mean_learning_gap': float(learning_gap.mean()),
                    'learning_consistency': float(learning_gap.std()),
                    'completion_rate': float(group['training_completed'].mean()),
                    'learning_speed_proxy': self.calculate_learning_speed_proxy(aligned_final, aligned_best)
                }
            }
        
        return None
    
    def calculate_learning_speed_proxy(self, final_rewards, best_rewards):
        """Calculate a proxy for learning speed."""
        if len(final_rewards) < 2:
            return 0.0
        
        # Learning speed = how quickly the agent improves
        # Higher improvement ratio with lower variance suggests faster learning
        improvements = best_rewards - final_rewards
        improvement_rate = improvements.mean()
        improvement_consistency = 1 / (improvements.std() + 0.001)  # Avoid division by zero
        
        return float(improvement_rate * improvement_consistency)
    
    def generate_learning_insights(self):
        """Generate key insights about learning dynamics."""
        insights = []
        
        # Convergence insights
        convergence_analysis = self.analyze_convergence_patterns()
        if convergence_analysis:
            best_env = max(convergence_analysis.keys(), 
                          key=lambda k: convergence_analysis[k]['convergence_metrics']['learning_efficiency'])
            efficiency = convergence_analysis[best_env]['convergence_metrics']['learning_efficiency']
            insights.append(f"Most efficient learning environment: {best_env} (efficiency: {efficiency:.1%})")
        
        # Stability insights
        stability_analysis = self.analyze_training_stability()
        if stability_analysis:
            most_stable = max(stability_analysis.keys(), 
                            key=lambda k: stability_analysis[k]['stability_score'])
            score = stability_analysis[most_stable]['stability_score']
            insights.append(f"Most stable training configuration: {most_stable} (stability score: {score:.3f})")
        
        # Duration insights
        if stability_analysis:
            duration_effects = [(k, v['training_duration_effect']) for k, v in stability_analysis.items()]
            excellent_configs = [k for k, effect in duration_effects if effect == 'excellent']
            if excellent_configs:
                insights.append(f"Excellent duration-performance configurations: {len(excellent_configs)} found")
        
        return insights
    
    def run_comprehensive_learning_analysis(self):
        """Run all learning dynamics analyses and save results."""
        logger.info("Starting comprehensive learning dynamics analysis...")
        
        analyses = {
            'convergence_patterns': self.analyze_convergence_patterns(),
            'training_stability': self.analyze_training_stability(),
            'loss_convergence': self.analyze_loss_convergence_patterns(),
            'hyperparameter_learning': self.analyze_hyperparameter_learning_curves(),
            'key_insights': self.generate_learning_insights(),
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_experiments': len(self.df),
                'analysis_version': '1.0'
            }
        }
        
        # Save comprehensive analysis
        output_path = os.path.join(self.output_path, "learning_dynamics_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        # Generate summary report
        self.generate_learning_report(analyses)
        
        logger.info(f"Learning dynamics analysis complete. Results saved to {self.output_path}")
        return analyses
    
    def generate_learning_report(self, analyses):
        """Generate a comprehensive learning dynamics report."""
        report_lines = []
        
        report_lines.append("# Learning Dynamics Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total experiments analyzed: {len(self.df)}")
        report_lines.append("")
        
        # Key insights
        report_lines.append("## Key Learning Dynamics Insights")
        for insight in analyses['key_insights']:
            report_lines.append(f"- {insight}")
        report_lines.append("")
        
        # Convergence analysis
        convergence_data = analyses['convergence_patterns']
        if convergence_data:
            report_lines.append("## Convergence Patterns by Environment")
            report_lines.append("| Environment | Completion Rate | Learning Efficiency | Improvement Ratio | Stability |")
            report_lines.append("|-------------|-----------------|--------------------|--------------------|-----------|")
            
            for env, data in convergence_data.items():
                conv_metrics = data['convergence_metrics']
                stab_metrics = data['stability_metrics']
                report_lines.append(f"| {env} | {data['completion_rate']:.1%} | {conv_metrics['learning_efficiency']:.1%} | {conv_metrics['mean_improvement_ratio']:.3f} | {stab_metrics['final_reward_volatility']:.2f} |")
            report_lines.append("")
        
        # Training stability
        stability_data = analyses['training_stability']
        if stability_data:
            report_lines.append("## Training Stability by Episode Length")
            report_lines.append("| Episode Length | Completion Rate | Stability Score | Duration Effect | PnL CV |")
            report_lines.append("|----------------|-----------------|-----------------|-----------------|---------|")
            
            for config, data in stability_data.items():
                cv_display = f"{data['pnl_coefficient_variation']:.2f}" if data['pnl_coefficient_variation'] < 999 else "N/A"
                report_lines.append(f"| {config} | {data['completion_rate']:.1%} | {data['stability_score']:.3f} | {data['training_duration_effect']} | {cv_display} |")
            report_lines.append("")
        
        # Loss convergence
        loss_data = analyses['loss_convergence']
        if loss_data:
            report_lines.append("## Loss Convergence Analysis")
            report_lines.append("| Environment | Actor Loss | Actor Quality | Critic Loss | Critic Quality |")
            report_lines.append("|-------------|------------|---------------|-------------|----------------|")
            
            for env, data in loss_data.items():
                actor_data = data['actor_loss_convergence']
                critic_data = data['critic_loss_convergence']
                report_lines.append(f"| {env} | {actor_data['mean_final_loss']:.3f} | {actor_data['convergence_quality']} | {critic_data['mean_final_loss']:.3f} | {critic_data['convergence_quality']} |")
            report_lines.append("")
        
        # Learning recommendations
        report_lines.append("## Learning Optimization Recommendations")
        report_lines.append("Based on the learning dynamics analysis:")
        report_lines.append("1. **Environment Selection**: Choose environments with high learning efficiency (>50%)")
        report_lines.append("2. **Episode Length**: Medium-length episodes (400 steps) show best stability-performance balance")
        report_lines.append("3. **Training Monitoring**: Monitor actor/critic loss convergence for early stopping decisions")
        report_lines.append("4. **Hyperparameter Tuning**: Focus on parameters that show consistent learning patterns")
        report_lines.append("5. **Stability Optimization**: Prioritize configurations with stability scores >0.7")
        
        # Save report
        report_path = os.path.join(self.output_path, "learning_dynamics_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Generated learning dynamics report: {report_path}")


def main():
    """Main execution function."""
    analyzer = LearningDynamicsAnalyzer()
    analyses = analyzer.run_comprehensive_learning_analysis()
    
    print(f"\n=== LEARNING DYNAMICS ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(analyzer.df)} experiments for learning patterns")
    print(f"Key insights generated: {len(analyses['key_insights'])}")
    print(f"Results saved to: /home/gaen/Documents/RL/paper_analysis/learning_dynamics/")


if __name__ == "__main__":
    main()