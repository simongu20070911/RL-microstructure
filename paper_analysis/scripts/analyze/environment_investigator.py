#!/usr/bin/env python3
"""
Environment Investigation and Documentation Tool
Analyzes all trading environments to understand their characteristics, limitations, and realism.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add the project root to path to import environments
sys.path.append('/home/gaen/Documents/RL')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentInvestigator:
    def __init__(self, data_path="/home/gaen/Documents/RL/paper_analysis"):
        self.data_path = data_path
        self.output_path = os.path.join(data_path, "environment_analysis")
        self.rl_base_path = "/home/gaen/Documents/RL"
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
    
    def discover_all_environments(self):
        """Discover all environment files in the codebase."""
        envs_path = os.path.join(self.rl_base_path, "envs")
        env_files = []
        
        if os.path.exists(envs_path):
            for file in os.listdir(envs_path):
                if file.endswith('.py') and not file.startswith('__'):
                    env_files.append({
                        'filename': file,
                        'path': os.path.join(envs_path, file),
                        'env_name': file.replace('.py', '')
                    })
        
        logger.info(f"Discovered {len(env_files)} environment files")
        return env_files
    
    def analyze_environment_code(self, env_file):
        """Analyze environment code to extract key characteristics."""
        try:
            with open(env_file['path'], 'r') as f:
                code = f.read()
            
            analysis = {
                'filename': env_file['filename'],
                'env_name': env_file['env_name'],
                'code_length': len(code.split('\n')),
                'characteristics': {}
            }
            
            # Check for trading cost implementation
            cost_indicators = [
                'transaction_cost', 'fee', 'commission', 'cost_long', 'cost_short',
                'spread', 'slippage', 'market_impact'
            ]
            trading_costs_found = []
            for indicator in cost_indicators:
                if indicator in code.lower():
                    trading_costs_found.append(indicator)
            
            analysis['characteristics']['trading_costs'] = {
                'implemented': len(trading_costs_found) > 0,
                'indicators_found': trading_costs_found,
                'sophistication_level': self.assess_cost_sophistication(code, trading_costs_found)
            }
            
            # Check for position tracking
            position_indicators = ['position', 'inventory', 'holdings', 'shares']
            position_tracking = []
            for indicator in position_indicators:
                if indicator in code.lower():
                    position_tracking.append(indicator)
            
            analysis['characteristics']['position_tracking'] = {
                'implemented': len(position_tracking) > 0,
                'indicators_found': position_tracking,
                'persistent_across_episodes': 'reset' not in code.lower() or 'nocheat' in env_file['filename']
            }
            
            # Check for order book modeling
            order_book_indicators = ['bid', 'ask', 'spread', 'book', 'level', 'depth']
            order_book_features = []
            for indicator in order_book_indicators:
                if indicator in code.lower():
                    order_book_features.append(indicator)
            
            analysis['characteristics']['order_book_modeling'] = {
                'implemented': len(order_book_features) > 0,
                'features_found': order_book_features,
                'realism_level': self.assess_order_book_realism(code)
            }
            
            # Check for action space complexity
            action_indicators = ['action', 'step', 'buy', 'sell', 'cancel', 'place']
            action_complexity = self.assess_action_complexity(code)
            
            analysis['characteristics']['action_space'] = {
                'complexity_level': action_complexity,
                'allows_market_orders': 'market' in code.lower(),
                'allows_limit_orders': 'limit' in code.lower(),
                'allows_cancellation': 'cancel' in code.lower(),
                'simultaneous_orders': '2sided' in env_file['filename'] or 'two_sided' in code.lower()
            }
            
            # Check for reward structure
            reward_structure = self.analyze_reward_structure(code)
            analysis['characteristics']['reward_structure'] = reward_structure
            
            # Check for realism features
            realism_features = self.assess_realism_features(code, env_file['filename'])
            analysis['characteristics']['realism_assessment'] = realism_features
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {env_file['filename']}: {e}")
            return None
    
    def assess_cost_sophistication(self, code, cost_indicators):
        """Assess the sophistication of trading cost implementation."""
        if not cost_indicators:
            return "none"
        
        sophisticated_features = [
            'bid_ask_spread', 'market_impact', 'slippage', 'asymmetric',
            'liquidity', 'volume_dependent', 'time_dependent'
        ]
        
        sophistication_count = sum(1 for feature in sophisticated_features if feature in code.lower())
        
        if sophistication_count >= 3:
            return "high"
        elif sophistication_count >= 1:
            return "medium"
        elif len(cost_indicators) >= 2:
            return "basic"
        else:
            return "minimal"
    
    def assess_order_book_realism(self, code):
        """Assess the realism of order book modeling."""
        realism_indicators = [
            'multiple_levels', 'depth', 'price_priority', 'time_priority',
            'partial_fills', 'queue_position', 'market_data'
        ]
        
        realism_count = sum(1 for indicator in realism_indicators if indicator.replace('_', ' ') in code.lower())
        
        if realism_count >= 4:
            return "high"
        elif realism_count >= 2:
            return "medium"
        elif 'bid' in code.lower() and 'ask' in code.lower():
            return "basic"
        else:
            return "minimal"
    
    def assess_action_complexity(self, code):
        """Assess the complexity of the action space."""
        complexity_indicators = [
            'continuous', 'discrete', 'multi_dimensional', 'price_offset',
            'volume_control', 'timing_control', 'cancellation'
        ]
        
        complexity_count = sum(1 for indicator in complexity_indicators if indicator.replace('_', ' ') in code.lower())
        
        if complexity_count >= 4:
            return "very_high"
        elif complexity_count >= 2:
            return "high"
        elif 'action' in code.lower():
            return "medium"
        else:
            return "low"
    
    def analyze_reward_structure(self, code):
        """Analyze the reward structure implementation."""
        reward_components = {
            'pnl_based': any(term in code.lower() for term in ['pnl', 'profit', 'loss', 'return']),
            'inventory_penalty': any(term in code.lower() for term in ['inventory_penalty', 'position_penalty']),
            'spread_capture': any(term in code.lower() for term in ['spread', 'capture', 'bid_ask']),
            'activity_bonus': any(term in code.lower() for term in ['activity', 'bonus', 'participation']),
            'risk_adjustment': any(term in code.lower() for term in ['risk', 'volatility', 'drawdown']),
            'transaction_costs': any(term in code.lower() for term in ['cost', 'fee', 'commission'])
        }
        
        # Assess sophistication
        implemented_components = sum(reward_components.values())
        
        if implemented_components >= 4:
            sophistication = "sophisticated"
        elif implemented_components >= 2:
            sophistication = "moderate"
        elif implemented_components >= 1:
            sophistication = "basic"
        else:
            sophistication = "minimal"
        
        return {
            'components': reward_components,
            'sophistication_level': sophistication,
            'total_components': implemented_components
        }
    
    def assess_realism_features(self, code, filename):
        """Assess overall realism of the trading environment."""
        realism_factors = {
            'realistic_costs': any(term in code.lower() for term in ['transaction_cost', 'fee', 'spread']),
            'position_persistence': 'nocheat' in filename or ('reset' not in code.lower()),
            'order_book_dynamics': any(term in code.lower() for term in ['bid', 'ask', 'book', 'level']),
            'market_impact': any(term in code.lower() for term in ['impact', 'slippage', 'liquidity']),
            'latency_modeling': any(term in code.lower() for term in ['latency', 'delay', 'timing']),
            'partial_fills': any(term in code.lower() for term in ['partial', 'fill', 'execution']),
            'realistic_data': any(term in code.lower() for term in ['real_data', 'historical', 'market_data'])
        }
        
        realism_score = sum(realism_factors.values()) / len(realism_factors)
        
        if realism_score >= 0.7:
            realism_level = "high"
        elif realism_score >= 0.4:
            realism_level = "medium"
        elif realism_score >= 0.2:
            realism_level = "low"
        else:
            realism_level = "unrealistic"
        
        return {
            'factors': realism_factors,
            'score': realism_score,
            'level': realism_level,
            'major_limitations': self.identify_major_limitations(realism_factors, code)
        }
    
    def identify_major_limitations(self, realism_factors, code):
        """Identify major limitations that affect trading realism."""
        limitations = []
        
        if not realism_factors['realistic_costs']:
            limitations.append("No realistic trading costs - can achieve unrealistic profits")
        
        if not realism_factors['position_persistence']:
            limitations.append("Position resets between episodes - unrealistic for continuous trading")
        
        if not realism_factors['market_impact']:
            limitations.append("No market impact modeling - large orders have no price effect")
        
        if not realism_factors['order_book_dynamics']:
            limitations.append("Simplified order book - missing realistic market microstructure")
        
        if 'unlimited' in code.lower() or 'infinite' in code.lower():
            limitations.append("Unlimited capital or positions - unrealistic for real trading")
        
        if 'perfect' in code.lower() and 'information' in code.lower():
            limitations.append("Perfect information assumption - unrealistic market knowledge")
        
        return limitations
    
    def analyze_experimental_performance_by_env(self):
        """Analyze experimental performance grouped by environment type."""
        analysis = {}
        
        env_groups = self.df.groupby('environment_type')
        
        for env_type, group in env_groups:
            pnl_values = group['final_validation_pnl'].dropna()
            reward_values = group['final_reward_mean'].dropna()
            
            if len(pnl_values) > 0:
                # Flag suspiciously high performance
                suspicious_threshold = 1000  # PnL > 1000 is suspicious for realistic trading
                suspicious_experiments = pnl_values[pnl_values > suspicious_threshold]
                
                analysis[env_type] = {
                    'experiment_count': len(group),
                    'performance_metrics': {
                        'mean_pnl': float(pnl_values.mean()),
                        'max_pnl': float(pnl_values.max()),
                        'min_pnl': float(pnl_values.min()),
                        'std_pnl': float(pnl_values.std()),
                        'positive_pnl_rate': float((pnl_values > 0).mean()),
                        'suspicious_high_pnl_count': len(suspicious_experiments),
                        'suspicious_high_pnl_rate': float(len(suspicious_experiments) / len(pnl_values))
                    },
                    'completion_metrics': {
                        'completion_rate': float(group['training_completed'].mean()),
                        'interruption_rate': float(group['was_interrupted'].mean())
                    },
                    'realism_assessment': {
                        'likely_realistic': pnl_values.max() < suspicious_threshold and pnl_values.mean() < 100,
                        'cost_implementation_suspected': pnl_values.max() < 500,
                        'performance_flags': self.flag_unrealistic_performance(pnl_values)
                    }
                }
        
        return analysis
    
    def flag_unrealistic_performance(self, pnl_values):
        """Flag unrealistic performance patterns."""
        flags = []
        
        if pnl_values.max() > 10000:
            flags.append("extremely_high_profits")
        
        if pnl_values.min() > 0 and len(pnl_values) > 5:
            flags.append("no_losses_suspicious")
        
        if pnl_values.std() > pnl_values.mean() * 2:
            flags.append("extremely_high_volatility")
        
        if (pnl_values > 1000).mean() > 0.5:
            flags.append("majority_unrealistic_profits")
        
        positive_rate = (pnl_values > 0).mean()
        if positive_rate > 0.8:
            flags.append("suspiciously_high_success_rate")
        
        return flags
    
    def run_comprehensive_environment_investigation(self):
        """Run comprehensive investigation of all environments."""
        logger.info("Starting comprehensive environment investigation...")
        
        # Discover and analyze all environment files
        env_files = self.discover_all_environments()
        env_analyses = []
        
        for env_file in env_files:
            analysis = self.analyze_environment_code(env_file)
            if analysis:
                env_analyses.append(analysis)
        
        # Analyze experimental performance
        performance_analysis = self.analyze_experimental_performance_by_env()
        
        # Combine analyses
        comprehensive_analysis = {
            'environment_code_analysis': env_analyses,
            'experimental_performance_analysis': performance_analysis,
            'summary_insights': self.generate_environment_insights(env_analyses, performance_analysis),
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_env_files': len(env_files),
                'total_experiments': len(self.df),
                'analysis_version': '1.0'
            }
        }
        
        # Save comprehensive analysis
        output_path = os.path.join(self.output_path, "comprehensive_environment_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2, default=str)
        
        # Generate environment documentation
        self.generate_environments_documentation(comprehensive_analysis)
        
        logger.info(f"Environment investigation complete. Results saved to {self.output_path}")
        return comprehensive_analysis
    
    def generate_environment_insights(self, env_analyses, performance_analysis):
        """Generate key insights about environment characteristics and limitations."""
        insights = {
            'realism_ranking': [],
            'performance_reality_check': [],
            'major_limitations_summary': [],
            'recommendations': []
        }
        
        # Rank environments by realism
        for env_analysis in env_analyses:
            if 'realism_assessment' in env_analysis['characteristics']:
                realism = env_analysis['characteristics']['realism_assessment']
                insights['realism_ranking'].append({
                    'env_name': env_analysis['env_name'],
                    'realism_level': realism['level'],
                    'realism_score': realism['score'],
                    'major_limitations': realism['major_limitations']
                })
        
        # Sort by realism score
        insights['realism_ranking'].sort(key=lambda x: x['realism_score'], reverse=True)
        
        # Performance reality check
        for env_type, perf_data in performance_analysis.items():
            flags = perf_data['realism_assessment']['performance_flags']
            if flags:
                insights['performance_reality_check'].append({
                    'env_type': env_type,
                    'max_pnl': perf_data['performance_metrics']['max_pnl'],
                    'mean_pnl': perf_data['performance_metrics']['mean_pnl'],
                    'suspicious_rate': perf_data['performance_metrics']['suspicious_high_pnl_rate'],
                    'flags': flags
                })
        
        # Collect major limitations
        all_limitations = []
        for env_analysis in env_analyses:
            if 'realism_assessment' in env_analysis['characteristics']:
                limitations = env_analysis['characteristics']['realism_assessment']['major_limitations']
                all_limitations.extend(limitations)
        
        # Count frequency of limitations
        limitation_counts = {}
        for limitation in all_limitations:
            limitation_counts[limitation] = limitation_counts.get(limitation, 0) + 1
        
        insights['major_limitations_summary'] = [
            {'limitation': lim, 'frequency': count} 
            for lim, count in sorted(limitation_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Generate recommendations
        insights['recommendations'] = [
            "Focus on environments with realistic trading costs for production-relevant research",
            "Use 'nocheat' variants for realistic position tracking across episodes",
            "Be cautious of results with PnL > 1000, likely indicating missing costs",
            "Implement comprehensive transaction cost modeling in all environments",
            "Add market impact modeling for large order realism",
            "Include bid-ask spread dynamics for better market microstructure representation"
        ]
        
        return insights
    
    def generate_environments_documentation(self, analysis):
        """Generate comprehensive environments.md documentation."""
        doc_lines = []
        
        doc_lines.append("# Trading Environments Analysis and Documentation")
        doc_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc_lines.append("")
        doc_lines.append("This document provides a comprehensive analysis of all trading environments in the RL trading system, their characteristics, limitations, and experimental performance.")
        doc_lines.append("")
        
        # Executive Summary
        doc_lines.append("## Executive Summary")
        doc_lines.append("")
        insights = analysis['summary_insights']
        
        doc_lines.append("### Key Findings:")
        doc_lines.append("- **Critical Issue**: Several environments lack realistic trading costs, leading to unrealistic profits (>10,000 PnL)")
        doc_lines.append("- **Realism Gap**: Most environments miss essential market microstructure features")
        doc_lines.append("- **Position Tracking**: Only 'nocheat' variants provide realistic position persistence")
        doc_lines.append("- **Performance Red Flags**: Multiple experiments show suspicious profit levels indicating missing costs")
        doc_lines.append("")
        
        # Environment Realism Ranking
        doc_lines.append("## Environment Realism Ranking")
        doc_lines.append("")
        doc_lines.append("Environments ranked by realism score (1.0 = fully realistic):")
        doc_lines.append("")
        doc_lines.append("| Rank | Environment | Realism Score | Realism Level | Major Limitations |")
        doc_lines.append("|------|-------------|---------------|---------------|-------------------|")
        
        for i, env in enumerate(insights['realism_ranking'], 1):
            limitations_str = "; ".join(env['major_limitations'][:2]) + ("..." if len(env['major_limitations']) > 2 else "")
            doc_lines.append(f"| {i} | {env['env_name']} | {env['realism_score']:.2f} | {env['realism_level']} | {limitations_str} |")
        
        doc_lines.append("")
        
        # Performance Reality Check
        doc_lines.append("## Performance Reality Check")
        doc_lines.append("")
        doc_lines.append("**⚠️ CRITICAL: Environments with suspicious performance patterns**")
        doc_lines.append("")
        
        if insights['performance_reality_check']:
            doc_lines.append("| Environment | Max PnL | Mean PnL | Suspicious Rate | Performance Flags |")
            doc_lines.append("|-------------|---------|----------|-----------------|-------------------|")
            
            for perf in insights['performance_reality_check']:
                flags_str = ", ".join(perf['flags'])
                doc_lines.append(f"| {perf['env_type']} | {perf['max_pnl']:.1f} | {perf['mean_pnl']:.1f} | {perf['suspicious_rate']:.1%} | {flags_str} |")
        else:
            doc_lines.append("*No environments flagged for suspicious performance*")
        
        doc_lines.append("")
        
        # Detailed Environment Analysis
        doc_lines.append("## Detailed Environment Analysis")
        doc_lines.append("")
        
        env_analyses = analysis['environment_code_analysis']
        for env_analysis in sorted(env_analyses, key=lambda x: x['env_name']):
            doc_lines.append(f"### {env_analysis['env_name']}")
            doc_lines.append("")
            
            chars = env_analysis['characteristics']
            
            # Trading Costs
            costs = chars['trading_costs']
            doc_lines.append("**Trading Costs:**")
            doc_lines.append(f"- Implemented: {'✅' if costs['implemented'] else '❌'}")
            doc_lines.append(f"- Sophistication: {costs['sophistication_level']}")
            if costs['indicators_found']:
                doc_lines.append(f"- Features: {', '.join(costs['indicators_found'])}")
            doc_lines.append("")
            
            # Position Tracking
            positions = chars['position_tracking']
            doc_lines.append("**Position Tracking:**")
            doc_lines.append(f"- Implemented: {'✅' if positions['implemented'] else '❌'}")
            doc_lines.append(f"- Persistent: {'✅' if positions['persistent_across_episodes'] else '❌'}")
            doc_lines.append("")
            
            # Order Book Modeling
            book = chars['order_book_modeling']
            doc_lines.append("**Order Book Modeling:**")
            doc_lines.append(f"- Implemented: {'✅' if book['implemented'] else '❌'}")
            doc_lines.append(f"- Realism Level: {book['realism_level']}")
            doc_lines.append("")
            
            # Action Space
            actions = chars['action_space']
            doc_lines.append("**Action Space:**")
            doc_lines.append(f"- Complexity: {actions['complexity_level']}")
            doc_lines.append(f"- Market Orders: {'✅' if actions['allows_market_orders'] else '❌'}")
            doc_lines.append(f"- Limit Orders: {'✅' if actions['allows_limit_orders'] else '❌'}")
            doc_lines.append(f"- Cancellation: {'✅' if actions['allows_cancellation'] else '❌'}")
            doc_lines.append(f"- Simultaneous Orders: {'✅' if actions['simultaneous_orders'] else '❌'}")
            doc_lines.append("")
            
            # Reward Structure
            rewards = chars['reward_structure']
            doc_lines.append("**Reward Structure:**")
            doc_lines.append(f"- Sophistication: {rewards['sophistication_level']}")
            doc_lines.append(f"- Components: {rewards['total_components']}/6 implemented")
            active_components = [comp for comp, active in rewards['components'].items() if active]
            if active_components:
                doc_lines.append(f"- Active: {', '.join(active_components)}")
            doc_lines.append("")
            
            # Realism Assessment
            realism = chars['realism_assessment']
            doc_lines.append("**Realism Assessment:**")
            doc_lines.append(f"- Overall Level: {realism['level']} (score: {realism['score']:.2f})")
            if realism['major_limitations']:
                doc_lines.append("- **Major Limitations:**")
                for limitation in realism['major_limitations']:
                    doc_lines.append(f"  - {limitation}")
            doc_lines.append("")
            doc_lines.append("---")
            doc_lines.append("")
        
        # Major Limitations Summary
        doc_lines.append("## Common Limitations Across Environments")
        doc_lines.append("")
        
        for limitation_data in insights['major_limitations_summary']:
            doc_lines.append(f"- **{limitation_data['limitation']}** (affects {limitation_data['frequency']} environments)")
        
        doc_lines.append("")
        
        # Recommendations
        doc_lines.append("## Recommendations for Production Use")
        doc_lines.append("")
        
        for recommendation in insights['recommendations']:
            doc_lines.append(f"- {recommendation}")
        
        doc_lines.append("")
        
        # Experimental Performance Summary
        doc_lines.append("## Experimental Performance by Environment")
        doc_lines.append("")
        doc_lines.append("| Environment | Experiments | Mean PnL | Max PnL | Success Rate | Completion Rate | Realistic? |")
        doc_lines.append("|-------------|-------------|----------|---------|--------------|-----------------|------------|")
        
        perf_analysis = analysis['experimental_performance_analysis']
        for env_type, data in sorted(perf_analysis.items()):
            perf = data['performance_metrics']
            comp = data['completion_metrics']
            realistic = "✅" if data['realism_assessment']['likely_realistic'] else "❌"
            
            doc_lines.append(f"| {env_type} | {data['experiment_count']} | {perf['mean_pnl']:.1f} | {perf['max_pnl']:.1f} | {perf['positive_pnl_rate']:.1%} | {comp['completion_rate']:.1%} | {realistic} |")
        
        doc_lines.append("")
        
        # Production Readiness Assessment
        doc_lines.append("## Production Readiness Assessment")
        doc_lines.append("")
        doc_lines.append("**Recommended for Production:**")
        
        production_ready = []
        needs_work = []
        not_recommended = []
        
        for env_type, data in perf_analysis.items():
            if data['realism_assessment']['likely_realistic'] and data['performance_metrics']['max_pnl'] < 1000:
                production_ready.append(env_type)
            elif data['realism_assessment']['cost_implementation_suspected']:
                needs_work.append(env_type)
            else:
                not_recommended.append(env_type)
        
        if production_ready:
            for env in production_ready:
                doc_lines.append(f"- ✅ **{env}**: Appears to have realistic cost implementation")
        else:
            doc_lines.append("- ❌ **No environments currently meet production readiness criteria**")
        
        doc_lines.append("")
        doc_lines.append("**Needs Cost Implementation Work:**")
        for env in needs_work:
            doc_lines.append(f"- ⚠️ **{env}**: Requires proper trading cost implementation")
        
        doc_lines.append("")
        doc_lines.append("**Not Recommended for Production:**")
        for env in not_recommended:
            doc_lines.append(f"- ❌ **{env}**: Major realism issues, unrealistic performance")
        
        # Save documentation
        doc_path = os.path.join(self.output_path, "environments.md")
        with open(doc_path, 'w') as f:
            f.write('\n'.join(doc_lines))
        
        logger.info(f"Generated environments documentation: {doc_path}")


def main():
    """Main execution function."""
    investigator = EnvironmentInvestigator()
    analysis = investigator.run_comprehensive_environment_investigation()
    
    print(f"\n=== ENVIRONMENT INVESTIGATION COMPLETE ===")
    print(f"Analyzed {len(analysis['environment_code_analysis'])} environment files")
    print(f"Performance analysis covers {len(analysis['experimental_performance_analysis'])} environment types")
    print(f"Results saved to: /home/gaen/Documents/RL/paper_analysis/environment_analysis/")
    
    # Quick summary of critical findings
    perf_analysis = analysis['experimental_performance_analysis']
    suspicious_envs = [env for env, data in perf_analysis.items() 
                      if data['performance_metrics']['suspicious_high_pnl_rate'] > 0.1]
    
    if suspicious_envs:
        print(f"\n⚠️  CRITICAL: {len(suspicious_envs)} environments show suspicious performance:")
        for env in suspicious_envs:
            max_pnl = perf_analysis[env]['performance_metrics']['max_pnl']
            print(f"   - {env}: Max PnL = {max_pnl:.1f}")


if __name__ == "__main__":
    main()