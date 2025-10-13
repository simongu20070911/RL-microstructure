#!/usr/bin/env python3
"""
Comprehensive Data Extractor for RL Trading Research Paper
Extracts and organizes data from all experimental runs for publication analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob
from tensorboard.backend.event_processing import event_accumulator
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentDataExtractor:
    def __init__(self, base_path="/home/gaen/Documents/RL", output_path="/home/gaen/Documents/RL/paper_analysis"):
        self.base_path = base_path
        self.output_path = output_path
        self.experiments_path = os.path.join(base_path, "experiments/prev_loggged")
        self.current_logs_path = os.path.join(base_path, "logs")
        
        # Create output directories
        os.makedirs(os.path.join(output_path, "raw_results"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "data_extraction"), exist_ok=True)
        
    def discover_experiments(self):
        """Discover all experimental runs from both prev_loggged and current logs."""
        experiments = []
        
        # Previous logged experiments
        if os.path.exists(self.experiments_path):
            for exp_dir in os.listdir(self.experiments_path):
                exp_path = os.path.join(self.experiments_path, exp_dir)
                if os.path.isdir(exp_path):
                    experiments.append({
                        'id': exp_dir,
                        'path': exp_path,
                        'source': 'prev_loggged',
                        'timestamp': exp_dir
                    })
        
        # Current logs
        if os.path.exists(self.current_logs_path):
            for exp_dir in os.listdir(self.current_logs_path):
                exp_path = os.path.join(self.current_logs_path, exp_dir)
                if os.path.isdir(exp_path):
                    experiments.append({
                        'id': exp_dir,
                        'path': exp_path,
                        'source': 'current_logs',
                        'timestamp': exp_dir
                    })
        
        logger.info(f"Discovered {len(experiments)} experiments")
        return experiments
    
    def extract_config(self, exp_path):
        """Extract configuration from experiment directory."""
        config_path = os.path.join(exp_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        return {}
    
    def extract_tensorboard_data(self, exp_path):
        """Extract all scalar data from TensorBoard event files."""
        event_files = glob(os.path.join(exp_path, "events.out.tfevents.*"))
        if not event_files:
            return {}
        
        all_scalars = {}
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(
                    event_file, 
                    size_guidance={'scalars': 0}
                )
                ea.Reload()
                
                scalar_tags = ea.Tags().get('scalars', [])
                for tag in scalar_tags:
                    scalars = ea.Scalars(tag)
                    all_scalars[tag] = [
                        {'step': s.step, 'value': s.value, 'wall_time': s.wall_time}
                        for s in scalars
                    ]
            except Exception as e:
                logger.warning(f"Failed to process {event_file}: {e}")
        
        return all_scalars
    
    def extract_progress_csv(self, exp_path):
        """Extract data from progress.csv if available."""
        progress_path = os.path.join(exp_path, "progress.csv")
        if os.path.exists(progress_path):
            try:
                return pd.read_csv(progress_path)
            except Exception as e:
                logger.warning(f"Failed to load progress CSV from {progress_path}: {e}")
        return None
    
    def categorize_experiment(self, config, exp_id):
        """Categorize experiment based on configuration."""
        category = {
            'environment_type': 'unknown',
            'agent_type': 'unknown',
            'fee_structure': 'unknown',
            'special_features': []
        }
        
        if config:
            # Environment categorization
            env_path = config.get('env_path', '')
            if '2sided' in env_path:
                if 'nocheat' in env_path:
                    category['environment_type'] = '2sided_nocheat'
                else:
                    category['environment_type'] = '2sided'
            elif 'taker_only' in env_path:
                category['environment_type'] = 'taker_only'
            elif 'post' in env_path:
                category['environment_type'] = 'post_only'
            
            # Fee structure
            long_cost = config.get('transaction_cost_long', 0)
            short_cost = config.get('transaction_cost_short', 0)
            if long_cost == short_cost and long_cost > 0:
                category['fee_structure'] = f'uniform_{long_cost}'
            elif long_cost != short_cost:
                category['fee_structure'] = f'asymmetric_L{long_cost}_S{short_cost}'
            else:
                category['fee_structure'] = 'no_fees'
            
            # Special features
            if config.get('quoting_reward_enabled', False):
                category['special_features'].append('quoting_rewards')
            if config.get('explicit_cancel_enabled', False):
                category['special_features'].append('explicit_cancel')
            if config.get('inventory_penalty', 0) > 0:
                category['special_features'].append('inventory_penalty')
        
        return category
    
    def extract_final_metrics(self, tensorboard_data, progress_data):
        """Extract key final performance metrics."""
        metrics = {}
        
        # From TensorBoard data
        if 'rollout/ep_rew_mean' in tensorboard_data:
            rewards = tensorboard_data['rollout/ep_rew_mean']
            if rewards:
                metrics['final_reward_mean'] = rewards[-1]['value']
                metrics['max_reward_mean'] = max(r['value'] for r in rewards)
                metrics['reward_trend'] = rewards[-1]['value'] - rewards[0]['value'] if len(rewards) > 1 else 0
        
        if 'validation/mean_pnl' in tensorboard_data:
            pnl_data = tensorboard_data['validation/mean_pnl']
            if pnl_data:
                metrics['final_validation_pnl'] = pnl_data[-1]['value']
                metrics['best_validation_pnl'] = max(p['value'] for p in pnl_data)
        
        if 'validation/mean_reward' in tensorboard_data:
            val_rewards = tensorboard_data['validation/mean_reward']
            if val_rewards:
                metrics['final_validation_reward'] = val_rewards[-1]['value']
                metrics['best_validation_reward'] = max(r['value'] for r in val_rewards)
        
        # Training stability metrics
        if 'train/actor_loss' in tensorboard_data:
            actor_losses = tensorboard_data['train/actor_loss']
            if actor_losses:
                metrics['final_actor_loss'] = actor_losses[-1]['value']
                metrics['actor_loss_stability'] = np.std([l['value'] for l in actor_losses[-100:]] if len(actor_losses) >= 100 else [l['value'] for l in actor_losses])
        
        if 'train/critic_loss' in tensorboard_data:
            critic_losses = tensorboard_data['train/critic_loss']
            if critic_losses:
                metrics['final_critic_loss'] = critic_losses[-1]['value']
        
        # From progress CSV
        if progress_data is not None and not progress_data.empty:
            if 'ep_rew_mean' in progress_data.columns:
                metrics['csv_final_reward'] = progress_data['ep_rew_mean'].iloc[-1]
                metrics['csv_reward_volatility'] = progress_data['ep_rew_mean'].std()
        
        return metrics
    
    def process_all_experiments(self):
        """Process all discovered experiments and create comprehensive dataset."""
        experiments = self.discover_experiments()
        
        results = []
        failed_experiments = []
        
        for i, exp in enumerate(experiments):
            logger.info(f"Processing experiment {i+1}/{len(experiments)}: {exp['id']}")
            
            try:
                # Extract all data
                config = self.extract_config(exp['path'])
                tensorboard_data = self.extract_tensorboard_data(exp['path'])
                progress_data = self.extract_progress_csv(exp['path'])
                category = self.categorize_experiment(config, exp['id'])
                final_metrics = self.extract_final_metrics(tensorboard_data, progress_data)
                
                # Check for model files
                model_files = {
                    'best_model': os.path.exists(os.path.join(exp['path'], 'best_model.zip')),
                    'final_model': os.path.exists(os.path.join(exp['path'], 'final_model.zip')),
                    'interrupted_model': os.path.exists(os.path.join(exp['path'], 'interrupted_model.zip'))
                }
                
                result = {
                    'experiment_id': exp['id'],
                    'experiment_path': exp['path'],
                    'source': exp['source'],
                    'timestamp': exp['timestamp'],
                    'config': config,
                    'category': category,
                    'final_metrics': final_metrics,
                    'model_files': model_files,
                    'tensorboard_tags': list(tensorboard_data.keys()),
                    'has_progress_csv': progress_data is not None,
                    'processing_time': datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Save individual experiment data
                exp_output_path = os.path.join(self.output_path, "raw_results", f"{exp['id']}.json")
                with open(exp_output_path, 'w') as f:
                    # Create a serializable version
                    serializable_result = result.copy()
                    serializable_result['tensorboard_data'] = tensorboard_data
                    if progress_data is not None:
                        serializable_result['progress_data'] = progress_data.to_dict('records')
                    json.dump(serializable_result, f, indent=2, default=str)
                
            except Exception as e:
                logger.error(f"Failed to process experiment {exp['id']}: {e}")
                failed_experiments.append({'id': exp['id'], 'error': str(e)})
        
        # Save summary
        summary = {
            'total_experiments': len(experiments),
            'successful_extractions': len(results),
            'failed_extractions': len(failed_experiments),
            'failed_experiments': failed_experiments,
            'extraction_time': datetime.now().isoformat(),
            'experiments_summary': results
        }
        
        summary_path = os.path.join(self.output_path, "data_extraction", "extraction_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Extraction complete: {len(results)} successful, {len(failed_experiments)} failed")
        return results, failed_experiments


def main():
    """Main execution function."""
    extractor = ExperimentDataExtractor()
    results, failures = extractor.process_all_experiments()
    
    print(f"\n=== EXTRACTION COMPLETE ===")
    print(f"Total experiments processed: {len(results) + len(failures)}")
    print(f"Successful extractions: {len(results)}")
    print(f"Failed extractions: {len(failures)}")
    
    if results:
        # Quick statistics
        categories = {}
        for result in results:
            env_type = result['category']['environment_type']
            categories[env_type] = categories.get(env_type, 0) + 1
        
        print(f"\nExperiment breakdown by environment:")
        for env_type, count in categories.items():
            print(f"  {env_type}: {count}")
    
    print(f"\nResults saved to: /home/gaen/Documents/RL/paper_analysis/")


if __name__ == "__main__":
    main()