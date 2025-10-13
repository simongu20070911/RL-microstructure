#!/usr/bin/env python3
"""
Extended RL Training for Rebated HFT Environment
Institutional-grade training with 50M timesteps and comprehensive monitoring
"""

import logging
import os
import time
from copy import deepcopy
import json
from datetime import datetime
import torch
import numpy as np
import psutil
import GPUtil
import gymnasium as gym
import importlib
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure as configure_logger
from stable_baselines3.common.callbacks import BaseCallback

# Import our optimized configuration
import sys
sys.path.append('/home/gaen/Documents/RL/envs/env_rebated')
from final_optimized_config import FINAL_OPTIMIZED_CONFIG

# Set up extended logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Extended configuration for institutional-grade training
extended_config = deepcopy(FINAL_OPTIMIZED_CONFIG)

# Update for extended training
extended_config.update({
    # === EXTENDED TRAINING PARAMETERS ===
    'csv_path': '/home/gaen/Documents/RL/orderbook_trimmed_large.csv',
    'initial_capital': 1000000,
    'max_steps': 50000,  # Use more data
    'episode_length': 2000,  # Longer episodes for complex strategies
    
    # === ENHANCED ARCHITECTURE ===
    'hidden_dim': 512,        # Larger hidden dimensions
    'features_dim': 256,      # Larger feature space
    'lstm_layers': 2,         # Deeper LSTM
    'num_heads': 8,           # More attention heads
    'net_arch_pi': [1024, 1024, 512, 256],  # Deeper policy network
    'net_arch_qf': [1024, 1024, 512, 256],  # Deeper value network
    
    # === EXTENDED TRAINING HYPERPARAMETERS ===
    'batch_size': None,       # Will be auto-calculated
    'total_timesteps': 50000000,  # 50M timesteps for deep learning
    'log_interval': 100,      # Log every 100 episodes
    'validation_frequency': 50000,    # Validate every 50k steps
    'checkpoint_frequency': 500000,   # Save every 500k steps
    
    # === ENVIRONMENT PATH ===
    'env_path': 'envs.env_rebated.env_rebated_unified',
    'env_class': 'RebatedHFTEnv',
})

def get_env_class(module_path, class_name):
    """Dynamically import the environment class."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        logging.error(f"Could not import environment module: {module_path}. Error: {e}")
        raise
    except AttributeError:
        logging.error(f"Could not find class '{class_name}' in module '{module_path}'.")
        raise

def get_optimal_batch_size(memory_per_sample_gb=0.002, max_default_bs=1024, reserve_factor=0.7):
    """Calculate optimal batch size for extended training."""
    if not torch.cuda.is_available():
        logging.warning("GPU not available, using CPU with smaller batch size.")
        return 256
    try:
        gpu = GPUtil.getGPUs()[0]
        free_memory_gb = (gpu.memoryFree / 1024) * reserve_factor
        optimal_size = int(free_memory_gb / memory_per_sample_gb)
        optimal_size = max(128, min(optimal_size, max_default_bs))
        logging.info(f"GPU Free Memory: {gpu.memoryFree/1024:.2f}GB. Calculated Optimal Batch Size: {optimal_size}")
        return optimal_size
    except Exception as e:
        logging.warning(f"GPU info unavailable, using default batch size. Error: {e}")
        return 512

def setup_extended_training_dir(base_log_dir="logs", config_to_save=None):
    """Create extended training directory with comprehensive logging."""
    run_name = f"extended_rebated_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    logging.info(f"Extended training log directory: {log_dir}")

    if config_to_save:
        config_path = os.path.join(log_dir, "extended_config.json")
        try:
            config_copy = deepcopy(config_to_save)
            with open(config_path, 'w') as f:
                json.dump(config_copy, f, indent=4)
            logging.info(f"Extended configuration saved to {config_path}")
        except Exception as e:
            logging.error(f"Failed to save extended configuration: {e}")

    return log_dir

class ExtendedValidationCallback(BaseCallback):
    """Extended validation callback for long-term training monitoring."""
    
    def __init__(self, eval_env, eval_freq=50000, n_eval_episodes=10, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.validation_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self.validation_count += 1
            
            # Run validation episodes
            episode_rewards = []
            episode_pnls = []
            total_rebates = []
            
            for episode in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0
                initial_capital = info.get('mtm', extended_config['initial_capital'])
                
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                
                final_mtm = info.get('mtm', initial_capital)
                episode_pnl = final_mtm - initial_capital
                rebates_earned = getattr(self.eval_env, 'total_rebates_earned', 0)
                
                episode_rewards.append(episode_reward)
                episode_pnls.append(episode_pnl)
                total_rebates.append(rebates_earned)
            
            # Calculate validation metrics
            mean_reward = np.mean(episode_rewards)
            mean_pnl = np.mean(episode_pnls)
            mean_rebates = np.mean(total_rebates)
            
            # Log extended validation metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_pnl", mean_pnl)
            self.logger.record("eval/mean_rebates", mean_rebates)
            self.logger.record("eval/validation_count", self.validation_count)
            self.logger.record("eval/total_timesteps", self.num_timesteps)
            
            # Progress tracking
            progress_pct = (self.num_timesteps / extended_config['total_timesteps']) * 100
            self.logger.record("progress/completion_pct", progress_pct)
            
            logging.info(f"Extended Validation #{self.validation_count} at {self.num_timesteps:,} steps:")
            logging.info(f"  Mean Reward: {mean_reward:.4f}")
            logging.info(f"  Mean PnL: {mean_pnl:.2f}")
            logging.info(f"  Mean Rebates: {mean_rebates:.4f}")
            logging.info(f"  Progress: {progress_pct:.1f}%")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.logger.get_dir(), "best_extended_model.zip")
                self.model.save(best_model_path)
                logging.info(f"New best model saved: {mean_reward:.4f}")
        
        return True

def create_extended_agent(env, config, log_dir):
    """Create SAC agent with extended architecture."""
    
    # Calculate optimal batch size
    if config["batch_size"] is None:
        config["batch_size"] = get_optimal_batch_size()
    
    logging.info(f"Creating extended SAC agent with batch size: {config['batch_size']}")
    
    # Import the enhanced feature extractor
    sys.path.append('/home/gaen/Documents/RL/agents')
    from agent_2sided import CachedLSTMAttention
    
    # Enhanced policy kwargs for extended training
    policy_kwargs = {
        "features_extractor_class": CachedLSTMAttention,
        "features_extractor_kwargs": {
            "features_dim": config["features_dim"],
            "hidden_dim": config["hidden_dim"],
            "lstm_layers": config["lstm_layers"],
            "num_heads": config["num_heads"],
            "max_attn_cache_len": 2000,  # Longer sequence for extended episodes
        },
        "net_arch": {
            "pi": config["net_arch_pi"],
            "qf": config["net_arch_qf"]
        }
    }
    
    # Create extended SAC agent
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=min(5000000, config["batch_size"] * 10000),  # 5M buffer for extended training
        batch_size=config["batch_size"],
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device='auto'
    )
    
    logging.info("Extended SAC agent created successfully")
    return model

def run_extended_training():
    """Run the extended RL training experiment."""
    
    print("🚀 STARTING EXTENDED REBATED HFT TRAINING")
    print("=" * 80)
    print(f"Target timesteps: {extended_config['total_timesteps']:,}")
    print(f"Episode length: {extended_config['episode_length']:,}")
    print(f"Expected runtime: 48-72 hours")
    print("=" * 80)
    
    # Setup training directory
    log_dir = setup_extended_training_dir(config_to_save=extended_config)
    
    # Create environment
    logging.info("Creating extended training environment...")
    env_class = get_env_class(extended_config["env_path"], extended_config["env_class"])
    env = env_class(extended_config)
    
    # Create evaluation environment
    eval_env = env_class(extended_config)
    
    logging.info(f"Environment created: {env}")
    logging.info(f"Observation space: {env.observation_space}")
    logging.info(f"Action space: {env.action_space}")
    
    # Create extended agent
    model = create_extended_agent(env, extended_config, log_dir)
    
    # Setup extended validation callback
    validation_callback = ExtendedValidationCallback(
        eval_env=eval_env,
        eval_freq=extended_config['validation_frequency'],
        n_eval_episodes=5,  # Fewer episodes but more frequent
        verbose=1
    )
    
    # Configure tensorboard logging
    tb_log_dir = os.path.join(log_dir, "tensorboard")
    model.set_logger(configure_logger(tb_log_dir, ["tensorboard", "csv"]))
    
    logging.info("Starting extended training...")
    
    # Record training start
    start_time = time.time()
    
    try:
        # Run extended training
        model.learn(
            total_timesteps=extended_config["total_timesteps"],
            callback=validation_callback,
            log_interval=extended_config["log_interval"],
            progress_bar=True
        )
        
        # Training completed successfully
        end_time = time.time()
        training_duration = end_time - start_time
        
        logging.info(f"🎉 Extended training completed successfully!")
        logging.info(f"Training duration: {training_duration/3600:.1f} hours")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_extended_model.zip")
        model.save(final_model_path)
        logging.info(f"Final model saved: {final_model_path}")
        
        # Save training summary
        summary = {
            "total_timesteps": extended_config["total_timesteps"],
            "training_duration_hours": training_duration / 3600,
            "average_timesteps_per_hour": extended_config["total_timesteps"] / (training_duration / 3600),
            "final_model_path": final_model_path,
            "log_directory": log_dir,
            "config": extended_config
        }
        
        summary_path = os.path.join(log_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n🎯 EXTENDED TRAINING COMPLETE")
        print(f"Duration: {training_duration/3600:.1f} hours")
        print(f"Final model: {final_model_path}")
        print(f"Logs: {log_dir}")
        print(f"TensorBoard: tensorboard --logdir={tb_log_dir}")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        interrupted_model_path = os.path.join(log_dir, "interrupted_extended_model.zip")
        model.save(interrupted_model_path)
        logging.info(f"Model saved before interruption: {interrupted_model_path}")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        error_model_path = os.path.join(log_dir, "error_extended_model.zip")
        model.save(error_model_path)
        logging.info(f"Model saved after error: {error_model_path}")
        raise
    
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    print("🏛️ EXTENDED REBATED HFT TRAINING")
    print("Institutional-grade RL training with 50M timesteps")
    print("All critical fixes applied and validated")
    print()
    
    # Display configuration summary
    print("📋 TRAINING CONFIGURATION:")
    print(f"  Timesteps: {extended_config['total_timesteps']:,}")
    print(f"  Episode Length: {extended_config['episode_length']:,}")
    print(f"  Initial Capital: ${extended_config['initial_capital']:,}")
    print(f"  Rebate Rate: {extended_config['rebate_rate_long']*10000:.1f} bps")
    print(f"  Architecture: LSTM + {extended_config['num_heads']}-head Attention")
    print(f"  Network Depth: {len(extended_config['net_arch_pi'])} layers")
    print()
    
    # Start extended training
    run_extended_training()