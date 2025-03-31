import logging
import os
import time
from copy import deepcopy
import json
from datetime import datetime

config = {
    #"csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_large.csv", # <<< --- UPDATE THIS PATH --- >>>
    "csv_path":"/home/gaen/Documents/billions_db/orderbooks/binance/futures/ethusdc/28-Mar-2025/binance_futures_ethusdc_orderbook_28-Mar-2025.csv",
    "initial_capital": 20000.0,
    "max_steps": 5000, # Reduced steps for quicker testing
    "order_book_levels": 9,
    "price_offset_ticks": 40,
    "max_order_volume": 2.5,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.005,
    "max_active_orders": 10, # Reduced for testing clarity
    "inventory_penalty": 0.0,
    "transaction_cost": 0.000, 
    "max_inventory": 5,
    "invalid_order_penalty": 0,
    "activity_bonus": 0.00,
    "episode_length": 400,
    "details": {
        "reward_type": "mtm if " 
    },
    #"env_path": "envs.ty_long_only_but_positive_rew",
    "env_path": "envs.post",
    "env_class": "HFTEnv",


    # --- NEW/UPDATED Parameters ---
    "latency_steps_long": 1,          # Latency (in steps) for BUY orders to become active
    "latency_steps_short": 1,         # Latency (in steps) for SELL orders to become active
    "transaction_cost_long": 0.000000,  # Percentage cost for BUY executions (e.g., 0.05%)
    "transaction_cost_short": 0.00, # Percentage cost for SELL executions (e.g., 0.05%)
    #"transaction_cost_long": 0.000000,  # Percentage cost for BUY executions (e.g., 0.05%)
    #"transaction_cost_short": 0.0, # Percentage cost for SELL executions (e.g., 0.05%)
    "inventory_penalty": 0.0,        # Penalty factor for holding inventory (quadratic)
    "invalid_order_penalty": 0.,     # Penalty for placing orders too far from BBO
    "activity_bonus": 0.00,          # Bonus per unit of volume executed
    "taker_penalty": 0.0005,
    "allowed_aggressiveness_ticks": 3,  # NEW: Allow up to 3 ticks of aggressiveness in placement

    # --- End NEW/UPDATED ---
}



import torch
import numpy as np
import psutil
import GPUtil
import gymnasium as gym
import importlib

def get_env_class(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

HFTEnv = get_env_class(config["env_path"], config["env_class"])
# Import your custom environment and configuration.
# Import the SAC agent creator from agent.py.
from agents.agent import create_sac_agent  



# Set up logging configuration.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def get_optimal_batch_size():
    """Calculate optimal batch size based on available GPU VRAM."""
    try:
        gpu = GPUtil.getGPUs()[0]
        free_memory = gpu.memoryFree * 0.8  # Use 80% of free VRAM
        # Approximate memory usage per sample in GB.
        memory_per_sample = 0.001  
        return min(256, int(free_memory / memory_per_sample))
    except Exception as e:
        logging.warning("GPU info unavailable, using default batch size. Error: %s", e)
        return 256

def estimate_training_time(data_size, batch_size):
    """Estimate training time based on data size and batch size."""
    samples_per_second = 1000 if torch.cuda.is_available() else 100
    total_samples = data_size * 1_000_000  # Assuming 1M steps per GB.
    estimated_seconds = total_samples / (samples_per_second * batch_size)
    return estimated_seconds

def setup_training_dir():
    """Create training directory and save config"""
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return log_dir

def train():
    # Ensure configuration contains 'csv_path'
    if 'csv_path' not in config:
        raise KeyError("Configuration must include 'csv_path' key pointing to the training data")
    
    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info("GPU Memory: %.2fGB", gpu_mem)
    sys_mem = psutil.virtual_memory().total / 1e9
    logging.info("System Memory: %.2fGB", sys_mem)
    
    # Determine batch size.
    batch_size = get_optimal_batch_size()
    logging.info("Using batch size: %d", batch_size)
    
    # Create the training environment.
    train_env = HFTEnv(config)
    val_env = HFTEnv(config)
    data_size = os.path.getsize(config['csv_path']) / 1e9  # File size in GB.
    logging.info("Dataset size: %.2fGB", data_size)
    
    # Estimate training time.
    est_seconds = estimate_training_time(data_size, batch_size)
    logging.info("Estimated training time: %.1f hours", est_seconds / 3600)
    
    # Replace log_dir creation with new setup function
    log_dir = setup_training_dir()
    
    # Create SAC agent with validation callback and logging.
    model, callback, _ = create_sac_agent(
        env=train_env,
        hidden_dim=256,
        features_dim=256,
        validation_start=0.8,
        batch_size=batch_size,
        log_dir=log_dir , # Pass log_dir to create_sac_agent
        episode_length=config["episode_length"]
    )
    
    start_time = time.time()
    # Scale total timesteps with data size (adjust as needed).

    total_timesteps = 180000000
    print(total_timesteps)
    logging.info("Starting training for %d timesteps", total_timesteps)
    
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    training_time = (time.time() - start_time) / 3600
    logging.info("Actual training time: %.1f hours", training_time)
    
    # Save the final model.
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    logging.info("Final model saved at: %s", final_model_path)
    
    return model, log_dir

def evaluate(model, env, n_episodes=10):
    """
    Evaluate the trained model and log both total and adjusted (per-step) rewards.
    """
    device = model.device
    episode_rewards = []
    adjusted_rewards = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        
        # Reset feature extractor states if available.
        if hasattr(model.policy, "features_extractor") and model.policy.features_extractor is not None:
            if hasattr(model.policy.features_extractor, "reset_cached_states"):
                model.policy.features_extractor.reset_cached_states(clear_memory=True)
        
        while not (done or truncated):
            step_count += 1
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32)

            action, _ = model.predict(obs_tensor.cpu(), deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            env.render()
        
        episode_rewards.append(total_reward)
        avg_reward = total_reward / step_count if step_count > 0 else 0
        adjusted_rewards.append(avg_reward)
        logging.info("Episode %d: Total Reward = %.2f, Steps = %d, Adjusted Reward = %.4f",
                     episode + 1, total_reward, step_count, avg_reward)
    
    mean_total = np.mean(episode_rewards)
    std_total = np.std(episode_rewards)
    mean_adjusted = np.mean(adjusted_rewards)
    std_adjusted = np.std(adjusted_rewards)
    
    logging.info("Mean Total Reward: %.2f +/- %.2f", mean_total, std_total)
    
    logging.info("Mean Adjusted Reward (per step): %.4f +/- %.4f", mean_adjusted, std_adjusted)
    
    return mean_total, std_total, mean_adjusted, std_adjusted

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Starting training script. Using device: %s", device)
    
    # Train the agent.
    model, log_dir = train()
    
    # Create test environment for evaluation.
    test_env = HFTEnv(config)
    
    logging.info("Evaluating final model:")
    evaluate(model, test_env)
    
    # If a best model was saved during validation, load and evaluate it.
    best_model_path = os.path.join(log_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        logging.info("Evaluating best model from: %s", best_model_path)
        best_model = model.__class__.load(best_model_path, device=device)
        evaluate(best_model, test_env)
