import os
import logging
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# Set up logging for this module.

def setup_logging():
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@dataclass
class CachedStates:
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    key_cache: torch.Tensor
    value_cache: torch.Tensor

@dataclass
class ValidationStates:
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    episode_returns: List[float]
    current_return: float = 0.0

class CachedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_length: int = 400):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_length = max_seq_length
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.key_cache = None
        self.value_cache = None
        self.cache_position = 0
        
    def reset_cache(self, batch_size: int, device: torch.device):
        self.key_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.cache_position = 0
        
    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        
        # Project queries, keys, and values.
        q = self.q_proj(x).view(B, L, H, -1).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, -1).transpose(1, 2)
        
        if use_cache and self.key_cache is not None:
            if self.cache_position + L <= self.max_seq_length:
                self.key_cache[:, self.cache_position:self.cache_position+L] = k.transpose(1, 2)
                self.value_cache[:, self.cache_position:self.cache_position+L] = v.transpose(1, 2)
                k_to_use = self.key_cache[:, :self.cache_position+L].transpose(1, 2)
                v_to_use = self.value_cache[:, :self.cache_position+L].transpose(1, 2)
                self.cache_position += L
            else:
                self.reset_cache(B, x.device)
                k_to_use, v_to_use = k, v
        else:
            k_to_use, v_to_use = k, v
        
        scores = torch.matmul(q, k_to_use.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_to_use)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class CachedLSTMAttention(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, hidden_dim=256, num_heads=8):
        super().__init__(observation_space, features_dim=features_dim)
        
        self.input_dim = observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self._features_dim = features_dim
        
        # LSTM layer.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        
        # Cached attention.
        self.attention = CachedMultiHeadAttention(hidden_dim, num_heads)
        
        # Final projection, normalization, and dropout.
        self.fc = nn.Linear(hidden_dim, self._features_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Cached states for recurrent layers.
        self.cached_states = None
        self.validation_states = None
        
        # Bind methods for compatibility.
        self.reset_cache = self.reset_cached_states
        self.reset = self.reset_cached_states
        setattr(self, 'reset_cached_states', self.reset_cached_states)
        setattr(self, 'reset', self.reset_cached_states)
        setattr(self, 'reset_cache', self.reset_cached_states)
    
    def reset_cached_states(self, batch_size: int = 1, clear_memory: bool = False):
        """Reset cached states with an option to clear memory."""
        device = next(self.parameters()).device
        if clear_memory or self.cached_states is None:
            lstm_state = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
            self.attention.reset_cache(batch_size, device)
        else:
            lstm_state = (
                self.cached_states.lstm_state[0].detach(),
                self.cached_states.lstm_state[1].detach()
            )
            if self.attention.key_cache is not None:
                self.attention.key_cache = self.attention.key_cache.detach()
                self.attention.value_cache = self.attention.value_cache.detach()
        
        self.cached_states = CachedStates(
            lstm_state=lstm_state,
            key_cache=self.attention.key_cache,
            value_cache=self.attention.value_cache
        )
    
    def init_validation(self):
        """Initialize validation states for use during evaluation."""
        device = next(self.parameters()).device
        self.validation_states = ValidationStates(
            lstm_state=(
                torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device)
            ),
            key_cache=torch.zeros(1, self.attention.max_seq_length,
                                  self.attention.num_heads, self.attention.head_dim, device=device),
            value_cache=torch.zeros(1, self.attention.max_seq_length,
                                    self.attention.num_heads, self.attention.head_dim, device=device),
            episode_returns=[]
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
            
        # Process through LSTM with state management.
        if self.training or self.cached_states is None:
            lstm_out, lstm_state = self.lstm(observations)
        else:
            lstm_out, lstm_state = self.lstm(observations, self.cached_states.lstm_state)
            self.cached_states.lstm_state = lstm_state
        
        # Process through attention (using cache when appropriate).
        attn_out = self.attention(lstm_out, use_cache=not self.training and self.cached_states is not None)
        combined = self.layer_norm(lstm_out + attn_out)
        combined = self.dropout(combined)
        
        # Final projection using the last time step.
        features = self.fc(combined[:, -1])
        return features

class ValidationCallback(BaseCallback):
    def __init__(self, eval_env, validation_freq=10000):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.validation_freq = validation_freq
        self.best_mean_reward = -np.inf
        self.log_dir = None
        
    def _init_callback(self) -> None:
        if self.model is None:
            raise ValueError("Model is not set!")
        if not hasattr(self.model, 'logger') or self.model.logger is None:
            self.log_dir = "runs"
        else:
            self.log_dir = self.model.logger.dir
    
    def _on_step(self):
        if self.n_calls % self.validation_freq == 0:
            n_eval_episodes = 5
            episode_rewards = []
            adjusted_rewards = []
            episode_pnls = []  # Add PnL tracking
            
            for _ in range(n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                episode_reward = 0
                step_count = 0
                while not (done or truncated):
                    step_count += 1
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    
                # Get final PnL from the last step's info
                if done or truncated:
                    final_pnl = info.get('episode_pnl', 0.0)
                    episode_pnls.append(final_pnl)
                
                episode_rewards.append(episode_reward)
                avg_reward = episode_reward / step_count if step_count > 0 else 0
                adjusted_rewards.append(avg_reward)
            
            # Calculate all statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_adjusted = np.mean(adjusted_rewards)
            std_adjusted = np.std(adjusted_rewards)
            mean_pnl = np.mean(episode_pnls)
            std_pnl = np.std(episode_pnls)
            
            # Log all metrics
            self.logger.record('validation/mean_reward', mean_reward)
            self.logger.record('validation/reward_std', std_reward)
            self.logger.record('validation/mean_adjusted_reward', mean_adjusted)
            self.logger.record('validation/adjusted_reward_std', std_adjusted)
            self.logger.record('validation/mean_pnl', mean_pnl)
            self.logger.record('validation/pnl_std', std_pnl)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.log_dir, 'best_model')
                self.model.save(model_path)
                self.logger.record('validation/best_mean_reward', self.best_mean_reward)
            
            self.logger.dump(self.n_calls)
        return True

class TimeSeriesEnvWrapper(gym.Wrapper):
    """
    A wrapper to manage episode length and reset feature extractor memory.
    """
    def __init__(self, env, feature_extractor, episode_length=400):
        super().__init__(env)
        self.feature_extractor = feature_extractor
        self.episode_length = episode_length
        self.steps = 0
        self.new_sequence = True
        self.current_start = 0  # Track starting position
        self.max_starts = env.max_steps - episode_length  # Maximum start position
        self.total_data_length = env.max_steps
        self.log_frequency = 1000  # Log every 1000 steps
        self.step_counter = 0
        
    def _process_observation(self, obs: Union[torch.Tensor, np.ndarray, Dict, List]) -> Union[np.ndarray, Dict, List]:
        """
        Ensure observation is in the correct format and on CPU.
        Handles single tensors, numpy arrays, dictionaries, and lists of tensors.
        """
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        elif isinstance(obs, dict):
            return {k: self._process_observation(v) for k, v in obs.items()}
        elif isinstance(obs, (list, tuple)):
            return [self._process_observation(x) for x in obs]
        elif isinstance(obs, np.ndarray):
            return obs
        return obs
        
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.steps = 0
        # Move start position forward
        self.current_start = (self.current_start + self.episode_length) % self.max_starts
        self.env.current_step = self.current_start
        
        logging.info(f"Starting new episode at data position {self.current_start:,} / {self.total_data_length:,} " 
                    f"({(self.current_start/self.total_data_length)*100:.1f}%)")
        
        if self.feature_extractor is not None:
            self.feature_extractor.reset_cached_states(clear_memory=self.new_sequence)
        self.new_sequence = False
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_observation(obs), info
    
    def step(self, action):
        self.steps += 1
        self.step_counter += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Log progress periodically
        if self.step_counter % self.log_frequency == 0:
            current_pos = self.env.current_step
            logging.info(f"Training at position {current_pos:,} / {self.total_data_length:,} "
                        f"({(current_pos/self.total_data_length)*100:.1f}%)")
            
        if self.steps >= self.episode_length:
            truncated = True
        return self._process_observation(obs), reward, terminated, truncated, info

def create_sac_agent(env, hidden_dim=256, features_dim=128, validation_start=0.8, device="auto", batch_size=256, log_dir=None, episode_length=400):
    """
    Create an SAC agent with custom feature extraction, validation, and TensorBoard logging.
    """
    try:
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use provided log_dir or create new one
        if log_dir is None:
            log_dir = setup_logging()
        
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        
        # Create deep copies for training and validation environments.
        train_env = deepcopy(env)
        val_env = deepcopy(env)
        
        total_steps = len(env.order_book_history)
        train_steps = int(total_steps * validation_start)
        
        # Split the data between training and validation.
        train_env.max_steps = train_steps
        val_env.order_book_history = val_env.order_book_history[train_steps:]
        val_env.max_steps = total_steps - train_steps
        
        policy_kwargs = dict(
            features_extractor_class=CachedLSTMAttention,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim
            ),
            net_arch=dict(pi=[512,512,256], qf=[512,512,256])
        )
        
        # Create a temporary extractor for verification.
        temp_extractor = CachedLSTMAttention(env.observation_space, features_dim, hidden_dim)
        assert hasattr(temp_extractor, 'reset_cached_states'), "Feature extractor initialization failed"
        
        # Wrap environments to manage sequence lengths.
        train_env = TimeSeriesEnvWrapper(train_env, None, episode_length=episode_length)
        val_env = TimeSeriesEnvWrapper(val_env, None, episode_length=episode_length)
        
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=min(1_000_000, int(1e5 * (batch_size / 256))),
            learning_starts=min(2000, batch_size * 4),
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
        
        # Assign the feature extractor to the environments.
        train_env.feature_extractor = model.policy.features_extractor
        val_env.feature_extractor = model.policy.features_extractor
        
        validation_callback = ValidationCallback(val_env, validation_freq=10000)
        model.set_logger(new_logger)
        
        return model, validation_callback, log_dir
        
    except Exception as e:
        logging.error("Error creating SAC agent: %s", e)
        raise

def create_test_env(env, model, episode_length=4000):
    """
    Create a properly wrapped test environment that ensures tensor conversions.
    
    Args:
        env: The base environment to wrap
        model: The trained model whose feature extractor will be used
        episode_length: Length of each episode
    
    Returns:
        A wrapped environment ready for testing
    """
    test_env = deepcopy(env)
    wrapped_env = TimeSeriesEnvWrapper(test_env, model.policy.features_extractor, episode_length=episode_length)
    return wrapped_env

