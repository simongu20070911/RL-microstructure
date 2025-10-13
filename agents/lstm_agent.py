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
class CachedState:
    lstm_state: Tuple[torch.Tensor, torch.Tensor]

class CachedLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    A feature extractor that only uses a cached LSTM.
    """
    def __init__(self, observation_space, features_dim=128, hidden_dim=256):
        super().__init__(observation_space, features_dim=features_dim)
        self.input_dim = observation_space.shape[0]
        self.hidden_dim = hidden_dim
        self._features_dim = features_dim

        # LSTM layer.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        
        # Final projection and dropout.
        self.fc = nn.Linear(hidden_dim, self._features_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Cached LSTM state.
        self.cached_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        
        # Bind methods for compatibility.
        self.reset_cache = self.reset_cached_states
        self.reset = self.reset_cached_states
        setattr(self, 'reset_cached_states', self.reset_cached_states)
        setattr(self, 'reset', self.reset_cached_states)
        setattr(self, 'reset_cache', self.reset_cached_states)
    
    def reset_cached_states(self, batch_size: int = 1, clear_memory: bool = False):
        """Reset cached LSTM state with an option to clear memory."""
        device = next(self.parameters()).device
        if clear_memory or self.cached_state is None:
            lstm_state = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
        else:
            lstm_state = (
                self.cached_state[0].detach(),
                self.cached_state[1].detach()
            )
        self.cached_state = lstm_state
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure observations have a time dimension.
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        
        # Process through LSTM with state management.
        if self.training or self.cached_state is None:
            lstm_out, lstm_state = self.lstm(observations)
        else:
            lstm_out, lstm_state = self.lstm(observations, self.cached_state)
        self.cached_state = lstm_state
        
        # Final projection using the last time step.
        features = self.fc(lstm_out[:, -1, :])
        features = self.dropout(features)
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
    Create an SAC agent with custom LSTM feature extraction, validation, and TensorBoard logging.
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
            features_extractor_class=CachedLSTMFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim
            ),
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])
        )
        
        # Create a temporary extractor for verification.
        temp_extractor = CachedLSTMFeatureExtractor(env.observation_space, features_dim, hidden_dim)
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
