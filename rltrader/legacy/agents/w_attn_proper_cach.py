# agents/agent.py

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
log = logging.getLogger(__name__)
# Example basic config if not configured elsewhere:
# logging.basicConfig(level=logging.INFO)

def setup_logging():
    """Sets up a timestamped log directory."""
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    # Configure SB3 logger here if needed, or rely on model's logger setup
    return log_dir

@dataclass
class CachedStates:
    """Holds the recurrent states needed between steps for CachedLSTMAttention."""
    # LSTM state is managed externally by the main feature extractor class
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    # Attention state (K/V cache, position, fullness) is now managed *internally*
    # within the CachedMultiHeadAttention module itself. No need to store here.

@dataclass
class ValidationStates:
    """Holds states specifically for validation runs (not used in training)."""
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    key_cache: torch.Tensor  # May be needed if validation needs access? Or rely on internal attn state.
    value_cache: torch.Tensor# May be needed if validation needs access? Or rely on internal attn state.
    attn_cache_position: int # May be needed if validation needs access? Or rely on internal attn state.
    attn_is_full: bool       # May be needed if validation needs access? Or rely on internal attn state.
    episode_returns: List[float]
    current_return: float = 0.0

class CachedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer with a sliding window key-value cache.

    Assumes inference is done step-by-step (L=1) when using the cache.
    """
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_length: int = 400):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        self.max_seq_length = max_seq_length # Cache size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # --- Internal Cache State ---
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        # `cache_position` points to the *next* index to write to.
        self.cache_position: int = 0
        # `is_full` tracks if max_seq_length elements have been written.
        self.is_full: bool = False
        # --------------------------

    def reset_cache(self, batch_size: int, device: torch.device):
        """Resets the cache tensors and state indicators."""
        log.debug(f"Resetting attention cache (Batch Size: {batch_size}, Device: {device}, Size: {self.max_seq_length}).")
        self.key_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.cache_position = 0
        self.is_full = False # Reset fullness flag

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass with optional sliding window caching.

        Args:
            x: Input tensor shape (B, L, D).
            use_cache: If True, use and update the internal sliding window cache.
                       Assumes L=1 if use_cache is True for correct window behavior.

        Returns:
            Output tensor shape (B, L, D).
        """
        B, L, D = x.shape
        H = self.num_heads
        head_dim = self.head_dim

        # --- Input Sanity Check for Sliding Window ---
        if use_cache and L != 1 and self.key_cache is not None:
            log.warning(f"CachedMultiHeadAttention sliding window received L={L} while cache is active. "
                        f"Designed for L=1 inference steps. Behavior may be unexpected.")
            # Depending on strictness, could raise error:
            # raise ValueError("Sliding window cache requires L=1 during cached inference steps.")

        # --- Project Q, K, V ---
        q = self.q_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]
        v = self.v_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]

        k_to_use = k
        v_to_use = v
        retrieved_len = L # Length of sequence to attend to

        # --- Cache Logic (if active and initialized) ---
        if use_cache and self.key_cache is not None and self.value_cache is not None:
            # Ensure cache matches current batch size
            if self.key_cache.shape[0] != B:
                log.warning(f"Resetting cache due to batch size mismatch (expected {self.key_cache.shape[0]}, got {B})")
                self.reset_cache(B, x.device)
                # Fallback to using current K,V only after reset if needed,
                # but reset_cache should make key_cache not None, so the logic below runs.

            # --- Update Cache (Sliding Window, assumes L=1) ---
            # We proceed assuming L=1 based on check/warning above.
            # If L > 1, this part would only cache the *last* element of the input sequence 'x'.
            current_k = k.transpose(1, 2)[:, -1] # Shape [B, H, Dh] - Use last K if L > 1
            current_v = v.transpose(1, 2)[:, -1] # Shape [B, H, Dh] - Use last V if L > 1

            # Store at the current cache_position
            self.key_cache[:, self.cache_position] = current_k
            self.value_cache[:, self.cache_position] = current_v

            # --- Update Position and Fullness Flag ---
            new_pos = (self.cache_position + 1) % self.max_seq_length
            # Check if we *just* became full (i.e., wrote to the last slot and wrapped)
            if not self.is_full and new_pos == 0:
                log.debug("Attention cache is now full.")
                self.is_full = True
            self.cache_position = new_pos

            # --- Retrieve K/V from Cache for Attention ---
            if self.is_full:
                # Cache is full, retrieve all max_seq_length items in correct order
                # Data wraps around: oldest data is at self.cache_position
                # Order required: [pos:max_len] followed by [0:pos]
                k_retrieved = torch.cat((self.key_cache[:, self.cache_position:], self.key_cache[:, :self.cache_position]), dim=1)
                v_retrieved = torch.cat((self.value_cache[:, self.cache_position:], self.value_cache[:, :self.cache_position]), dim=1)
                retrieved_len = self.max_seq_length
            else:
                # Cache not full yet, retrieve only valid entries [0:pos]
                # self.cache_position points to the *next empty* slot
                k_retrieved = self.key_cache[:, :self.cache_position]
                v_retrieved = self.value_cache[:, :self.cache_position]
                retrieved_len = self.cache_position # Current number of items in cache

            # Transpose back for attention: [B, retrieved_len, H, Dh] -> [B, H, retrieved_len, Dh]
            k_to_use = k_retrieved.transpose(1, 2)
            v_to_use = v_retrieved.transpose(1, 2)

        # --- Attention Calculation ---
        # q: [B, H, L, Dh]
        # k_to_use: [B, H, retrieved_len, Dh]
        # scores: [B, H, L, retrieved_len]
        if k_to_use.shape[2] == 0: # Handle case where cache is empty (step 0) and use_cache=True
             # Should not happen if reset_cache is called correctly before first step,
             # but as a safeguard, perform attention only over current input.
             log.warning("Attention calculation called with empty cache history.")
             k_to_use = k # Use original K, V from current input only
             v_to_use = v
             retrieved_len = L

        scores = torch.matmul(q, k_to_use.transpose(-2, -1)) / np.sqrt(head_dim)
        attn_weights = torch.softmax(scores, dim=-1) # Softmax over retrieved_len dimension

        # attn_weights: [B, H, L, retrieved_len]
        # v_to_use: [B, H, retrieved_len, Dh]
        # out: [B, H, L, Dh]
        out = torch.matmul(attn_weights, v_to_use)

        # --- Output Projection ---
        # out.transpose: [B, L, H, Dh] -> view: [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, D) # Use D (hidden_dim)
        return self.out_proj(out)


class CachedLSTMAttention(BaseFeaturesExtractor):
    """
    Feature extractor using LSTM and CachedMultiHeadAttention with sliding window.

    Manages the LSTM hidden state cache externally and delegates attention
    caching to the attention module internally.
    """
    def __init__(self, observation_space: spaces.Space, features_dim: int = 128,
                 hidden_dim: int = 256, num_heads: int = 8, max_seq_length: int = 400):
        """
        Args:
            observation_space: Environment observation space.
            features_dim: Dimension of the output features.
            hidden_dim: Dimension of the LSTM and Attention hidden states.
            num_heads: Number of attention heads.
            max_seq_length: Max history length for the attention sliding window cache.
        """
        super().__init__(observation_space, features_dim=features_dim) # features_dim passed to super

        # Ensure observation space is Box
        if not isinstance(observation_space, spaces.Box):
            raise ValueError(f"Expected observation space to be Box, got {type(observation_space)}")
        if len(observation_space.shape) != 1:
             raise ValueError(f"Expected 1D observation space, got shape {observation_space.shape}")

        self.input_dim = observation_space.shape[0]
        self.hidden_dim = hidden_dim
        # Note: self._features_dim is automatically set by BaseFeaturesExtractor(..., features_dim=features_dim)

        # LSTM layer.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # Cached attention with sliding window.
        self.attention = CachedMultiHeadAttention(hidden_dim, num_heads, max_seq_length)

        # Final projection, normalization, and dropout.
        self.fc = nn.Linear(hidden_dim, self._features_dim) # Project attention output to features_dim
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1) # Dropout is automatically handled by model.eval()

        # Cached states for LSTM recurrent layer.
        self.cached_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_cached_states(self, batch_size: int = 1):
        """
        Reset cached LSTM states and the internal attention cache.
        Should be called at the start of each episode during inference/evaluation.
        """
        device = next(self.parameters()).device
        log.debug(f"Resetting feature extractor states (Batch Size: {batch_size}).")

        # Reset LSTM state
        self.cached_lstm_state = (
            torch.zeros(1, batch_size, self.hidden_dim, device=device), # (num_layers, batch, hidden_size)
            torch.zeros(1, batch_size, self.hidden_dim, device=device)
        )

        # Reset the internal cache of the attention module
        # Need to determine batch size and device correctly.
        self.attention.reset_cache(batch_size, device)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass logic. Uses cached states if model is in eval mode.

        Args:
            observations: Input observations tensor (B, D) or (B, L, D).

        Returns:
            Extracted features tensor (B, features_dim).
        """
        # Reshape input if needed: (B, D) -> (B, 1, D)
        # This is typical for step-by-step processing in RL.
        if len(observations.shape) == observations.spaces.shape[0]: # Check if it matches obs space dim directly
             observations = observations.unsqueeze(1)
        elif len(observations.shape) == 2: # Assume (B, D)
             observations = observations.unsqueeze(1) # -> (B, 1, D)
        elif len(observations.shape) == 3: # Assume (B, L, D)
             pass # Already has sequence length
        else:
             raise ValueError(f"Unexpected observation shape: {observations.shape}")

        B, L, _ = observations.shape

        # --- Determine if cache should be used ---
        # Cache is used if:
        # 1. Not in training mode (model.eval() has been called)
        # 2. Cached LSTM state exists (implies reset_cached_states was called)
        use_lstm_cache = not self.training and self.cached_lstm_state is not None
        # Attention cache use is determined *inside* its forward method based on the flag below
        use_attn_cache_flag = not self.training and self.attention.key_cache is not None # Check if attn cache initialized

        # --- LSTM Layer ---
        if use_lstm_cache:
            # We need to ensure the cache batch size matches the input batch size
            if self.cached_lstm_state[0].shape[1] != B:
                 log.warning(f"LSTM cache batch size ({self.cached_lstm_state[0].shape[1]}) != "
                             f"Input batch size ({B}). Resetting state.")
                 self.reset_cached_states(batch_size=B)
                 # After reset, use_lstm_cache is still True, state matches B
            
            lstm_out, new_lstm_state = self.lstm(observations, self.cached_lstm_state)
            self.cached_lstm_state = new_lstm_state # Update LSTM state cache for next step
        else:
            # No state passed, LSTM computes initial state implicitly (zeros).
            # Or if training, state is not carried over BPTT boundaries handled by sampler.
            lstm_out, _ = self.lstm(observations) # Discard state if not caching/training

        # --- Attention Layer ---
        # The attention layer manages its own internal cache state.
        # We just tell it whether it *should* use its cache.
        attn_out = self.attention(lstm_out, use_cache=use_attn_cache_flag)

        # --- Combine & Project ---
        # Add & Norm: Residual connection between LSTM output and Attention output
        combined = self.layer_norm(lstm_out + attn_out)
        # Dropout (automatically disabled in eval mode)
        combined = self.dropout(combined)

        # Final projection: Use the features from the last time step
        # If L=1 (typical eval step), this just takes the only time step.
        features = self.fc(combined[:, -1, :]) # Output shape: [B, features_dim]
        return features

# --- Callbacks and Environment Wrappers ---

class ValidationCallback(BaseCallback):
    """
    Callback for evaluating the agent on a validation environment periodically.

    Logs validation metrics to SB3 logger. Saves the best model based on mean reward.
    Includes tracking of episode PnL if available in `info` dict.
    """
    def __init__(self, eval_env, validation_freq=10000, n_eval_episodes=5, log_path=None):
        """
        Args:
            eval_env: The validation environment.
            validation_freq: Evaluate every N steps.
            n_eval_episodes: Number of episodes to run for evaluation.
            log_path: Path to save the best model. If None, uses model's log dir.
        """
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.validation_freq = validation_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        """Initialize logger and save path."""
        if self.model is None:
            raise ValueError("ValidationCallback received no model!")
        # Setup log path
        if self.log_path is None:
            if self.model.logger is not None:
                self.log_path = self.model.logger.get_dir()
                log.info(f"Validation logs and best model will be saved to: {self.log_path}")
            else:
                log.warning("No logger found in model and no log_path provided. Best model will not be saved.")
                self.log_path = "." # Default to current dir if no logger

        # Ensure the feature extractor has the reset method if it's custom
        if isinstance(self.model.policy.features_extractor, CachedLSTMAttention):
             if not hasattr(self.model.policy.features_extractor, 'reset_cached_states'):
                 raise AttributeError("Feature extractor needs 'reset_cached_states' method for validation.")


    def _on_step(self) -> bool:
        """Trigger evaluation when validation frequency is reached."""
        if self.n_calls % self.validation_freq == 0:
            log.info(f"Running validation at step {self.n_calls}...")
            
            # Important: Ensure model is in evaluation mode
            self.model.policy.set_training_mode(False) 
            
            episode_rewards = []
            episode_lengths = []
            episode_pnls = [] # Track Profit and Loss per episode

            for episode in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                # --- Reset Feature Extractor State ---
                # Determine batch size (usually 1 for eval)
                batch_size = 1 # Assuming eval runs one episode at a time
                # Call reset method if it exists
                if hasattr(self.model.policy.features_extractor, 'reset_cached_states'):
                    self.model.policy.features_extractor.reset_cached_states(batch_size=batch_size)
                # ------------------------------------

                done = False
                truncated = False
                current_reward = 0.0
                current_length = 0
                
                while not (done or truncated):
                    # Use deterministic actions for evaluation
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    current_reward += reward
                    current_length += 1
                    
                    # Check for termination or truncation
                    if done or truncated:
                        final_pnl = info.get('episode_pnl', np.nan) # Get PnL if available
                        episode_pnls.append(final_pnl)
                        log.debug(f"Eval Episode {episode+1} finished. Reward: {current_reward:.2f}, Length: {current_length}, PnL: {final_pnl:.4f}")
                        break # Exit while loop

                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)

            # Calculate statistics
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            # Handle potential NaNs in PnL if 'episode_pnl' key wasn't in info
            mean_pnl = np.nanmean(episode_pnls) if len(episode_pnls) > 0 else np.nan
            std_pnl = np.nanstd(episode_pnls) if len(episode_pnls) > 0 else np.nan

            # Log metrics
            log.info(f"Validation Results: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}, Mean Length: {mean_length:.1f}, Mean PnL: {mean_pnl:.4f} +/- {std_pnl:.4f}")
            if self.model.logger is not None:
                self.logger.record('validation/mean_reward', mean_reward)
                self.logger.record('validation/std_reward', std_reward)
                self.logger.record('validation/mean_episode_length', mean_length)
                self.logger.record('validation/mean_pnl', mean_pnl)
                self.logger.record('validation/std_pnl', std_pnl)
                self.logger.dump(step=self.num_timesteps) # Use num_timesteps for SB3 log step

            # Save best model
            if mean_reward > self.best_mean_reward:
                log.info(f"New best mean reward: {mean_reward:.2f} (previous: {self.best_mean_reward:.2f}). Saving best model...")
                self.best_mean_reward = mean_reward
                if self.log_path is not None:
                    save_path = os.path.join(self.log_path, 'best_model')
                    self.model.save(save_path)
                    log.info(f"Best model saved to {save_path}")
                self.logger.record('validation/best_mean_reward', self.best_mean_reward) # Log new best

            # Important: Set model back to training mode
            self.model.policy.set_training_mode(True)

        return True # Continue training


class TimeSeriesEnvWrapper(gym.Wrapper):
    """
    A wrapper for time-series environments to manage episode length and
    reset feature extractor memory correctly.

    This wrapper assumes the underlying environment (`env`) handles the
    actual time series data and stepping logic. It focuses on episode
    truncation and coordinating state resets.
    """
    def __init__(self, env: gym.Env, feature_extractor: Optional[BaseFeaturesExtractor], episode_length: int = 400):
        """
        Args:
            env: The environment to wrap.
            feature_extractor: The feature extractor instance (used to call reset).
                               Can be None if reset is not needed or handled elsewhere.
            episode_length: Maximum steps per episode before truncation.
        """
        super().__init__(env)
        # It's better practice for the wrapper *not* to hold a direct reference
        # to the feature extractor if possible, as it couples them tightly.
        # The reset should ideally be called by the training/evaluation loop.
        # However, if explicit reset per episode start *within the wrapper* is
        # desired (e.g., if the loop doesn't handle it), keep the reference.
        # For now, let's assume the training/evaluation loop handles the reset.
        # self.feature_extractor = feature_extractor
        self.episode_length = episode_length
        self.steps_this_episode = 0

        # --- Logging setup for wrapper ---
        self.log_frequency = 5000 # Log internal counter every N steps
        self.total_steps_counter = 0 # Track total steps across episodes
        # Get total data length if available from underlying env for context
        self.total_data_length = getattr(env, 'max_steps', None) or \
                                 getattr(env.unwrapped, 'max_steps', 'Unknown')


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment and the episode step counter."""
        # Pass seed and options to the underlying environment's reset
        super().reset(seed=seed) # Pass seed to wrapped env
        obs, info = self.env.reset(seed=seed, options=options)

        self.steps_this_episode = 0
        log.debug(f"Environment reset. Episode step counter reset.")

        # --- Feature Extractor Reset ---
        # This should ideally be called *outside* the wrapper by the logic
        # that controls episodes (e.g., SB3 rollout collector or evaluation loop).
        # If it *must* be done here:
        # if self.feature_extractor is not None and hasattr(self.feature_extractor, 'reset_cached_states'):
        #     batch_size = 1 # Assuming reset is for a single environment instance
        #     self.feature_extractor.reset_cached_states(batch_size=batch_size)
        #     log.debug("Feature extractor state reset via wrapper.")
        # else:
        #     log.debug("Feature extractor not provided or has no reset method; skipping reset in wrapper.")
        # ----------------------------

        # Log underlying env's starting position if available
        current_pos = getattr(self.env, 'current_step', None) or \
                      getattr(self.env.unwrapped, 'current_step', None)
        if current_pos is not None:
             log.info(f"Starting new episode at data index {current_pos:,} / {self.total_data_length:,} "
                      f"({(current_pos / self.total_data_length * 100):.1f}% progress in data)" if isinstance(self.total_data_length, int) else "")

        return self._process_observation(obs), info

    def step(self, action):
        """Steps the environment, increments counter, and checks for truncation."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_this_episode += 1
        self.total_steps_counter += 1

        # Log progress periodically based on total steps
        if self.total_steps_counter % self.log_frequency == 0:
             current_pos = getattr(self.env, 'current_step', None) or \
                           getattr(self.env.unwrapped, 'current_step', None)
             if current_pos is not None:
                 log.info(f"Wrapper Step {self.total_steps_counter:,}: Env at data index {current_pos:,} / {self.total_data_length:,} "
                         f"({(current_pos / self.total_data_length * 100):.1f}% progress in data)" if isinstance(self.total_data_length, int) else "")
             else:
                 log.info(f"Wrapper Step Counter: {self.total_steps_counter:,}")

        # Apply episode length truncation condition
        if self.steps_this_episode >= self.episode_length:
            log.debug(f"Episode truncated at step {self.steps_this_episode} (max length {self.episode_length}).")
            truncated = True # Override truncation based on wrapper's limit

        return self._process_observation(obs), reward, terminated, truncated, info

    def _process_observation(self, obs: Union[torch.Tensor, np.ndarray, Dict, List]) -> Union[np.ndarray, Dict, List]:
        """Ensures observation is a numpy array (or dict/list of arrays) on CPU."""
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        elif isinstance(obs, dict):
            return {k: self._process_observation(v) for k, v in obs.items()}
        elif isinstance(obs, (list, tuple)):
            # Convert tuple to list to ensure mutability if needed downstream
            return [self._process_observation(x) for x in obs]
        elif isinstance(obs, np.ndarray):
            return obs # Already numpy
        else:
             # Attempt conversion for other types if possible, or return as is
             try:
                 return np.asarray(obs)
             except Exception:
                 log.warning(f"Could not convert observation of type {type(obs)} to numpy array.")
                 return obs


# --- Agent Creation Factory ---

def create_sac_agent(
    env: gym.Env,
    hidden_dim: int = 256,
    features_dim: int = 128,
    num_heads: int = 8,
    max_seq_length: int = 400, # Attention cache size
    validation_start: Optional[float] = 0.8, # Percentage of data for training
    device: Union[str, torch.device] = "auto",
    batch_size: int = 256,
    buffer_size: Optional[int] = None,
    learning_starts: Optional[int] = None,
    log_dir: Optional[str] = None,
    episode_length: int = 400, # Wrapper episode length
    validation_freq: int = 10000,
    n_eval_episodes: int = 5
    ):
    """
    Creates an SAC agent with CachedLSTMAttention feature extraction.

    Sets up training and validation environments (if validation_start is provided),
    configures logging, and includes a validation callback.

    Args:
        env: The base Gymnasium environment instance. Should have observation_space.
        hidden_dim: Hidden dimension for LSTM and Attention.
        features_dim: Output dimension of the feature extractor.
        num_heads: Number of attention heads.
        max_seq_length: History length for the attention sliding window.
        validation_start: Fraction of data (if applicable) to use for training (e.g., 0.8 for 80%).
                          If None, uses the full environment for training and validation.
                          Requires env to have specific attributes for splitting (e.g., max_steps, order_book_history).
        device: PyTorch device ('cpu', 'cuda', 'auto').
        batch_size: Training batch size.
        buffer_size: Replay buffer size. Defaults based on batch size.
        learning_starts: Number of steps before learning starts. Defaults based on batch size.
        log_dir: Directory for logs and saved models. If None, creates a timestamped dir.
        episode_length: Max steps per episode for the TimeSeriesEnvWrapper.
        validation_freq: How often to run validation (in training steps).
        n_eval_episodes: Number of episodes for each validation run.

    Returns:
        Tuple: (model, validation_callback, log_dir)
    """
    try:
        # --- Device Setup ---
        if device == "auto":
            pytorch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            pytorch_device = torch.device(device)
        log.info(f"Using device: {pytorch_device}")

        # --- Logging Setup ---
        if log_dir is None:
            log_dir = setup_logging()
        log.info(f"Logging directory: {log_dir}")
        # Configure SB3 logger to use this directory
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

        # --- Environment Splitting (Optional) ---
        train_env = deepcopy(env) # Start with copies
        val_env = deepcopy(env)

        # Attempt to split data based on env attributes if validation_start is set
        can_split = validation_start is not None and \
                    hasattr(env, 'max_steps') and isinstance(env.max_steps, int) and \
                    hasattr(env, 'order_book_history') # Example attribute needed for splitting
        
        if can_split:
            total_steps = env.max_steps # Use max_steps from original env
            train_steps = int(total_steps * validation_start)
            val_steps = total_steps - train_steps

            log.info(f"Splitting data: Training on first {train_steps:,} steps ({validation_start*100:.1f}%), "
                     f"Validating on last {val_steps:,} steps.")

            # Modify train_env (e.g., limit its max_steps)
            # This depends heavily on the environment's implementation.
            # Example: Assuming env uses max_steps to limit its data view
            train_env.max_steps = train_steps

            # Modify val_env (e.g., adjust its data source and max_steps)
            # Example: Assuming order_book_history holds the data
            val_env.order_book_history = val_env.order_book_history[train_steps:] # Adjust data slice
            val_env.current_step = 0 # Reset internal step counter for validation slice
            val_env.max_steps = val_steps # Set max steps for validation slice
        elif validation_start is not None:
            log.warning("Could not split environment data for validation (required attributes missing/invalid). "
                        "Training and validation will use the full environment.")
        else:
            log.info("No validation split requested. Training and validation use the full environment.")


        # --- Policy Kwargs ---
        policy_kwargs = dict(
            features_extractor_class=CachedLSTMAttention,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                max_seq_length=max_seq_length # Pass cache size here
            ),
            net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]) # Example architecture
        )
        log.info(f"Policy kwargs: {policy_kwargs}")

        # --- Buffer and Learning Starts Defaults ---
        if buffer_size is None:
            # Scale buffer size slightly with batch size, cap at 1M
            buffer_size = min(1_000_000, int(1e5 * (batch_size / 256)))
            log.info(f"Using default buffer_size: {buffer_size:,}")
        if learning_starts is None:
             # Ensure enough samples for a few batches before starting
             learning_starts = max(1000, batch_size * 8)
             log.info(f"Using default learning_starts: {learning_starts:,}")


        # --- Wrap Environments ---
        # The feature extractor instance will be created *inside* the SAC model.
        # We don't pass it to the wrapper here. Reset will be handled by SB3/callback.
        train_env_wrapped = TimeSeriesEnvWrapper(train_env, feature_extractor=None, episode_length=episode_length)
        val_env_wrapped = TimeSeriesEnvWrapper(val_env, feature_extractor=None, episode_length=episode_length)


        # --- Create SAC Model ---
        model = SAC(
            "MlpPolicy", # Base policy type, feature extractor is specified in policy_kwargs
            train_env_wrapped,
            learning_rate=3e-4,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1, # Train after each step
            gradient_steps=1, # Single gradient update per training step
            policy_kwargs=policy_kwargs,
            verbose=1,
            # tensorboard_log=log_dir, # Logger handles this now
            device=pytorch_device,
            seed=np.random.randint(0, 10000) # Set a random seed
        )

        # Set the configured logger
        model.set_logger(new_logger)
        log.info("SAC model created.")

        # --- Create Validation Callback ---
        validation_callback = ValidationCallback(
            val_env_wrapped,
            validation_freq=validation_freq,
            n_eval_episodes=n_eval_episodes,
            log_path=log_dir # Save best model in the same log directory
        )
        log.info("Validation callback created.")

        return model, validation_callback, log_dir

    except Exception as e:
        log.error(f"Error creating SAC agent: {e}", exc_info=True) # Log traceback
        raise

# --- Test Environment Creation Helper ---

def create_test_env(env: gym.Env, model: SAC, episode_length: int = 4000) -> gym.Env:
    """
    Creates a properly wrapped environment for testing/inference.

    Crucially, it does *not* provide the feature extractor reference,
    as the evaluation loop should handle resetting the model's state.

    Args:
        env: The base environment instance.
        model: The trained SB3 model (used only to potentially get info, not passed to wrapper).
        episode_length: Maximum steps per episode for the test wrapper.

    Returns:
        A Gymnasium environment wrapped for time series testing.
    """
    test_env = deepcopy(env)
    # Pass None for feature_extractor; reset should be handled externally
    # before starting evaluation episodes using model.policy.features_extractor.reset_cached_states()
    wrapped_env = TimeSeriesEnvWrapper(test_env, feature_extractor=None, episode_length=episode_length)
    log.info(f"Test environment created with episode length {episode_length}.")
    return wrapped_env

# --- Example Usage (Optional Guard) ---
if __name__ == '__main__':
    # This block is optional, demonstrates basic usage pattern.
    # Requires a concrete environment definition (e.g., DummyVecEnv or your custom env).
    print("Agent script loaded. Contains classes and factory functions.")
    print("To run training, import create_sac_agent and provide an environment.")

    # Example:
    # from stable_baselines3.common.env_util import make_vec_env
    # dummy_env_id = "Pendulum-v1" # Replace with your actual environment
    # base_env = make_vec_env(dummy_env_id, n_envs=1)
    # base_env = gym.make("YourTimeSeriesEnv-v0") # Use your custom env

    # try:
    #     # Assuming gym.make("YourTimeSeriesEnv-v0") works and returns an env with correct space
    #     # env = gym.make("YourTimeSeriesEnv-v0")
    #     # model, callback, logdir = create_sac_agent(env)
    #     # model.learn(total_timesteps=1_000_000, callback=callback)
    #     # print(f"Training finished. Logs in: {logdir}")
    # except NameError:
    #      print("Define 'YourTimeSeriesEnv-v0' or replace with a valid gym environment to run example.")
    # except Exception as e:
    #      print(f"An error occurred during example execution: {e}")