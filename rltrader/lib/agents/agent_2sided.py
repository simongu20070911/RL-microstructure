# agent.py
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
# Removed: from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
# Removed: from stable_baselines3.common.vec_env import DummyVecEnv # No longer needed for validation

# --- Action Space Definition (Example) ---
# Stays the same - defined by the environment
# action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
# Interpretation:
# action[0]: Buy Offset Signal
# action[1]: Sell Offset Signal
# action[2]: Buy Size Signal
# action[3]: Sell Size Signal
# action[4]: Replacement/Cancel Signal
# action[5]: Do Nothing Signal
# --- END MODIFIED ---


# Set up logging for this module.
def setup_logging():
    log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    # Basic logging config - can be customized further
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Log to console
    return log_dir

@dataclass
class CachedStates:
    """Holds cached hidden states for recurrent layers."""
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    key_cache: Optional[torch.Tensor] = None
    value_cache: Optional[torch.Tensor] = None
    cache_position: int = 0

# Removed ValidationStates dataclass

class CachedMultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional KV caching for efficient inference."""
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_length: int = 400):
        super().__init__()
        if hidden_dim % num_heads != 0:
             raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_seq_length = max_seq_length # Max length for the cache

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.key_cache = None
        self.value_cache = None
        self.cache_position = 0

    def reset_cache(self, batch_size: int, device: torch.device):
        """Resets the KV cache tensors."""
        logging.debug(f"Resetting Attention Cache for batch size {batch_size} on device {device}. Max length: {self.max_seq_length}")
        self.key_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.cache_position = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """Forward pass with optional KV caching."""
        B, L, D = x.shape
        H = self.num_heads

        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)

        if use_cache:
            if self.key_cache is None or self.value_cache is None:
                logging.warning("Attention cache used before reset. Re-initializing.")
                self.reset_cache(B, x.device)
                return self._compute_attention(q, k, v, B, L, D)

            if L != 1:
                 logging.warning(f"Using attention cache with sequence length L={L} > 1. Cache may behave unexpectedly.")

            if self.cache_position >= self.max_seq_length:
                # FIX: Use circular buffer instead of position reset to maintain temporal continuity
                shift_size = self.max_seq_length // 2
                logging.info(f"Attention cache full (pos {self.cache_position} >= max {self.max_seq_length}). Shifting cache by {shift_size} positions to maintain temporal continuity.")
                
                # Shift existing cache contents left by shift_size
                self.key_cache[:, :-shift_size] = self.key_cache[:, shift_size:].clone()
                self.value_cache[:, :-shift_size] = self.value_cache[:, shift_size:].clone()
                
                # Zero out the new space at the end
                self.key_cache[:, -shift_size:] = 0
                self.value_cache[:, -shift_size:] = 0
                
                # Continue from the new position
                self.cache_position = self.max_seq_length - shift_size
                logging.debug(f"Cache shifted. New position: {self.cache_position}")

            current_k = k.transpose(1, 2)
            current_v = v.transpose(1, 2)
            end_pos = self.cache_position + L
            if end_pos > self.max_seq_length:
                 logging.error(f"Cache update exceeds max length ({end_pos} > {self.max_seq_length}) unexpectedly.")
                 end_pos = self.max_seq_length
                 current_k = current_k[:, :(self.max_seq_length - self.cache_position)]
                 current_v = current_v[:, :(self.max_seq_length - self.cache_position)]

            if current_k.shape[1:] != self.key_cache[:, self.cache_position:end_pos].shape[1:] :
                 logging.error(f"Shape mismatch during cache update. K shape: {current_k.shape}, Cache slice shape: {self.key_cache[:, self.cache_position:end_pos].shape}")
                 return self._compute_attention(q, k, v, B, L, D)

            self.key_cache[:, self.cache_position:end_pos] = current_k
            self.value_cache[:, self.cache_position:end_pos] = current_v

            k_to_use = self.key_cache[:, :end_pos].transpose(1, 2)
            v_to_use = self.value_cache[:, :end_pos].transpose(1, 2)

            # FIX: Update cache position only after successful attention computation  
            try:
                result = self._compute_attention(q, k_to_use, v_to_use, B, L, D)
                # Update position only if attention computation succeeds
                self.cache_position = end_pos
                return result
            except Exception as e:
                logging.error(f'Cache attention computation failed: {e}')
                # Don't update cache_position if attention failed
                raise

        else: # Not using cache
            k_to_use, v_to_use = k, v
            return self._compute_attention(q, k_to_use, v_to_use, B, L, D)

    def _compute_attention(self, q, k, v, B, L, D):
        """Helper function to compute attention output."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


class CachedLSTMAttention(BaseFeaturesExtractor):
    """
    Feature extractor using LSTM followed by CachedMultiHeadAttention.
    Manages hidden states for LSTM and KV cache for Attention.
    """
    def __init__(self, observation_space: gym.spaces.Space,
                 features_dim: int = 128, hidden_dim: int = 256, num_heads: int = 8, lstm_layers: int = 1, dropout: float = 0.1, max_attn_cache_len: int = 400):
        super().__init__(observation_space, features_dim=features_dim)

        if isinstance(observation_space, spaces.Box):
            self.input_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space type: {type(observation_space)}")

        self.hidden_dim = hidden_dim
        self._features_dim = features_dim
        self.lstm_layers = lstm_layers

        # --- Layers ---
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.attention = CachedMultiHeadAttention(hidden_dim, num_heads, max_seq_length=max_attn_cache_len)
        self.layer_norm_lstm = nn.LayerNorm(hidden_dim)
        self.layer_norm_attn = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, self._features_dim)
        # --- End Layers ---

        self.cached_states: Optional[CachedStates] = None

    def reset_cached_states(self, batch_size: int = 1, device: Optional[torch.device] = None, clear_memory: bool = True):
        """Reset cached states for LSTM and Attention."""
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                 logging.warning("Could not get device from model parameters in reset_cached_states. Defaulting to CPU.")
                 device = torch.device("cpu")

        logging.debug(f"Resetting feature extractor cache (LSTM & Attention) for batch size {batch_size} on {device}. Clear memory: {clear_memory}")

        # --- LSTM State Reset ---
        new_lstm_state = None
        if clear_memory or self.cached_states is None or self.cached_states.lstm_state is None:
            h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
            new_lstm_state = (h_0, c_0)
            logging.debug("Created new zero LSTM state.")
        else:
            try:
                h_old, c_old = self.cached_states.lstm_state
                if h_old.shape[1] == batch_size:
                     new_lstm_state = (h_old.detach(), c_old.detach())
                     logging.debug("Detached existing LSTM state.")
                else:
                     logging.warning(f"Batch size mismatch during LSTM state reset ({h_old.shape[1]} vs {batch_size}). Creating new zero state.")
                     h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
                     c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
                     new_lstm_state = (h_0, c_0)
            except Exception as e:
                 logging.error(f"Error detaching LSTM state: {e}. Creating new zero state.")
                 h_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
                 c_0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
                 new_lstm_state = (h_0, c_0)

        # --- Attention Cache Reset ---
        if clear_memory:
            self.attention.reset_cache(batch_size, device)
            logging.debug("Attention cache reset (clear_memory=True).")
        else:
            # Preserve attention cache but ensure it's properly initialized
            if self.attention.key_cache is None or self.attention.value_cache is None:
                self.attention.reset_cache(batch_size, device)
                logging.debug("Attention cache initialized (was None).")
            else:
                logging.debug("Attention cache preserved (clear_memory=False).")

        # --- Update CachedStates Object ---
        if self.cached_states is None:
            self.cached_states = CachedStates()
        self.cached_states.lstm_state = new_lstm_state
        self.cached_states.key_cache = self.attention.key_cache
        self.cached_states.value_cache = self.attention.value_cache
        self.cached_states.cache_position = self.attention.cache_position

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM and Attention."""
        is_batched = observations.dim() == 3
        if not is_batched:
             observations = observations.unsqueeze(1)

        B, L, D = observations.shape

        # --- LSTM Processing ---
        lstm_state_input = None
        use_lstm_cache = not self.training and self.cached_states is not None and self.cached_states.lstm_state is not None

        if use_lstm_cache:
            cached_h, cached_c = self.cached_states.lstm_state
            if cached_h.shape[1] == B:
                lstm_state_input = self.cached_states.lstm_state
                logging.debug(f"Using cached LSTM state for forward pass (Input L={L}).")
            else:
                logging.warning(f"LSTM cache batch size mismatch ({cached_h.shape[1]}) vs input ({B}). Ignoring cache for this pass.")

        lstm_out, lstm_state_output = self.lstm(observations, lstm_state_input)

        if use_lstm_cache:
            # FIX: Detach gradients to prevent memory accumulation
            h_detached = lstm_state_output[0].detach()
            c_detached = lstm_state_output[1].detach()
            self.cached_states.lstm_state = (h_detached, c_detached)
            logging.debug("Updated cached LSTM state with gradient detachment.")

        lstm_out_norm = self.layer_norm_lstm(lstm_out)

        # --- Attention Processing ---
        use_attn_cache = not self.training and self.cached_states is not None
        attn_out = self.attention(lstm_out_norm, use_cache=use_attn_cache)
        attn_out_norm = self.layer_norm_attn(attn_out)

        # --- Combine and Final Layers ---
        combined = lstm_out_norm + attn_out_norm
        combined_dropped = self.dropout(combined)
        combined_last_step = combined_dropped[:, -1, :]
        features = self.fc(combined_last_step)

        return features

    @property
    def state_size(self) -> Optional[Tuple[torch.Tensor, ...]]:
         if self.cached_states:
             return self.cached_states.lstm_state
         return None

    @property
    def attention_cache_size(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
         if self.cached_states:
             return self.cached_states.key_cache, self.cached_states.value_cache
         return None


# --- Removed ValidationCallback Class ---


class TimeSeriesEnvWrapper(gym.Wrapper):
    """
    A wrapper primarily for managing episode length and potentially handling
    feature extractor state resets between episodes that span across large datasets.

    Args:
        env: The environment to wrap.
        feature_extractor: The feature extractor instance (e.g., CachedLSTMAttention)
                           used by the policy. Required for state resets.
        episode_length (int): The maximum number of steps per episode.
        log_frequency (int): How often (in steps) to log progress messages.
    """
    def __init__(self, env, feature_extractor: Optional[BaseFeaturesExtractor], episode_length: int = 400, log_frequency: int = 1000):
        super().__init__(env)
        if feature_extractor is not None and not hasattr(feature_extractor, "reset_cached_states"):
            logging.warning(f"Provided feature_extractor of type {type(feature_extractor)} lacks 'reset_cached_states' method. State resets won't occur via wrapper.")
            self.feature_extractor = None
        else:
            self.feature_extractor = feature_extractor

        self.episode_length = episode_length
        self.steps_in_current_episode = 0

        self.total_data_length = getattr(env.unwrapped, "max_steps", None) or \
                                 getattr(env.unwrapped, "current_step", 0)
        if self.total_data_length == 0 and hasattr(env.unwrapped, "order_book_history"):
             history = getattr(env.unwrapped, "order_book_history")
             if history is not None and len(history) > 0:
                self.total_data_length = len(env.unwrapped.order_book_history)


        self.log_frequency = log_frequency
        self.total_step_counter_across_episodes = 0

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        logging.info(f"TimeSeriesEnvWrapper initialized. Episode length: {self.episode_length}. Total data steps: {self.total_data_length}")


    def _process_observation(self, obs: Union[torch.Tensor, np.ndarray, Dict, List]) -> Union[np.ndarray, Dict, List]:
        """Ensures observation is a NumPy array on CPU for SB3 compatibility."""
        if isinstance(obs, torch.Tensor):
            return obs.detach().cpu().numpy()
        elif isinstance(obs, dict):
            return {k: self._process_observation(v) for k, v in obs.items()}
        elif isinstance(obs, (list, tuple)):
            return [self._process_observation(x) for x in obs]
        elif isinstance(obs, np.ndarray):
            return obs
        else:
            try:
                return np.asarray(obs)
            except Exception as e:
                logging.error(f"Failed to convert observation of type {type(obs)} to NumPy array: {e}")
                return obs


    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Resets the underlying environment and the wrapper's step counter.
        Also resets the feature extractor's hidden state.
        """
        self.steps_in_current_episode = 0

        current_pos_in_data = getattr(self.env.unwrapped, "current_step", "N/A")
        data_perc = "N/A"
        if isinstance(current_pos_in_data, int) and self.total_data_length and self.total_data_length > 0:
            data_perc = f"{(current_pos_in_data / self.total_data_length) * 100:.1f}%"
        logging.info(f"Resetting episode. Starting at data step {current_pos_in_data} / {self.total_data_length} ({data_perc})")

        # Reset the feature extractor's state (crucial for LSTM/Attention)
        if self.feature_extractor is not None:
            try:
                batch_size = getattr(self.env, "num_envs", 1) # Assumes env might be VecEnv later
                device = next(self.feature_extractor.parameters()).device
                
                # Check if environment wants to preserve memory across episodes
                # First check the options passed to reset, then check environment attribute
                preserve_memory = False
                if options and 'preserve_memory' in options:
                    preserve_memory = options['preserve_memory']
                else:
                    preserve_memory = getattr(self.env.unwrapped, 'preserve_memory', False)
                clear_memory = not preserve_memory
                
                self.feature_extractor.reset_cached_states(batch_size=batch_size, device=device, clear_memory=clear_memory)
                logging.debug(f"Feature extractor state reset via wrapper. Clear memory: {clear_memory}")
            except Exception as e:
                logging.error(f"Error resetting feature extractor state in wrapper: {e}")

        obs, info = self.env.reset(seed=seed, options=options)
        processed_obs = self._process_observation(obs)

        # Add initial PnL to info for logging (optional, but good practice)
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'config'):
            initial_capital = self.env.unwrapped.config.get("initial_capital", 0.0)
            info['episode_pnl'] = info.get('mtm', 0.0) - initial_capital
        else:
            info['episode_pnl'] = info.get('mtm', 0.0)

        return processed_obs, info


    def step(self, action):
        """
        Steps the underlying environment, increments counters, handles truncation,
        and processes observation/info.
        """
        self.steps_in_current_episode += 1
        self.total_step_counter_across_episodes += 1

        if self.total_step_counter_across_episodes % self.log_frequency == 0:
            current_pos = getattr(self.env.unwrapped, "current_step", "N/A")
            perc = "N/A"
            if isinstance(current_pos, int) and self.total_data_length and self.total_data_length > 0:
                 perc = f"{(current_pos / self.total_data_length) * 100:.1f}%"
            logging.info(f"Wrapper Step: {self.total_step_counter_across_episodes}. Env Data Step: {current_pos}/{self.total_data_length} ({perc}). Ep Step: {self.steps_in_current_episode}")

            # --- NaN/Inf Checks ---
            # Check action before passing to env
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                 logging.error(f"!!! NaN/Inf detected in ACTION before step {self.total_step_counter_across_episodes} !!! Action: {action}")
                 # FIX: Replace invalid actions with safe 'do nothing' action
                 original_action = action.copy()
                 action = np.clip(np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
                 action[5] = 0.9  # Force 'do nothing' mode for safety
                 logging.warning(f"Replaced invalid action {original_action} with safe action {action}")

        obs, reward, terminated, truncated, info = self.env.step(action)

        # --- Post-step NaN/Inf Checks ---
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            logging.error(f"!!! NaN/Inf detected in observation at step {self.total_step_counter_across_episodes} !!! Obs: {obs}")
        if np.isnan(reward) or np.isinf(reward):
            logging.error(f"!!! NaN/Inf detected in reward at step {self.total_step_counter_across_episodes} !!! Reward: {reward}")


        if self.steps_in_current_episode >= self.episode_length:
            truncated = True
            logging.debug(f"Episode truncated by wrapper at {self.steps_in_current_episode} steps.")
            info["TimeLimit.truncated"] = info.get("TimeLimit.truncated", False) or truncated

        processed_obs = self._process_observation(obs)

        if terminated or truncated:
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'config'):
                 initial_capital = self.env.unwrapped.config.get("initial_capital", 0.0)
                 # Use info.get('mtm', initial_capital) to handle cases where mtm might be missing on final step
                 final_mtm = info.get('mtm', initial_capital)
                 info['episode_pnl'] = final_mtm - initial_capital
            else:
                 info['episode_pnl'] = info.get('mtm', 0.0)


        return processed_obs, reward, terminated, truncated, info


def create_sac_agent(env: gym.Env, # Env should already have the 6D action space
                     hidden_dim: int = 256,
                     features_dim: int = 128,
                     lstm_layers: int = 1,
                     num_heads: int = 8,
                     # Removed validation_start parameter
                     device: Union[torch.device, str] = "auto",
                     batch_size: int = 256,
                     buffer_size_mb: int = 1000,
                     learning_starts_multiplier: int = 4,
                     log_dir: Optional[str] = None,
                     episode_length: int = 400,
                     net_arch_pi: List[int] = [256, 256],
                     net_arch_qf: List[int] = [256, 256]
                    ) -> Tuple[SAC, str]: # Updated return type hint
    """
    Creates an SAC agent with CachedLSTMAttention feature extraction,
    sets up the training environment, and logging. Trains on the full dataset.

    Args:
        env: The base Gymnasium environment instance. MUST have specific attributes
             like `order_book_history`, `max_steps`, and allow deepcopying.
             Should also have a `config` attribute for re-initialization.
             *** IMPORTANT: This env instance MUST already have the correct (6D) action space defined. ***
        hidden_dim: Dimension for LSTM hidden state and Attention layers.
        features_dim: Output dimension of the feature extractor.
        lstm_layers: Number of layers in the LSTM.
        num_heads: Number of heads in the Multi-Head Attention.
        device: PyTorch device ("cuda", "cpu", "auto").
        batch_size: Batch size for training.
        buffer_size_mb: Replay buffer size in Megabytes (approx).
        learning_starts_multiplier: Multiplier for batch_size to determine learning_starts.
        log_dir: Directory for TensorBoard logs and saved models. If None, created automatically.
        episode_length: Maximum steps per episode for the TimeSeriesEnvWrapper.
        net_arch_pi: Architecture of the policy network (actor) after feature extraction.
        net_arch_qf: Architecture of the value network (critic) after feature extraction.

    Returns:
        Tuple containing the SAC model and the log directory path.
    """
    try:
        # --- Setup Device and Logging ---
        if device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        if log_dir is None:
            log_dir = setup_logging()
        else:
            os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Logging directory: {log_dir}")

        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

        # --- Environment Setup (Using Full Data) ---
        if not hasattr(env, 'order_book_history') or not hasattr(env, 'config'):
             raise AttributeError("The base environment must have 'order_book_history' and 'config' attributes.")

        if not env.order_book_history:
             logging.warning("Base environment 'order_book_history' is empty. Attempting to load using config...")
             try:
                env._load_order_book_data()
                if not env.order_book_history:
                     raise ValueError("Failed to load order book data even after attempting reload.")
             except Exception as e:
                 raise ValueError(f"Base environment 'order_book_history' is empty and reload failed: {e}")


        total_steps = len(env.order_book_history)
        if total_steps <= episode_length:
             raise ValueError(f"Total data steps ({total_steps}) must be greater than episode length ({episode_length}).")

        logging.info(f"Total data steps available for training: {total_steps}.")

        # Create the training environment by re-initializing (ensures clean state)
        base_config = deepcopy(env.config)
        env_class = type(env)
        try:
            # Use the full history for the training environment
            train_env_base = env_class(base_config)
            train_env_base.order_book_history = env.order_book_history # Assign the full history
            train_env_base.max_steps = total_steps # Set max steps accordingly
            # train_env_base.reset() # Optional, depends if env needs reset after history assignment
        except Exception as e:
            logging.error(f"Failed to re-initialize training environment using type {env_class} and config. Error: {e}")
            raise

        logging.info(f"Train env created using full data length: {len(train_env_base.order_book_history)}")


        # --- SAC Model Configuration ---
        policy_kwargs = dict(
            features_extractor_class=CachedLSTMAttention,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                lstm_layers=lstm_layers,
                max_attn_cache_len=episode_length + 10
            ),
            net_arch=dict(pi=net_arch_pi, qf=net_arch_qf)
        )

        # Estimate buffer size
        if not isinstance(train_env_base.action_space, spaces.Box):
             raise TypeError(f"Expected Box action space, got {type(train_env_base.action_space)}")
        if not isinstance(train_env_base.observation_space, spaces.Box):
             raise TypeError(f"Expected Box observation space, got {type(train_env_base.observation_space)}")

        obs_bytes = train_env_base.observation_space.shape[0] * train_env_base.observation_space.dtype.itemsize
        action_bytes = train_env_base.action_space.shape[0] * train_env_base.action_space.dtype.itemsize
        reward_bytes = 4
        done_bytes = 1
        bytes_per_transition_approx = (obs_bytes + action_bytes + reward_bytes + done_bytes + obs_bytes) * 1.1

        if bytes_per_transition_approx <= 0:
            raise ValueError("Calculated bytes_per_transition_approx is zero or negative.")

        buffer_size_transitions = int((buffer_size_mb * 1024 * 1024) / bytes_per_transition_approx)
        learning_starts = batch_size * learning_starts_multiplier
        buffer_size = max(buffer_size_transitions, learning_starts + batch_size)
        logging.info(f"Obs dim: {train_env_base.observation_space.shape[0]}, Action dim: {train_env_base.action_space.shape[0]}")
        logging.info(f"Approx. Bytes/Transition: {bytes_per_transition_approx:.1f}. Target Buffer Size: {buffer_size_transitions:,} transitions (~{buffer_size_mb}MB). Actual Buffer Size: {buffer_size:,}")
        logging.info(f"Learning Starts: {learning_starts:,}")


        # --- Create SAC Agent ---
        model = SAC(
            "MlpPolicy",
            train_env_base, # Pass base env initially
            learning_rate=3e-4,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            seed=np.random.randint(0, 10000)
        )

        # --- Wrap Training Environment and Assign Feature Extractor ---
        feature_extractor = model.policy.features_extractor
        if not isinstance(feature_extractor, CachedLSTMAttention):
             logging.warning(f"Model's feature extractor is not CachedLSTMAttention (type: {type(feature_extractor)}). Wrapper state resets might not work as expected.")

        train_env_wrapped = TimeSeriesEnvWrapper(train_env_base, feature_extractor, episode_length=episode_length)


        # --- No Validation Callback Setup ---

        # Assign the logger to the model
        model.set_logger(new_logger)

        # Set the model's environment to the *wrapped* training environment
        model.set_env(train_env_wrapped)
        logging.info("Assigned wrapped training environment to the SAC model.")

        logging.info("SAC agent created successfully (without validation callback).")
        # Return model and log_dir
        return model, log_dir

    except AttributeError as ae:
        logging.error(f"Attribute error during agent creation: {ae}. Does the base env have required attributes (e.g., 'order_book_history', 'config')?")
        raise
    except ValueError as ve:
        logging.error(f"Value error during agent creation: {ve}.")
        raise
    except Exception as e:
        logging.error(f"Unexpected error creating SAC agent: {e}", exc_info=True)
        raise


def create_test_env(env_config: dict, model: SAC, episode_length: int = 4000) -> gym.Env:
    """
    Creates a properly wrapped test environment using the trained model's
    feature extractor for state resets. Requires HFTEnv class to be available.

    Args:
        env_config: Configuration dictionary for the base HFTEnv.
        model: The trained SAC model (its feature extractor will be used).
        episode_length: Max length of each test episode.

    Returns:
        A wrapped Gymnasium environment ready for testing.
    """
    logging.info("Creating test environment...")
    if not hasattr(model.policy, "features_extractor") or \
       not isinstance(model.policy.features_extractor, CachedLSTMAttention):
        logging.warning("Model policy does not have a CachedLSTMAttention feature extractor. Test env wrapper might not reset state correctly.")
        feature_extractor = None
    else:
        feature_extractor = model.policy.features_extractor

    try:
        # Assumes HFTEnv class is correctly imported where this function is called
        from rltrader.envs import TwoSidedMarketEnv as HFTEnv  # Updated import path
    except ImportError:
        logging.error("Could not import HFTEnv for test environment creation. Ensure it's in the Python path.")
        raise

    test_env_base = HFTEnv(env_config)
    wrapped_env = TimeSeriesEnvWrapper(test_env_base, feature_extractor, episode_length=episode_length)
    logging.info("Test environment created and wrapped.")
    return wrapped_env

# Example Usage (Illustrative - Adapt to your main script)
if __name__ == '__main__':
    # Activate anomaly detection for debugging NaN issues during training
    # torch.autograd.set_detect_anomaly(True) # Uncomment for detailed NaN tracebacks (slows training)

    # --- 1. Define Configuration ---
    config = {
        "csv_path":"/path/to/your/orderbook_data.csv", # <<< --- UPDATE THIS PATH --- >>>
        "initial_capital": 20000.0,
        "max_steps": 50000,
        "order_book_levels": 9,
        "price_offset_ticks": 10,
        "max_order_volume": 2.5,
        "tick_size": 0.01,
        "lot_size": 0.005,
        "max_active_orders": 10,
        "max_inventory": 5.0,
        "episode_length": 400,
        "env_path": "rltrader.envs.two_sided",
        "env_class": "HFTEnv",
        "latency_steps_long": 1,
        "latency_steps_short": 1,
        "transaction_cost_long": 0.0004,
        "transaction_cost_short": 0.0004,
        "inventory_penalty": 0.00001,
        "invalid_order_penalty": 0.001,
        "activity_bonus": 0.0001,
        "taker_penalty": 0.005,
        "allowed_aggressiveness_ticks": 5,
        "quoting_reward_enabled": True,
        "quoting_reward_amount": 0.00002,
        "quoting_reward_max_ticks": 3,
        "explicit_cancel_enabled": True,
        "explicit_cancel_threshold": 0.5,
        "explicit_cancel_penalty": 0.00005,
        "explicit_cancel_clears_pending": True,
        "do_nothing_threshold": 0.7,
    }

    # --- 2. Import the Environment Class ---
    try:
        from rltrader.envs import TwoSidedMarketEnv as HFTEnv  # Updated import path
    except ImportError:
        logging.error("Could not import HFTEnv. Make sure env_2sided.py is accessible.")
        exit()

    # --- 3. Create the Base Environment Instance ---
    try:
        base_env = HFTEnv(config)
        logging.info(f"Base environment created. Action space: {base_env.action_space}")
        logging.info(f"Observation space: {base_env.observation_space}")
    except Exception as e:
        logging.error(f"Failed to create base HFTEnv: {e}", exc_info=True)
        exit()

    # --- 4. Create the SAC Agent ---
    try:
        # Removed validation_start, now returns model, log_dir
        model, log_dir = create_sac_agent(
            env=base_env,
            hidden_dim=256,
            features_dim=128,
            lstm_layers=1,
            num_heads=4,
            device="auto",
            batch_size=128,
            buffer_size_mb=500,
            learning_starts_multiplier=10,
            log_dir=None, # Auto-generate log directory
            episode_length=config["episode_length"],
            net_arch_pi=[128, 128],
            net_arch_qf=[128, 128]
        )
    except Exception as e:
        logging.error(f"Failed to create SAC agent: {e}", exc_info=True)
        exit()

    # --- 5. Train the Agent ---
    TOTAL_TIMESTEPS = 1_000_000 # Example total steps
    logging.info(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        # Removed callback argument
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            log_interval=10, # Log training stats every 10 episodes (or adjust as needed)
            progress_bar=True # Show progress bar
        )
        logging.info("Training finished.")
        # Save the final model
        final_model_path = os.path.join(log_dir, "final_model.zip")
        model.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

    # --- 6. Test the Trained (Final) Agent ---
    logging.info(f"Loading final model from {final_model_path} for testing...")
    if os.path.exists(final_model_path):
        loaded_model = SAC.load(final_model_path)

        # Create a separate test environment
        test_config = config.copy()
        # Optionally change config for testing (e.g., different data, evaluation settings)
        # test_config["csv_path"] = "/path/to/your/test_data.csv"
        # test_config["max_steps"] = 10000 # Example: limit test length

        try:
            test_env = create_test_env(test_config, loaded_model, episode_length=2000)
            obs, info = test_env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            episode_pnl = 0 # Track PnL specifically for testing output

            while not terminated and not truncated:
                action, _states = loaded_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                step_count += 1
                if step_count % 100 == 0:
                    # Log PnL from info if available
                    current_pnl = info.get('episode_pnl', 'N/A')
                    logging.info(f"Test Step: {step_count}, Current Ep PnL: {current_pnl}")


            # Capture final PnL
            episode_pnl = info.get('episode_pnl', 'N/A')
            logging.info(f"Test finished. Steps: {step_count}, Total Reward: {total_reward:.4f}, Final Episode PnL: {episode_pnl}, Final Info: {info}")
            test_env.close()
        except Exception as e:
            logging.error(f"Error during testing: {e}", exc_info=True)
    else:
        logging.warning(f"Final model file not found at {final_model_path}, skipping testing.")
