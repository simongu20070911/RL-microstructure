import time
import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import logging

# Import your agent module (ensure agent.py is in your PYTHONPATH or same folder)
import agents.agent as agent

# =============================================================================
# Custom SAC Policy forcing the use of CachedLSTMAttention as the feature extractor.
#
# This subclass overrides the features_extractor settings so that even for a 
# low-dimensional observation space, your custom extractor is built.
# =============================================================================
from stable_baselines3.sac.sac import SACPolicy

class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        # Force our custom extractor settings:
        kwargs["features_extractor_class"] = agent.CachedLSTMAttention
        # Set default kwargs for the extractor; adjust as needed.
        kwargs["features_extractor_kwargs"] = {
            "features_dim": 128,
            "hidden_dim": 256,
            "num_heads": 8
        }
        super(CustomSACPolicy, self).__init__(*args, **kwargs)
        if self.features_extractor is None:
            raise ValueError("features_extractor is None; custom extractor was not built!")
        else:
            print(f"Using custom features_extractor: {self.features_extractor.__class__.__name__}")

# =============================================================================
# Helper: Create a custom SAC agent using our CustomSACPolicy.
#
# This function builds train and validation environments, splits the dummy
# order book history, and passes our custom policy via policy_kwargs.
# =============================================================================
def create_custom_sac_agent(env, device="auto", batch_size=256, log_dir=None):
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if log_dir is None:
        log_dir = agent.setup_logging()
    new_logger = agent.configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    from copy import deepcopy
    train_env = deepcopy(env)
    val_env = deepcopy(env)
    
    total_steps = len(env.order_book_history)
    train_steps = int(total_steps * 0.8)
    train_env.max_steps = train_steps
    val_env.order_book_history = val_env.order_book_history[train_steps:]
    val_env.max_steps = total_steps - train_steps
    
    # Wrap environments to enforce a fixed episode length.
    train_env = agent.TimeSeriesEnvWrapper(train_env, None, episode_length=400)
    val_env = agent.TimeSeriesEnvWrapper(val_env, None, episode_length=400)
    
    # Even though our CustomSACPolicy forces the extractor, we still pass net_arch.
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])
    )
    
    from stable_baselines3 import SAC
    model = SAC(
        CustomSACPolicy,
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
    
    # Attach the features extractor to the env wrappers (for cache resets).
    train_env.feature_extractor = model.policy.features_extractor
    val_env.feature_extractor = model.policy.features_extractor
    
    validation_callback = agent.ValidationCallback(val_env, validation_freq=10000)
    model.set_logger(new_logger)
    return model, validation_callback, log_dir

# =============================================================================
# Dummy Time-Series Environment
#
# The observation space is defined as a vector of length 76 (not a sequence).
# The TimeSeriesEnvWrapper later stacks observations into a sequence.
# =============================================================================
class DummyTimeSeriesEnv(gym.Env):
    def __init__(self):
        super(DummyTimeSeriesEnv, self).__init__()
        self.feature_size = 76
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.feature_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.order_book_history = np.zeros((10000, self.feature_size), dtype=np.float32)
        self.max_steps = len(self.order_book_history)
        self.current_step = 0
        
    def reset(self, seed=None, options=None):
        self.current_step = 0
        # Return a random observation (vector of length 76)
        obs = np.random.uniform(-1, 1, size=(self.feature_size,)).astype(np.float32)
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        obs = np.random.uniform(-1, 1, size=(self.feature_size,)).astype(np.float32)
        reward = np.random.rand()
        done = self.current_step >= self.max_steps
        truncated = False
        info = {"episode_pnl": np.random.randn()}
        return obs, reward, done, truncated, info

# =============================================================================
# Test the Cached Feature Extractor forward pass.
#
# This function runs a batch forward pass (simulating a full sequence of 400 steps)
# and sequential one-step calls while printing the internal cache position.
# =============================================================================
def test_feature_extractor_forward_pass(device: torch.device, batch_size: int, sequence_length: int, feature_size: int):
    print(f"\n=== Testing Feature Extractor on {device} ===")
    # The observation space is a vector of length `feature_size`
    observation_space = spaces.Box(low=-1, high=1, shape=(feature_size,), dtype=np.float32)
    
    extractor = agent.CachedLSTMAttention(
        observation_space=observation_space,
        features_dim=128,
        hidden_dim=256,
        num_heads=8
    ).to(device)
    extractor.eval()
    
    # Reset cached states (simulate starting a new sequence)
    extractor.reset_cached_states(batch_size=batch_size, clear_memory=True)
    print(f"Initial cache position: {extractor.attention.cache_position}")
    
    # --- Batch forward pass ---
    # Simulate a full sequence by stacking `sequence_length` observations: shape (batch, sequence_length, feature_size)
    dummy_input = torch.randn(batch_size, sequence_length, feature_size, device=device)
    with torch.no_grad():
        _ = extractor(dummy_input)  # Warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output = extractor(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"Batch forward pass: Input shape {dummy_input.shape} -> Output shape {output.shape}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Cache position after batch pass: {extractor.attention.cache_position}")
    
    # --- Sequential forward passes (simulate one-step calls using the cache) ---
    num_steps = 10
    print("\nSequential forward pass timings (using cached states):")
    for step in range(num_steps):
        # One-step input: shape (batch_size, 1, feature_size)
        dummy_step = torch.randn(batch_size, 1, feature_size, device=device)
        t0 = time.time()
        with torch.no_grad():
            out_step = extractor(dummy_step)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        cache_pos = extractor.attention.cache_position
        print(f" Step {step}: {t1 - t0:.6f} sec, Output shape: {out_step.shape}, Cache position: {cache_pos}")

# =============================================================================
# Test the full SAC Policy forward pass.
#
# This function creates the custom SAC agent, resets the extractor’s cache,
# and times both an initial and sequential policy forward pass.
# =============================================================================
def test_policy_forward_pass(device: torch.device):
    print(f"\n=== Testing SAC Policy Forward Pass on {device} ===")
    env = DummyTimeSeriesEnv()
    model, validation_callback, log_dir = create_custom_sac_agent(
        env, device=device, batch_size=1
    )
    model.policy.eval()
    
    # Get a dummy observation (vector of length 76) and add batch dimension -> (1, 76)
    obs, _ = env.reset()
    dummy_obs = torch.tensor(obs, device=device).float().unsqueeze(0)
    
    # Reset cached states in the feature extractor.
    if hasattr(model.policy, "features_extractor") and model.policy.features_extractor is not None:
        model.policy.features_extractor.reset_cached_states(batch_size=1, clear_memory=True)
        print(f"Initial policy extractor cache position: {model.policy.features_extractor.attention.cache_position}")
    else:
        raise ValueError("features_extractor not found in policy!")
    
    # Warm-up pass.
    with torch.no_grad():
        _ = model.policy(dummy_obs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        action = model.policy(dummy_obs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    print(f"Initial policy forward pass: Action output shape: {action.shape}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Policy extractor cache position after initial pass: {model.policy.features_extractor.attention.cache_position}")
    
    # --- Sequential forward passes (simulate cached calls in the policy) ---
    num_steps = 10
    print("\nSequential policy forward pass timings (with cached states):")
    for step in range(num_steps):
        obs_sample, _ = env.reset()
        dummy_obs = torch.tensor(obs_sample, device=device).float().unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            out = model.policy(dummy_obs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        cache_pos = model.policy.features_extractor.attention.cache_position
        print(f" Step {step}: {t1 - t0:.6f} sec, Action output shape: {out.shape}, Cache position: {cache_pos}")

# =============================================================================
# Main execution: run tests on CPU and CUDA (if available).
# =============================================================================
if __name__ == "__main__":
    batch_size = 1        # Use a single-sample batch for these tests.
    sequence_length = 400 # Simulate a full sequence of 400 time steps.
    feature_size = 76     # Expected feature dimension.
    
    print("===== Running tests on CPU =====")
    test_feature_extractor_forward_pass(torch.device("cpu"), batch_size, sequence_length, feature_size)
    test_policy_forward_pass(torch.device("cpu"))
    
    if torch.cuda.is_available():
        print("\n===== Running tests on CUDA =====")
        test_feature_extractor_forward_pass(torch.device("cuda"), batch_size, sequence_length, feature_size)
        test_policy_forward_pass(torch.device("cuda"))
    else:
        print("\nCUDA not available. Only CPU tests were run.")