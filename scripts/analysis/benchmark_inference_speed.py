# benchmark_inference_speed.py

import os
import time
import logging
from datetime import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union, Dict

# Assume gymnasium is installed, but we only need spaces for instantiation here.
# If not available, replace with a dummy class having shape attribute.
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Warning: gymnasium not found. Using dummy space.")
    class DummySpace:
        def __init__(self, shape):
            self.shape = shape
    spaces = DummySpace # type: ignore

import numpy as np
import torch
import torch.nn as nn

# --- Paste or Import your agent code here ---
# Option 1: Paste directly (if agent.py is simple enough)
# Option 2: Ensure agents/ is in python path and import (Recommended)
# Make sure the script can find the 'agents' directory.
# If running from the directory containing 'agents':
# from agents.agent import (
#     CachedStates, ValidationStates, CachedMultiHeadAttention, CachedLSTMAttention
# )
# If running from *inside* the agents directory:
# from agent import (
#      CachedStates, ValidationStates, CachedMultiHeadAttention, CachedLSTMAttention
# )
# Or adjust sys.path if needed:
# import sys
# sys.path.append('.') # If running from parent directory of 'agents'

# --- For demonstration, pasting the necessary classes directly ---
# (In a real project, use imports as shown above)

@dataclass
class CachedStates:
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    key_cache: torch.Tensor
    value_cache: torch.Tensor

# ValidationStates not needed for benchmark, but included for completeness if copy-pasting
@dataclass
class ValidationStates:
    lstm_state: Tuple[torch.Tensor, torch.Tensor]
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    episode_returns: List[float]
    current_return: float = 0.0

class CachedMultiHeadAttention(nn.Module):
    # --- (Paste the full CachedMultiHeadAttention class code here) ---
    def __init__(self, hidden_dim: int, num_heads: int, max_seq_length: int = 400):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
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
        # Important: Cache shape uses self.head_dim
        self.key_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.cache_position = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H = self.num_heads
        # Make sure head_dim is correct if hidden_dim/num_heads was adjusted
        head_dim = D // H # Recalculate based on actual input dim D passed to linear layers

        # Project queries, keys, and values.
        q = self.q_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]
        v = self.v_proj(x).view(B, L, H, head_dim).transpose(1, 2) # [B, H, L, Dh]

        k_to_use = k
        v_to_use = v

        if use_cache and self.key_cache is not None and self.value_cache is not None:
            # Check if cache size matches batch size
            if self.key_cache.shape[0] != B:
                 print(f"Warning: Resetting cache due to batch size mismatch (expected {self.key_cache.shape[0]}, got {B})")
                 self.reset_cache(B, x.device) # Or handle error

            if self.cache_position + L <= self.max_seq_length:
                # Store k, v transposed to match cache shape [B, max_len, H, Dh]
                self.key_cache[:, self.cache_position : self.cache_position + L] = k.transpose(1, 2)
                self.value_cache[:, self.cache_position : self.cache_position + L] = v.transpose(1, 2)

                # Retrieve cached k, v for attention calculation, transpose back to [B, H, len, Dh]
                k_to_use = self.key_cache[:, : self.cache_position + L].transpose(1, 2)
                v_to_use = self.value_cache[:, : self.cache_position + L].transpose(1, 2)

                self.cache_position += L
            else:
                # Cache overflow - handle as needed (e.g., reset, use rolling window)
                print("Warning: Attention cache overflow. Resetting cache.")
                self.reset_cache(B, x.device)
                # Use current k, v only after reset
                k_to_use = k
                v_to_use = v


        # --- Attention Calculation ---
        # q: [B, H, L, Dh], k_to_use: [B, H, cache_len, Dh] -> k_to_use.transpose: [B, H, Dh, cache_len]
        # scores: [B, H, L, cache_len]
        scores = torch.matmul(q, k_to_use.transpose(-2, -1)) / np.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1) # Softmax over cache_len dimension

        # attn: [B, H, L, cache_len], v_to_use: [B, H, cache_len, Dh]
        # out: [B, H, L, Dh]
        out = torch.matmul(attn, v_to_use)

        # Concatenate heads and project
        # out.transpose: [B, L, H, Dh] -> view: [B, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# NOTE: BaseFeaturesExtractor requires observation_space and features_dim
# We need a dummy BaseFeaturesExtractor or just inherit from nn.Module for the benchmark
# Let's use nn.Module for simplicity in the benchmark context.
class BenchmarkingFeatureExtractor(nn.Module):
    # --- (Paste the relevant parts of CachedLSTMAttention here) ---
    def __init__(self, observation_space_shape, features_dim=128, hidden_dim=256, num_heads=8, max_seq_length=400):
        super().__init__()

        # Use shape directly instead of observation_space object
        self.input_dim = observation_space_shape[0]
        self.hidden_dim = hidden_dim
        self._features_dim = features_dim # Renamed to avoid clash

        # LSTM layer.
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)

        # Cached attention.
        self.attention = CachedMultiHeadAttention(hidden_dim, num_heads, max_seq_length=max_seq_length)

        # Final projection, normalization, and dropout.
        self.fc = nn.Linear(hidden_dim, self._features_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Dropout is typically disabled during eval, but we keep it for structural consistency
        self.dropout = nn.Dropout(0.1)

        # Cached states for recurrent layers.
        self.cached_states = None
        # self.validation_states = None # Not needed for benchmark

        # Track if model is in training mode (influences caching)
        self.training = False # Default to eval mode for benchmark

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        self.training = mode
        super().train(mode)
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)


    def reset_cached_states(self, batch_size: int = 1, clear_memory: bool = False):
        """Reset cached states with an option to clear memory."""
        device = next(self.parameters()).device
        # Always create new tensors for benchmark consistency unless explicitly reusing
        # For benchmark, `clear_memory=True` should likely always be true between sequences
        # if clear_memory or self.cached_states is None: # Original logic
        if True: # Simplified for benchmark: always reset fully or create if None
            lstm_state = (
                torch.zeros(1, batch_size, self.hidden_dim, device=device),
                torch.zeros(1, batch_size, self.hidden_dim, device=device)
            )
            # Reset attention cache state as well
            self.attention.reset_cache(batch_size, device)

            # Assign the newly created cache references
            self.cached_states = CachedStates(
                lstm_state=lstm_state,
                key_cache=self.attention.key_cache, # Reference the reset cache
                value_cache=self.attention.value_cache # Reference the reset cache
            )
        # The 'else' part (reusing detached states) is less relevant for a clean benchmark start
        # else: # Reuse existing states but detached
        #     lstm_state = (
        #         self.cached_states.lstm_state[0].detach().clone(), # Clone to be safe
        #         self.cached_states.lstm_state[1].detach().clone()
        #     )
        #     # Keep existing attention cache, just detach? Be careful with positions
        #     if self.attention.key_cache is not None:
        #         self.attention.key_cache = self.attention.key_cache.detach().clone()
        #         self.attention.value_cache = self.attention.value_cache.detach().clone()
        #     # Update cached_states tuple with new detached tensor references
        #     self.cached_states = CachedStates(
        #         lstm_state=lstm_state,
        #         key_cache=self.attention.key_cache,
        #         value_cache=self.attention.value_cache
        #     )


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure input has sequence length dimension (L=1 for step-by-step)
        if len(observations.shape) == 2: # (B, D)
            observations = observations.unsqueeze(1) # -> (B, 1, D)
        elif len(observations.shape) != 3: # (B, L, D)
             raise ValueError(f"Expected 2D or 3D input, got shape {observations.shape}")

        # --- Determine if cache should be used ---
        # Cache is used if:
        # 1. Not in training mode (self.training is False)
        # 2. Cached states exist (self.cached_states is not None)
        use_lstm_cache = not self.training and self.cached_states is not None
        use_attn_cache = not self.training and self.cached_states is not None \
                         and self.attention.key_cache is not None \
                         and self.attention.value_cache is not None

        # --- LSTM Layer ---
        if use_lstm_cache:
            # Pass previous state
            lstm_out, new_lstm_state = self.lstm(observations, self.cached_states.lstm_state)
            # Update cache FOR THE NEXT STEP
            self.cached_states.lstm_state = new_lstm_state
        else:
            # No state passed, LSTM computes initial state implicitly
            lstm_out, new_lstm_state = self.lstm(observations)
            # If we start caching mid-sequence (unlikely in benchmark), init cache here
            # if not self.training and self.cached_states is None:
            #    self.reset_cached_states(batch_size=observations.shape[0])
            #    self.cached_states.lstm_state = new_lstm_state

        # --- Attention Layer ---
        # Pass use_cache flag explicitly to attention forward
        attn_out = self.attention(lstm_out, use_cache=use_attn_cache)

        # --- Combine & Project ---
        # Residual connection + Norm
        combined = self.layer_norm(lstm_out + attn_out)
        # Dropout (will be inactive in eval mode)
        combined = self.dropout(combined)

        # Final projection using the last time step feature (or the only one if L=1)
        features = self.fc(combined[:, -1, :]) # Output shape: [B, features_dim]
        return features

# --- Benchmark Configuration ---
INPUT_DIM = 76
HIDDEN_DIM = 256
FEATURES_DIM = 128
NUM_HEADS = 8
MAX_SEQ_LENGTH = 400 # Max length for attention cache
BENCHMARK_SEQ_LENGTH = 200 # How many steps to simulate in sequence
N_RUNS = 10           # How many sequences to average over
DEVICE = 'cpu'        # Focus on CPU inference

print(f"--- Benchmark Parameters ---")
print(f"Device:           {DEVICE}")
print(f"Input Dimension:  {INPUT_DIM}")
print(f"Hidden Dimension: {HIDDEN_DIM}")
print(f"Features Dim:     {FEATURES_DIM}")
print(f"Num Heads:        {NUM_HEADS}")
print(f"Max Seq Length:   {MAX_SEQ_LENGTH}")
print(f"Benchmark Seq Len:{BENCHMARK_SEQ_LENGTH}")
print(f"Number of Runs:   {N_RUNS}")
print("-" * 30)

# --- Model Initialization ---
# Create a dummy observation space shape tuple
dummy_obs_space_shape = (INPUT_DIM,)

model = BenchmarkingFeatureExtractor(
    observation_space_shape=dummy_obs_space_shape,
    features_dim=FEATURES_DIM,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    max_seq_length=MAX_SEQ_LENGTH
)
model.to(DEVICE)
# Ensure dropout and batchnorm are in eval mode if they existed
# Our custom class handles this via self.training flag check

# --- Benchmark: No Cache ---
# Simulate by being in 'training' mode or re-creating state each time
print("\n--- Benchmarking WITHOUT Cache ---")
model.train() # Force training mode to disable internal caching logic
non_cached_times = []
with torch.no_grad(): # Still disable gradient calculation for fair comparison
    for i in range(N_RUNS):
        # Generate one full sequence of dummy data
        # Note: For non-cached, we could process the whole sequence at once
        # But to compare step-by-step vs step-by-step, we loop here too.
        # IMPORTANT: In a truly non-cached step-by-step, the LSTM state would
        # be implicitly reset each time, and attention would only look at L=1.
        # Setting model.train() is the easiest way to disable our explicit cache.

        # Reset internal LSTM state implicitly by not providing one
        # And attention cache won't be used because model.training is True
        start_time = time.perf_counter()
        current_lstm_state = None # Simulate no state carry-over
        for step in range(BENCHMARK_SEQ_LENGTH):
            # Input shape: (batch_size=1, seq_len=1, input_dim)
            dummy_obs = torch.randn(1, 1, INPUT_DIM, device=DEVICE)
            _ = model(dummy_obs) # Forward pass, result ignored
            # Note: model.cached_states is NOT updated because model.training is True

        end_time = time.perf_counter()
        non_cached_times.append(end_time - start_time)
        print(f"Run {i+1}/{N_RUNS} (No Cache): {non_cached_times[-1]:.4f} seconds")

avg_non_cached_time = sum(non_cached_times) / N_RUNS
print(f"Average time WITHOUT cache: {avg_non_cached_time:.4f} seconds per sequence")
print(f"Average time per step:      {avg_non_cached_time / BENCHMARK_SEQ_LENGTH:.6f} seconds")


# --- Benchmark: With Cache ---
print("\n--- Benchmarking WITH Cache ---")
model.eval() # IMPORTANT: Set to eval mode to enable caching logic
cached_times = []
with torch.no_grad():
    for i in range(N_RUNS):
        # IMPORTANT: Reset the cache *before* starting the sequence!
        # Use batch_size=1 for our step-by-step inference benchmark.
        model.reset_cached_states(batch_size=1, clear_memory=True)

        start_time = time.perf_counter()
        for step in range(BENCHMARK_SEQ_LENGTH):
            # Input shape: (batch_size=1, seq_len=1, input_dim)
            dummy_obs = torch.randn(1, 1, INPUT_DIM, device=DEVICE)

            # Forward pass. Because model is in eval mode and cache was reset,
            # the internal logic should use and update the cache correctly.
            _ = model(dummy_obs)

        end_time = time.perf_counter()
        cached_times.append(end_time - start_time)
        print(f"Run {i+1}/{N_RUNS} (With Cache): {cached_times[-1]:.4f} seconds")

avg_cached_time = sum(cached_times) / N_RUNS
print(f"Average time WITH cache:    {avg_cached_time:.4f} seconds per sequence")
print(f"Average time per step:      {avg_cached_time / BENCHMARK_SEQ_LENGTH:.6f} seconds")

# --- Results ---
print("\n--- Comparison ---")
print(f"Average time WITHOUT cache: {avg_non_cached_time:.4f} s")
print(f"Average time WITH cache:    {avg_cached_time:.4f} s")

if avg_cached_time > 0:
    speedup_factor = avg_non_cached_time / avg_cached_time
    print(f"\nSpeedup Factor (No Cache / With Cache): {speedup_factor:.2f}x")
else:
    print("\nCould not calculate speedup factor (cached time was zero or negative).")

print("\nBenchmark finished.")