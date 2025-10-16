#!/usr/bin/env python3
"""
Test script to verify that agent memory (LSTM states and attention cache) 
is preserved across episodes for sequential market playback.
"""

import numpy as np
import torch
import logging
from rltrader.agents import CachedLSTMAttention, TimeSeriesEnvWrapper
from rltrader.envs import RebatedMarketEnv
from rltrader.configs import FINAL_OPTIMIZED_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_memory_preservation():
    """Test that agent memory is preserved across episodes when requested."""
    
    print("🧠 Testing Agent Memory Preservation Across Episodes")
    print("=" * 60)
    
    # Create environment with memory preservation enabled
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config['episode_length'] = 50  # Short episodes for testing
    
    env = RebatedMarketEnv(config)
    
    # Create feature extractor
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=200
    )
    
    # Wrap environment
    wrapped_env = TimeSeriesEnvWrapper(
        env=env,
        feature_extractor=feature_extractor,
        episode_length=50
    )
    
    print(f"✓ Environment and feature extractor created")
    print(f"  - LSTM hidden dim: {feature_extractor.hidden_dim}")
    print(f"  - LSTM layers: {feature_extractor.lstm_layers}")
    print(f"  - Attention cache max length: {feature_extractor.attention.max_seq_length}")
    print()
    
    # Test 1: Memory preservation enabled (default)
    print("🔄 Test 1: Memory Preservation ENABLED")
    print("-" * 40)
    
    obs1, info1 = wrapped_env.reset(options={'preserve_memory': True})
    
    # Process some observations to build up memory
    dummy_obs = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0)
    
    # Get initial memory states
    initial_lstm_state = feature_extractor.cached_states.lstm_state if feature_extractor.cached_states else None
    initial_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  Initial LSTM state shape: {initial_lstm_state[0].shape if initial_lstm_state else 'None'}")
    print(f"  Initial attention cache position: {initial_cache_pos}")
    
    # Forward pass to populate memory
    features1 = feature_extractor(dummy_obs)
    
    # Get memory states after forward pass
    after_forward_lstm_state = feature_extractor.cached_states.lstm_state
    after_forward_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  After forward pass:")
    print(f"    LSTM state norm: {torch.norm(after_forward_lstm_state[0]).item():.4f}")
    print(f"    Attention cache position: {after_forward_cache_pos}")
    
    # Reset episode with memory preservation
    obs2, info2 = wrapped_env.reset(options={'preserve_memory': True})
    
    # Check if memory was preserved
    preserved_lstm_state = feature_extractor.cached_states.lstm_state
    preserved_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  After reset with preserve_memory=True:")
    print(f"    LSTM state norm: {torch.norm(preserved_lstm_state[0]).item():.4f}")
    print(f"    Attention cache position: {preserved_cache_pos}")
    
    # Check if states are similar (detached but preserved)
    lstm_preserved = torch.allclose(after_forward_lstm_state[0], preserved_lstm_state[0], atol=1e-6)
    cache_preserved = (after_forward_cache_pos == preserved_cache_pos)
    
    print(f"  ✓ LSTM state preserved: {lstm_preserved}")
    print(f"  ✓ Attention cache position preserved: {cache_preserved}")
    print()
    
    # Test 2: Memory preservation disabled
    print("🔄 Test 2: Memory Preservation DISABLED")
    print("-" * 40)
    
    # Reset with memory clearing
    obs3, info3 = wrapped_env.reset(options={'preserve_memory': False})
    
    # Check if memory was cleared
    cleared_lstm_state = feature_extractor.cached_states.lstm_state
    cleared_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  After reset with preserve_memory=False:")
    print(f"    LSTM state norm: {torch.norm(cleared_lstm_state[0]).item():.4f}")
    print(f"    Attention cache position: {cleared_cache_pos}")
    
    # Check if states were cleared
    lstm_cleared = torch.allclose(cleared_lstm_state[0], torch.zeros_like(cleared_lstm_state[0]), atol=1e-6)
    cache_cleared = (cleared_cache_pos == 0)
    
    print(f"  ✓ LSTM state cleared: {lstm_cleared}")
    print(f"  ✓ Attention cache position cleared: {cache_cleared}")
    print()
    
    # Test 3: Default behavior (should preserve memory)
    print("🔄 Test 3: Default Behavior (Should Preserve Memory)")
    print("-" * 40)
    
    # Forward pass to populate memory again
    features3 = feature_extractor(dummy_obs)
    default_lstm_state = feature_extractor.cached_states.lstm_state
    default_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  Before default reset:")
    print(f"    LSTM state norm: {torch.norm(default_lstm_state[0]).item():.4f}")
    print(f"    Attention cache position: {default_cache_pos}")
    
    # Reset with default options (should preserve memory)
    obs4, info4 = wrapped_env.reset()
    
    final_lstm_state = feature_extractor.cached_states.lstm_state
    final_cache_pos = feature_extractor.attention.cache_position
    
    print(f"  After default reset:")
    print(f"    LSTM state norm: {torch.norm(final_lstm_state[0]).item():.4f}")
    print(f"    Attention cache position: {final_cache_pos}")
    
    # Check if default preserves memory
    default_preserves = torch.allclose(default_lstm_state[0], final_lstm_state[0], atol=1e-6)
    default_cache_preserves = (default_cache_pos == final_cache_pos)
    
    print(f"  ✓ Default preserves LSTM state: {default_preserves}")
    print(f"  ✓ Default preserves cache position: {default_cache_preserves}")
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✓ Memory preservation when enabled: {lstm_preserved and cache_preserved}")
    print(f"✓ Memory clearing when disabled: {lstm_cleared and cache_cleared}")
    print(f"✓ Default behavior preserves memory: {default_preserves and default_cache_preserves}")
    
    success = all([
        lstm_preserved and cache_preserved,
        lstm_cleared and cache_cleared,
        default_preserves and default_cache_preserves
    ])
    
    if success:
        print("🎉 ALL TESTS PASSED! Memory preservation is working correctly.")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    wrapped_env.close()
    return success

if __name__ == "__main__":
    success = test_memory_preservation()
    sys.exit(0 if success else 1)
