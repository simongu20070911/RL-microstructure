#!/usr/bin/env python3
"""
Comprehensive test suite for memory preservation implementation.
Tests edge cases, potential issues, and verifies correctness.
"""

import numpy as np
import torch
import logging
from rltrader.agents import CachedLSTMAttention, TimeSeriesEnvWrapper
from rltrader.envs import RebatedHFTEnv
from rltrader.configs import FINAL_OPTIMIZED_CONFIG

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def test_comprehensive_memory_preservation():
    """Comprehensive test suite for memory preservation."""
    
    print("🔬 COMPREHENSIVE MEMORY PRESERVATION TEST SUITE")
    print("=" * 80)
    
    issues_found = []
    
    # Create environment
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config['episode_length'] = 30
    
    env = RebatedHFTEnv(config)
    
    # Create feature extractor
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=100  # Smaller cache for testing overflow
    )
    
    # Initialize with random weights
    for param in feature_extractor.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    feature_extractor.eval()  # Enable caching
    
    # Wrap environment
    wrapped_env = TimeSeriesEnvWrapper(
        env=env,
        feature_extractor=feature_extractor,
        episode_length=30
    )
    
    print("✓ Test environment created")
    print()
    
    # Test 1: Basic Memory Preservation
    print("🧪 Test 1: Basic Memory Preservation")
    print("-" * 50)
    
    obs1, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    # Build up significant memory
    for i in range(15):
        obs_tensor = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0)
        features = feature_extractor(obs_tensor)
        action = env.action_space.sample()
        obs1, reward, done, truncated, info = wrapped_env.step(action)
        if done or truncated:
            break
    
    # Get memory after episode 1
    lstm_h1 = feature_extractor.cached_states.lstm_state[0].clone()
    lstm_c1 = feature_extractor.cached_states.lstm_state[1].clone()
    cache_pos1 = feature_extractor.attention.cache_position
    key_cache1 = feature_extractor.attention.key_cache.clone() if feature_extractor.attention.key_cache is not None else None
    
    print(f"  Episode 1 memory state:")
    print(f"    LSTM H norm: {torch.norm(lstm_h1).item():.4f}")
    print(f"    LSTM C norm: {torch.norm(lstm_c1).item():.4f}")
    print(f"    Cache position: {cache_pos1}")
    print(f"    Key cache norm: {torch.norm(key_cache1).item() if key_cache1 is not None else 'None'}")
    
    # Reset with memory preservation
    obs2, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    # Check preservation
    lstm_h2 = feature_extractor.cached_states.lstm_state[0]
    lstm_c2 = feature_extractor.cached_states.lstm_state[1]
    cache_pos2 = feature_extractor.attention.cache_position
    key_cache2 = feature_extractor.attention.key_cache
    
    h_preserved = torch.allclose(lstm_h1, lstm_h2, atol=1e-6)
    c_preserved = torch.allclose(lstm_c1, lstm_c2, atol=1e-6)
    pos_preserved = (cache_pos1 == cache_pos2)
    key_preserved = torch.allclose(key_cache1, key_cache2, atol=1e-6) if key_cache1 is not None and key_cache2 is not None else (key_cache1 is None and key_cache2 is None)
    
    print(f"  After reset with preserve_memory=True:")
    print(f"    LSTM H preserved: {h_preserved}")
    print(f"    LSTM C preserved: {c_preserved}")
    print(f"    Cache position preserved: {pos_preserved}")
    print(f"    Key cache preserved: {key_preserved}")
    
    if not (h_preserved and c_preserved and pos_preserved and key_preserved):
        issues_found.append("Basic memory preservation failed")
    
    print()
    
    # Test 2: Memory Clearing
    print("🧪 Test 2: Memory Clearing")
    print("-" * 50)
    
    obs3, _ = wrapped_env.reset(options={'preserve_memory': False})
    
    lstm_h3 = feature_extractor.cached_states.lstm_state[0]
    lstm_c3 = feature_extractor.cached_states.lstm_state[1]
    cache_pos3 = feature_extractor.attention.cache_position
    
    h_cleared = torch.allclose(lstm_h3, torch.zeros_like(lstm_h3), atol=1e-6)
    c_cleared = torch.allclose(lstm_c3, torch.zeros_like(lstm_c3), atol=1e-6)
    pos_cleared = (cache_pos3 == 0)
    
    print(f"  After reset with preserve_memory=False:")
    print(f"    LSTM H cleared: {h_cleared}")
    print(f"    LSTM C cleared: {c_cleared}")
    print(f"    Cache position cleared: {pos_cleared}")
    
    if not (h_cleared and c_cleared and pos_cleared):
        issues_found.append("Memory clearing failed")
    
    print()
    
    # Test 3: Cache Overflow Behavior
    print("🧪 Test 3: Cache Overflow Behavior")
    print("-" * 50)
    
    obs4, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    # Fill cache beyond capacity (max_attn_cache_len = 100)
    for i in range(120):  # Exceed cache capacity
        obs_tensor = torch.tensor(obs4, dtype=torch.float32).unsqueeze(0)
        features = feature_extractor(obs_tensor)
        action = env.action_space.sample()
        obs4, reward, done, truncated, info = wrapped_env.step(action)
        if done or truncated:
            obs4, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    cache_pos_overflow = feature_extractor.attention.cache_position
    
    print(f"  After {120} steps (exceeding cache capacity of 100):")
    print(f"    Final cache position: {cache_pos_overflow}")
    print(f"    Expected: <= 100 due to circular buffering")
    
    if cache_pos_overflow > 100:
        issues_found.append(f"Cache overflow not handled correctly: position {cache_pos_overflow} > 100")
    
    # Test preservation after overflow
    lstm_overflow = feature_extractor.cached_states.lstm_state[0].clone()
    cache_pos_before_reset = cache_pos_overflow
    
    obs5, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    lstm_after_reset = feature_extractor.cached_states.lstm_state[0]
    cache_pos_after_reset = feature_extractor.attention.cache_position
    
    overflow_h_preserved = torch.allclose(lstm_overflow, lstm_after_reset, atol=1e-6)
    overflow_pos_preserved = (cache_pos_before_reset == cache_pos_after_reset)
    
    print(f"  Memory preservation after overflow:")
    print(f"    LSTM H preserved: {overflow_h_preserved}")
    print(f"    Cache position preserved: {overflow_pos_preserved}")
    
    if not overflow_h_preserved:
        issues_found.append("LSTM state not preserved after cache overflow")
    
    print()
    
    # Test 4: Gradient Detachment
    print("🧪 Test 4: Gradient Detachment")
    print("-" * 50)
    
    feature_extractor.train()  # Switch to training mode
    
    obs6, _ = wrapped_env.reset(options={'preserve_memory': True})
    obs_tensor = torch.tensor(obs6, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    
    # Forward pass in training mode (should not use cache)
    features = feature_extractor(obs_tensor)
    loss = features.sum()
    loss.backward()
    
    # Check if cached states have gradients (they shouldn't after detachment)
    if feature_extractor.cached_states and feature_extractor.cached_states.lstm_state:
        h_has_grad = feature_extractor.cached_states.lstm_state[0].requires_grad
        c_has_grad = feature_extractor.cached_states.lstm_state[1].requires_grad
        
        print(f"  LSTM hidden state requires_grad: {h_has_grad}")
        print(f"  LSTM cell state requires_grad: {c_has_grad}")
        
        if h_has_grad or c_has_grad:
            issues_found.append("Cached LSTM states still have gradients (detachment failed)")
    
    feature_extractor.eval()  # Back to eval mode
    print()
    
    # Test 5: Batch Size Mismatch
    print("🧪 Test 5: Batch Size Mismatch Handling")
    print("-" * 50)
    
    # Reset with batch size 1
    obs7, _ = wrapped_env.reset(options={'preserve_memory': True})
    obs_tensor = torch.tensor(obs7, dtype=torch.float32).unsqueeze(0)
    features = feature_extractor(obs_tensor)
    
    original_batch_size = feature_extractor.cached_states.lstm_state[0].shape[1]
    print(f"  Original batch size: {original_batch_size}")
    
    # Manually reset with different batch size
    try:
        feature_extractor.reset_cached_states(batch_size=2, clear_memory=False)
        new_batch_size = feature_extractor.cached_states.lstm_state[0].shape[1]
        print(f"  After reset with batch_size=2: {new_batch_size}")
        
        if new_batch_size != 2:
            issues_found.append(f"Batch size mismatch not handled: expected 2, got {new_batch_size}")
    except Exception as e:
        print(f"  Exception during batch size mismatch test: {e}")
        issues_found.append(f"Exception during batch size handling: {e}")
    
    print()
    
    # Test 6: Default Behavior
    print("🧪 Test 6: Default Behavior Verification")
    print("-" * 50)
    
    # Test default environment behavior
    obs8, _ = wrapped_env.reset()  # No options provided
    default_preserve = getattr(wrapped_env.env.unwrapped, 'preserve_memory', False)
    
    print(f"  Default preserve_memory setting: {default_preserve}")
    
    # Test default wrapper behavior
    obs9, _ = wrapped_env.reset(options=None)
    wrapper_default_preserve = getattr(wrapped_env.env.unwrapped, 'preserve_memory', False)
    
    print(f"  Wrapper default behavior: {wrapper_default_preserve}")
    
    if not default_preserve:
        issues_found.append("Default behavior should preserve memory for sequential playback")
    
    print()
    
    # Test 7: Memory State Consistency
    print("🧪 Test 7: Memory State Consistency")
    print("-" * 50)
    
    obs10, _ = wrapped_env.reset(options={'preserve_memory': True})
    
    # Check if cached_states object is consistent
    cached_states = feature_extractor.cached_states
    attention_cache = feature_extractor.attention
    
    if cached_states:
        cached_key = cached_states.key_cache
        cached_value = cached_states.value_cache
        cached_pos = cached_states.cache_position
        
        actual_key = attention_cache.key_cache
        actual_value = attention_cache.value_cache
        actual_pos = attention_cache.cache_position
        
        key_consistent = torch.equal(cached_key, actual_key) if cached_key is not None and actual_key is not None else (cached_key is None and actual_key is None)
        value_consistent = torch.equal(cached_value, actual_value) if cached_value is not None and actual_value is not None else (cached_value is None and actual_value is None)
        pos_consistent = (cached_pos == actual_pos)
        
        print(f"  Key cache consistency: {key_consistent}")
        print(f"  Value cache consistency: {value_consistent}")
        print(f"  Position consistency: {pos_consistent}")
        
        if not (key_consistent and value_consistent and pos_consistent):
            issues_found.append("Cached states object inconsistent with actual attention cache")
    
    print()
    
    # Summary
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print()
        print("🔧 RECOMMENDATION: Fix these issues before using memory preservation in production.")
    else:
        print("✅ ALL TESTS PASSED!")
        print("🎉 Memory preservation implementation is robust and ready for use.")
    
    wrapped_env.close()
    return len(issues_found) == 0

if __name__ == "__main__":
    success = test_comprehensive_memory_preservation()
    sys.exit(0 if success else 1)
