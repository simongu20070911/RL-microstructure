#!/usr/bin/env python3
"""
Test edge cases and potential issues in memory preservation implementation.
"""

import sys
sys.path.append('/home/gaen/Documents/RL')

import numpy as np
import torch
import logging
from agents.agent_2sided import CachedLSTMAttention, TimeSeriesEnvWrapper
from envs.env_rebated.env_rebated_unified import RebatedHFTEnv
from envs.env_rebated.final_optimized_config import FINAL_OPTIMIZED_CONFIG

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def test_edge_cases():
    """Test edge cases in memory preservation."""
    
    print("⚠️  MEMORY PRESERVATION EDGE CASE TESTING")
    print("=" * 60)
    
    issues_found = []
    
    # Test 1: Attention Cache Key/Value None Handling
    print("🧪 Test 1: Attention Cache None Handling")
    print("-" * 40)
    
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config['episode_length'] = 10
    
    env = RebatedHFTEnv(config)
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=50
    )
    
    # Manually set attention cache to None to test edge case
    feature_extractor.attention.key_cache = None
    feature_extractor.attention.value_cache = None
    feature_extractor.eval()
    
    wrapped_env = TimeSeriesEnvWrapper(env, feature_extractor, episode_length=10)
    
    try:
        obs, _ = wrapped_env.reset(options={'preserve_memory': True})
        print("  ✓ Handled None attention cache gracefully")
    except Exception as e:
        issues_found.append(f"Failed to handle None attention cache: {e}")
        print(f"  ❌ Exception: {e}")
    
    wrapped_env.close()
    print()
    
    # Test 2: Memory Leaks During Long Sequences
    print("🧪 Test 2: Memory Leak Detection")
    print("-" * 40)
    
    env = RebatedHFTEnv(config)
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=50
    )
    
    # Initialize with random weights
    for param in feature_extractor.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    
    feature_extractor.eval()
    wrapped_env = TimeSeriesEnvWrapper(env, feature_extractor, episode_length=10)
    
    # Track memory usage
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Run many episodes with memory preservation
    for ep in range(100):
        obs, _ = wrapped_env.reset(options={'preserve_memory': True})
        for step in range(5):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            features = feature_extractor(obs_tensor)
            action = env.action_space.sample()
            obs, reward, done, truncated, info = wrapped_env.step(action)
            if done or truncated:
                break
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_increase = final_memory - initial_memory
    
    print(f"  Memory increase over 100 episodes: {memory_increase} bytes")
    
    # Check if memory increase is reasonable (should be minimal for preserved states)
    if memory_increase > 100 * 1024 * 1024:  # 100MB threshold
        issues_found.append(f"Potential memory leak: {memory_increase} bytes over 100 episodes")
    else:
        print("  ✓ No significant memory leak detected")
    
    wrapped_env.close()
    print()
    
    # Test 3: Training/Eval Mode Switching
    print("🧪 Test 3: Training/Eval Mode Switching")
    print("-" * 40)
    
    env = RebatedHFTEnv(config)
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=50
    )
    
    # Initialize with random weights
    for param in feature_extractor.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    
    wrapped_env = TimeSeriesEnvWrapper(env, feature_extractor, episode_length=10)
    
    # Test eval mode (should use cache)
    feature_extractor.eval()
    obs, _ = wrapped_env.reset(options={'preserve_memory': True})
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    features_eval = feature_extractor(obs_tensor)
    
    # Get cache state
    cache_pos_eval = feature_extractor.attention.cache_position
    
    # Switch to training mode (should not use cache)
    feature_extractor.train()
    features_train = feature_extractor(obs_tensor)
    cache_pos_train = feature_extractor.attention.cache_position
    
    print(f"  Eval mode cache position: {cache_pos_eval}")
    print(f"  Train mode cache position: {cache_pos_train}")
    
    # In training mode, cache position should not advance
    if cache_pos_train != cache_pos_eval:
        issues_found.append(f"Cache advanced in training mode: {cache_pos_eval} -> {cache_pos_train}")
    else:
        print("  ✓ Cache correctly preserved in training mode")
    
    wrapped_env.close()
    print()
    
    # Test 4: Concurrent Environment Access
    print("🧪 Test 4: Multiple Environment Instances")
    print("-" * 40)
    
    try:
        # Create multiple environments with shared feature extractor
        env1 = RebatedHFTEnv(config)
        env2 = RebatedHFTEnv(config)
        
        feature_extractor = CachedLSTMAttention(
            observation_space=env1.observation_space,
            features_dim=128,
            hidden_dim=256,
            lstm_layers=2,
            num_heads=8,
            max_attn_cache_len=50
        )
        
        # Initialize with random weights
        for param in feature_extractor.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
        
        feature_extractor.eval()
        
        wrapped_env1 = TimeSeriesEnvWrapper(env1, feature_extractor, episode_length=10)
        wrapped_env2 = TimeSeriesEnvWrapper(env2, feature_extractor, episode_length=10)
        
        # Reset both environments
        obs1, _ = wrapped_env1.reset(options={'preserve_memory': True})
        obs2, _ = wrapped_env2.reset(options={'preserve_memory': True})
        
        print("  ✓ Multiple environments created successfully")
        
        wrapped_env1.close()
        wrapped_env2.close()
        
    except Exception as e:
        issues_found.append(f"Failed with multiple environments: {e}")
        print(f"  ❌ Exception: {e}")
    
    print()
    
    # Test 5: Invalid Options Handling
    print("🧪 Test 5: Invalid Options Handling")
    print("-" * 40)
    
    env = RebatedHFTEnv(config)
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=50
    )
    
    feature_extractor.eval()
    wrapped_env = TimeSeriesEnvWrapper(env, feature_extractor, episode_length=10)
    
    # Test invalid option types - should now be handled gracefully
    try:
        obs, _ = wrapped_env.reset(options={'preserve_memory': 'invalid'})
        # Check if it was coerced to default value
        actual_preserve = getattr(wrapped_env.env.unwrapped, 'preserve_memory', False)
        if actual_preserve == True:  # Should default to True
            print("  ✓ Invalid preserve_memory value handled gracefully (defaulted to True)")
        else:
            issues_found.append("Invalid preserve_memory value not handled correctly")
    except Exception as e:
        print(f"  ✓ Correctly rejected invalid preserve_memory: {e}")
    
    # Test None options
    try:
        obs, _ = wrapped_env.reset(options=None)
        print("  ✓ Handled None options gracefully")
    except Exception as e:
        issues_found.append(f"Failed to handle None options: {e}")
        print(f"  ❌ Exception with None options: {e}")
    
    wrapped_env.close()
    print()
    
    # Test 6: Device Mismatch
    print("🧪 Test 6: Device Consistency")
    print("-" * 40)
    
    env = RebatedHFTEnv(config)
    feature_extractor = CachedLSTMAttention(
        observation_space=env.observation_space,
        features_dim=128,
        hidden_dim=256,
        lstm_layers=2,
        num_heads=8,
        max_attn_cache_len=50
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_extractor.to(device)
    feature_extractor.eval()
    
    wrapped_env = TimeSeriesEnvWrapper(env, feature_extractor, episode_length=10)
    
    try:
        obs, _ = wrapped_env.reset(options={'preserve_memory': True})
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        features = feature_extractor(obs_tensor)
        
        # Check if cached states are on the same device
        if feature_extractor.cached_states and feature_extractor.cached_states.lstm_state:
            cached_device = feature_extractor.cached_states.lstm_state[0].device
            # Compare device types rather than exact string match
            if cached_device.type != device.type:
                issues_found.append(f"Device mismatch: feature extractor on {device}, cache on {cached_device}")
            else:
                print(f"  ✓ All tensors correctly on {device.type} device")
        
    except Exception as e:
        issues_found.append(f"Device consistency test failed: {e}")
        print(f"  ❌ Exception: {e}")
    
    wrapped_env.close()
    print()
    
    # Summary
    print("📊 EDGE CASE TEST RESULTS")
    print("=" * 60)
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        print()
        print("🔧 RECOMMENDATION: Address these edge cases for production use.")
        return False
    else:
        print("✅ ALL EDGE CASE TESTS PASSED!")
        print("🎉 Implementation is robust against edge cases.")
        return True

if __name__ == "__main__":
    success = test_edge_cases()
    sys.exit(0 if success else 1)