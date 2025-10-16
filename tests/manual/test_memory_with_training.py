#!/usr/bin/env python3
"""
Test memory preservation with actual network training to verify LSTM states are meaningful.
"""

import numpy as np
import torch
import logging
from rltrader.agents import CachedLSTMAttention, TimeSeriesEnvWrapper
from rltrader.envs import RebatedMarketEnv
from rltrader.configs import FINAL_OPTIMIZED_CONFIG

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def test_memory_with_actual_network():
    """Test memory preservation with actual network forward passes that generate meaningful states."""
    
    print("🧠 Testing Memory Preservation with Actual Network Activity")
    print("=" * 70)
    
    # Create environment
    config = FINAL_OPTIMIZED_CONFIG.copy()
    config['episode_length'] = 20  # Short episodes for testing
    
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
        episode_length=20
    )
    
    print(f"✓ Environment and feature extractor created")
    print(f"  - LSTM hidden dim: {feature_extractor.hidden_dim}")
    print(f"  - LSTM layers: {feature_extractor.lstm_layers}")
    print(f"  - Attention cache max length: {feature_extractor.attention.max_seq_length}")
    print()
    
    # Initialize with some random weights to make the network active
    for param in feature_extractor.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)
    
    # Set to eval mode to enable caching
    feature_extractor.eval()
    
    # Test memory preservation across episodes
    print("🔄 Testing Memory Preservation Across Episodes")
    print("-" * 50)
    
    # Start first episode
    obs1, info1 = wrapped_env.reset(options={'preserve_memory': True})
    
    # Run through several steps to build up meaningful memory
    observations = []
    for step in range(10):
        obs_tensor = torch.tensor(obs1, dtype=torch.float32).unsqueeze(0)
        observations.append(obs_tensor)
        
        # Forward pass
        features = feature_extractor(obs_tensor)
        
        # Take a step in environment
        action = env.action_space.sample()  # Random action
        obs1, reward, done, truncated, info = wrapped_env.step(action)
        
        if done or truncated:
            break
    
    # Get memory state after first episode
    lstm_state_ep1 = feature_extractor.cached_states.lstm_state
    cache_pos_ep1 = feature_extractor.attention.cache_position
    
    print(f"Episode 1 completed:")
    print(f"  - LSTM hidden state norm: {torch.norm(lstm_state_ep1[0]).item():.4f}")
    print(f"  - LSTM cell state norm: {torch.norm(lstm_state_ep1[1]).item():.4f}")
    print(f"  - Attention cache position: {cache_pos_ep1}")
    print(f"  - Attention cache norm: {torch.norm(feature_extractor.attention.key_cache).item():.4f}")
    print()
    
    # Reset for second episode WITH memory preservation
    obs2, info2 = wrapped_env.reset(options={'preserve_memory': True})
    
    # Get memory state after reset with preservation
    lstm_state_ep2_preserved = feature_extractor.cached_states.lstm_state
    cache_pos_ep2_preserved = feature_extractor.attention.cache_position
    
    print(f"Episode 2 start (preserve_memory=True):")
    print(f"  - LSTM hidden state norm: {torch.norm(lstm_state_ep2_preserved[0]).item():.4f}")
    print(f"  - LSTM cell state norm: {torch.norm(lstm_state_ep2_preserved[1]).item():.4f}")
    print(f"  - Attention cache position: {cache_pos_ep2_preserved}")
    print(f"  - Attention cache norm: {torch.norm(feature_extractor.attention.key_cache).item():.4f}")
    
    # Check if memory was preserved
    lstm_hidden_preserved = torch.allclose(lstm_state_ep1[0], lstm_state_ep2_preserved[0], atol=1e-6)
    lstm_cell_preserved = torch.allclose(lstm_state_ep1[1], lstm_state_ep2_preserved[1], atol=1e-6)
    cache_pos_preserved = (cache_pos_ep1 == cache_pos_ep2_preserved)
    
    print(f"  ✓ LSTM hidden state preserved: {lstm_hidden_preserved}")
    print(f"  ✓ LSTM cell state preserved: {lstm_cell_preserved}")
    print(f"  ✓ Attention cache position preserved: {cache_pos_preserved}")
    print()
    
    # Reset for third episode WITHOUT memory preservation
    obs3, info3 = wrapped_env.reset(options={'preserve_memory': False})
    
    # Get memory state after reset without preservation
    lstm_state_ep3_cleared = feature_extractor.cached_states.lstm_state
    cache_pos_ep3_cleared = feature_extractor.attention.cache_position
    
    print(f"Episode 3 start (preserve_memory=False):")
    print(f"  - LSTM hidden state norm: {torch.norm(lstm_state_ep3_cleared[0]).item():.4f}")
    print(f"  - LSTM cell state norm: {torch.norm(lstm_state_ep3_cleared[1]).item():.4f}")
    print(f"  - Attention cache position: {cache_pos_ep3_cleared}")
    print(f"  - Attention cache norm: {torch.norm(feature_extractor.attention.key_cache).item():.4f}")
    
    # Check if memory was cleared
    lstm_hidden_cleared = torch.allclose(lstm_state_ep3_cleared[0], torch.zeros_like(lstm_state_ep3_cleared[0]), atol=1e-6)
    lstm_cell_cleared = torch.allclose(lstm_state_ep3_cleared[1], torch.zeros_like(lstm_state_ep3_cleared[1]), atol=1e-6)
    cache_pos_cleared = (cache_pos_ep3_cleared == 0)
    
    print(f"  ✓ LSTM hidden state cleared: {lstm_hidden_cleared}")
    print(f"  ✓ LSTM cell state cleared: {lstm_cell_cleared}")
    print(f"  ✓ Attention cache position cleared: {cache_pos_cleared}")
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 70)
    
    memory_meaningful = (torch.norm(lstm_state_ep1[0]).item() > 0.01)  # Check if we got meaningful states
    preservation_works = lstm_hidden_preserved and lstm_cell_preserved and cache_pos_preserved
    clearing_works = lstm_hidden_cleared and lstm_cell_cleared and cache_pos_cleared
    
    print(f"✓ Network generated meaningful memory states: {memory_meaningful}")
    print(f"✓ Memory preservation works correctly: {preservation_works}")
    print(f"✓ Memory clearing works correctly: {clearing_works}")
    
    success = memory_meaningful and preservation_works and clearing_works
    
    if success:
        print("🎉 ALL TESTS PASSED! Memory preservation is working with actual network activity.")
        print("   The agent will now remember market patterns across episodes for sequential playback.")
    else:
        print("❌ Some tests failed. Check implementation.")
    
    wrapped_env.close()
    return success

if __name__ == "__main__":
    success = test_memory_with_actual_network()
    sys.exit(0 if success else 1)
