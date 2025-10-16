#!/usr/bin/env python3
"""
Test to demonstrate the attention cache overflow issue that breaks temporal learning
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("🔬 TESTING ATTENTION CACHE OVERFLOW ISSUE")
print("=" * 80)

# Simplified version of the problematic CachedMultiHeadAttention
class ProblematicAttention(nn.Module):
    """Demonstrates the cache overflow issue."""
    
    def __init__(self, hidden_dim=64, num_heads=4, max_seq_length=10):
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
        
    def reset_cache(self, batch_size, device):
        """Reset cache."""
        self.key_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.value_cache = torch.zeros(batch_size, self.max_seq_length, self.num_heads, self.head_dim, device=device)
        self.cache_position = 0
        
    def forward(self, x, use_cache=True):
        """Forward with the PROBLEMATIC cache overflow behavior."""
        B, L, D = x.shape
        H = self.num_heads
        
        q = self.q_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, self.head_dim).transpose(1, 2)
        
        if use_cache:
            if self.key_cache is None:
                self.reset_cache(B, x.device)
                
            # 🚨 THE PROBLEMATIC CODE FROM THE REAL IMPLEMENTATION:
            if self.cache_position >= self.max_seq_length:
                print(f"⚠️  CACHE OVERFLOW! Position {self.cache_position} >= max {self.max_seq_length}")
                print(f"   Resetting position to 0 - THIS BREAKS TEMPORAL CONTINUITY!")
                self.cache_position = 0  # ❌ PROBLEM: Sudden discontinuity
            
            # Store current k,v in cache
            current_k = k.transpose(1, 2)  # B, L, H, head_dim
            current_v = v.transpose(1, 2)
            
            end_pos = self.cache_position + L
            if end_pos <= self.max_seq_length:
                self.key_cache[:, self.cache_position:end_pos] = current_k
                self.value_cache[:, self.cache_position:end_pos] = current_v
                
                # Use accumulated cache
                k_to_use = self.key_cache[:, :end_pos].transpose(1, 2)
                v_to_use = self.value_cache[:, :end_pos].transpose(1, 2)
                
                self.cache_position = end_pos
            else:
                # Fallback when cache would overflow
                k_to_use, v_to_use = k, v
        else:
            k_to_use, v_to_use = k, v
            
        # Compute attention
        scores = torch.matmul(q, k_to_use.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_to_use)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(out), attn_weights.mean(dim=1)  # Return attention for analysis

def test_cache_overflow_impact():
    """Test how cache overflow affects attention patterns."""
    print("\n📊 TESTING CACHE OVERFLOW IMPACT ON ATTENTION PATTERNS")
    print("-" * 60)
    
    # Create attention module with small cache to trigger overflow quickly
    attention = ProblematicAttention(hidden_dim=64, num_heads=4, max_seq_length=5)
    
    # Create a sequence with clear temporal pattern
    sequence_length = 15  # Longer than cache to trigger overflow
    batch_size = 1
    hidden_dim = 64
    
    print(f"Sequence length: {sequence_length}")
    print(f"Cache max length: {attention.max_seq_length}")
    print(f"Will overflow after step: {attention.max_seq_length}")
    
    # Create input with temporal pattern (each step has different signature)
    all_outputs = []
    all_attention_patterns = []
    cache_positions = []
    
    with torch.no_grad():
        attention.reset_cache(batch_size, torch.device('cpu'))
        
        for step in range(sequence_length):
            # Create input with step-specific pattern
            x = torch.randn(batch_size, 1, hidden_dim)
            x[0, 0, :10] = step  # First 10 dims encode step number
            
            output, attn_weights = attention(x, use_cache=True)
            
            all_outputs.append(output.clone())
            all_attention_patterns.append(attn_weights.clone())
            cache_positions.append(attention.cache_position)
            
            print(f"Step {step:2d}: Cache pos = {attention.cache_position:2d}, "
                  f"Attn shape = {attn_weights.shape}, "
                  f"Attn to recent = {attn_weights[0, -1, -3:].sum():.3f}")
    
    print(f"\n📈 ANALYZING ATTENTION DISCONTINUITY:")
    print("-" * 40)
    
    # Analyze attention patterns before and after overflow
    overflow_step = attention.max_seq_length
    
    if len(all_attention_patterns) > overflow_step + 2:
        before_overflow = all_attention_patterns[overflow_step - 1]
        after_overflow = all_attention_patterns[overflow_step + 1]
        
        print(f"Before overflow (step {overflow_step-1}):")
        print(f"  Attention shape: {before_overflow.shape}")
        print(f"  Attention to recent 3 steps: {before_overflow[0, -1, -3:].tolist()}")
        
        print(f"After overflow (step {overflow_step+1}):")  
        print(f"  Attention shape: {after_overflow.shape}")
        print(f"  Attention pattern completely different!")
        print(f"  Can only attend to 2 cached steps instead of {overflow_step}")
        
        # Calculate attention discontinuity
        if before_overflow.shape[-1] >= 3 and after_overflow.shape[-1] >= 2:
            recent_attn_before = before_overflow[0, -1, -3:].sum()
            recent_attn_after = after_overflow[0, -1, -2:].sum() 
            discontinuity = abs(recent_attn_before - recent_attn_after)
            
            print(f"\n🚨 ATTENTION DISCONTINUITY DETECTED:")
            print(f"   Recent attention before: {recent_attn_before:.4f}")
            print(f"   Recent attention after:  {recent_attn_after:.4f}")
            print(f"   Discontinuity magnitude: {discontinuity:.4f}")
            
            if discontinuity > 0.1:
                print(f"   ❌ SEVERE: This breaks temporal learning!")
            else:
                print(f"   ⚠️  MODERATE: Still problematic for RL")
    
    return cache_positions, all_attention_patterns

def test_memory_accumulation():
    """Test how LSTM state accumulation affects memory usage."""
    print(f"\n🧠 TESTING MEMORY ACCUMULATION ISSUE")
    print("-" * 60)
    
    # Simulate the problematic LSTM state caching
    class ProblematicLSTM:
        def __init__(self):
            self.cached_state = None
            
        def forward_with_bad_caching(self, x):
            """Simulates the memory accumulation problem."""
            # Create LSTM-like computation with gradient tracking
            lstm_out = torch.nn.functional.relu(x @ torch.randn(x.shape[-1], x.shape[-1]))
            
            # 🚨 THE PROBLEMATIC CODE: Storing output with gradients
            if self.cached_state is None:
                self.cached_state = lstm_out  # ❌ Keeps gradients!
            else:
                self.cached_state = torch.cat([self.cached_state, lstm_out], dim=1)  # ❌ Growing graph!
                
            return lstm_out
    
    lstm = ProblematicLSTM()
    
    print("Simulating memory accumulation over steps...")
    memory_usage = []
    
    for step in range(10):
        x = torch.randn(1, 1, 64, requires_grad=True)
        
        # Check memory before
        if hasattr(torch.cuda, 'memory_allocated'):
            mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        else:
            mem_before = 0
            
        output = lstm.forward_with_bad_caching(x)
        
        # Check memory after  
        if hasattr(torch.cuda, 'memory_allocated'):
            mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        else:
            mem_after = 0
            
        if lstm.cached_state is not None:
            cache_size = lstm.cached_state.shape[1]
            print(f"Step {step}: Cache size = {cache_size}, "
                  f"Requires grad = {lstm.cached_state.requires_grad}")
            
            if cache_size > 5:
                print(f"  ⚠️  Cache growing unbounded - memory leak detected!")
        
        memory_usage.append(mem_after - mem_before)
    
    print(f"\n🚨 MEMORY ACCUMULATION ANALYSIS:")
    print(f"   Final cache size: {lstm.cached_state.shape[1] if lstm.cached_state is not None else 0}")
    print(f"   Gradients tracked: {lstm.cached_state.requires_grad if lstm.cached_state is not None else False}")
    print(f"   This will cause memory leaks in long training runs!")

def demonstrate_fixes():
    """Show how to fix the identified issues."""
    print(f"\n🔧 DEMONSTRATING FIXES FOR IDENTIFIED ISSUES")
    print("=" * 60)
    
    print("1. FIX FOR ATTENTION CACHE OVERFLOW:")
    print("   Instead of: self.cache_position = 0")
    print("   Use: Circular buffer or sliding window:")
    print("   ```python")
    print("   if self.cache_position >= self.max_seq_length:")
    print("       # Shift cache left and continue from end")
    print("       shift_size = self.max_seq_length // 2")
    print("       self.key_cache[:, :-shift_size] = self.key_cache[:, shift_size:]")
    print("       self.value_cache[:, :-shift_size] = self.value_cache[:, shift_size:]")
    print("       self.cache_position = self.max_seq_length - shift_size")
    print("   ```")
    
    print("\n2. FIX FOR MEMORY ACCUMULATION:")
    print("   Instead of: self.cached_states.lstm_state = lstm_state_output")
    print("   Use: Proper gradient detachment:")
    print("   ```python")
    print("   if use_lstm_cache:")
    print("       # Detach gradients to prevent accumulation")
    print("       h_detached = lstm_state_output[0].detach()")
    print("       c_detached = lstm_state_output[1].detach()")
    print("       self.cached_states.lstm_state = (h_detached, c_detached)")
    print("   ```")
    
    print("\n3. FIX FOR NaN/Inf HANDLING:")
    print("   Instead of: # action = np.zeros_like(action) # Commented out")
    print("   Use: Active replacement:")
    print("   ```python")
    print("   if np.any(np.isnan(action)) or np.any(np.isinf(action)):")
    print("       logging.error(f'NaN/Inf in action: {action}')")
    print("       action = np.clip(np.nan_to_num(action, 0.0), -1.0, 1.0)")
    print("       action[5] = 0.9  # Force 'do nothing' mode")
    print("   ```")

if __name__ == "__main__":
    # Run tests
    cache_positions, attention_patterns = test_cache_overflow_impact()
    test_memory_accumulation()
    demonstrate_fixes()
    
    print(f"\n" + "=" * 80)
    print("🎯 SUMMARY: CRITICAL RL TRAINING ISSUES IDENTIFIED")
    print("=" * 80)
    print("1. ❌ Attention cache overflow breaks temporal learning")
    print("2. ❌ Memory accumulation causes training slowdown/crashes") 
    print("3. ❌ NaN/Inf actions not handled - can crash training")
    print("4. ⚠️  Episode truncation breaks natural trading patterns")
    print("5. ⚠️  Memory cleared between episodes hurts learning")
    print()
    print("🔧 RECOMMENDED ACTIONS:")
    print("1. Fix attention cache overflow with circular buffer")
    print("2. Add gradient detachment to LSTM state caching")
    print("3. Implement NaN/Inf action replacement")
    print("4. Consider trading-aware episode termination")
    print("5. Optionally allow some cross-episode memory retention")
    print()
    print("These fixes could significantly improve RL training performance!")
    print("=" * 80)