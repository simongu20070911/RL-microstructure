#!/usr/bin/env python3
"""
Confirm the volume tracking bug in baseline environment
"""

import sys
import os
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_unified import RebatedHFTEnv, get_unified_config

def test_volume_attributes():
    """Test if volume attributes exist in different environments."""
    print("🧪 TESTING VOLUME ATTRIBUTE EXISTENCE")
    
    # Create minimal test data
    data = []
    for i in range(5):
        row = {'datetime': i}
        for level in range(1, 11):
            row[f'bid{level}'] = 1799.99 - (level-1) * 0.01
            row[f'bidqty{level}'] = 100.0
            row[f'ask{level}'] = 1800.01 + (level-1) * 0.01  
            row[f'askqty{level}'] = 100.0
        data.append(row)
    
    df = pd.DataFrame(data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        # Test baseline
        print(f"\n--- BASELINE ENVIRONMENT ---")
        config = get_unified_config("baseline", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"has maker_volume: {hasattr(env, 'maker_volume')}")
        print(f"has taker_volume: {hasattr(env, 'taker_volume')}")
        print(f"is_rebated: {env.is_rebated}")
        
        if hasattr(env, 'maker_volume'):
            print(f"maker_volume value: {env.maker_volume}")
        if hasattr(env, 'taker_volume'):
            print(f"taker_volume value: {env.taker_volume}")
        
        env.close()
        
        # Test rebated
        print(f"\n--- REBATED ENVIRONMENT ---")
        config = get_unified_config("rebate_6bps", False)
        config["csv_path"] = temp_file.name
        config["episode_length"] = 5
        
        env = RebatedHFTEnv(config)
        obs, info = env.reset()
        
        print(f"has maker_volume: {hasattr(env, 'maker_volume')}")
        print(f"has taker_volume: {hasattr(env, 'taker_volume')}")
        print(f"is_rebated: {env.is_rebated}")
        
        if hasattr(env, 'maker_volume'):
            print(f"maker_volume value: {env.maker_volume}")
        if hasattr(env, 'taker_volume'):
            print(f"taker_volume value: {env.taker_volume}")
        
        env.close()
        
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

if __name__ == "__main__":
    test_volume_attributes()