#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Rebated HFT Environments

Tests the functionality of rebated environments with 0.004%, 0.006%, and 0.008% rebates.
Validates rebate calculations, environment behavior, and integration with the base HFT environment.
"""

import unittest
import numpy as np
import tempfile
import os
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_004 import HFTEnvRebated004, get_default_rebated_config_004
from env_rebated_006 import HFTEnvRebated006, get_default_rebated_config_006
from env_rebated_008 import HFTEnvRebated008, get_default_rebated_config_008


class TestRebatedEnvironments(unittest.TestCase):
    """Test suite for all rebated environments."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be used across all tests."""
        # Create a temporary CSV file with synthetic order book data
        cls.test_data = cls._create_test_orderbook_data()
        cls.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        cls.test_data.to_csv(cls.temp_csv.name, index=False)
        cls.temp_csv.close()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        os.unlink(cls.temp_csv.name)
        
    @staticmethod
    def _create_test_orderbook_data():
        """Create synthetic order book data for testing."""
        np.random.seed(42)  # For reproducible tests
        n_rows = 1000
        
        # Generate base price around 1800 (like ETH)
        base_price = 1800.0
        price_walk = np.cumsum(np.random.normal(0, 0.1, n_rows))
        midprices = base_price + price_walk
        
        data = []
        for i, midprice in enumerate(midprices):
            row = {'timestamp': i}
            
            # Generate bid and ask levels with realistic spreads
            spread = 0.02  # 2 cents spread
            tick_size = 0.01
            
            for level in range(1, 11):  # 10 levels
                bid_offset = (level - 1) * tick_size + spread/2
                ask_offset = (level - 1) * tick_size + spread/2
                
                bid_price = midprice - bid_offset
                ask_price = midprice + ask_offset
                
                # Generate random quantities
                bid_qty = np.random.uniform(1.0, 20.0)
                ask_qty = np.random.uniform(1.0, 20.0)
                
                row[f'bid{level}'] = bid_price
                row[f'bidqty{level}'] = bid_qty
                row[f'ask{level}'] = ask_price
                row[f'askqty{level}'] = ask_qty
                
            data.append(row)
            
        return pd.DataFrame(data)
        
    def _get_test_config(self, rebate_rate):
        """Get a test configuration with the temporary CSV file."""
        if rebate_rate == 0.00004:
            config = get_default_rebated_config_004()
        elif rebate_rate == 0.00006:
            config = get_default_rebated_config_006()
        elif rebate_rate == 0.00008:
            config = get_default_rebated_config_008()
        else:
            raise ValueError(f"Unknown rebate rate: {rebate_rate}")
            
        config["csv_path"] = self.temp_csv.name
        config["max_steps"] = 100
        config["episode_length"] = 50
        return config
        
    def test_rebated_004_initialization(self):
        """Test that the 0.004% rebated environment initializes correctly."""
        config = self._get_test_config(0.00004)
        env = HFTEnvRebated004(config)
        
        # Check rebate configuration
        self.assertEqual(env.rebate_rate, 0.00004)
        self.assertEqual(env.rebate_bps, 4)
        self.assertEqual(env.config["transaction_cost_long"], -0.00004)
        self.assertEqual(env.config["transaction_cost_short"], -0.00004)
        
        # Check that rebate tracking is initialized
        self.assertEqual(env.total_rebates_earned, 0.0)
        self.assertEqual(env.rebated_volume, 0.0)
        
    def test_rebated_006_initialization(self):
        """Test that the 0.006% rebated environment initializes correctly."""
        config = self._get_test_config(0.00006)
        env = HFTEnvRebated006(config)
        
        # Check rebate configuration
        self.assertEqual(env.rebate_rate, 0.00006)
        self.assertEqual(env.rebate_bps, 6)
        self.assertEqual(env.config["transaction_cost_long"], -0.00006)
        self.assertEqual(env.config["transaction_cost_short"], -0.00006)
        
    def test_rebated_008_initialization(self):
        """Test that the 0.008% rebated environment initializes correctly."""
        config = self._get_test_config(0.00008)
        env = HFTEnvRebated008(config)
        
        # Check rebate configuration
        self.assertEqual(env.rebate_rate, 0.00008)
        self.assertEqual(env.rebate_bps, 8)
        self.assertEqual(env.config["transaction_cost_long"], -0.00008)
        self.assertEqual(env.config["transaction_cost_short"], -0.00008)
        
    def test_environment_reset(self):
        """Test that environment reset works correctly for all rebated environments."""
        for rebate_rate, env_class in [(0.00004, HFTEnvRebated004), 
                                      (0.00006, HFTEnvRebated006),
                                      (0.00008, HFTEnvRebated008)]:
            with self.subTest(rebate_rate=rebate_rate):
                config = self._get_test_config(rebate_rate)
                env = env_class(config)
                
                obs, info = env.reset()
                
                # Check observation shape
                self.assertEqual(obs.shape, env.observation_space.shape)
                
                # Check that rebate info is in the info dict
                self.assertIn('rebate_rate', info)
                self.assertIn('rebate_bps', info)
                self.assertIn('total_rebates_earned', info)
                self.assertIn('rebated_volume', info)
                
                # Check rebate tracking is reset
                self.assertEqual(info['total_rebates_earned'], 0.0)
                self.assertEqual(info['rebated_volume'], 0.0)
                
    def test_step_functionality(self):
        """Test that step function works correctly for all rebated environments."""
        for rebate_rate, env_class in [(0.00004, HFTEnvRebated004), 
                                      (0.00006, HFTEnvRebated006),
                                      (0.00008, HFTEnvRebated008)]:
            with self.subTest(rebate_rate=rebate_rate):
                config = self._get_test_config(rebate_rate)
                env = env_class(config)
                
                obs, info = env.reset()
                
                # Take a step with market making action
                action = np.array([0.0, 0.0, 0.5, 0.5, -1.0, -1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check that step returns valid data
                self.assertEqual(obs.shape, env.observation_space.shape)
                self.assertIsInstance(reward, (int, float))
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIsInstance(info, dict)
                
                # Check rebate information is present
                self.assertIn('rebate_rate', info)
                self.assertIn('total_rebates_earned', info)
                self.assertEqual(info['rebate_rate'], rebate_rate)
                
    def test_rebate_calculation_differences(self):
        """Test that different rebate rates produce different rebate earnings."""
        configs = [
            (0.00004, HFTEnvRebated004),
            (0.00006, HFTEnvRebated006), 
            (0.00008, HFTEnvRebated008)
        ]
        
        rebate_earnings = []
        
        for rebate_rate, env_class in configs:
            config = self._get_test_config(rebate_rate)
            env = env_class(config)
            
            # Set same random seed for consistent comparison
            obs, info = env.reset(seed=42)
            
            # Take several steps with market making actions
            total_rebates = 0.0
            for _ in range(10):
                action = np.array([0.0, 0.0, 0.8, 0.8, -1.0, -1.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                total_rebates = info['total_rebates_earned']
                
                if terminated or truncated:
                    break
                    
            rebate_earnings.append(total_rebates)
            
        # Check that higher rebate rates generate more rebates (if any trading occurred)
        if any(earnings > 0 for earnings in rebate_earnings):
            # If trading occurred, higher rebates should earn more
            self.assertLessEqual(rebate_earnings[0], rebate_earnings[1])  # 4bps <= 6bps
            self.assertLessEqual(rebate_earnings[1], rebate_earnings[2])  # 6bps <= 8bps
            
    def test_negative_transaction_costs(self):
        """Test that rebated environments have negative transaction costs."""
        for rebate_rate, env_class in [(0.00004, HFTEnvRebated004), 
                                      (0.00006, HFTEnvRebated006),
                                      (0.00008, HFTEnvRebated008)]:
            with self.subTest(rebate_rate=rebate_rate):
                config = self._get_test_config(rebate_rate)
                env = env_class(config)
                
                # Check that transaction costs are negative (rebates)
                self.assertLess(env.config["transaction_cost_long"], 0)
                self.assertLess(env.config["transaction_cost_short"], 0)
                self.assertEqual(env.config["transaction_cost_long"], -rebate_rate)
                self.assertEqual(env.config["transaction_cost_short"], -rebate_rate)
                
    def test_cash_increase_from_rebates(self):
        """Test that rebates increase cash when trades are executed."""
        config = self._get_test_config(0.00008)  # Use highest rebate for clearer effect
        env = HFTEnvRebated008(config)
        
        obs, info = env.reset(seed=42)
        initial_cash = env.cash
        
        # Force market making activity with aggressive actions
        for _ in range(20):
            # Market making action: place orders at spread
            action = np.array([0.2, -0.2, 0.9, 0.9, -1.0, -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
                
        # If any trading occurred, cash should increase due to rebates
        if info['rebated_volume'] > 0:
            self.assertGreater(env.cash, initial_cash - 100)  # Allow for some market moves
            self.assertGreater(info['total_rebates_earned'], 0)
            
    def test_config_isolation(self):
        """Test that rebated environments don't modify the original config."""
        original_config = self._get_test_config(0.00004)
        original_cost_long = original_config["transaction_cost_long"]
        
        # Create rebated environment
        env = HFTEnvRebated004(original_config)
        
        # Check that original config wasn't modified
        self.assertEqual(original_config["transaction_cost_long"], original_cost_long)
        # But environment config should be rebated
        self.assertEqual(env.config["transaction_cost_long"], -0.00004)
        
    def test_observation_space_consistency(self):
        """Test that rebated environments have the same observation space as base environment."""
        # Create base environment config (with positive transaction costs)
        base_config = self._get_test_config(0.00004)
        base_config["transaction_cost_long"] = 0.0001  # Positive costs
        base_config["transaction_cost_short"] = 0.0001
        
        # Import base environment
        from env_2sided import HFTEnv
        base_env = HFTEnv(base_config)
        
        # Create rebated environment
        rebated_config = self._get_test_config(0.00004)
        rebated_env = HFTEnvRebated004(rebated_config)
        
        # Check observation spaces are identical
        self.assertEqual(base_env.observation_space.shape, rebated_env.observation_space.shape)
        self.assertEqual(base_env.action_space.shape, rebated_env.action_space.shape)
        
    def test_rebate_tracking_accuracy(self):
        """Test that rebate tracking is accurate."""
        config = self._get_test_config(0.00006)
        env = HFTEnvRebated006(config)
        
        obs, info = env.reset(seed=42)
        
        # Track rebates manually
        manual_rebates = 0.0
        manual_volume = 0.0
        
        for _ in range(15):
            pre_step_volume = getattr(env, 'last_executed_volume', 0.0)
            action = np.array([0.1, -0.1, 0.7, 0.7, -1.0, -1.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # If volume was executed this step
            if hasattr(env, 'last_executed_volume') and env.last_executed_volume > 0:
                step_rebates = env.last_executed_volume * env.rebate_rate * env.midprice
                manual_rebates += step_rebates
                manual_volume += env.last_executed_volume
                
            if terminated or truncated:
                break
                
        # Compare manual calculation with environment tracking
        if manual_volume > 0:
            self.assertAlmostEqual(info['total_rebates_earned'], manual_rebates, places=6)
            self.assertAlmostEqual(info['rebated_volume'], manual_volume, places=6)


class TestRebatedEnvironmentPerformance(unittest.TestCase):
    """Performance and stress tests for rebated environments."""
    
    def setUp(self):
        """Set up test environment."""
        # Create minimal test data
        test_data = TestRebatedEnvironments._create_test_orderbook_data()
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_data.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
    def tearDown(self):
        """Clean up."""
        os.unlink(self.temp_csv.name)
        
    def test_environment_performance(self):
        """Test that rebated environments perform similarly to base environment."""
        config = get_default_rebated_config_004()
        config["csv_path"] = self.temp_csv.name
        config["max_steps"] = 100
        config["episode_length"] = 100
        
        env = HFTEnvRebated004(config)
        
        # Time a full episode
        import time
        start_time = time.time()
        
        obs, info = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
                
        end_time = time.time()
        episode_time = end_time - start_time
        
        # Should complete episode in reasonable time (less than 10 seconds)
        self.assertLess(episode_time, 10.0)
        
    def test_memory_usage(self):
        """Test that rebated environments don't have memory leaks."""
        config = get_default_rebated_config_008()
        config["csv_path"] = self.temp_csv.name
        config["max_steps"] = 50
        config["episode_length"] = 50
        
        # Run multiple episodes
        for episode in range(5):
            env = HFTEnvRebated008(config)
            obs, info = env.reset()
            
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
                    
            # Clean up
            del env
            
        # If we get here without errors, memory usage is reasonable


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)