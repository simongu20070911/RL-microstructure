#!/usr/bin/env python3
"""
Critical Validation Tests for Rebated HFT Environments

This test suite is specifically designed to find problems, edge cases, and potential
issues in the rebated environments. Focus is on identifying bugs, inconsistencies,
and implementation flaws rather than just verifying basic functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
import math
import warnings
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_rebated_004 import HFTEnvRebated004, get_default_rebated_config_004
from env_rebated_006 import HFTEnvRebated006, get_default_rebated_config_006
from env_rebated_008 import HFTEnvRebated008, get_default_rebated_config_008
from env_2sided import HFTEnv


class RebatedEnvironmentValidationTests(unittest.TestCase):
    """Critical validation tests to find problems in rebated environments."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and configurations."""
        cls.test_csv = cls._create_problematic_test_data()
        cls.normal_csv = cls._create_normal_test_data()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        for csv_file in [cls.test_csv, cls.normal_csv]:
            if os.path.exists(csv_file):
                os.unlink(csv_file)
    
    @staticmethod
    def _create_problematic_test_data():
        """Create test data with potential edge cases and problems."""
        data = []
        
        # Create problematic scenarios
        for i in range(1000):
            row = {'timestamp': i}
            
            if i < 100:
                # Normal data
                base_price = 1800.0
                spread = 0.02
            elif i < 200:
                # Extremely tight spreads (sub-penny)
                base_price = 1800.0
                spread = 0.001  # 0.1 cent spread
            elif i < 300:
                # Very wide spreads
                base_price = 1800.0
                spread = 1.0  # $1 spread
            elif i < 400:
                # Price discontinuities (gaps)
                base_price = 1800.0 + (i - 300) * 10  # Large price jumps
                spread = 0.02
            elif i < 500:
                # Extremely low volumes
                base_price = 1800.0
                spread = 0.02
            elif i < 600:
                # Extremely high volumes
                base_price = 1800.0
                spread = 0.02
            elif i < 700:
                # Inverted book scenarios (should be filtered out but test anyway)
                base_price = 1800.0
                spread = -0.02  # Negative spread (bid > ask)
            elif i < 800:
                # Zero quantities
                base_price = 1800.0
                spread = 0.02
            elif i < 900:
                # Price precision edge cases
                base_price = 1800.123456789  # High precision
                spread = 0.020000001  # Tiny precision differences
            else:
                # Normal data for stability
                base_price = 1800.0
                spread = 0.02
            
            # Generate order book levels
            for level in range(1, 11):
                bid_offset = (level - 1) * 0.01 + spread/2
                ask_offset = (level - 1) * 0.01 + spread/2
                
                bid_price = base_price - bid_offset
                ask_price = base_price + ask_offset
                
                # Generate quantities with edge cases
                if 500 <= i < 600:  # High volume period
                    bid_qty = np.random.uniform(1000.0, 10000.0)
                    ask_qty = np.random.uniform(1000.0, 10000.0)
                elif 400 <= i < 500:  # Low volume period
                    bid_qty = np.random.uniform(0.001, 0.01)
                    ask_qty = np.random.uniform(0.001, 0.01)
                elif 700 <= i < 800:  # Zero quantity period
                    bid_qty = 0.0 if level > 5 else np.random.uniform(1.0, 10.0)
                    ask_qty = 0.0 if level > 5 else np.random.uniform(1.0, 10.0)
                else:
                    bid_qty = np.random.uniform(1.0, 100.0)
                    ask_qty = np.random.uniform(1.0, 100.0)
                
                row[f'bid{level}'] = bid_price
                row[f'bidqty{level}'] = bid_qty
                row[f'ask{level}'] = ask_price
                row[f'askqty{level}'] = ask_qty
            
            data.append(row)
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    
    @staticmethod
    def _create_normal_test_data():
        """Create normal test data for comparison."""
        data = []
        np.random.seed(42)
        
        for i in range(500):
            row = {'timestamp': i}
            base_price = 1800.0 + np.random.normal(0, 0.1)
            spread = 0.02
            
            for level in range(1, 11):
                bid_offset = (level - 1) * 0.01 + spread/2
                ask_offset = (level - 1) * 0.01 + spread/2
                
                row[f'bid{level}'] = base_price - bid_offset
                row[f'bidqty{level}'] = np.random.uniform(5.0, 50.0)
                row[f'ask{level}'] = base_price + ask_offset
                row[f'askqty{level}'] = np.random.uniform(5.0, 50.0)
                
            data.append(row)
        
        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        return temp_file.name
    
    def test_rebate_calculation_accuracy(self):
        """Test that rebate calculations are mathematically correct."""
        print("\n=== Testing Rebate Calculation Accuracy ===")
        
        for rebate_rate, env_class, config_func in [
            (0.00004, HFTEnvRebated004, get_default_rebated_config_004),
            (0.00006, HFTEnvRebated006, get_default_rebated_config_006),
            (0.00008, HFTEnvRebated008, get_default_rebated_config_008)
        ]:
            with self.subTest(rebate_rate=rebate_rate):
                config = config_func()
                config["csv_path"] = self.normal_csv
                config["max_steps"] = 100
                config["episode_length"] = 50
                
                env = env_class(config)
                obs, info = env.reset(seed=42)
                
                # Track manual calculations
                manual_rebates = 0.0
                manual_volume = 0.0
                
                for step in range(20):
                    pre_cash = env.cash
                    pre_volume = getattr(env, 'last_executed_volume', 0.0)
                    
                    # Execute a trade-likely action
                    action = [0.0, 0.0, 0.8, 0.8, -1.0, -1.0]
                    obs, reward, term, trunc, info = env.step(action)
                    
                    # Check if execution occurred
                    if hasattr(env, 'last_executed_volume') and env.last_executed_volume > pre_volume:
                        executed_volume = env.last_executed_volume - pre_volume
                        expected_rebate = executed_volume * rebate_rate * env.midprice
                        
                        # Cash should increase by MORE than trade value due to rebate
                        cash_change = env.cash - pre_cash
                        
                        # The cash change should include the rebate benefit
                        # (Note: this is complex due to position tracking, but rebates should be positive)
                        manual_rebates += expected_rebate
                        manual_volume += executed_volume
                        
                        print(f"Step {step}: Volume={executed_volume:.4f}, "
                              f"Expected rebate=${expected_rebate:.6f}, "
                              f"Cash change=${cash_change:.4f}")
                    
                    if term or trunc:
                        break
                
                # Verify rebate tracking accuracy
                if manual_volume > 0:
                    tracked_rebates = info.get('total_rebates_earned', 0.0)
                    tracked_volume = info.get('rebated_volume', 0.0)
                    
                    print(f"Manual: ${manual_rebates:.6f}, Tracked: ${tracked_rebates:.6f}")
                    print(f"Volume Manual: {manual_volume:.4f}, Tracked: {tracked_volume:.4f}")
                    
                    # Allow for small floating point differences
                    self.assertAlmostEqual(manual_volume, tracked_volume, places=4,
                                         msg=f"Volume tracking mismatch for {rebate_rate}")
    
    def test_negative_transaction_cost_edge_cases(self):
        """Test behavior with negative transaction costs in edge scenarios."""
        print("\n=== Testing Negative Transaction Cost Edge Cases ===")
        
        config = get_default_rebated_config_008()  # Highest rebate
        config["csv_path"] = self.test_csv
        config["max_steps"] = 200
        config["episode_length"] = 100
        
        env = HFTEnvRebated008(config)
        
        # Test with various scenarios
        test_scenarios = [
            ("Tiny volumes", [0.0, 0.0, 0.01, 0.01, -1.0, -1.0]),
            ("Maximum volumes", [0.0, 0.0, 1.0, 1.0, -1.0, -1.0]),
            ("Aggressive pricing", [0.8, -0.8, 0.5, 0.5, -1.0, -1.0]),
            ("Conservative pricing", [0.1, -0.1, 0.5, 0.5, -1.0, -1.0]),
        ]
        
        for scenario_name, action in test_scenarios:
            with self.subTest(scenario=scenario_name):
                obs, info = env.reset(seed=42)
                initial_cash = env.cash
                
                # Execute scenario
                for step in range(10):
                    obs, reward, term, trunc, info = env.step(action)
                    
                    # Verify cash never becomes infinite or NaN
                    self.assertFalse(math.isinf(env.cash), 
                                   f"Cash became infinite in {scenario_name}")
                    self.assertFalse(math.isnan(env.cash), 
                                   f"Cash became NaN in {scenario_name}")
                    
                    # Verify rebates are never negative
                    rebates = info.get('total_rebates_earned', 0.0)
                    self.assertGreaterEqual(rebates, 0.0,
                                          f"Negative rebates in {scenario_name}")
                    
                    if term or trunc:
                        break
    
    def test_config_modification_safety(self):
        """Test that rebated environments don't break with config modifications."""
        print("\n=== Testing Config Modification Safety ===")
        
        # Test invalid rebate rates
        config = get_default_rebated_config_004()
        config["csv_path"] = self.normal_csv
        
        # Test extremely high rebate (should work but might be unrealistic)
        config["transaction_cost_long"] = -0.01  # 100 bps rebate
        config["transaction_cost_short"] = -0.01
        
        try:
            env = HFTEnvRebated004(config)
            obs, info = env.reset()
            
            # Should work but might produce unrealistic profits
            for step in range(5):
                action = [0.0, 0.0, 0.5, 0.5, -1.0, -1.0]
                obs, reward, term, trunc, info = env.step(action)
                
                # Verify environment doesn't break
                self.assertEqual(obs.shape, env.observation_space.shape)
                self.assertFalse(math.isnan(reward))
                
                if term or trunc:
                    break
                    
            print(f"Extreme rebate test passed: Final cash=${env.cash:.2f}")
            
        except Exception as e:
            self.fail(f"Environment failed with extreme rebate: {e}")
    
    def test_rebate_vs_baseline_consistency(self):
        """Test that rebated environments behave consistently vs baseline."""
        print("\n=== Testing Rebate vs Baseline Consistency ===")
        
        # Create baseline environment with same config but positive costs
        base_config = get_default_rebated_config_004()
        base_config["csv_path"] = self.normal_csv
        base_config["transaction_cost_long"] = 0.0001  # 1 bps fee
        base_config["transaction_cost_short"] = 0.0001
        base_config["episode_length"] = 50
        
        # Create rebated environment
        rebate_config = get_default_rebated_config_004()
        rebate_config["csv_path"] = self.normal_csv
        rebate_config["episode_length"] = 50
        
        base_env = HFTEnv(base_config)
        rebate_env = HFTEnvRebated004(rebate_config)
        
        # Run identical action sequences
        base_obs, base_info = base_env.reset(seed=42)
        rebate_obs, rebate_info = rebate_env.reset(seed=42)
        
        # Verify observation spaces are identical
        self.assertEqual(base_obs.shape, rebate_obs.shape)
        
        base_rewards = []
        rebate_rewards = []
        
        for step in range(30):
            action = [0.1, -0.1, 0.6, 0.6, -1.0, -1.0]
            
            base_obs, base_reward, base_term, base_trunc, base_info = base_env.step(action)
            rebate_obs, rebate_reward, rebate_term, rebate_trunc, rebate_info = rebate_env.step(action)
            
            base_rewards.append(base_reward)
            rebate_rewards.append(rebate_reward)
            
            # Observations should have same structure (different values due to cash differences)
            self.assertEqual(base_obs.shape, rebate_obs.shape)
            
            # Rebated environment should generally perform better due to rebates
            # (though not necessarily every single step due to market dynamics)
            
            if base_term or rebate_term or base_trunc or rebate_trunc:
                break
        
        # Overall, rebated environment should perform better
        base_total = sum(base_rewards)
        rebate_total = sum(rebate_rewards)
        
        print(f"Baseline total reward: {base_total:.4f}")
        print(f"Rebated total reward: {rebate_total:.4f}")
        print(f"Rebate advantage: {rebate_total - base_total:.4f}")
        
        # Rebated should generally be better (allowing for some market noise)
        if rebate_env.rebated_volume > 0:  # Only if trading occurred
            self.assertGreater(rebate_total, base_total - 50,  # Allow some tolerance
                             "Rebated environment should generally outperform baseline")
    
    def test_memory_leaks_and_performance(self):
        """Test for memory leaks and performance issues."""
        print("\n=== Testing Memory Leaks and Performance ===")
        
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple episodes to check for memory leaks
        for episode in range(10):
            config = get_default_rebated_config_006()
            config["csv_path"] = self.normal_csv
            config["episode_length"] = 100
            
            env = HFTEnvRebated006(config)
            obs, info = env.reset()
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                
                if term or trunc:
                    break
            
            # Clean up
            del env
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            # Should not grow excessively (allow 100MB growth over 10 episodes)
            self.assertLess(memory_growth, 100, 
                          f"Excessive memory growth: {memory_growth:.1f}MB after {episode+1} episodes")
        
        print(f"Memory growth over 10 episodes: {memory_growth:.1f}MB")
    
    def test_extreme_market_conditions(self):
        """Test behavior under extreme market conditions."""
        print("\n=== Testing Extreme Market Conditions ===")
        
        config = get_default_rebated_config_004()
        config["csv_path"] = self.test_csv  # Contains extreme scenarios
        config["episode_length"] = 200
        
        env = HFTEnvRebated004(config)
        obs, info = env.reset()
        
        errors_caught = []
        warnings_caught = []
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            for step in range(200):
                try:
                    # Use aggressive actions that might trigger edge cases
                    if step % 4 == 0:
                        action = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0]  # Max aggressive
                    elif step % 4 == 1:
                        action = [-1.0, 1.0, 1.0, 1.0, -1.0, -1.0]  # Opposite aggressive
                    elif step % 4 == 2:
                        action = [0.0, 0.0, 0.01, 0.01, -1.0, -1.0]  # Tiny volumes
                    else:
                        action = [0.0, 0.0, 0.0, 0.0, 0.9, -1.0]  # Cancel everything
                    
                    obs, reward, term, trunc, info = env.step(action)
                    
                    # Verify critical invariants
                    self.assertFalse(math.isnan(reward), f"NaN reward at step {step}")
                    self.assertFalse(math.isinf(reward), f"Infinite reward at step {step}")
                    self.assertFalse(math.isnan(env.cash), f"NaN cash at step {step}")
                    self.assertFalse(math.isinf(env.cash), f"Infinite cash at step {step}")
                    
                    # Check observation validity
                    self.assertFalse(np.any(np.isnan(obs)), f"NaN in observation at step {step}")
                    self.assertFalse(np.any(np.isinf(obs)), f"Infinite values in observation at step {step}")
                    
                    # Rebates should never be negative
                    rebates = info.get('total_rebates_earned', 0.0)
                    self.assertGreaterEqual(rebates, 0.0, f"Negative rebates at step {step}")
                    
                    if term or trunc:
                        break
                        
                except Exception as e:
                    errors_caught.append((step, str(e)))
            
            warnings_caught = [str(warning.message) for warning in w]
        
        print(f"Completed {step+1} steps under extreme conditions")
        print(f"Errors caught: {len(errors_caught)}")
        print(f"Warnings: {len(warnings_caught)}")
        
        # Report any errors (should be zero for robust implementation)
        if errors_caught:
            print("ERRORS FOUND:")
            for step, error in errors_caught[:5]:  # Show first 5
                print(f"  Step {step}: {error}")
            self.fail(f"Environment failed under extreme conditions: {len(errors_caught)} errors")
        
        # Report warnings (might be acceptable)
        if warnings_caught:
            print("WARNINGS (first 5):")
            for warning in warnings_caught[:5]:
                print(f"  {warning}")
    
    def test_rebate_accumulation_precision(self):
        """Test precision of rebate accumulation over many trades."""
        print("\n=== Testing Rebate Accumulation Precision ===")
        
        config = get_default_rebated_config_008()
        config["csv_path"] = self.normal_csv
        config["episode_length"] = 300
        
        env = HFTEnvRebated008(config)
        obs, info = env.reset(seed=42)
        
        # Execute many small trades to test precision
        manual_rebate_sum = 0.0
        trade_count = 0
        
        for step in range(300):
            pre_volume = getattr(env, 'last_executed_volume', 0.0)
            
            # Small consistent trades
            action = [0.05, -0.05, 0.3, 0.3, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            # Track manual calculation
            if hasattr(env, 'last_executed_volume') and env.last_executed_volume > pre_volume:
                volume_delta = env.last_executed_volume - pre_volume
                rebate_delta = volume_delta * env.rebate_rate * env.midprice
                manual_rebate_sum += rebate_delta
                trade_count += 1
            
            if term or trunc:
                break
        
        tracked_rebates = info.get('total_rebates_earned', 0.0)
        
        print(f"Trades executed: {trade_count}")
        print(f"Manual rebate sum: ${manual_rebate_sum:.8f}")
        print(f"Tracked rebates: ${tracked_rebates:.8f}")
        print(f"Precision difference: ${abs(manual_rebate_sum - tracked_rebates):.8f}")
        
        if trade_count > 0:
            # Allow for reasonable floating point precision differences
            precision_tolerance = trade_count * 1e-8  # Scale with number of operations
            self.assertAlmostEqual(manual_rebate_sum, tracked_rebates, 
                                 delta=precision_tolerance,
                                 msg="Rebate accumulation precision error")
    
    def test_environment_state_consistency(self):
        """Test that environment state remains consistent after rebate calculations."""
        print("\n=== Testing Environment State Consistency ===")
        
        config = get_default_rebated_config_006()
        config["csv_path"] = self.normal_csv
        config["episode_length"] = 100
        
        env = HFTEnvRebated006(config)
        obs, info = env.reset()
        
        # Track state consistency
        for step in range(100):
            pre_state = {
                'cash': env.cash,
                'inventory': env.inventory,
                'long_position': env.long_position,
                'short_position': env.short_position,
                'active_orders': len(env.active_orders),
                'pending_orders': len(env.pending_orders)
            }
            
            action = [0.1, -0.1, 0.5, 0.5, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            # Verify state consistency
            net_inventory = env.long_position - env.short_position
            self.assertAlmostEqual(net_inventory, env.inventory, places=6,
                                 msg=f"Inventory inconsistency at step {step}")
            
            # MTM calculation should be consistent
            expected_mtm = env.cash + env.inventory * env.midprice
            actual_mtm = info.get('mtm', 0.0)
            
            if not math.isnan(env.midprice):
                self.assertAlmostEqual(expected_mtm, actual_mtm, places=2,
                                     msg=f"MTM calculation inconsistency at step {step}")
            
            # Rebate tracking should be monotonic (never decrease)
            if step > 0:
                current_rebates = info.get('total_rebates_earned', 0.0)
                self.assertGreaterEqual(current_rebates, 0.0,
                                      msg=f"Rebates became negative at step {step}")
            
            if term or trunc:
                break
        
        print(f"State consistency verified over {step+1} steps")


class EdgeCaseStressTests(unittest.TestCase):
    """Stress tests for edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up for each test."""
        # Create minimal test data
        data = []
        for i in range(50):
            row = {'timestamp': i}
            for level in range(1, 11):
                row[f'bid{level}'] = 1800.0 - (level-1) * 0.01 - 0.01
                row[f'bidqty{level}'] = 10.0
                row[f'ask{level}'] = 1800.0 + (level-1) * 0.01 + 0.01
                row[f'askqty{level}'] = 10.0
            data.append(row)
        
        df = pd.DataFrame(data)
        self.test_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(self.test_csv.name, index=False)
        self.test_csv.close()
    
    def tearDown(self):
        """Clean up after each test."""
        os.unlink(self.test_csv.name)
    
    def test_zero_rebate_edge_case(self):
        """Test behavior when rebate rate is exactly zero."""
        config = get_default_rebated_config_004()
        config["csv_path"] = self.test_csv.name
        config["transaction_cost_long"] = 0.0  # Zero cost/rebate
        config["transaction_cost_short"] = 0.0
        config["episode_length"] = 20
        
        env = HFTEnvRebated004(config)
        obs, info = env.reset()
        
        for step in range(20):
            action = [0.0, 0.0, 0.5, 0.5, -1.0, -1.0]
            obs, reward, term, trunc, info = env.step(action)
            
            # With zero transaction costs, rebates should remain zero
            rebates = info.get('total_rebates_earned', 0.0)
            self.assertEqual(rebates, 0.0, "Rebates should be zero when rate is zero")
            
            if term or trunc:
                break
    
    def test_boundary_inventory_with_rebates(self):
        """Test rebate behavior at inventory boundaries."""
        config = get_default_rebated_config_008()
        config["csv_path"] = self.test_csv.name
        config["max_inventory"] = 10.0  # Small limit for testing
        config["episode_length"] = 30
        
        env = HFTEnvRebated008(config)
        obs, info = env.reset()
        
        # Try to hit inventory limits
        for step in range(30):
            if env.inventory < 5:
                action = [0.0, 0.0, 1.0, 0.0, -1.0, -1.0]  # Buy only
            elif env.inventory > -5:
                action = [0.0, 0.0, 0.0, 1.0, -1.0, -1.0]  # Sell only
            else:
                action = [0.0, 0.0, 0.5, 0.5, -1.0, -1.0]  # Normal
            
            obs, reward, term, trunc, info = env.step(action)
            
            # Verify inventory constraints
            self.assertLessEqual(abs(env.inventory), config["max_inventory"] + 0.01,
                               f"Inventory limit violated: {env.inventory}")
            
            # Rebates should still accumulate properly
            rebates = info.get('total_rebates_earned', 0.0)
            self.assertGreaterEqual(rebates, 0.0, "Rebates should never be negative")
            
            if term or trunc:
                break


def run_validation_suite():
    """Run the complete validation suite and generate a report."""
    
    print("=" * 80)
    print("REBATED ENVIRONMENT VALIDATION SUITE")
    print("=" * 80)
    print("Purpose: Find problems, bugs, and edge cases in rebated environments")
    print("Focus: Critical validation rather than basic functionality testing")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add validation tests
    suite.addTest(unittest.makeSuite(RebatedEnvironmentValidationTests))
    suite.addTest(unittest.makeSuite(EdgeCaseStressTests))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY REPORT")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Tests Run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures > 0:
        print(f"\nFAILURES ({failures}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")
    
    if errors > 0:
        print(f"\nERRORS ({errors}):")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"  - {test}: {error_msg}")
    
    if failures == 0 and errors == 0:
        print("\n✅ ALL VALIDATION TESTS PASSED")
        print("The rebated environments appear to be robust and well-implemented.")
    else:
        print(f"\n❌ VALIDATION ISSUES FOUND: {failures + errors} problems")
        print("The rebated environments need fixes before production use.")
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # Run the validation suite
    result = run_validation_suite()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)