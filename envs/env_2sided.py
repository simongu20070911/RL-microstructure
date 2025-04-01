# env.py
# <<< Includes Latency=1, Reduced Aggression, Zero Taker Penalty (as set in config) >>>
# <<< ADDED Reward for Quoting >>>

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging
import math # Import math for isnan check

# Configure logging (ensure level is handled by main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HFTEnv(gym.Env):
    """
    A High-Frequency Trading Environment for Gymnasium.
    (Docstring includes Quoting Reward now conceptually)
    ... (rest of docstring) ...
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.config = config
        self._validate_config() # Validation updated below

        # Load and validate order book data
        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config.get("max_steps", len(self.order_book_history)), len(self.order_book_history))
        if not self.order_book_history:
             raise ValueError("Failed to load any valid order book data. Cannot initialize environment.")

        # --- Action Space ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # --- END Action Space ---

        self.observation_space = self._create_observation_space()

        # Initialize running statistics for observation normalization
        self.running_stats = {
            'mean': None,
            'var': None,
            'count': 0
        }
        self.decay = 0.999

        # Position tracking and other state variables (initialized in reset)
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.cash = 0.0
        self.inventory = 0.0
        self.active_orders = []
        self.pending_orders = []
        self.order_id_counter = 0
        self.current_step = 0
        self.total_steps_elapsed_in_episode = 0
        self.last_executed_volume = 0.0
        self.bids = np.array([])
        self.asks = np.array([])
        self.best_bid = np.nan
        self.best_ask = np.nan
        self.midprice = np.nan
        self.spread = np.nan
        self._last_valid_midprice = np.nan

        self.reset()

    def _validate_config(self):
        """Validates the configuration dictionary."""
        required_keys = [
            "csv_path", "initial_capital", "order_book_levels",
            "max_order_volume",
            "latency_steps_long", "latency_steps_short",
            "tick_size", "lot_size", "max_active_orders", "inventory_penalty",
            "transaction_cost_long", "transaction_cost_short",
            "max_inventory", "invalid_order_penalty", "activity_bonus",
            "taker_penalty",
            "price_offset_ticks",
            "allowed_aggressiveness_ticks",
            # --- NEW QUOTING REWARD KEYS ---
            "quoting_reward_enabled",
            "quoting_reward_amount",
            "quoting_reward_max_ticks"
        ]
        missing_keys = set(required_keys) - set(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Validate types and ranges
        if not isinstance(self.config["quoting_reward_enabled"], bool):
             raise ValueError("quoting_reward_enabled must be a boolean.")
        if self.config["quoting_reward_amount"] < 0:
             raise ValueError("quoting_reward_amount cannot be negative.")
        if self.config["quoting_reward_max_ticks"] < 0:
             raise ValueError("quoting_reward_max_ticks cannot be negative.")
        if self.config["latency_steps_long"] < 0 or self.config["latency_steps_short"] < 0:
             raise ValueError("Latency steps cannot be negative.")
        if self.config["transaction_cost_long"] < 0 or self.config["transaction_cost_short"] < 0:
             raise ValueError("Transaction costs cannot be negative.")
        if self.config["allowed_aggressiveness_ticks"] < 0:
             raise ValueError("Allowed aggressiveness ticks cannot be negative.")
        if self.config["price_offset_ticks"] < 0:
             raise ValueError("Price offset ticks cannot be negative.")
        if self.config["max_active_orders"] <= 0:
            raise ValueError("max_active_orders must be positive.")
        if self.config["tick_size"] <= 0:
            raise ValueError("tick_size must be positive.")
        if self.config["lot_size"] <= 0:
            raise ValueError("lot_size must be positive.")

        logging.info("Configuration validated successfully.")

    # ... ( _load_order_book_data, _create_observation_space remain the same ) ...
    def _load_order_book_data(self):
        """Loads and preprocesses order book data from a CSV file."""
        path = self.config["csv_path"]
        levels = self.config["order_book_levels"]
        logging.info(f"Loading order book data from: {path} (Levels: {levels})")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            logging.error(f"CSV file not found at: {path}")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise

        required_columns = ["timestamp"]
        for i in range(1, levels + 1):
            required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            # Try to load even with missing levels if possible? No, require all configured levels.
            raise ValueError(f"Missing required LOB columns in CSV data: {missing_cols}")

        # Convert necessary columns to numeric, coercing errors
        numeric_cols = [col for col in required_columns if col != "timestamp"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN in critical price/qty columns after coercion
        initial_rows = len(df)
        df.dropna(subset=numeric_cols, inplace=True)
        rows_after_nan_drop = len(df)
        logging.info(f"Dropped {initial_rows - rows_after_nan_drop} rows due to non-numeric values in LOB columns.")

        if df.empty:
            logging.error("No valid numeric data found in required LOB columns after cleaning.")
            return []

        history = []
        rows_processed = 0
        rows_skipped_validation = 0

        for index, row in df.iterrows():
            rows_processed += 1
            try:
                bids_data = [[row[f"bid{i}"], row[f"bidqty{i}"]] for i in range(1, levels + 1)]
                asks_data = [[row[f"ask{i}"], row[f"askqty{i}"]] for i in range(1, levels + 1)]

                bids = np.array(bids_data, dtype=np.float32)
                asks = np.array(asks_data, dtype=np.float32)

                # --- Rigorous Data Validation ---
                # Check for non-positive prices or negative quantities
                # Check for NaN values again (should be caught by dropna, but safety check)
                if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any() or \
                   (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any() or \
                   np.isnan(bids).any() or np.isnan(asks).any():
                    rows_skipped_validation += 1
                    continue

                # Check order book structure (decreasing bids, increasing asks)
                # Allow equal prices within bids/asks, but not across best bid/ask
                if not np.all(np.diff(bids[:, 0]) <= 1e-9) or \
                   not np.all(np.diff(asks[:, 0]) >= -1e-9): # Allow for float tolerance
                    rows_skipped_validation += 1
                    continue

                # Check for crossed book (best bid >= best ask)
                if bids[0, 0] >= asks[0, 0] - 1e-9: # Allow for float tolerance
                    rows_skipped_validation += 1
                    continue

                history.append({"bids": bids, "asks": asks})

            except KeyError as e:
                logging.error(f"Row {index}: Missing expected column key: {e}. Skipping.")
                rows_skipped_validation += 1
                continue
            except Exception as e:
                logging.error(f"Row {index}: Error processing row: {e}. Skipping.")
                rows_skipped_validation += 1
                continue

        if not history:
            logging.error("No valid order book snapshots could be processed after structure/value validation.")
            return [] # Return empty list if no valid rows remain

        logging.info(f"Loaded {len(history)} valid LOB snapshots ({rows_skipped_validation} rows skipped during validation).")
        return history

    def _create_observation_space(self):
        """Creates the observation space based on configuration."""
        # Recalculate size based on features included in _get_raw_observation
        portfolio_size = 9 # mtm, inv, active_ord, long_pos, short_pos, long_cost, short_cost, bid_dev, ask_dev
        orders_size = 2 * self.config["max_active_orders"] # price, volume per order
        book_size = 4 * self.config["order_book_levels"] # bid_px, bid_qty, ask_px, ask_qty per level
        market_size = 1 # spread

        obs_size = portfolio_size + orders_size + book_size + market_size
        logging.info(f"Observation space size: {obs_size}")
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        logging.info("Resetting environment state.")

        # Reset running statistics for normalization
        self.running_stats = {'mean': None, 'var': None, 'count': 0}

        # Liquidate positions (if any) - important if reset is called mid-run
        if hasattr(self, 'long_position') and (abs(self.long_position) > 1e-9 or abs(self.short_position) > 1e-9):
            logging.warning("Reset called with active positions. Liquidating before reset.")
            # Need market data to liquidate, use current if available, else first point
            if self.current_step > 0 and self.current_step < len(self.order_book_history):
                pass # Use existing market state for liquidation
            elif self.order_book_history:
                 first_lob_data = self.order_book_history[0]
                 self.bids = first_lob_data["bids"].copy()
                 self.asks = first_lob_data["asks"].copy()
                 self._update_market_state()
            else:
                 logging.error("Cannot liquidate positions during reset: No market data available.")
                 # Proceed with reset, but positions might not be properly zeroed if liquidation fails
            self._liquidate_positions()

        self.current_step = 0 # Index into order_book_history
        self.total_steps_elapsed_in_episode = 0 # Steps within this episode

        # Ensure data is loaded and valid
        if not self.order_book_history:
             logging.error("Cannot reset environment: Order book history is empty or invalid.")
             obs_shape = self.observation_space.shape
             dummy_obs = np.zeros(obs_shape, dtype=self.observation_space.dtype)
             dummy_info = self._get_info() # Get info based on zeroed state
             dummy_info["error"] = "No order book data loaded."
             return dummy_obs, dummy_info

        # Initialize portfolio state
        self.cash = self.config["initial_capital"]
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.inventory = 0.0

        self.active_orders = []
        self.pending_orders = []
        self.order_id_counter = 0
        self.last_executed_volume = 0.0

        # Initialize market state from the first valid data point
        first_lob_data = self.order_book_history[0]
        self.bids = first_lob_data["bids"].copy()
        self.asks = first_lob_data["asks"].copy()
        self._update_market_state() # Calculate midprice, best bid/ask etc.

        observation = self._get_observation()
        info = self._get_info()
        logging.info(f"Reset complete. Initial MTM: {info.get('mtm', 'N/A'):.2f}")
        return observation, info

    def _update_market_state(self):
        """
        Updates internal market state variables based on current LOB. Handles invalid states.
        """
        # Check if bids/asks arrays are valid and have data
        if self.bids is None or self.asks is None or self.bids.shape[0] == 0 or self.asks.shape[0] == 0:
            logging.warning(f"Step {self.current_step}: Bids/Asks array is invalid or empty. Using last valid state.")
            if math.isnan(self._last_valid_midprice):
                 logging.error("FATAL: Initial market state is invalid and no fallback exists.")
                 self.best_bid = 1.0; self.best_ask = 1.01; self.midprice = 1.005; self.spread = 0.01
            else:
                self.midprice = self._last_valid_midprice
                if not (hasattr(self, 'best_bid') and isinstance(self.best_bid, float) and not math.isnan(self.best_bid)):
                    self.best_bid = self.midprice - (self.spread / 2 if hasattr(self, 'spread') and not math.isnan(self.spread) else self.config['tick_size'])
                if not (hasattr(self, 'best_ask') and isinstance(self.best_ask, float) and not math.isnan(self.best_ask)):
                     self.best_ask = self.midprice + (self.spread / 2 if hasattr(self, 'spread') and not math.isnan(self.spread) else self.config['tick_size'])
            return

        # Check for valid quantities at BBO
        if self.bids[0, 1] < 1e-9 or self.asks[0, 1] < 1e-9:
             logging.warning(f"Step {self.current_step}: Zero quantity at BBO (Bid Qty: {self.bids[0, 1]:.4f}, Ask Qty: {self.asks[0, 1]:.4f}). Using last valid state.")
             if math.isnan(self._last_valid_midprice):
                  logging.error("FATAL: Initial BBO has zero quantity and no fallback exists.")
                  self.best_bid = 1.0; self.best_ask = 1.01; self.midprice = 1.005; self.spread = 0.01
             else:
                self.midprice = self._last_valid_midprice
                if not (hasattr(self, 'best_bid') and isinstance(self.best_bid, float) and not math.isnan(self.best_bid)):
                    self.best_bid = self.midprice - (self.spread / 2 if hasattr(self, 'spread') and not math.isnan(self.spread) else self.config['tick_size'])
                if not (hasattr(self, 'best_ask') and isinstance(self.best_ask, float) and not math.isnan(self.best_ask)):
                     self.best_ask = self.midprice + (self.spread / 2 if hasattr(self, 'spread') and not math.isnan(self.spread) else self.config['tick_size'])
             return

        # We have valid BBO data with quantity
        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        # Check for crossed/locked book
        if self.best_bid >= self.best_ask - 1e-9:
            logging.warning(f"Step {self.current_step}: Book crossed/locked (Bid {self.best_bid:.2f} >= Ask {self.best_ask:.2f}). Using mid = avg, spread = 0.")
            self.midprice = (self.best_bid + self.best_ask) / 2.0
            self.spread = max(0.0, self.best_ask - self.best_bid)
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2.0
            self.spread = self.best_ask - self.best_bid

        # Final check for valid midprice
        if math.isnan(self.midprice) or math.isinf(self.midprice):
             logging.error(f"Step {self.current_step}: Invalid midprice ({self.midprice}) calculated. BBO: Bid={self.best_bid}, Ask={self.best_ask}. Using last valid midprice.")
             if hasattr(self, '_last_valid_midprice') and not math.isnan(self._last_valid_midprice):
                  self.midprice = self._last_valid_midprice
             else:
                  self.midprice = 1.0
                  logging.warning("Falling back to midprice = 1.0")
             if not (hasattr(self, 'spread') and isinstance(self.spread, float) and not math.isnan(self.spread)):
                 self.spread = self.config['tick_size'] * 2

        else:
             self._last_valid_midprice = self.midprice

    def step(self, action):
        """Executes one time step within the environment."""
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
             logging.error(f"Step {self.current_step}: Received action with incorrect shape {action.shape}, expected {self.action_space.shape}. Taking no action.")
             action = np.array([0.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float32)

        self.total_steps_elapsed_in_episode += 1
        logging.debug(f"--- Step {self.current_step} (Episode Step {self.total_steps_elapsed_in_episode}/{self.config.get('episode_length', self.max_steps)}) ---")

        logging.debug(f"Step {self.current_step} - Raw Action: {action}")
        try:
            buy_offset_action, sell_offset_action, buy_size_signal, sell_size_signal, replacement_signal = action
            buy_volume_scaled = np.clip((buy_size_signal + 1) / 2.0, 0.0, 1.0)
            sell_volume_scaled = np.clip((sell_size_signal + 1) / 2.0, 0.0, 1.0)
            logging.debug(f"Step {self.current_step} - Actions Parsed: BuyOffset={buy_offset_action:.3f}, SellOffset={sell_offset_action:.3f}, BuySize={buy_volume_scaled:.4f}, SellSize={sell_volume_scaled:.4f}, ReplaceSig={replacement_signal:.6f}") # Increased precision
            bbo_bid_str = f"{self.best_bid:.2f}" if not math.isnan(self.best_bid) else "NaN"
            bbo_ask_str = f"{self.best_ask:.2f}" if not math.isnan(self.best_ask) else "NaN"
            logging.debug(f"Step {self.current_step} - BBO Before Placement: Bid={bbo_bid_str}, Ask={bbo_ask_str}")
        except ValueError as e:
            logging.error(f"Step {self.current_step}: Error unpacking action {action}: {e}. Taking no action.")
            buy_offset_action, sell_offset_action, buy_size_signal, sell_size_signal, replacement_signal = 0,0,-1,-1,-1
            buy_volume_scaled, sell_volume_scaled = 0.0, 0.0

        # 1. Process Action: Cancel/Place Orders
        order_penalty = 0.0
        cancellation_penalty = 0.0 # Initialize cancellation penalty
        num_cancelled = 0
        #################
        replacement_threshold = 0
        #################
        if replacement_signal > replacement_threshold:
            logging.debug(f"Replacement signal {replacement_signal:.6f} > {replacement_threshold}. Triggering cancellation and placement.") # Log includes threshold
            num_cancelled = len(self.active_orders) + len(self.pending_orders)
            if num_cancelled > 0:
                # Apply cancellation penalty (using new config key)
                # penalty_val = self.config.get("cancellation_penalty", 0.0) # Simple flat penalty
                # cancellation_penalty = penalty_val
                # logging.debug(f"Applying cancellation penalty: {-cancellation_penalty:.6f} for cancelling {num_cancelled} orders.")
                logging.debug(f"Cancelling {len(self.active_orders)} active and {len(self.pending_orders)} pending orders.")
                self.active_orders = []
                self.pending_orders = []
            else:
                 logging.debug("Replacement signal positive, but no orders to cancel.")

            order_placed_buy = False
            order_placed_sell = False
            if buy_volume_scaled > 1e-9:
                 order_placed_buy = self._place_single_order(True, buy_offset_action, buy_volume_scaled)
            if sell_volume_scaled > 1e-9:
                 order_placed_sell = self._place_single_order(False, sell_offset_action, sell_volume_scaled)

            if buy_volume_scaled > 1e-9 and not order_placed_buy:
                 order_penalty += self.config["invalid_order_penalty"]
                 logging.debug("Failed to place intended BUY order (invalid order penalty applied).")
            if sell_volume_scaled > 1e-9 and not order_placed_sell:
                 order_penalty += self.config["invalid_order_penalty"]
                 logging.debug("Failed to place intended SELL order (invalid order penalty applied).")
        else:
            logging.debug(f"Replacement signal {replacement_signal:.6f} <= {replacement_threshold}. No order replacement action.")


        # 2. Process Latency
        self._process_pending_orders()

        # 3. Execute Active Orders
        realized_pnl_from_fills = self._execute_orders()

        # 4. Update Inventory
        self.inventory = self.long_position - self.short_position

        # 5. Check Termination/Truncation Conditions
        terminated_data = self.current_step >= len(self.order_book_history) - 1
        max_ep_len = self.config.get('episode_length', self.max_steps)
        truncated_len = self.total_steps_elapsed_in_episode >= max_ep_len
        terminated_cash = self.cash < 0
        terminated_inv = abs(self.inventory) > self.config["max_inventory"] + 1e-9
        terminated = terminated_cash or terminated_inv or terminated_data
        truncated = truncated_len

        # 6. Liquidate Positions at the End
        liquidation_pnl = 0.0
        if terminated or truncated:
            reasons = []
            if terminated_cash: reasons.append("Cash<0")
            if terminated_inv: reasons.append("InvLimit")
            if terminated_data: reasons.append("EndData")
            if truncated_len: reasons.append("MaxEpLen")
            logging.info(f"Episode end condition met ({', '.join(reasons)}). Term: {terminated}, Trunc: {truncated}. Liquidating.")
            liquidation_pnl = self._liquidate_positions()
            self.inventory = 0.0

        # 7. Calculate Mark-to-Market (MTM)
        midprice_mtm = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
        if math.isnan(midprice_mtm): midprice_mtm = 0.0
        current_inventory_value = self.inventory * midprice_mtm
        mtm = self.cash + current_inventory_value

        # 8. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty()
        activity_bonus = self.config["activity_bonus"] * self.last_executed_volume
        # --- Calculate Quoting Reward ---
        quoting_reward = self._calculate_quoting_reward()

        # 9. Calculate Final Reward
        reward = (realized_pnl_from_fills +
                  liquidation_pnl +
                  activity_bonus +
                  quoting_reward - # Add quoting reward
                  risk_penalty -
                  order_penalty -
                  cancellation_penalty) # Subtract cancellation penalty if implemented
        logging.debug(f"Step {self.current_step} Rewards - Fills: {realized_pnl_from_fills:+.4f}, Liq: {liquidation_pnl:+.4f}, Quote: {quoting_reward:+.6f}, ActBonus: {activity_bonus:+.4f}, RiskPen: {-risk_penalty:.4f}, OrderPen: {-order_penalty:.4f}, CancelPen: {-cancellation_penalty:.4f} => Total: {reward:+.4f}")


        # 10. Advance Market Data
        if not (terminated or truncated):
             self._update_step_state()

        # 11. Get Observation and Info
        observation = self._get_observation()
        info = self._get_info()
        info['episode_pnl'] = mtm - self.config["initial_capital"]
        info['quoting_reward_step'] = quoting_reward # Add step reward to info

        # Log final state
        final_inv = self.long_position - self.short_position
        log_mtm = info.get('mtm', 0.0); log_cash = info.get('cash', 0.0)
        log_long_cost = info.get('long_avg_cost', 0.0); log_short_cost = info.get('short_avg_cost', 0.0)
        logging.debug(f"Step {self.current_step if not (terminated or truncated) else 'Final'} End State - MTM: {log_mtm:.2f}, Inv: {final_inv:.4f}, Cash: {log_cash:.2f}, Long@{log_long_cost:.2f}, Short@{log_short_cost:.2f}")

        # Ensure observation matches space definition
        if not self.observation_space.contains(observation):
             logging.error(f"Step {self.current_step}: Observation {observation} is not contained in space {self.observation_space}. Clipping or fixing is needed.")
             observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
             observation = np.nan_to_num(observation, nan=0.0, posinf=self.observation_space.high[0], neginf=self.observation_space.low[0])

        return observation, reward, terminated, truncated, info

    # ==========================================================================
    # Order Placement Logic (Hybrid Approach) - Remains the same
    # ==========================================================================
    def _place_single_order(self, is_buy, price_offset_action, volume_scaled):
        """Places a single order using price_offset_ticks for range,
           clipped by allowed_aggressiveness_ticks for maximum aggression.
        """
        if len(self.active_orders) + len(self.pending_orders) >= self.config["max_active_orders"]:
            logging.debug(f"Max orders ({self.config['max_active_orders']}) reached. Cannot place {'BUY' if is_buy else 'SELL'} order.")
            return False

        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0.0, math.floor(volume / self.config["lot_size"]) * self.config["lot_size"])

        if volume <= 1e-9:
            logging.debug(f"Scaled volume {volume_scaled:.4f} -> {volume:.6f} is effectively zero. Not placing.")
            return False

        max_passive_offset_ticks = self.config["price_offset_ticks"]
        max_aggressive_placement_ticks = self.config["allowed_aggressiveness_ticks"]
        tick_size = self.config["tick_size"]

        best_bid_eff = self.best_bid if not math.isnan(self.best_bid) else self._last_valid_midprice - tick_size
        best_ask_eff = self.best_ask if not math.isnan(self.best_ask) else self._last_valid_midprice + tick_size
        if math.isnan(best_bid_eff) or math.isnan(best_ask_eff):
             logging.error("Cannot determine effective BBO for placement, BBO and fallback are NaN.")
             return False

        if best_bid_eff >= best_ask_eff - 1e-9:
             mid = (best_bid_eff + best_ask_eff) / 2.0
             best_bid_ref_calc = mid
             best_ask_ref_calc = mid
             clip_bid_ref = best_bid_eff
             clip_ask_ref = best_ask_eff
             logging.warning(f"Book crossed/locked. Using Midprice ({mid:.2f}) as reference for offset calc.")
        else:
             best_bid_ref_calc = best_bid_eff
             best_ask_ref_calc = best_ask_eff
             clip_bid_ref = best_bid_eff
             clip_ask_ref = best_ask_eff

        target_limit_price = 0.0; calculated_tick_offset = 0.0
        action_type = ""; ref_price_calc = 0.0

        if is_buy:
            calculated_tick_offset = price_offset_action * max_passive_offset_ticks
            target_limit_price = best_bid_ref_calc + calculated_tick_offset * tick_size
            action_type = "BUY"; ref_price_calc = best_bid_ref_calc
        else:
            calculated_tick_offset = price_offset_action * max_passive_offset_ticks
            target_limit_price = best_ask_ref_calc + calculated_tick_offset * tick_size
            action_type = "SELL"; ref_price_calc = best_ask_ref_calc

        logging.debug(f"Placing {action_type} Order Attempt: Vol={volume:.4f}, Action={price_offset_action:.3f}")
        logging.debug(f"  MaxPassiveOffset={max_passive_offset_ticks}, CalcTickOffset={calculated_tick_offset:.2f}")
        logging.debug(f"  Ref Price ({('Bid' if is_buy else 'Ask')} Calc): {ref_price_calc:.2f} -> Target Price: {target_limit_price:.2f}")

        max_allowed_aggressive_buy_price = clip_ask_ref + max_aggressive_placement_ticks * tick_size
        min_allowed_aggressive_sell_price = clip_bid_ref - max_aggressive_placement_ticks * tick_size
        min_allowed_aggressive_sell_price = max(tick_size, min_allowed_aggressive_sell_price)

        clipped_limit_price = target_limit_price; clipped = False
        if is_buy:
            if target_limit_price > max_allowed_aggressive_buy_price:
                clipped_limit_price = max_allowed_aggressive_buy_price; clipped = True
            logging.debug(f"  Buy Aggro Limit (Ask+{max_aggressive_placement_ticks}*T): {max_allowed_aggressive_buy_price:.2f}")
        else:
            if target_limit_price < min_allowed_aggressive_sell_price:
                clipped_limit_price = min_allowed_aggressive_sell_price; clipped = True
            logging.debug(f"  Sell Aggro Limit (Bid-{max_aggressive_placement_ticks}*T): {min_allowed_aggressive_sell_price:.2f}")

        if clipped:
            logging.debug(f"  Target price clipped by aggro limit. Clipped Price: {clipped_limit_price:.2f}")

        final_limit_price = self._round_to_tick(clipped_limit_price)
        final_limit_price = max(tick_size, final_limit_price)

        logging.debug(f"  Final Rounded Price: {final_limit_price:.2f}")

        latency = self.config["latency_steps_long"] if is_buy else self.config["latency_steps_short"]
        target_step = self.current_step + latency

        order = {
            "id": self.order_id_counter, "price": final_limit_price, "volume": volume,
            "is_buy": is_buy, "timestamp_placed": self.current_step, "target_step": target_step
        }
        self.order_id_counter += 1
        self.pending_orders.append(order)
        logging.debug(f"  Added Pending Order {order['id']}: {'Buy' if is_buy else 'Sell'} {volume:.4f} @ {final_limit_price:.2f}. Target Step: {target_step}")
        return True

    def _round_to_tick(self, price):
        """Rounds a price to the nearest tick size."""
        if math.isnan(price) or self.config["tick_size"] <= 0: return price
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    # ==========================================================================
    # Order Processing and Execution - Remains the same
    # ==========================================================================
    def _process_pending_orders(self):
        """Moves matured pending orders to active orders list."""
        still_pending = []; activated_count = 0
        current_active_count = len(self.active_orders)
        max_active = self.config["max_active_orders"]
        for order in self.pending_orders:
            if self.current_step >= order["target_step"]:
                if current_active_count < max_active:
                    self.active_orders.append(order); current_active_count += 1; activated_count += 1
                else:
                    logging.warning(f"Order {order['id']} matured but max active orders ({max_active}) reached. Order discarded.")
            else:
                still_pending.append(order)
        if activated_count > 0:
             logging.debug(f"Activated {activated_count} orders from pending list. Total active: {len(self.active_orders)}")
        self.pending_orders = still_pending

    def _execute_orders(self):
        """Matches active orders against LOB, applying maker checks and penalties."""
        realized_pnl = 0.0; total_transaction_costs = 0.0
        self.last_executed_volume = 0.0; new_active_orders = []
        best_bid_exec = self.best_bid if not math.isnan(self.best_bid) else -np.inf
        best_ask_exec = self.best_ask if not math.isnan(self.best_ask) else np.inf

        logging.debug(f"--- Exec Check Step {self.current_step} ---")
        bid_str_exec = f"{best_bid_exec:.2f}" if best_bid_exec > -np.inf else "NaN"
        ask_str_exec = f"{best_ask_exec:.2f}" if best_ask_exec < np.inf else "NaN"
        logging.debug(f"BBO for Exec: Bid={bid_str_exec}, Ask={ask_str_exec}")
        active_orders_summary = [f"ID:{o['id']},{'B' if o['is_buy'] else 'S'},P:{o['price']:.2f},V:{o['volume']:.4f}" for o in self.active_orders]
        logging.debug(f"Active Orders ({len(self.active_orders)}): [{'; '.join(active_orders_summary)}]")

        for order in self.active_orders:
            is_buy = order["is_buy"]; limit_price = order["price"]; order_volume_at_start = order["volume"]
            logging.debug(f"Checking Order {order['id']}: {'Buy' if is_buy else 'Sell'} {order['volume']:.4f} @ {limit_price:.2f}")

            is_taker = False; taker_reason = ""
            if is_buy and limit_price >= best_ask_exec - 1e-9:
                is_taker = True; taker_reason = f"Buy Price {limit_price:.2f} >= Ask {ask_str_exec}"
            elif not is_buy and limit_price <= best_bid_exec + 1e-9:
                is_taker = True; taker_reason = f"Sell Price {limit_price:.2f} <= Bid {bid_str_exec}"

            if is_taker:
                logging.debug(f"Order {order['id']} flagged TAKER ({taker_reason}). Applying penalty & cancelling.")
                realized_pnl -= self.config.get("taker_penalty", 0.0) * order_volume_at_start
                continue

            logging.debug(f"Order {order['id']} passed Taker check.")
            remaining_volume = order["volume"]
            levels_to_match = self.asks if is_buy else self.bids
            cost_rate = self.config["transaction_cost_long"] if is_buy else self.config["transaction_cost_short"]
            match_side = 'Asks' if is_buy else 'Bids'
            levels_str = ", ".join([f"P:{p:.2f},Q:{q:.4f}" for p, q in levels_to_match[:min(3, len(levels_to_match))]]) if len(levels_to_match) > 0 else "Empty"
            logging.debug(f"Matching Order {order['id']} ({'Buy' if is_buy else 'Sell'}) against {match_side}: [{levels_str}, ...]")

            for level_idx in range(len(levels_to_match)):
                if level_idx >= len(levels_to_match): break
                level_price, level_qty = levels_to_match[level_idx]
                if math.isnan(level_price) or math.isnan(level_qty) or level_qty < 1e-9: continue

                logging.debug(f"  Level {level_idx}: Price={level_price:.2f}, Qty={level_qty:.4f}")
                if remaining_volume <= 1e-9: break

                can_fill_at_level = (is_buy and limit_price >= level_price - 1e-9) or \
                                    (not is_buy and limit_price <= level_price + 1e-9)
                logging.debug(f"  Can fill at this level? {'YES' if can_fill_at_level else 'NO'}. (OrderLimit={limit_price:.2f}, LevelPrice={level_price:.2f}, IsBuy={is_buy})")

                if can_fill_at_level:
                    logging.debug(f"    Fill attempt at level price {level_price:.2f}...")
                    fill_qty_possible = min(remaining_volume, level_qty)
                    current_net_inventory = self.long_position - self.short_position
                    inventory_change = fill_qty_possible if is_buy else -fill_qty_possible
                    potential_new_inventory = current_net_inventory + inventory_change
                    max_inv = self.config["max_inventory"]
                    fill_qty = fill_qty_possible

                    if abs(potential_new_inventory) > max_inv + 1e-9:
                         if potential_new_inventory > max_inv:
                            allowed_increase = max(0.0, max_inv - current_net_inventory)
                            fill_qty = min(fill_qty_possible, allowed_increase)
                         elif potential_new_inventory < -max_inv:
                             allowed_decrease = max(0.0, current_net_inventory - (-max_inv))
                             fill_qty = min(fill_qty_possible, allowed_decrease)
                         if fill_qty < fill_qty_possible - 1e-9:
                              logging.debug(f"    Inventory limit ({max_inv}) reached. Fill adjusted: {fill_qty_possible:.4f} -> {fill_qty:.4f}.")
                    fill_qty = max(0.0, fill_qty)

                    if fill_qty > 1e-9:
                        fill_price = level_price
                        executed_value = fill_qty * fill_price
                        transaction_cost = cost_rate * executed_value
                        self.cash -= transaction_cost
                        total_transaction_costs += transaction_cost
                        if is_buy: self.cash -= executed_value
                        else: self.cash += executed_value
                        pnl_from_this_fill = self._process_fill(fill_qty, fill_price, is_buy)
                        realized_pnl += pnl_from_this_fill
                        remaining_volume -= fill_qty
                        self.last_executed_volume += fill_qty
                        levels_to_match[level_idx, 1] = max(0.0, levels_to_match[level_idx, 1] - fill_qty)
                        logging.debug(f"    EXECUTED Fill: {fill_qty:.4f} @ {fill_price:.2f}. Val={executed_value:.2f}, Cost={transaction_cost:.4f}, FillPNL={pnl_from_this_fill:+.4f}. Cash={self.cash:.2f}. LvlQtyLeft={levels_to_match[level_idx, 1]:.4f}")

            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)
            else:
                logging.debug(f"Order {order['id']} fully filled or consumed.")

        self.active_orders = new_active_orders
        logging.debug(f"End Exec Check. Step PNL(fills): {realized_pnl:+.4f}. Costs: {total_transaction_costs:.4f}. Exec Vol: {self.last_executed_volume:.4f}. Rem Act Ord: {len(self.active_orders)}")
        logging.debug(f"--- End Exec Check Step {self.current_step} ---")
        return realized_pnl

    def _process_fill(self, fill_qty, price, is_buy):
        """Updates positions based on a fill and calculates realized PnL for that fill."""
        realized_pnl = 0.0
        if is_buy:
            if self.short_position > 1e-9:
                cover_qty = min(fill_qty, self.short_position)
                if self.short_avg_cost > 0:
                    realized_pnl += (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                logging.debug(f"  Fill covered {cover_qty:.4f} short. PNL: {realized_pnl:+.4f}. Short Pos Left: {self.short_position:.4f}")
                if self.short_position <= 1e-9: self.short_position = 0.0; self.short_avg_cost = 0.0
                remaining_fill = fill_qty - cover_qty
                if remaining_fill > 1e-9: self._add_to_long(remaining_fill, price)
            else: self._add_to_long(fill_qty, price)
        else:
            if self.long_position > 1e-9:
                close_qty = min(fill_qty, self.long_position)
                if self.long_avg_cost > 0:
                    realized_pnl += (price - self.long_avg_cost) * close_qty
                self.long_position -= close_qty
                logging.debug(f"  Fill closed {close_qty:.4f} long. PNL: {realized_pnl:+.4f}. Long Pos Left: {self.long_position:.4f}")
                if self.long_position <= 1e-9: self.long_position = 0.0; self.long_avg_cost = 0.0
                remaining_fill = fill_qty - close_qty
                if remaining_fill > 1e-9: self._add_to_short(remaining_fill, price)
            else: self._add_to_short(fill_qty, price)
        return realized_pnl

    def _add_to_long(self, qty, price):
        """Helper to add to long position and update average cost."""
        if qty <= 1e-9: return
        new_total_qty = self.long_position + qty
        if self.long_position <= 1e-9: self.long_avg_cost = price
        elif new_total_qty > 1e-9: self.long_avg_cost = ((self.long_avg_cost * self.long_position) + (price * qty)) / new_total_qty
        else: logging.warning("Zero total quantity encountered when adding to long position."); self.long_avg_cost = price
        self.long_position = new_total_qty
        logging.debug(f"  Added {qty:.4f} to long @ {price:.2f}. New Long: {self.long_position:.4f}, Avg Cost: {self.long_avg_cost:.2f}")

    def _add_to_short(self, qty, price):
        """Helper to add to short position and update average cost."""
        if qty <= 1e-9: return
        new_total_qty = self.short_position + qty
        if self.short_position <= 1e-9: self.short_avg_cost = price
        elif new_total_qty > 1e-9: self.short_avg_cost = ((self.short_avg_cost * self.short_position) + (price * qty)) / new_total_qty
        else: logging.warning("Zero total quantity encountered when adding to short position."); self.short_avg_cost = price
        self.short_position = new_total_qty
        logging.debug(f"  Added {qty:.4f} to short @ {price:.2f}. New Short: {self.short_position:.4f}, Avg Cost: {self.short_avg_cost:.2f}")

    # ==========================================================================
    # Reward Calculation Helpers
    # ==========================================================================
    def _calculate_risk_penalty(self):
        """Calculates inventory risk penalty."""
        inv_penalty_factor = self.config.get("inventory_penalty", 0.0)
        max_inv = self.config.get("max_inventory", 0.0)
        if inv_penalty_factor <= 1e-9 or max_inv <= 1e-9: return 0.0

        net_inventory = self.long_position - self.short_position
        normalized_inventory = net_inventory / max_inv
        midprice_val = abs(self.midprice) if not math.isnan(self.midprice) else abs(self._last_valid_midprice)
        if math.isnan(midprice_val) or midprice_val <= 0: midprice_val = 1.0

        penalty = inv_penalty_factor * (normalized_inventory ** 2) * midprice_val
        return abs(penalty)

    def _calculate_quoting_reward(self):
        """Calculates a reward for maintaining valid maker orders near the BBO."""
        if not self.config.get("quoting_reward_enabled", False):
            return 0.0

        quoting_reward = 0.0
        reward_amount = self.config["quoting_reward_amount"]
        max_ticks = self.config["quoting_reward_max_ticks"]
        tick_size = self.config["tick_size"]

        # Use effective BBO, handling NaN
        best_bid_eff = self.best_bid if not math.isnan(self.best_bid) else -np.inf
        best_ask_eff = self.best_ask if not math.isnan(self.best_ask) else np.inf

        if best_bid_eff == -np.inf or best_ask_eff == np.inf:
             logging.debug("Quoting reward skipped: Invalid BBO.")
             return 0.0 # Cannot calculate reward without valid BBO

        best_active_buy_price = -np.inf
        best_active_sell_price = np.inf
        found_buy = False
        found_sell = False

        for order in self.active_orders:
            if order["is_buy"]:
                best_active_buy_price = max(best_active_buy_price, order["price"])
                found_buy = True
            else:
                best_active_sell_price = min(best_active_sell_price, order["price"])
                found_sell = True

        # Check buy side
        if found_buy:
            is_buy_maker = best_active_buy_price < best_ask_eff - 1e-9 # Check if it doesn't cross ask
            is_buy_close = (best_bid_eff - best_active_buy_price) <= max_ticks * tick_size + 1e-9 # Check distance from bid
            # Buy price must also be less than or equal to best bid
            is_buy_at_or_below_bid = best_active_buy_price <= best_bid_eff + 1e-9

            if is_buy_maker and is_buy_close and is_buy_at_or_below_bid:
                quoting_reward += reward_amount
                logging.debug(f"  Quoting reward added for BUY side (Price: {best_active_buy_price:.2f}, BBO: {best_bid_eff:.2f}/{best_ask_eff:.2f}, MaxTicks: {max_ticks})")
            # else: # Optional: Log why it failed
            #      logging.debug(f"  Buy quote failed: Price={best_active_buy_price:.2f}, Maker={is_buy_maker}, Close={is_buy_close}, AtOrBelowBid={is_buy_at_or_below_bid}")


        # Check sell side
        if found_sell:
            is_sell_maker = best_active_sell_price > best_bid_eff + 1e-9 # Check if it doesn't cross bid
            is_sell_close = (best_active_sell_price - best_ask_eff) <= max_ticks * tick_size + 1e-9 # Check distance from ask
            # Sell price must also be greater than or equal to best ask
            is_sell_at_or_above_ask = best_active_sell_price >= best_ask_eff - 1e-9

            if is_sell_maker and is_sell_close and is_sell_at_or_above_ask:
                quoting_reward += reward_amount
                logging.debug(f"  Quoting reward added for SELL side (Price: {best_active_sell_price:.2f}, BBO: {best_bid_eff:.2f}/{best_ask_eff:.2f}, MaxTicks: {max_ticks})")
            # else: # Optional: Log why it failed
            #      logging.debug(f"  Sell quote failed: Price={best_active_sell_price:.2f}, Maker={is_sell_maker}, Close={is_sell_close}, AtOrAboveAsk={is_sell_at_or_above_ask}")


        if quoting_reward > 0:
             logging.debug(f"Total quoting reward for step: {quoting_reward:.6f}")

        return quoting_reward

    # ==========================================================================
    # State, Observation, Info, Rendering - Remain the same
    # ==========================================================================
    def _update_step_state(self) -> None:
        """Advances market data by one step."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            current_lob_data = self.order_book_history[self.current_step]
            self.bids = current_lob_data["bids"].copy()
            self.asks = current_lob_data["asks"].copy()
            self._update_market_state()
        else:
             logging.warning(f"Attempted to update step state beyond end of data (Step {self.current_step}).")

    def _get_raw_observation(self):
        """Constructs the raw observation vector before normalization."""
        midprice_safe = self.midprice if (not math.isnan(self.midprice) and self.midprice > 0) else self._last_valid_midprice
        if math.isnan(midprice_safe) or midprice_safe <= 0: midprice_safe = 1.0

        best_bid_safe = self.best_bid if not math.isnan(self.best_bid) else midprice_safe - self.config["tick_size"]
        best_ask_safe = self.best_ask if not math.isnan(self.best_ask) else midprice_safe + self.config["tick_size"]
        if best_bid_safe >= best_ask_safe - 1e-9: best_ask_safe = best_bid_safe + self.config["tick_size"]
        spread_safe = best_ask_safe - best_bid_safe

        current_inventory = self.long_position - self.short_position
        inventory_value = current_inventory * midprice_safe
        mtm = self.cash + inventory_value

        init_capital = self.config["initial_capital"] if self.config["initial_capital"] > 0 else 1.0
        max_inv = self.config["max_inventory"] if self.config["max_inventory"] > 0 else 1.0
        max_active_ord = self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 1.0
        max_vol_safe = self.config["max_order_volume"] if self.config["max_order_volume"] > 0 else 1.0

        mtm_norm = mtm / init_capital
        inventory_norm = np.clip(current_inventory / max_inv, -1.5, 1.5)
        active_orders_norm = len(self.active_orders) / max_active_ord
        long_pos_norm = np.clip(self.long_position / max_inv, 0.0, 1.5)
        short_pos_norm = np.clip(self.short_position / max_inv, 0.0, 1.5)
        long_avg_cost_norm = (self.long_avg_cost / midprice_safe) - 1.0 if self.long_position > 1e-9 else 0.0
        short_avg_cost_norm = (self.short_avg_cost / midprice_safe) - 1.0 if self.short_position > 1e-9 else 0.0
        bid_dev_norm = (best_bid_safe - midprice_safe) / midprice_safe
        ask_dev_norm = (best_ask_safe - midprice_safe) / midprice_safe

        portfolio_features = [
            mtm_norm, inventory_norm, active_orders_norm, long_pos_norm, short_pos_norm,
            long_avg_cost_norm, short_avg_cost_norm, bid_dev_norm, ask_dev_norm
        ]

        order_prices_norm = []; order_volumes_norm = []
        num_active = len(self.active_orders)
        active_orders_to_encode = self.active_orders[:self.config["max_active_orders"]]

        for order in active_orders_to_encode:
            order_prices_norm.append((order["price"] - midprice_safe) / midprice_safe)
            norm_vol = (order["volume"] / max_vol_safe) * (1 if order['is_buy'] else -1)
            order_volumes_norm.append(norm_vol)

        padding_count = self.config["max_active_orders"] - num_active
        order_prices_norm.extend([0.0] * padding_count)
        order_volumes_norm.extend([0.0] * padding_count)

        book_features = []
        ref_price_book = midprice_safe
        levels = self.config["order_book_levels"]
        bids_len = len(self.bids) if self.bids is not None else 0
        asks_len = len(self.asks) if self.asks is not None else 0

        for level in range(levels):
            bid_price_norm, bid_qty_norm = 0.0, 0.0
            ask_price_norm, ask_qty_norm = 0.0, 0.0
            if level < bids_len:
                bid_price = self.bids[level, 0]; bid_qty = self.bids[level, 1]
                if not math.isnan(bid_price) and bid_price > 0:
                    bid_price_norm = (bid_price - ref_price_book) / ref_price_book
                    bid_qty_norm = bid_qty / max_vol_safe
            if level < asks_len:
                ask_price = self.asks[level, 0]; ask_qty = self.asks[level, 1]
                if not math.isnan(ask_price) and ask_price > 0:
                    ask_price_norm = (ask_price - ref_price_book) / ref_price_book
                    ask_qty_norm = ask_qty / max_vol_safe
            book_features.extend([bid_price_norm, bid_qty_norm, ask_price_norm, ask_qty_norm])

        spread_norm = spread_safe / midprice_safe if midprice_safe > 0 else 0.0
        raw_obs_list = portfolio_features + order_prices_norm + order_volumes_norm + book_features + [spread_norm]
        raw_obs_array = np.array(raw_obs_list, dtype=np.float32)

        if not np.all(np.isfinite(raw_obs_array)):
             logging.warning(f"Step {self.current_step}: Observation contains NaN/Inf values before return. Replacing with 0.")
             raw_obs_array = np.nan_to_num(raw_obs_array, nan=0.0, posinf=0.0, neginf=0.0)

        expected_shape = self.observation_space.shape
        if raw_obs_array.shape != expected_shape:
             logging.error(f"Observation shape mismatch! Expected {expected_shape}, got {raw_obs_array.shape}. Padding/truncating.")
             current_len = raw_obs_array.shape[0]; target_len = expected_shape[0]
             if current_len > target_len: raw_obs_array = raw_obs_array[:target_len]
             elif current_len < target_len:
                 padding = np.zeros(target_len - current_len, dtype=np.float32)
                 raw_obs_array = np.concatenate((raw_obs_array, padding))
        return raw_obs_array

    def _get_observation(self):
        """Gets the current observation."""
        return self._get_raw_observation() # Return raw observations, normalization disabled

    # --- Normalization Methods (Optional, currently unused) ---
    def _update_running_stats(self, obs): pass
    def _normalize_observation(self, obs): return obs
    # --- End Normalization Methods ---

    def _get_info(self):
        """Returns auxiliary information."""
        midprice_safe = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
        if math.isnan(midprice_safe): midprice_safe = 0.0
        inventory_value = (self.long_position - self.short_position) * midprice_safe
        mtm = self.cash + inventory_value

        info = {
            "mtm": mtm, "cash": self.cash,
            "net_inventory": self.long_position - self.short_position,
            "long_position": self.long_position, "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost if self.long_position > 1e-9 else 0.0,
            "short_avg_cost": self.short_avg_cost if self.short_position > 1e-9 else 0.0,
            "active_orders_count": len(self.active_orders), "pending_orders_count": len(self.pending_orders),
            "best_bid": self.best_bid if not math.isnan(self.best_bid) else None,
            "best_ask": self.best_ask if not math.isnan(self.best_ask) else None,
            "mid_price": self.midprice if not math.isnan(self.midprice) else None,
            "spread": self.spread if not math.isnan(self.spread) else None,
            "current_step": self.current_step,
            "total_steps_elapsed": self.total_steps_elapsed_in_episode,
        }
        if 'episode_pnl' not in info: info['episode_pnl'] = mtm - self.config["initial_capital"]
        # Add quoting reward info from the step if calculated
        if 'quoting_reward_step' not in info: info['quoting_reward_step'] = 0.0 # Default if not set in step yet
        return info

    def render(self, mode="human"):
        """Renders the environment state (prints basic info to console)."""
        if mode == "human":
            info = self._get_info()
            bid_str = f"{info['best_bid']:.2f}" if info['best_bid'] is not None else "N/A"
            ask_str = f"{info['best_ask']:.2f}" if info['best_ask'] is not None else "N/A"
            pnl_str = f"{info.get('episode_pnl', 0.0):,.2f}"
            quote_rew_str = f"{info.get('quoting_reward_step', 0.0):.6f}" # Show quoting reward

            render_msg = (
                f"Step: {info['current_step']:<6} EpStep: {info['total_steps_elapsed']:<4} | "
                f"BBO: {bid_str}/{ask_str} | "
                f"MTM: {info['mtm']:<10,.2f} | PnL: {pnl_str:<9} | "
                f"Inv: {info['net_inventory']:<+7.4f} | "
                f"Ord: A={info['active_orders_count']}/P={info['pending_orders_count']} | "
                f"QuoteRw: {quote_rew_str}" # Add quoting reward display
            )
            logging.info(render_msg)
        else:
            super().render(mode=mode)

    # ==========================================================================
    # Liquidation and Cleanup - Remains the same
    # ==========================================================================
    def _liquidate_positions(self):
        """Liquidates all open positions at current best market prices and cancels orders."""
        logging.info("Liquidating positions...")
        liquidation_pnl = 0.0; total_liq_cost = 0.0

        best_bid_liq = self.best_bid if not math.isnan(self.best_bid) else self._last_valid_midprice - self.config['tick_size']
        best_ask_liq = self.best_ask if not math.isnan(self.best_ask) else self._last_valid_midprice + self.config['tick_size']

        if math.isnan(best_bid_liq) or math.isnan(best_ask_liq) or best_bid_liq <= 0 or best_ask_liq <= 0 or best_bid_liq >= best_ask_liq - 1e-9:
            logging.error(f"Cannot determine valid liquidation prices (Bid: {best_bid_liq}, Ask: {best_ask_liq}). Using fallback.")
            mid = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
            if math.isnan(mid) or mid <=0: mid = 1.0
            best_bid_liq = max(self.config['tick_size'], mid - self.config['tick_size'])
            best_ask_liq = mid + self.config['tick_size']

        logging.info(f"Liquidation Prices: Bid={best_bid_liq:.2f}, Ask={best_ask_liq:.2f}")

        if self.long_position > 1e-9:
            qty = self.long_position; price = best_bid_liq; value = qty * price
            cost = self.config["transaction_cost_short"] * value
            pnl = (price - self.long_avg_cost) * qty if self.long_avg_cost > 0 else 0.0
            self.cash += value - cost; liquidation_pnl += (pnl - cost); total_liq_cost += cost
            logging.info(f"Liq LONG: {qty:.4f} @ {price:.2f}. Value={value:.2f}, Cost={cost:.2f}. PNL(Net)={pnl-cost:.2f}")
            self.long_position = 0.0; self.long_avg_cost = 0.0

        if self.short_position > 1e-9:
            qty = self.short_position; price = best_ask_liq; value = qty * price
            cost = self.config["transaction_cost_long"] * value
            pnl = (self.short_avg_cost - price) * qty if self.short_avg_cost > 0 else 0.0
            self.cash -= (value + cost); liquidation_pnl += (pnl - cost); total_liq_cost += cost
            logging.info(f"Liq SHORT: {qty:.4f} @ {price:.2f}. Value={value:.2f}, Cost={cost:.2f}. PNL(Net)={pnl-cost:.2f}")
            self.short_position = 0.0; self.short_avg_cost = 0.0

        active_cancel_count = len(self.active_orders); pending_cancel_count = len(self.pending_orders)
        if active_cancel_count > 0 or pending_cancel_count > 0:
             logging.info(f"Cancelling remaining {active_cancel_count} active and {pending_cancel_count} pending orders during liquidation.")
             self.active_orders = []; self.pending_orders = []

        self.inventory = 0.0
        logging.info(f"Liquidation complete. Total Liq PNL(Net): {liquidation_pnl:.2f}. Total Liq Cost: {total_liq_cost:.2f}. Final Cash: {self.cash:.2f}")
        return liquidation_pnl

    def close(self):
        """Clean up any resources (if any were used)."""
        logging.info("Closing HFT Environment.")
        pass