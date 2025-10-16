# -*- coding: utf-8 -*-
"""
DOGE/USDT Market Making Environment with Binance Fee Structure
Based on env_2sided_nocheat.py with crypto-specific enhancements.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque # Can be useful for efficient FIFO conceptually
import pandas as pd
import logging
import math
import warnings # To manage potential NaN warnings during calculations
import os # For dummy data check

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')


class HFTEnv(gym.Env):
    """
    HFT Environment with Constant Placement, FIFO Pruning, Explicit Cancellation,
    and an optional "Do Nothing" action override.

    MODIFIED TO CARRY FORWARD POSITIONS/ORDERS/INVENTORY BETWEEN EPISODES.
    ONLY CASH AND EPISODE STEP COUNT ARE RESET.

    Action Space (6 dimensions, Box [-1, 1]):
        action[0]: Buy Offset Signal (-1 to 1) -> Controls price offset ticks
        action[1]: Sell Offset Signal (-1 to 1) -> Controls price offset ticks
        action[2]: Buy Size Signal (-1 to 1) -> Controls volume (0% to 100% of max_order_volume)
        action[3]: Sell Size Signal (-1 to 1) -> Controls volume (0% to 100% of max_order_volume)
        action[4]: Explicit Cancel Signal (-1 to 1). If > cancel_threshold, cancels active/pending orders.
        action[5]: Do Nothing Signal (-1 to 1). If > do_nothing_threshold, ignore actions 0-4 and step forward.

    Observation Normalization (Revised):
        - Prices (book, orders, costs, BBO): Normalized by tick size relative to midprice ((price - midprice) / tick_size), then scaled down by obs_price_norm_scale.
        - Quantities (book, orders): Normalized by max_order_volume (qty / max_order_volume), then scaled by obs_qty_norm_scale, clipped.
        - Inventory/Positions: Normalized by max_inventory.
        - MTM: Normalized by initial_capital.
        - Spread: Normalized by tick_size (spread / tick_size).
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.config = config
        self._validate_config() # Includes validation for new keys

        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config.get("max_steps", len(self.order_book_history)), len(self.order_book_history))
        if not self.order_book_history:
             raise ValueError("Failed to load any valid order book data.")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = self._create_observation_space()

        # --- State variables initialized here, some persist across resets ---
        # Persistent state (Market Footprint)
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.inventory = 0.0
        self.active_orders = []
        self.pending_orders = []
        self.order_id_counter = 0 # Persists to avoid ID reuse

        # Persistent state (Market Data Position)
        self.current_step = 0 # Tracks position in the historical data

        # Reset per episode
        self.cash = 0.0 # Will be set in reset()
        self.total_steps_elapsed_in_episode = 0

        # Reset per step
        self.last_executed_volume = 0.0
        self.bids = np.array([]); self.asks = np.array([])
        self.best_bid = np.nan; self.best_ask = np.nan
        self.midprice = np.nan; self.spread = np.nan
        self._last_valid_midprice = np.nan # Store the last known good midprice

        # --- Flag for first reset ---
        # This ensures positions/orders are truly zero only at the very beginning
        self._is_first_reset = True

        # Call reset once during initialization to set initial state properly
        # self.reset() # Avoid calling reset here, let the user call it first.

    def _validate_config(self):
        """Validates the configuration dictionary."""
        required_keys = [
            "csv_path", "initial_capital", "order_book_levels", "max_order_volume",
            "latency_steps_long", "latency_steps_short", "tick_size", "lot_size",
            "max_active_orders", "inventory_penalty", "transaction_cost_long",
            "transaction_cost_short", "max_inventory", "invalid_order_penalty",
            "activity_bonus", "taker_penalty", "price_offset_ticks",
            "allowed_aggressiveness_ticks", "quoting_reward_enabled",
            "quoting_reward_amount", "quoting_reward_max_ticks",
            "explicit_cancel_enabled", "explicit_cancel_threshold",
            "explicit_cancel_penalty", "explicit_cancel_clears_pending",
            "do_nothing_threshold", "episode_length",
            "obs_price_norm_scale", "obs_qty_norm_scale" # Added obs scaling factors
        ]
        missing_keys = set(required_keys) - set(self.config.keys())
        if missing_keys: raise ValueError(f"Missing required config keys: {missing_keys}")

        # ... (rest of validation remains the same) ...
        if not isinstance(self.config["explicit_cancel_enabled"], bool): raise TypeError("explicit_cancel_enabled must be boolean")
        if not isinstance(self.config["explicit_cancel_threshold"], (int, float)): raise TypeError("explicit_cancel_threshold must be numeric")
        if self.config["explicit_cancel_penalty"] < 0: raise ValueError("explicit_cancel_penalty cannot be negative")
        if not isinstance(self.config["explicit_cancel_clears_pending"], bool): raise TypeError("explicit_cancel_clears_pending must be boolean")
        if not isinstance(self.config["quoting_reward_enabled"], bool): raise ValueError("quoting_reward_enabled must be a boolean.")
        if self.config["quoting_reward_amount"] < 0: raise ValueError("quoting_reward_amount cannot be negative.")
        if self.config["quoting_reward_max_ticks"] < 0: raise ValueError("quoting_reward_max_ticks cannot be negative.")
        if self.config["latency_steps_long"] < 0 or self.config["latency_steps_short"] < 0: raise ValueError("Latency steps cannot be negative.")
        if self.config["allowed_aggressiveness_ticks"] < 0: raise ValueError("Allowed aggressiveness ticks cannot be negative.")
        if self.config["price_offset_ticks"] < 0: raise ValueError("Price offset ticks cannot be negative.")
        if self.config["max_active_orders"] <= 0: raise ValueError("max_active_orders must be positive.")
        if self.config["tick_size"] <= 0: raise ValueError("tick_size must be positive.")
        if self.config["lot_size"] <= 0: raise ValueError("lot_size must be positive.")
        if not isinstance(self.config["do_nothing_threshold"], (int, float)): raise TypeError("do_nothing_threshold must be numeric")
        if self.config["episode_length"] <= 0: raise ValueError("episode_length must be positive.")
        if self.config["max_inventory"] <= 0: raise ValueError("max_inventory must be positive.")
        if self.config["max_order_volume"] <= 0: raise ValueError("max_order_volume must be positive.")
        if self.config["initial_capital"] <= 0: raise ValueError("initial_capital must be positive.")
        if not isinstance(self.config["obs_price_norm_scale"], (int, float)) or self.config["obs_price_norm_scale"] <= 0: raise ValueError("obs_price_norm_scale must be a positive number.")
        if not isinstance(self.config["obs_qty_norm_scale"], (int, float)) or self.config["obs_qty_norm_scale"] <= 0: raise ValueError("obs_qty_norm_scale must be a positive number.")

        logging.info("Configuration validated successfully.")

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
            raise ValueError(f"Missing required LOB columns in CSV data: {missing_cols}")

        numeric_cols = [col for col in required_columns if col != "timestamp"]
        for col in numeric_cols:
            # Use errors='coerce' to turn non-numeric into NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_rows = len(df)
        # Drop rows where *any* required numeric LOB column is NaN
        df.dropna(subset=numeric_cols, inplace=True)
        rows_after_nan_drop = len(df)
        if initial_rows - rows_after_nan_drop > 0:
             logging.info(f"Dropped {initial_rows - rows_after_nan_drop} rows due to non-numeric values in LOB columns.")

        if df.empty:
            logging.error("No valid numeric data found in required LOB columns after cleaning.")
            return []

        history = []
        rows_processed = 0
        rows_skipped_validation = 0
        tick_size = self.config['tick_size'] # For validation

        for index, row in df.iterrows():
            rows_processed += 1
            try:
                bids_data = [[row[f"bid{i}"], row[f"bidqty{i}"]] for i in range(1, levels + 1)]
                asks_data = [[row[f"ask{i}"], row[f"askqty{i}"]] for i in range(1, levels + 1)]
                bids = np.array(bids_data, dtype=np.float32)
                asks = np.array(asks_data, dtype=np.float32)

                # Basic value checks (already handled NaNs, check for zeros/negatives)
                if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any() or \
                   (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any(): # Allow zero quantity, but not negative
                    rows_skipped_validation += 1
                    continue

                # Structure checks: Bid prices decreasing, Ask prices increasing
                # Use tolerance for floating point comparisons
                if not np.all(np.diff(bids[:, 0]) <= 1e-9) or \
                   not np.all(np.diff(asks[:, 0]) >= -1e-9):
                    rows_skipped_validation += 1
                    continue

                # Spread check: Best bid must be less than best ask
                if bids[0, 0] >= asks[0, 0] - 1e-9: # Use tolerance
                    rows_skipped_validation += 1
                    continue

                # Valid snapshot found
                history.append({"bids": bids, "asks": asks})

            except Exception as e:
                logging.debug(f"Row {index}: Error processing row: {e}. Skipping.") # Debug level for less noise
                rows_skipped_validation += 1
                continue

        if not history:
            logging.error("No valid order book snapshots could be processed after structure/value validation.")
            return []

        logging.info(f"Loaded {len(history)} valid LOB snapshots ({rows_skipped_validation} rows skipped during validation).")
        return history

    def _create_observation_space(self):
        """Creates the observation space based on configuration."""
        # Portfolio State (9 features)
        # mtm_norm, inv_norm, active_orders_count_norm, long_pos_norm, short_pos_norm,
        # long_cost_norm_scaled, short_cost_norm_scaled, bid_dev_norm_scaled, ask_dev_norm_scaled
        portfolio_size = 9
        # Active Orders (2 features per order: price_norm_scaled, volume_norm_scaled)
        # *** NOTE: We don't add the is_taker_at_activation flag to the observation space ***
        orders_size = 2 * self.config["max_active_orders"]
        # Order Book (4 features per level: bid_price_norm_scaled, bid_qty_norm_scaled, ask_price_norm_scaled, ask_qty_norm_scaled)
        book_size = 4 * self.config["order_book_levels"]
        # Market State (1 feature: spread_norm_ticks)
        market_size = 1
        obs_size = portfolio_size + orders_size + book_size + market_size
        logging.info(f"Observation space size: {obs_size}")
        # Use broad bounds initially, as normalization handles scaling. Actual bounds depend on clipping inside _get_raw_observation.
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Resets the environment state for a new episode.
        - Carries over positions, orders, and inventory from the previous episode.
        - Resets cash to initial_capital.
        - Resets episode step counter.
        - Continues from the current position in the dataset (`self.current_step`).
        - Performs full state initialization only on the very first call.
        """
        super().reset(seed=seed)
        logging.debug("Resetting environment state for new episode...")

        # --- 1. Handle End-of-Data Condition ---
        if not hasattr(self, 'order_book_history') or not self.order_book_history:
             logging.error("Cannot reset: Order book history is empty.")
             dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype) \
                 if hasattr(self, 'observation_space') else None
             return dummy_obs, {"error": "No order book data.",
                              "current_step": self.current_step if hasattr(self, 'current_step') else -1}

        data_len = len(self.order_book_history)
        if self.current_step >= data_len:
            logging.warning(f"Reached end of data (current_step {self.current_step} >= data_len {data_len}). "
                           "Resetting to step 0 (Looping dataset).")
            self.current_step = 0
        elif self.current_step < 0:
            logging.warning(f"Invalid current_step ({self.current_step}) during reset. Setting to 0.")
            self.current_step = 0

        # --- 2. Reset Agent Financials & Episode Tracking ---
        self.cash = self.config["initial_capital"]
        self.total_steps_elapsed_in_episode = 0
        # Reset transient step variables (although they should be reset in step anyway)
        self.last_executed_volume = 0.0

        # --- 3. Handle First Reset vs. Subsequent Resets ---
        if self._is_first_reset:
            logging.info("First reset: Initializing positions, inventory, and orders.")
            # Initialize positions, costs, inventory
            self.long_position = 0.0
            self.short_position = 0.0
            self.long_avg_cost = 0.0
            self.short_avg_cost = 0.0
            self.inventory = 0.0
            # Initialize order lists and counter
            self.active_orders = []
            self.pending_orders = []
            self.order_id_counter = 0
            # Mark that the first reset has occurred
            self._is_first_reset = False
        else:
            logging.info("Subsequent reset: Carrying over existing positions, inventory, and orders.")
            # Positions, costs, inventory, orders, order_id_counter are NOT reset.
            # Recalculate inventory just in case (should be consistent if logic is correct)
            self.inventory = self.long_position - self.short_position
            logging.debug(f" Carried over state: Inv={self.inventory:.4f}, Long={self.long_position:.4f}@{self.long_avg_cost:.2f}, Short={self.short_position:.4f}@{self.short_avg_cost:.2f}, ActOrd={len(self.active_orders)}, PendOrd={len(self.pending_orders)}")

        logging.info(f"Resetting episode. Starting at data step {self.current_step}/{data_len}")

        # --- 4. Load Market State for the NEW Current Step ---
        try:
            current_lob_data = self.order_book_history[self.current_step]
            self.bids = current_lob_data["bids"].copy()
            self.asks = current_lob_data["asks"].copy()
            self._update_market_state()
            logging.debug(f"Loaded market state for step {self.current_step}. Midprice: {self.midprice:.2f}")
        except IndexError:
            logging.critical(f"Reset failed: current_step {self.current_step} is out of bounds for "
                            f"order_book_history (len={data_len}).")
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, {"error": "Data index out of bounds during reset.",
                              "current_step": self.current_step}
        except Exception as e:
            logging.error(f"Reset failed: Error loading market state for step {self.current_step}: {e}",
                         exc_info=True)
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return dummy_obs, {"error": f"State load failed: {e}",
                              "current_step": self.current_step}

        # --- 5. Get Initial Observation and Info ---
        observation = self._get_observation()
        info = self._get_info()
        # Episode PnL is relative to the newly reset capital
        info['episode_pnl'] = info.get('mtm', self.config["initial_capital"]) - self.config["initial_capital"]

        logging.info(f"Reset complete. Start Step: {self.current_step}. Initial MTM: {info.get('mtm', 'N/A'):.2f}. "
                    f"Initial Cash: {self.cash:.2f}. Obs Shape: {observation.shape}")

        # --- 6. Final Observation Validation ---
        if not self.observation_space.contains(observation):
             logging.error("Observation generated during reset is outside defined space bounds! Fixing...")
             low_bounds = np.where(np.isneginf(self.observation_space.low), -1e10, self.observation_space.low)
             high_bounds = np.where(np.isposinf(self.observation_space.high), 1e10, self.observation_space.high)
             observation = np.clip(observation, low_bounds, high_bounds)
             observation = np.nan_to_num(observation, nan=0.0, posinf=high_bounds[0], neginf=low_bounds[0])
             if observation.shape != self.observation_space.shape:
                 logging.critical("Observation SHAPE incorrect even after fixing during reset!")
                 info["error"] = "Observation shape mismatch during reset fix"
                 # Ensure returned observation has the correct shape, even if it's zeros
                 return np.zeros_like(self.observation_space.sample()), info

        return observation, info

    def _update_market_state(self):
        """Updates internal market state variables based on current LOB."""
        # Use warning level for potentially recoverable issues
        if not hasattr(self, 'bids') or not hasattr(self, 'asks') or self.bids is None or self.asks is None or self.bids.shape[0] == 0 or self.asks.shape[0] == 0:
            logging.warning(f"Step {self.current_step}: Bids/Asks array invalid/empty. Using last valid midprice if available.")
            if hasattr(self, '_last_valid_midprice') and not math.isnan(self._last_valid_midprice):
                self.midprice = self._last_valid_midprice
                # Try to estimate BBO around last midprice if they are currently NaN
                tick_size = self.config.get('tick_size', 0.01)
                if not (hasattr(self, 'best_bid') and not math.isnan(self.best_bid)):
                    self.best_bid = self.midprice - tick_size / 2.0
                if not (hasattr(self, 'best_ask') and not math.isnan(self.best_ask)):
                    self.best_ask = self.midprice + tick_size / 2.0
                self.spread = max(0.0, self.best_ask - self.best_bid) # Recalculate spread based on estimates
            else:
                 # Absolute fallback if no history
                 fallback_tick = self.config.get('tick_size', 0.01)
                 self.best_bid, self.best_ask, self.midprice, self.spread = 1.0, 1.0 + fallback_tick, 1.0 + fallback_tick / 2.0, fallback_tick
                 logging.warning("No valid previous market state, using default fallback BBO/Mid/Spread.")
            return

        # Check for valid quantities at BBO
        # Allow zero quantity, but log it. Assume prices are still meaningful.
        if self.bids[0, 1] < 1e-9:
             logging.debug(f"Step {self.current_step}: Zero quantity at best bid ({self.bids[0, 0]:.2f}).")
        if self.asks[0, 1] < 1e-9:
             logging.debug(f"Step {self.current_step}: Zero quantity at best ask ({self.asks[0, 0]:.2f}).")

        # Update BBO
        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        # Check for crossed or locked book
        if self.best_bid >= self.best_ask - 1e-9: # Use tolerance
            logging.warning(f"Step {self.current_step}: Book crossed/locked (Bid {self.best_bid:.2f} >= Ask {self.best_ask:.2f}). Using midpoint for midprice, spread is zero/negative.")
            self.midprice = (self.best_bid + self.best_ask) / 2.0
            self.spread = max(0.0, self.best_ask - self.best_bid) # Spread is 0 or negative if crossed
        else:
            # Normal case: calculate midprice and spread
            self.midprice = (self.best_bid + self.best_ask) / 2.0
            self.spread = self.best_ask - self.best_bid

        # Validate calculated midprice and store if valid
        if math.isnan(self.midprice) or math.isinf(self.midprice) or self.midprice <=0:
             logging.error(f"Step {self.current_step}: Invalid midprice calculated ({self.midprice}). Attempting to use last valid.")
             if hasattr(self, '_last_valid_midprice') and not math.isnan(self._last_valid_midprice):
                 self.midprice = self._last_valid_midprice
                 # Re-estimate spread if current spread is invalid
                 if not (hasattr(self, 'spread') and isinstance(self.spread, float) and not math.isnan(self.spread) and self.spread >= 0):
                      self.spread = self.config.get('tick_size', 0.01) # Minimal estimate
                      logging.warning(f"Spread recalculated to fallback value: {self.spread:.2f}")
             else:
                 # Absolute fallback if no history
                 fallback_tick = self.config.get('tick_size', 0.01)
                 self.midprice = 1.0 # Arbitrary positive value
                 self.spread = fallback_tick * 2
                 logging.error("No valid previous midprice, using absolute fallback midprice/spread.")
        else:
            # Midprice is valid, store it for future fallbacks
            self._last_valid_midprice = self.midprice


    def step(self, action):
        """Executes one time step: place, prune, activate, execute, reward.
           LIQUIDATION LOGIC REMOVED FROM TERMINATION/TRUNCATION."""
        # --- Action Validation ---
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
             logging.error(f"Step {self.current_step}: Incorrect action shape {action.shape}. Expected {self.action_space.shape}. Taking NO action.")
             # Create a "do nothing" effective action to prevent crashes downstream
             action = np.array([0.0] * 4 + [-1.0, 1.0], dtype=np.float32) # Ensure cancel=off, do_nothing=on

        self.total_steps_elapsed_in_episode += 1
        logging.debug(f"--- Step {self.current_step} (EpStep {self.total_steps_elapsed_in_episode}/{self.config.get('episode_length', self.max_steps)}) ---")
        logging.debug(f"Raw Action: {action}")

        # --- Unpack Action ---
        try:
            buy_offset_action, sell_offset_action, buy_size_signal, \
            sell_size_signal, explicit_cancel_signal, do_nothing_signal = action
        except ValueError as e:
            logging.error(f"Action unpacking error (expected 6 values): {e}. Returning error state.")
            return self._handle_step_error("Action unpacking failed")

        # --- Determine Action Path (Do Nothing or Market Interaction) ---
        do_nothing_threshold = self.config.get("do_nothing_threshold", 1.1) # Default > 1 disables if unset
        do_nothing_triggered = bool(do_nothing_signal > do_nothing_threshold) # Ensure boolean

        # --- Initialize Step Variables ---
        order_penalty = 0.0
        explicit_cancel_penalty = 0.0
        realized_pnl_from_fills = 0.0
        self.last_executed_volume = 0.0 # Reset executed volume for the step
        explicit_cancel_triggered = False
        # liquidation_pnl = 0.0 # REMOVED - Liquidation not automatic here
        quoting_reward = 0.0
        activity_bonus = 0.0
        risk_penalty = 0.0

        # --- Process Pending Orders (Maturation & Taker Status Check) ---
        self._process_pending_orders()

        # --- Action Execution ---
        if do_nothing_triggered:
            logging.debug(f"Do Nothing signal {do_nothing_signal:.3f} > threshold {do_nothing_threshold}. Skipping market actions.")
            quoting_reward = self._calculate_quoting_reward() # Evaluate existing quotes

        else:
            # --- Explicit Cancellation ---
            logging.debug(f"Do Nothing signal {do_nothing_signal:.3f} <= threshold {do_nothing_threshold}. Processing normal actions.")
            if self.config.get("explicit_cancel_enabled", False):
                cancel_threshold = self.config.get("explicit_cancel_threshold", 0.0)
                if explicit_cancel_signal > cancel_threshold:
                    explicit_cancel_triggered = True
                    logging.debug(f"Explicit Cancel signal {explicit_cancel_signal:.6f} > {cancel_threshold}. Triggering cancellation.")
                    num_cancelled_active = len(self.active_orders)
                    num_cancelled_pending = 0
                    if self.config.get("explicit_cancel_clears_pending", True):
                        num_cancelled_pending = len(self.pending_orders)
                        if num_cancelled_pending > 0:
                             self.pending_orders = []
                             logging.debug(f"Cleared {num_cancelled_pending} pending orders.")

                    if num_cancelled_active > 0:
                        self.active_orders = []
                        logging.debug(f"Cleared {num_cancelled_active} active orders.")

                    if num_cancelled_active + num_cancelled_pending > 0:
                        explicit_cancel_penalty = self.config.get("explicit_cancel_penalty", 0.0)
                        logging.debug(f"Applying explicit cancel penalty: {-explicit_cancel_penalty:.6f}")
                else:
                     logging.debug(f"Explicit Cancel signal {explicit_cancel_signal:.6f} <= {cancel_threshold}. No explicit cancellation.")

            # --- Order Placement ---
            buy_volume_scaled = np.clip((buy_size_signal + 1) / 2.0, 0.0, 1.0)
            sell_volume_scaled = np.clip((sell_size_signal + 1) / 2.0, 0.0, 1.0)
            logging.debug(f"Actions Parsed: BuyOffset={buy_offset_action:.3f}, SellOffset={sell_offset_action:.3f}, BuySize={buy_volume_scaled:.4f}, SellSize={sell_volume_scaled:.4f}")

            bbo_bid_str = f"{self.best_bid:.2f}" if not math.isnan(self.best_bid) else "NaN"
            bbo_ask_str = f"{self.best_ask:.2f}" if not math.isnan(self.best_ask) else "NaN"
            logging.debug(f"BBO Before Orders: Bid={bbo_bid_str}, Ask={bbo_ask_str}")

            order_placed_buy = False
            order_placed_sell = False
            if buy_volume_scaled > 1e-9:
                order_placed_buy = self._place_single_order(True, buy_offset_action, buy_volume_scaled)
                if not order_placed_buy:
                    order_penalty += self.config["invalid_order_penalty"]
                    logging.debug("Applying invalid order penalty for failed buy placement.")
            if sell_volume_scaled > 1e-9:
                order_placed_sell = self._place_single_order(False, sell_offset_action, sell_volume_scaled)
                if not order_placed_sell:
                    order_penalty += self.config["invalid_order_penalty"]
                    logging.debug("Applying invalid order penalty for failed sell placement.")

            # --- Order Pruning (FIFO if limit exceeded) ---
            self._prune_oldest_orders()

            # --- Order Execution (Matching Active Orders) ---
            realized_pnl_from_fills = self._execute_orders() # Includes taker penalty logic within

            # --- Calculate Rewards/Penalties specific to this path ---
            activity_bonus = self.config["activity_bonus"] * self.last_executed_volume
            quoting_reward = self._calculate_quoting_reward() # Evaluate remaining active quotes

        # --- Update State & Calculate Common Rewards/Penalties ---
        self.inventory = self.long_position - self.short_position
        risk_penalty = self._calculate_risk_penalty() # Inventory penalty applies regardless
        logging.info(f"Step {self.current_step} - Inventory: {self.inventory:.4f}, Cash: {self.cash:.2f}")

        # --- Check Termination/Truncation Conditions ---
        terminated_data = self.current_step >= len(self.order_book_history) - 1
        max_ep_len = self.config.get('episode_length', self.max_steps)
        truncated_len = self.total_steps_elapsed_in_episode >= max_ep_len
        terminated_cash = self.cash < -1e-9 # Terminate if bankrupt
        terminated_inv = abs(self.inventory) > self.config["max_inventory"] + 1e-9 # Terminate if inventory limit breached

        terminated = terminated_cash or terminated_inv or terminated_data
        truncated = truncated_len and not terminated # Truncate only if not already terminated

        # --- Liquidation REMOVED from end of episode ---
        if terminated or truncated:
            reasons = []
            if terminated_cash: reasons.append("Cash<0")
            if terminated_inv: reasons.append("InvLimit")
            if terminated_data: reasons.append("EndData")
            if truncated_len: reasons.append("MaxEpLen")
            # Note: No liquidation call here anymore. Positions/orders persist.
            logging.info(f"Episode end ({', '.join(reasons)}). Term={terminated}, Trunc={truncated}. NO automatic liquidation.")
            # If terminated due to cash/inventory, state is already 'bad'
            # If terminated due to data end or truncated due to length, state carries over.

        # --- Calculate Total Reward for the Step ---
        # Note: liquidation_pnl is removed from the sum
        reward = (realized_pnl_from_fills + activity_bonus + quoting_reward
                  - risk_penalty - order_penalty - explicit_cancel_penalty)

        log_msg_prefix = f"Step {self.current_step} Rewards"
        if do_nothing_triggered: log_msg_prefix += " (DO NOTHING)"
        # Adjusted log message (removed Liq)
        logging.debug(f"{log_msg_prefix} - Fills: {realized_pnl_from_fills:+.4f}, Quote: {quoting_reward:+.6f}, ActBonus: {activity_bonus:+.4f}, RiskPen: {-risk_penalty:.4f}, OrderPen: {-order_penalty:.4f}, ExpCancelPen: {-explicit_cancel_penalty:.4f} => Total: {reward:+.4f}")

        # --- Prepare Return Values ---
        observation = self._get_observation()
        info = self._get_info()

        # Update final MTM and PnL in info dict
        midprice_mtm = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
        if math.isnan(midprice_mtm) or midprice_mtm <=0 : midprice_mtm = 1.0 # Use 1.0 if midprice still unknown
        # Calculate MTM based on current cash and persistent inventory
        final_mtm = self.cash + (self.long_position - self.short_position) * midprice_mtm
        info['mtm'] = final_mtm
        # Episode PnL is relative to the initial capital *for this episode*
        info['episode_pnl'] = final_mtm - self.config["initial_capital"]
        info['quoting_reward_step'] = quoting_reward
        info['explicit_cancel_triggered'] = explicit_cancel_triggered
        info['do_nothing_triggered'] = do_nothing_triggered

        # --- Advance Market Data ---
        # Increment step counter regardless of termination/truncation IF NOT at data end
        # The reset() method will handle looping or stopping based on this incremented value
        # on the *next* call if terminated/truncated.
        if self.current_step < len(self.order_book_history) - 1:
             self._update_step_state() # This increments self.current_step internally
             step_num_log = self.current_step -1 # Log state *before* increment
        else:
             step_num_log = self.current_step # Already at the end
             logging.debug("Reached end of data, not advancing market state further in this step.")


        # Log final state summary (before potential reset on next cycle)
        final_inv_log = self.long_position - self.short_position
        log_mtm = info.get('mtm', 0.0)
        log_cash = info.get('cash', 0.0)
        log_long_cost = info.get('long_avg_cost', 0.0)
        log_short_cost = info.get('short_avg_cost', 0.0)
        term_trunc_flag = f" (Term={terminated}, Trunc={truncated})" if terminated or truncated else ""
        logging.debug(f"Step {step_num_log} End State{term_trunc_flag} - MTM: {log_mtm:.2f}, Inv: {final_inv_log:.4f}, Cash: {log_cash:.2f}, Long@{log_long_cost:.2f}, Short@{log_short_cost:.2f}")

        # --- Observation Validation (Final Check) ---
        if not self.observation_space.contains(observation):
             logging.error(f"Step {self.current_step}: Observation is outside defined space bounds! Shape: {observation.shape}. Attempting to clip/fix.")
             low_bounds = np.where(np.isneginf(self.observation_space.low), -1e10, self.observation_space.low)
             high_bounds = np.where(np.isposinf(self.observation_space.high), 1e10, self.observation_space.high)
             observation = np.clip(observation, low_bounds, high_bounds)
             observation = np.nan_to_num(observation, nan=0.0, posinf=high_bounds[0], neginf=low_bounds[0]) # Use first bound element as example limit
             logging.warning(f"Observation clipped and NaNs/Infs replaced. Final shape: {observation.shape}")
             if observation.shape != self.observation_space.shape:
                  logging.critical(f"Observation SHAPE still incorrect after fixing! Expected {self.observation_space.shape}, Got {observation.shape}. Returning error state.")
                  return self._handle_step_error("Observation shape mismatch after fix")

        return observation, reward, terminated, truncated, info

    def _handle_step_error(self, error_msg="Unknown error during step"):
         """Handles errors during step, returning a valid but neutral state."""
         logging.error(f"Error during step processing: {error_msg}. Returning zero reward and current observation.")
         # Try to get current observation, default to zeros if impossible
         try:
             obs = self._get_observation()
         except Exception as e:
             logging.error(f"Failed to get observation during error handling: {e}. Returning zeros.")
             obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

         # Try to get current info, default to basic dict if impossible
         try:
             info = self._get_info()
         except Exception as e:
             logging.error(f"Failed to get info during error handling: {e}. Returning basic info.")
             # Ensure essential keys exist even in basic info
             info = {
                 'mtm': self.cash + (self.long_position - self.short_position) * (self._last_valid_midprice if not math.isnan(self._last_valid_midprice) else 1.0),
                 'cash': self.cash,
                 'net_inventory': self.inventory,
                 'long_position': self.long_position,
                 'short_position': self.short_position,
                 'long_avg_cost': self.long_avg_cost,
                 'short_avg_cost': self.short_avg_cost,
                 'current_step': self.current_step,
                 'total_steps_elapsed': self.total_steps_elapsed_in_episode
                 # Add other essential keys expected by the user if possible
             }
         info["step_error"] = True
         info["error_message"] = error_msg

         # Check termination/truncation based on current state
         terminated_cash = self.cash < -1e-9
         terminated_inv = abs(self.inventory) > self.config["max_inventory"] + 1e-9
         # Check if current_step exists before accessing history length
         terminated_data = hasattr(self, 'current_step') and hasattr(self, 'order_book_history') and self.current_step >= len(self.order_book_history) - 1

         terminated = terminated_cash or terminated_inv or terminated_data

         max_ep_len = self.config.get('episode_length', self.max_steps)
         truncated = hasattr(self, 'total_steps_elapsed_in_episode') and self.total_steps_elapsed_in_episode >= max_ep_len
         truncated = truncated and not terminated # Only truncate if not terminated

         # Return neutral reward, current state, and error flag
         return obs, 0.0, terminated, truncated, info

    # --- Methods from _place_single_order to _add_to_short remain unchanged ---
    # (No modifications needed in these helper functions for the carry-over logic)
    def _place_single_order(self, is_buy, price_offset_action, volume_scaled):
        """Places a single order using price_offset_ticks for range,
           clipped by allowed_aggressiveness_ticks for maximum aggression. Adds to pending.
           Order dictionary now includes placeholder for 'is_taker_at_activation'.
        """
        # --- Calculate Volume ---
        max_vol = self.config["max_order_volume"]
        lot_size = self.config["lot_size"]
        volume = volume_scaled * max_vol
        # Ensure volume is a multiple of lot size and non-negative
        volume = max(0.0, math.floor(volume / lot_size + 1e-9) * lot_size) # Add tolerance for floor
        if volume <= 1e-9:
             logging.debug("Placement skipped: Volume is effectively zero after lot size rounding.")
             return False # No volume, placement fails

        # --- Determine Price ---
        max_passive_offset_ticks = self.config["price_offset_ticks"]
        max_aggressive_placement_ticks = self.config["allowed_aggressiveness_ticks"]
        tick_size = self.config["tick_size"]

        # Get reliable BBO or fallback based on last valid midprice
        midprice_fallback = self._last_valid_midprice if not math.isnan(self._last_valid_midprice) else 1.0 # Absolute fallback
        best_bid_eff = self.best_bid if not math.isnan(self.best_bid) else midprice_fallback - tick_size / 2.0
        best_ask_eff = self.best_ask if not math.isnan(self.best_ask) else midprice_fallback + tick_size / 2.0

        # Ensure effective BBO is valid and not crossed for calculations
        if math.isnan(best_bid_eff) or math.isnan(best_ask_eff) or best_bid_eff <= 0 or best_ask_eff <= 0:
             logging.error("_place_single_order: Cannot get valid BBO or fallback. Placement failed.")
             return False
        if best_bid_eff >= best_ask_eff - 1e-9: # Locked/crossed book
             mid = (best_bid_eff + best_ask_eff) / 2.0
             best_bid_ref_calc, best_ask_ref_calc = mid, mid # Use midprice as reference for offset
             clip_bid_ref, clip_ask_ref = best_bid_eff, best_ask_eff # Use actual BBO for clipping limit
             logging.warning(f"_place_single_order: Book crossed/locked. Using Mid ({mid:.2f}) for offset calculation.")
        else:
             best_bid_ref_calc, best_ask_ref_calc = best_bid_eff, best_ask_eff # Use BBO for offset
             clip_bid_ref, clip_ask_ref = best_bid_eff, best_ask_eff # Use BBO for clipping limit


        target_limit_price = 0.0
        calculated_tick_offset = 0.0
        action_type = ""

        # Calculate target price based on action and reference prices
        if is_buy:
            # Action range [-1, 1] maps to offset ticks [-max_passive, +max_passive] relative to BEST BID
            calculated_tick_offset = price_offset_action * max_passive_offset_ticks
            target_limit_price = best_bid_ref_calc + calculated_tick_offset * tick_size
            action_type = "BUY"
        else: # is_sell
            # Action range [-1, 1] maps to offset ticks [-max_passive, +max_passive] relative to BEST ASK
            # Sell: Target = RefAsk + OffsetTicks * TickSize (Higher signal = higher sell price / less aggressive)
            calculated_tick_offset = price_offset_action * max_passive_offset_ticks
            target_limit_price = best_ask_ref_calc + calculated_tick_offset * tick_size
            action_type = "SELL"

        # --- Aggressiveness Clipping ---
        # Max buy price allowed: Best Ask + Aggressiveness Ticks
        max_allowed_aggressive_buy_price = clip_ask_ref + max_aggressive_placement_ticks * tick_size
        # Min sell price allowed: Best Bid - Aggressiveness Ticks (but not below tick_size)
        min_allowed_aggressive_sell_price = max(tick_size, clip_bid_ref - max_aggressive_placement_ticks * tick_size)

        clipped_limit_price = target_limit_price
        is_clipped = False
        if is_buy and target_limit_price > max_allowed_aggressive_buy_price:
            clipped_limit_price = max_allowed_aggressive_buy_price
            is_clipped = True
            logging.debug(f"  Buy price clipped by aggressiveness. Target={target_limit_price:.2f}, ClipLimit={max_allowed_aggressive_buy_price:.2f}")
        elif not is_buy and target_limit_price < min_allowed_aggressive_sell_price:
            clipped_limit_price = min_allowed_aggressive_sell_price
            is_clipped = True
            logging.debug(f"  Sell price clipped by aggressiveness. Target={target_limit_price:.2f}, ClipLimit={min_allowed_aggressive_sell_price:.2f}")

        # Round final price to nearest tick, ensuring it's positive
        final_limit_price = max(tick_size, self._round_to_tick(clipped_limit_price))

        clip_msg = f" (Clipped from {target_limit_price:.2f})" if is_clipped else ""
        logging.debug(f"Placing {action_type}: Vol={volume:.4f}, Action={price_offset_action:.3f} -> OffsetTicks={calculated_tick_offset:.2f} -> Final={final_limit_price:.2f}{clip_msg}")

        # --- Inventory Limit Check ---
        current_net_inventory = self.long_position - self.short_position
        potential_inventory_change = volume if is_buy else -volume
        max_inv = self.config["max_inventory"]

        if is_buy and current_net_inventory + potential_inventory_change > max_inv + 1e-9:
             logging.debug(f"Placement blocked: Buy order for {volume:.4f} would exceed max inventory {max_inv:.4f} (Current: {current_net_inventory:.4f}).")
             return False # Placement fails due to inventory limit
        if not is_buy and current_net_inventory + potential_inventory_change < -max_inv - 1e-9:
             logging.debug(f"Placement blocked: Sell order for {volume:.4f} would exceed max inventory {-max_inv:.4f} (Current: {current_net_inventory:.4f}).")
             return False # Placement fails due to inventory limit

        # --- Add to Pending Orders ---
        latency = self.config["latency_steps_long"] if is_buy else self.config["latency_steps_short"]
        # Order becomes active at the START of the target step
        target_step = self.current_step + latency + 1 # +1 because current_step increments AFTER this step logic

        order = {
            "id": self.order_id_counter,
            "price": final_limit_price,
            "volume": volume,           # Current remaining volume
            "initial_volume": volume,   # Volume when placed (for rewards/penalties)
            "is_buy": is_buy,
            "timestamp_placed": self.current_step,
            "target_step": target_step, # Step when it should become active
            "is_taker_at_activation": None # *** ADDED: Placeholder, will be set in _process_pending_orders ***
        }
        self.order_id_counter += 1
        self.pending_orders.append(order)
        logging.debug(f"  Added Pending Order {order['id']}: {'Buy' if is_buy else 'Sell'} {volume:.4f} @ {final_limit_price:.2f}. Target Step: {target_step}")
        return True # Placement successful

    def _prune_oldest_orders(self):
        """If total orders exceed max_active_orders, remove the oldest by timestamp (FIFO)
           considering both pending and active orders."""
        max_orders = self.config["max_active_orders"]
        total_orders = len(self.active_orders) + len(self.pending_orders)
        num_to_prune = total_orders - max_orders

        if num_to_prune <= 0:
            return # No pruning needed

        logging.warning(f"Order limit {max_orders} exceeded ({total_orders} total). Pruning {num_to_prune} oldest orders (FIFO).") # Changed to warning

        # Combine orders with their source and timestamp for sorting
        combined_orders = []
        for order in self.pending_orders:
            # Pending orders use their placement timestamp
            combined_orders.append({'order': order, 'source': 'pending', 'ts': order['timestamp_placed']})
        for order in self.active_orders:
            # Active orders also use their original placement timestamp
            ts = order.get('timestamp_placed', -1) # Fallback if missing
            combined_orders.append({'order': order, 'source': 'active', 'ts': ts})

        # Sort by timestamp (oldest first). Handle potential missing timestamps.
        combined_orders.sort(key=lambda x: x['ts'] if x['ts'] >= 0 else float('inf'))

        # Identify the IDs of the oldest orders to prune
        pruned_pending_ids = set()
        pruned_active_ids = set()
        pruned_count = 0
        for item in combined_orders:
            if pruned_count >= num_to_prune:
                break
            order_id = item['order']['id']
            if item['source'] == 'pending':
                pruned_pending_ids.add(order_id)
            else: # source == 'active'
                pruned_active_ids.add(order_id)
            logging.debug(f"  Pruning Order ID {order_id} (From {item['source']}, Timestamp: {item['ts']}) due to capacity limit.") # Changed to warning
            pruned_count += 1

        # Remove the identified orders from the respective lists
        if pruned_pending_ids:
            self.pending_orders = [o for o in self.pending_orders if o['id'] not in pruned_pending_ids]
        if pruned_active_ids:
            self.active_orders = [o for o in self.active_orders if o['id'] not in pruned_active_ids]

        logging.debug(f"Pruning complete. Pending: {len(self.pending_orders)}, Active: {len(self.active_orders)}") # Changed to warning

    def _process_pending_orders(self):
        """Moves matured pending orders to active orders list, respecting max_active_orders limit.
           *** MODIFICATION: Calculates and stores 'is_taker_at_activation' status. ***
        """
        still_pending = []
        activated_count = 0
        discarded_count = 0 # Keep track of discarded orders
        max_active = self.config["max_active_orders"]

        # --- Get BBO at the current step (activation time) for taker check ---
        midprice_fallback = self._last_valid_midprice if not math.isnan(self._last_valid_midprice) else 1.0
        best_bid_activation = self.best_bid if not math.isnan(self.best_bid) else midprice_fallback - self.config['tick_size']/2.0
        best_ask_activation = self.best_ask if not math.isnan(self.best_ask) else midprice_fallback + self.config['tick_size']/2.0
        # Ensure BBO is somewhat valid for checks
        best_bid_activation = max(self.config['tick_size'] / 10.0, best_bid_activation)
        best_ask_activation = max(best_bid_activation + self.config['tick_size'] / 10.0, best_ask_activation)
        logging.debug(f"  BBO for Activation Check (Step {self.current_step}): Bid={best_bid_activation:.2f}, Ask={best_ask_activation:.2f}")

        # Sort pending orders by target step to process earliest first
        self.pending_orders.sort(key=lambda x: x['target_step'])

        for order in self.pending_orders:
            # Check if the order should become active at the beginning of the current step
            if self.current_step >= order["target_step"]:
                # Check if there's space in the active orders list
                if len(self.active_orders) < max_active:
                    # *** Calculate Taker Status at Activation Time ***
                    is_taker_activation = False
                    taker_reason = "Maker"
                    if order["is_buy"] and order["price"] >= best_ask_activation - 1e-9:
                        is_taker_activation = True
                        taker_reason = f"Buy Price {order['price']:.2f} >= Ask@Act {best_ask_activation:.2f}"
                    elif not order["is_buy"] and order["price"] <= best_bid_activation + 1e-9:
                        is_taker_activation = True
                        taker_reason = f"Sell Price {order['price']:.2f} <= Bid@Act {best_bid_activation:.2f}"

                    # Store the calculated status in the order dictionary
                    order["is_taker_at_activation"] = is_taker_activation

                    # Move to active list
                    self.active_orders.append(order)
                    activated_count += 1
                    logging.debug(f"  Activated Order {order['id']} ({'Buy' if order['is_buy'] else 'Sell'} {order['volume']:.4f} @ {order['price']:.2f}). Taker@Activation: {is_taker_activation} ({taker_reason})")

                else:
                    # Max active limit reached, discard the matured pending order
                    logging.warning(f"Order {order['id']} matured but max active limit ({max_active}) reached. Discarding order.")
                    discarded_count += 1
                    # Do not add to still_pending, effectively removing it
            else:
                # Order hasn't matured yet, keep it in pending
                still_pending.append(order)

        if activated_count > 0:
             logging.debug(f"Activated {activated_count} orders from pending. Total active now: {len(self.active_orders)}")
        if discarded_count > 0:
             logging.warning(f"Discarded {discarded_count} matured pending orders due to active order limit ({max_active}).")


        self.pending_orders = still_pending # Update pending list


    def _round_to_tick(self, price):
        """Rounds a price to the nearest tick size."""
        tick_size = self.config["tick_size"]
        if math.isnan(price) or tick_size <= 0:
             logging.warning(f"Attempted to round invalid price {price} or zero/negative tick size.")
             return price # Return original invalid price
        return np.round(price / tick_size) * tick_size


    def _execute_orders(self):
        """Matches active orders against LOB.
           Applies penalty ONLY if the order was flagged as 'taker' when it became active.
           All active orders proceed to matching regardless of the flag.
           Applies transaction costs and updates positions/cash.
           Returns realized PnL from fills (excluding taker penalties)."""
        realized_pnl_from_fills = 0.0
        total_transaction_costs = 0.0
        taker_penalty_applied_total = 0.0
        self.last_executed_volume = 0.0 # Reset volume counter for this step
        orders_to_match = [] # List for ALL active orders proceeding to matching phase

        logging.debug(f"--- Exec Check Step {self.current_step} ---")
        active_orders_summary = [
            f"ID:{o['id']},{'B' if o['is_buy'] else 'S'},P:{o['price']:.2f},V:{o['volume']:.4f},Tkr@Act:{o.get('is_taker_at_activation', 'N/A')}"
            for o in self.active_orders
        ]
        logging.debug(f"Active Orders In ({len(self.active_orders)}): [{'; '.join(active_orders_summary)}]")

        # --- Apply Penalty Based on Stored Flag & Prepare for Matching ---
        for order in self.active_orders:
            is_taker_activation = order.get("is_taker_at_activation", False) # Default to False if somehow missing

            if is_taker_activation:
                penalty = self.config.get("taker_penalty", 0.0) * order["volume"]
                if penalty > 0: # Only apply if penalty is configured and volume > 0
                    self.cash -= penalty # Directly reduce cash by penalty amount
                    taker_penalty_applied_total += penalty
                    logging.debug(f"Order {order['id']} was Taker at Activation. Applying penalty {-penalty:.6f}. Order WILL proceed to matching.")
                else:
                    logging.debug(f"Order {order['id']} was Taker at Activation, but penalty is zero. Order WILL proceed to matching.")
            else:
                logging.debug(f"Order {order['id']} was Maker at Activation. No penalty applied.")

            orders_to_match.append(order)

        if taker_penalty_applied_total > 0:
             logging.debug(f"Total taker penalty applied this step (based on activation status): {-taker_penalty_applied_total:.6f}. Cash impact: {-taker_penalty_applied_total:.2f}")

        # --- Order Matching (Applies to all orders in orders_to_match) ---
        final_active_orders = [] # Use a fresh list for orders remaining after matching
        current_bids = self.bids.copy() if self.bids is not None and self.bids.size > 0 else np.array([])
        current_asks = self.asks.copy() if self.asks is not None and self.asks.size > 0 else np.array([])

        orders_to_match.sort(key=lambda x: (-x['price'] if x['is_buy'] else x['price']))
        logging.debug(f"Sorted {len(orders_to_match)} orders for matching.")

        for order in orders_to_match: # Iterate through the combined list
            order_id = order['id']
            is_buy = order["is_buy"]
            limit_price = order["price"]
            remaining_volume = order["volume"]

            if remaining_volume <= 1e-9:
                continue

            logging.debug(f"Matching Order {order_id}: {'Buy' if is_buy else 'Sell'} {remaining_volume:.4f} @ {limit_price:.2f}")

            levels_to_match = current_asks if is_buy else current_bids
            cost_rate = self.config["transaction_cost_long"] if is_buy else self.config["transaction_cost_short"]
            match_side = 'Asks' if is_buy else 'Bids'

            if levels_to_match.shape[0] == 0:
                 logging.debug(f"  No liquidity on {match_side} side to match against Order {order_id}.")
                 final_active_orders.append(order) # Keep unmatched order
                 continue

            levels_str = ", ".join([f"P:{p:.2f},Q:{q:.4f}" for p, q in levels_to_match[:min(3, levels_to_match.shape[0])]])
            logging.debug(f"  Matching Order {order_id} against {match_side}: [{levels_str}, ...]")

            volume_filled_this_order = 0.0
            order_fully_filled = False

            for level_idx in range(levels_to_match.shape[0]):
                if remaining_volume <= 1e-9: # Order already filled
                    order_fully_filled = True
                    break

                level_price, level_qty = levels_to_match[level_idx]

                if math.isnan(level_price) or math.isnan(level_qty) or level_qty < 1e-9:
                    logging.debug(f"  Skipping {match_side} Level {level_idx}: Invalid price/qty ({level_price:.2f}, {level_qty:.4f}) or zero qty.")
                    continue

                logging.debug(f"  Level {level_idx}: Price={level_price:.2f}, Qty={level_qty:.4f}")

                can_fill_at_level = False
                if is_buy: # Buy order matches if limit price is >= ask level price
                    can_fill_at_level = limit_price >= level_price - 1e-9
                else: # Sell order matches if limit price is <= bid level price
                    can_fill_at_level = limit_price <= level_price + 1e-9

                logging.debug(f"  Can fill at this level? {'YES' if can_fill_at_level else 'NO'}. (OrderLimit={limit_price:.2f}, LevelPrice={level_price:.2f}, IsBuy={is_buy})")

                if can_fill_at_level:
                    fill_price = level_price # Fill occurs at the book level's price
                    fill_qty_possible = min(remaining_volume, level_qty)

                    current_net_inventory = self.long_position - self.short_position
                    inventory_change = fill_qty_possible if is_buy else -fill_qty_possible
                    potential_new_inventory = current_net_inventory + inventory_change
                    max_inv = self.config["max_inventory"]

                    fill_qty = fill_qty_possible # Assume full possible fill initially
                    logging.debug(f"  Fill possible: {fill_qty_possible:.4f}. CurrentInv={current_net_inventory:.4f}, Change={inventory_change:.4f}, PotentialNewInv={potential_new_inventory:.4f}")

                    inventory_limit_hit = False
                    if is_buy and potential_new_inventory > max_inv + 1e-9:
                        fill_qty = max(0.0, max_inv - current_net_inventory) # Fill only up to the limit
                        inventory_limit_hit = True
                    elif not is_buy and potential_new_inventory < -max_inv - 1e-9:
                        fill_qty = max(0.0, current_net_inventory - (-max_inv)) # Fill only down to the limit (e.g. sell from -4 to -5 max = 1)
                        inventory_limit_hit = True

                    if inventory_limit_hit:
                         if fill_qty < fill_qty_possible - 1e-9: # Check if adjustment was significant
                              logging.warning(f"    Fill quantity for Order {order_id} adjusted due to inventory limit ({max_inv:.4f}): {fill_qty_possible:.4f} -> {fill_qty:.4f}.")
                         else: # fill_qty is near zero or zero
                              logging.warning(f"    Fill blocked entirely for Order {order_id} at this level due to inventory limit ({max_inv:.4f}).")
                              break # Stop trying to fill this order at subsequent levels *this step* if limit blocks here

                    if fill_qty > 1e-9:
                        executed_value = fill_qty * fill_price
                        transaction_cost = cost_rate * executed_value
                        self.cash -= transaction_cost
                        total_transaction_costs += transaction_cost

                        if is_buy:
                            self.cash -= executed_value # Buying costs cash
                        else:
                            self.cash += executed_value # Selling provides cash

                        pnl_from_this_fill = self._process_fill(fill_qty, fill_price, is_buy)
                        realized_pnl_from_fills += pnl_from_this_fill

                        remaining_volume -= fill_qty
                        volume_filled_this_order += fill_qty
                        self.last_executed_volume += fill_qty # Accumulate total executed volume for the step

                        levels_to_match[level_idx, 1] = max(0.0, levels_to_match[level_idx, 1] - fill_qty)

                        logging.debug(f"    EXECUTED Fill Order {order_id}: {fill_qty:.4f} @ {fill_price:.2f}. Val={executed_value:.2f}, Cost={transaction_cost:.4f}, FillPNL={pnl_from_this_fill:+.4f}. Cash={self.cash:.2f}. LvlQtyLeft={levels_to_match[level_idx, 1]:.4f}")
                    else:
                         logging.debug(f"    Skipping fill for Order {order_id} at this level (Qty is zero after limits or adjustment).")
                         pass # No fill occurs if adjusted quantity is zero

                else:
                    logging.debug(f"  Order {order_id} cannot fill at {match_side} Level {level_idx} price {level_price:.2f}. Stopping match for this order.")
                    break # Limit price condition not met, won't be met at worse prices

            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume # Update remaining volume in the order dict
                final_active_orders.append(order) # Keep partially filled or unfilled order active
                logging.debug(f"  Order {order_id} remains active. Remaining Vol: {remaining_volume:.4f}")
            else:
                logging.debug(f"  Order {order_id} fully filled or consumed.")

        self.active_orders = final_active_orders

        logging.debug(f"End Exec Check. Realized PNL(fills): {realized_pnl_from_fills:+.4f}. Costs: {total_transaction_costs:.4f}. TakerPen Applied: {-taker_penalty_applied_total:.4f}. Exec Vol: {self.last_executed_volume:.4f}. Rem Act Ord: {len(self.active_orders)}")
        logging.debug(f"--- End Exec Check Step {self.current_step} ---")

        return realized_pnl_from_fills

    def _process_fill(self, fill_qty, price, is_buy):
        """Updates long/short positions based on a fill and calculates realized PnL for that specific fill."""
        realized_pnl = 0.0
        if is_buy:
            if self.short_position > 1e-9:
                cover_qty = min(fill_qty, self.short_position)
                if self.short_avg_cost > 0: # Ensure avg cost is valid before PnL calc
                     realized_pnl += (self.short_avg_cost - price) * cover_qty
                else:
                     logging.warning(f"Calculating short cover PNL with zero avg cost for qty {cover_qty:.4f}.")

                self.short_position -= cover_qty
                logging.debug(f"  Fill covered {cover_qty:.4f} short. Fill PNL: {realized_pnl:+.4f}. Short Pos Left: {self.short_position:.4f}")

                if self.short_position <= 1e-9:
                    self.short_position = 0.0
                    self.short_avg_cost = 0.0

                remaining_fill_to_long = fill_qty - cover_qty
                if remaining_fill_to_long > 1e-9:
                    self._add_to_long(remaining_fill_to_long, price)
            else:
                self._add_to_long(fill_qty, price)
        else: # is_sell
            if self.long_position > 1e-9:
                close_qty = min(fill_qty, self.long_position)
                if self.long_avg_cost > 0: # Ensure avg cost is valid
                    realized_pnl += (price - self.long_avg_cost) * close_qty
                else:
                     logging.warning(f"Calculating long close PNL with zero avg cost for qty {close_qty:.4f}.")

                self.long_position -= close_qty
                logging.debug(f"  Fill closed {close_qty:.4f} long. Fill PNL: {realized_pnl:+.4f}. Long Pos Left: {self.long_position:.4f}")

                if self.long_position <= 1e-9:
                    self.long_position = 0.0
                    self.long_avg_cost = 0.0

                remaining_fill_to_short = fill_qty - close_qty
                if remaining_fill_to_short > 1e-9:
                    self._add_to_short(remaining_fill_to_short, price)
            else:
                self._add_to_short(fill_qty, price)

        self.inventory = self.long_position - self.short_position
        return realized_pnl # Return PnL generated *by this specific fill*

    def _add_to_long(self, qty, price):
        """Helper to add to long position and update average cost."""
        if qty <= 1e-9: return
        new_total_value = (self.long_avg_cost * self.long_position) + (price * qty)
        new_total_qty = self.long_position + qty
        if new_total_qty <= 1e-9: # Avoid division by zero if adding tiny/negative qty somehow
            self.long_avg_cost = 0.0
            self.long_position = 0.0
            logging.warning("_add_to_long: Resulting quantity is near zero. Resetting long position.")
        else:
            self.long_avg_cost = new_total_value / new_total_qty
            self.long_position = new_total_qty
        logging.debug(f"  Added {qty:.4f} to long @ {price:.2f}. New Long: {self.long_position:.4f}, Avg Cost: {self.long_avg_cost:.2f}")

    def _add_to_short(self, qty, price):
        """Helper to add to short position and update average cost."""
        if qty <= 1e-9: return
        new_total_value = (self.short_avg_cost * self.short_position) + (price * qty)
        new_total_qty = self.short_position + qty
        if new_total_qty <= 1e-9:
            self.short_avg_cost = 0.0
            self.short_position = 0.0
            logging.warning("_add_to_short: Resulting quantity is near zero. Resetting short position.")
        else:
            self.short_avg_cost = new_total_value / new_total_qty
            self.short_position = new_total_qty
        logging.debug(f"  Added {qty:.4f} to short @ {price:.2f}. New Short: {self.short_position:.4f}, Avg Cost: {self.short_avg_cost:.2f}")


    # --- Methods _calculate_risk_penalty, _calculate_quoting_reward remain unchanged ---
    def _calculate_risk_penalty(self):
        """Calculates inventory risk penalty based on squared normalized inventory and midprice."""
        inv_penalty_factor = self.config.get("inventory_penalty", 0.0)
        max_inv = self.config.get("max_inventory", 1.0) # Use 1.0 as fallback if not configured

        if inv_penalty_factor <= 1e-9 or max_inv <= 1e-9:
             return 0.0

        net_inventory = self.long_position - self.short_position
        normalized_inventory = net_inventory / max_inv
        inventory_risk_component = normalized_inventory ** 2

        midprice_val = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
        if math.isnan(midprice_val) or midprice_val <= 0:
             midprice_val = 1.0 # Fallback to avoid zero/negative scaling
             logging.debug("Using fallback midprice (1.0) for risk penalty calculation.")

        penalty = inv_penalty_factor * inventory_risk_component * midprice_val
        return max(0.0, penalty)

    def _calculate_quoting_reward(self):
        """Calculates a reward for maintaining valid maker orders near the BBO.
           *** MODIFICATION: Considers the 'is_taker_at_activation' flag. Only rewards orders that were makers at activation. ***
        """
        if not self.config.get("quoting_reward_enabled", False) or not self.active_orders:
            return 0.0 # No reward if disabled or no active orders

        quoting_reward_total = 0.0
        reward_amount = self.config["quoting_reward_amount"]
        max_ticks = self.config["quoting_reward_max_ticks"]
        tick_size = self.config["tick_size"]
        max_dist_from_bbo = max_ticks * tick_size

        best_bid_eff = self.best_bid if not math.isnan(self.best_bid) else -np.inf
        best_ask_eff = self.best_ask if not math.isnan(self.best_ask) else np.inf

        if best_bid_eff == -np.inf or best_ask_eff == np.inf or best_bid_eff >= best_ask_eff - 1e-9:
            logging.debug("Quoting reward skipped: Invalid or crossed/locked BBO.")
            return 0.0

        best_active_maker_buy_price = -np.inf
        best_active_maker_sell_price = np.inf
        active_maker_buy_order_present = False
        active_maker_sell_order_present = False

        for order in self.active_orders:
            if order['volume'] > 1e-9 and not order.get("is_taker_at_activation", True): # Default to True (taker) if flag missing
                if order["is_buy"]:
                    best_active_maker_buy_price = max(best_active_maker_buy_price, order["price"])
                    active_maker_buy_order_present = True
                else:
                    best_active_maker_sell_price = min(best_active_maker_sell_price, order["price"])
                    active_maker_sell_order_present = True

        buy_rewarded = False
        if active_maker_buy_order_present:
            is_buy_maker_now = best_active_maker_buy_price < best_ask_eff - 1e-9
            is_buy_close_now = (best_bid_eff - best_active_maker_buy_price) <= max_dist_from_bbo + 1e-9
            is_buy_at_or_below_bid_now = best_active_maker_buy_price <= best_bid_eff + 1e-9

            if is_buy_maker_now and is_buy_close_now and is_buy_at_or_below_bid_now:
                quoting_reward_total += reward_amount
                buy_rewarded = True
                logging.debug(f"  Quoting reward applied for Buy side (Maker@Act, Price: {best_active_maker_buy_price:.2f})")
            else:
                 fail_reasons = ["WasMaker@Act"] # Start assuming it passed the activation check
                 if not is_buy_maker_now: fail_reasons.append(f"NotMakerNow (P={best_active_maker_buy_price:.2f} vs Ask={best_ask_eff:.2f})")
                 if not is_buy_close_now: fail_reasons.append(f"NotCloseNow (Dist={(best_bid_eff - best_active_maker_buy_price):.2f} > Max={max_dist_from_bbo:.2f})")
                 if not is_buy_at_or_below_bid_now: fail_reasons.append(f"NotPassiveNow (P={best_active_maker_buy_price:.2f} > Bid={best_bid_eff:.2f})")
                 logging.debug(f"  Buy Quote Failed: {', '.join(fail_reasons)}")

        sell_rewarded = False
        if active_maker_sell_order_present:
            is_sell_maker_now = best_active_maker_sell_price > best_bid_eff + 1e-9
            is_sell_close_now = (best_active_maker_sell_price - best_ask_eff) <= max_dist_from_bbo + 1e-9
            is_sell_at_or_above_ask_now = best_active_maker_sell_price >= best_ask_eff - 1e-9

            if is_sell_maker_now and is_sell_close_now and is_sell_at_or_above_ask_now:
                quoting_reward_total += reward_amount
                sell_rewarded = True
                logging.debug(f"  Quoting reward applied for Sell side (Maker@Act, Price: {best_active_maker_sell_price:.2f})")
            else:
                 fail_reasons = ["WasMaker@Act"]
                 if not is_sell_maker_now: fail_reasons.append(f"NotMakerNow (P={best_active_maker_sell_price:.2f} vs Bid={best_bid_eff:.2f})")
                 if not is_sell_close_now: fail_reasons.append(f"NotCloseNow (Dist={(best_active_maker_sell_price - best_ask_eff):.2f} > Max={max_dist_from_bbo:.2f})")
                 if not is_sell_at_or_above_ask_now: fail_reasons.append(f"NotPassiveNow (P={best_active_maker_sell_price:.2f} < Ask={best_ask_eff:.2f})")
                 logging.debug(f"  Sell Quote Failed: {', '.join(fail_reasons)}")


        if quoting_reward_total > 0:
             logging.debug(f"  Total quoting reward this step: {quoting_reward_total:.6f} (Buy OK: {buy_rewarded}, Sell OK: {sell_rewarded})")

        return quoting_reward_total


    def _update_step_state(self) -> None:
        """Advances to the next step in the order book history and updates market state."""
        # This method now only advances the step and loads data.
        # The check for data end is implicitly handled by the caller (step function)
        # and the reset logic.
        next_step_index = self.current_step + 1
        if next_step_index < len(self.order_book_history):
            self.current_step = next_step_index # Increment step counter
            try:
                current_lob_data = self.order_book_history[self.current_step]
                # Validate data format before assigning
                if isinstance(current_lob_data.get("bids"), np.ndarray) and \
                   isinstance(current_lob_data.get("asks"), np.ndarray) and \
                   current_lob_data["bids"].ndim == 2 and current_lob_data["asks"].ndim == 2 and \
                   current_lob_data["bids"].shape[1] == 2 and current_lob_data["asks"].shape[1] == 2:

                    self.bids = current_lob_data["bids"].copy()
                    self.asks = current_lob_data["asks"].copy()
                    self._update_market_state() # Update BBO, midprice, spread based on new data
                else:
                     logging.error(f"Invalid data format encountered at step {self.current_step}. Bids/Asks are not valid 2D numpy arrays with shape (levels, 2).")
                     # Attempt to update market state using fallback mechanisms
                     self._update_market_state()
            except IndexError:
                 # This should ideally not happen if the initial check is correct, but handle defensively.
                 logging.error(f"Attempted to access index {self.current_step} which is out of bounds for order_book_history (len={len(self.order_book_history)}) during state update.")
                 self._update_market_state() # Use fallback
            except Exception as e:
                 logging.error(f"Unexpected error updating step state at step {self.current_step}: {e}")
                 self._update_market_state() # Use fallback
        else:
             # Log that we are at the end, but don't change current_step here.
             # The termination logic in step() and reset() handles what happens next.
             logging.debug(f"Reached end of data (Step {self.current_step}). No further LOB update.")
             # Market state (BBO, midprice) remains as it was at the last valid step unless updated by fallback.
             self._update_market_state()


    # --- Methods _get_raw_observation, _get_observation, _get_info remain unchanged ---
    def _get_raw_observation(self):
        """
        Constructs the raw observation vector with revised normalization and scaling.
        (No changes needed here due to the taker flag implementation)
        """
        # --- Safe References & Normalization Factors ---
        tick_size = self.config["tick_size"]
        midprice_safe = self.midprice if (not math.isnan(self.midprice) and self.midprice > 0) else self._last_valid_midprice
        # Absolute fallback if no valid midprice ever seen
        if math.isnan(midprice_safe) or midprice_safe <= 0:
            midprice_safe = 1.0
            logging.warning(f"Step {self.current_step}: Midprice fallback (1.0) used for observation normalization.")

        # Ensure tick_size is valid for division
        if tick_size <= 1e-9:
             logging.error("Tick size is zero or negative, cannot normalize prices by tick. Returning zeros.")
             # Ensure the returned array has the correct shape
             return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Normalization references
        norm_ref_price_tick = tick_size
        norm_ref_qty = max(1e-9, self.config["max_order_volume"]) # Avoid zero division
        norm_ref_inv = max(1e-9, self.config["max_inventory"])
        norm_ref_mtm = max(1.0, self.config["initial_capital"]) # Use 1.0 as minimum
        # Added scaling factors from config
        obs_price_scale = self.config["obs_price_norm_scale"]
        obs_qty_scale = self.config["obs_qty_norm_scale"]

        # Get safe BBO using the safe midprice as fallback reference
        best_bid_safe = self.best_bid if not math.isnan(self.best_bid) else midprice_safe - norm_ref_price_tick / 2.0
        best_ask_safe = self.best_ask if not math.isnan(self.best_ask) else midprice_safe + norm_ref_price_tick / 2.0

        # Ensure BBO are positive and spread is sensible
        best_bid_safe = max(norm_ref_price_tick / 10.0, best_bid_safe) # Min bid slightly positive
        best_ask_safe = max(best_bid_safe + norm_ref_price_tick / 10.0, best_ask_safe) # Ensure ask > bid
        spread_safe = best_ask_safe - best_bid_safe

        # --- Portfolio Features ---
        current_inventory = self.long_position - self.short_position
        # Calculate MTM using the safe midprice
        inventory_value = current_inventory * midprice_safe
        mtm = self.cash + inventory_value

        mtm_norm = mtm / norm_ref_mtm
        inventory_norm = np.clip(current_inventory / norm_ref_inv, -1.5, 1.5) # Clip potential outliers
        active_orders_count_norm = len(self.active_orders) / max(1.0, self.config["max_active_orders"]) # Normalize by max allowed
        long_pos_norm = np.clip(self.long_position / norm_ref_inv, 0.0, 1.5)
        short_pos_norm = np.clip(self.short_position / norm_ref_inv, 0.0, 1.5)

        # Normalize avg costs as ticks away from safe midprice, then scale
        default_cost_norm_scaled = 0.0 # Default is zero scaled value
        price_clip_ticks = 500 # Clip raw tick deviations before scaling

        long_avg_cost_norm_scaled = default_cost_norm_scaled
        if self.long_position > 1e-9 and self.long_avg_cost > 0:
            raw_ticks_dev = (self.long_avg_cost - midprice_safe) / norm_ref_price_tick
            clipped_ticks_dev = np.clip(raw_ticks_dev, -price_clip_ticks, price_clip_ticks)
            long_avg_cost_norm_scaled = clipped_ticks_dev / obs_price_scale # Apply scaling

        short_avg_cost_norm_scaled = default_cost_norm_scaled
        if self.short_position > 1e-9 and self.short_avg_cost > 0:
            raw_ticks_dev = (self.short_avg_cost - midprice_safe) / norm_ref_price_tick
            clipped_ticks_dev = np.clip(raw_ticks_dev, -price_clip_ticks, price_clip_ticks)
            short_avg_cost_norm_scaled = clipped_ticks_dev / obs_price_scale # Apply scaling

        # Normalize BBO deviation as ticks away from safe midprice, then scale
        bid_dev_norm_ticks_scaled = ((best_bid_safe - midprice_safe) / norm_ref_price_tick) / obs_price_scale
        ask_dev_norm_ticks_scaled = ((best_ask_safe - midprice_safe) / norm_ref_price_tick) / obs_price_scale

        portfolio_features = [
            mtm_norm, inventory_norm, active_orders_count_norm,
            long_pos_norm, short_pos_norm, long_avg_cost_norm_scaled, short_avg_cost_norm_scaled,
            bid_dev_norm_ticks_scaled, ask_dev_norm_ticks_scaled
        ]

        # --- Active Order Features ---
        order_prices_norm_scaled = []
        order_volumes_norm_scaled = []
        # Sort orders? No, take first max_active_orders as they are for simplicity
        active_orders_to_encode = self.active_orders[:self.config["max_active_orders"]]

        vol_clip = 1.5 # Clip normalized *raw* order volume before scaling

        for order in active_orders_to_encode:
            # Normalize price as ticks from midprice, clip, then scale
            raw_price_ticks_dev = (order["price"] - midprice_safe) / norm_ref_price_tick
            clipped_price_ticks_dev = np.clip(raw_price_ticks_dev, -price_clip_ticks, price_clip_ticks)
            price_norm_scaled = clipped_price_ticks_dev / obs_price_scale
            order_prices_norm_scaled.append(price_norm_scaled)

            # Normalize volume relative to max order size, clip, then scale
            raw_vol_norm = (order["volume"] / norm_ref_qty) * (1 if order['is_buy'] else -1)
            clipped_vol_norm = np.clip(raw_vol_norm, -vol_clip, vol_clip)
            vol_norm_scaled = clipped_vol_norm * obs_qty_scale # Apply scaling
            order_volumes_norm_scaled.append(vol_norm_scaled)

        # Pad if fewer orders than max_active_orders
        padding_count = self.config["max_active_orders"] - len(active_orders_to_encode)
        order_prices_norm_scaled.extend([0.0] * padding_count) # Pad scaled prices with 0
        order_volumes_norm_scaled.extend([0.0] * padding_count) # Pad scaled volumes with 0

        # --- Order Book Features ---
        book_features = []
        levels = self.config["order_book_levels"]
        bids_safe = self.bids if self.bids is not None and self.bids.ndim == 2 and self.bids.shape[1] == 2 else np.empty((0,2))
        asks_safe = self.asks if self.asks is not None and self.asks.ndim == 2 and self.asks.shape[1] == 2 else np.empty((0,2))
        bids_len = bids_safe.shape[0]
        asks_len = asks_safe.shape[0]

        qty_clip = 5.0 # Clip raw normalized book qty before scaling

        for level in range(levels):
            bid_price_norm_scaled, bid_qty_norm_scaled = 0.0, 0.0
            ask_price_norm_scaled, ask_qty_norm_scaled = 0.0, 0.0

            # Process bid side
            if level < bids_len:
                bid_price, bid_qty = bids_safe[level, 0], bids_safe[level, 1]
                # Normalize, clip price ticks, then scale
                if not math.isnan(bid_price) and bid_price > 0:
                    raw_bid_ticks = (bid_price - midprice_safe) / norm_ref_price_tick
                    clipped_bid_ticks = np.clip(raw_bid_ticks, -price_clip_ticks, price_clip_ticks)
                    bid_price_norm_scaled = clipped_bid_ticks / obs_price_scale
                # Normalize, clip qty, then scale
                if not math.isnan(bid_qty) and bid_qty >= 0:
                     raw_bid_qty_norm = bid_qty / norm_ref_qty
                     clipped_bid_qty_norm = np.clip(raw_bid_qty_norm, 0.0, qty_clip)
                     bid_qty_norm_scaled = clipped_bid_qty_norm * obs_qty_scale

            # Process ask side
            if level < asks_len:
                ask_price, ask_qty = asks_safe[level, 0], asks_safe[level, 1]
                # Normalize, clip price ticks, then scale
                if not math.isnan(ask_price) and ask_price > 0:
                    raw_ask_ticks = (ask_price - midprice_safe) / norm_ref_price_tick
                    clipped_ask_ticks = np.clip(raw_ask_ticks, -price_clip_ticks, price_clip_ticks)
                    ask_price_norm_scaled = clipped_ask_ticks / obs_price_scale
                # Normalize, clip qty, then scale
                if not math.isnan(ask_qty) and ask_qty >= 0:
                    raw_ask_qty_norm = ask_qty / norm_ref_qty
                    clipped_ask_qty_norm = np.clip(raw_ask_qty_norm, 0.0, qty_clip)
                    ask_qty_norm_scaled = clipped_ask_qty_norm * obs_qty_scale

            book_features.extend([bid_price_norm_scaled, bid_qty_norm_scaled, ask_price_norm_scaled, ask_qty_norm_scaled])

        # --- Market Features ---
        # Normalize spread by tick size (spread in ticks) - NO additional scaling applied here by default
        spread_norm_ticks = max(0.0, spread_safe / norm_ref_price_tick) # Ensure non-negative

        # --- Combine All Features ---
        raw_obs_list = (portfolio_features +
                        order_prices_norm_scaled + order_volumes_norm_scaled +
                        book_features +
                        [spread_norm_ticks])
        raw_obs_array = np.array(raw_obs_list, dtype=np.float32)

        # --- Final Validation and Cleanup ---
        if not np.all(np.isfinite(raw_obs_array)):
             logging.warning(f"Step {self.current_step}: Observation contains NaN/Inf values AFTER normalization/scaling. Replacing with 0.")
             raw_obs_array = np.nan_to_num(raw_obs_array, nan=0.0, posinf=0.0, neginf=0.0)

        expected_shape = self.observation_space.shape
        if raw_obs_array.shape != expected_shape:
             logging.critical(f"Observation shape mismatch! Expected {expected_shape}, got {raw_obs_array.shape}. This indicates a bug in observation construction. Padding/truncating as fallback.")
             current_len = raw_obs_array.shape[0]
             target_len = expected_shape[0]
             if current_len > target_len:
                 raw_obs_array = raw_obs_array[:target_len] # Truncate
             elif current_len < target_len:
                 padding = np.zeros(target_len - current_len, dtype=np.float32)
                 raw_obs_array = np.concatenate((raw_obs_array, padding)) # Pad with zeros
             logging.critical(f"Shape corrected to {raw_obs_array.shape}")

        return raw_obs_array

    def _get_observation(self):
        """Returns the current observation using the revised normalization and scaling."""
        obs = self._get_raw_observation()
        return obs

    def _get_info(self):
        """Returns dictionary with auxiliary information about the state."""
        midprice_safe = self.midprice if not math.isnan(self.midprice) else self._last_valid_midprice
        if math.isnan(midprice_safe) or midprice_safe <= 0: midprice_safe = 0.0 # Use 0.0 for info calc if truly unknown

        mtm = self.cash + (self.long_position - self.short_position) * midprice_safe
        info = {
            "mtm": mtm,
            "cash": self.cash,
            "net_inventory": self.long_position - self.short_position,
            "long_position": self.long_position,
            "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost if self.long_position > 1e-9 else 0.0,
            "short_avg_cost": self.short_avg_cost if self.short_position > 1e-9 else 0.0,
            "active_orders_count": len(self.active_orders),
            "pending_orders_count": len(self.pending_orders),
            "best_bid": self.best_bid if not math.isnan(self.best_bid) else None,
            "best_ask": self.best_ask if not math.isnan(self.best_ask) else None,
            "mid_price": self.midprice if not math.isnan(self.midprice) else None, # Report actual midprice used
            "last_valid_mid_price": self._last_valid_midprice if not math.isnan(self._last_valid_midprice) else None, # Report fallback
            "spread": self.spread if not math.isnan(self.spread) else None,
            "current_step": self.current_step, # Step within the data history slice
            "total_steps_elapsed": self.total_steps_elapsed_in_episode, # Steps within current episode
            # Default values, updated in step() before returning
            "quoting_reward_step": 0.0,
            "explicit_cancel_triggered": False,
            "do_nothing_triggered": False,
            "step_error": False,
            "error_message": "" # Store potential error messages
        }
        # Calculate episode PnL relative to the episode's starting capital
        info['episode_pnl'] = mtm - self.config["initial_capital"]
        return info

    def render(self, mode="human"):
        """Renders the current state, typically prints info to console."""
        if mode == "human":
            info = self._get_info() # Get current info dictionary
            bid_str = f"{info['best_bid']:.2f}" if info['best_bid'] is not None else "N/A"
            ask_str = f"{info['best_ask']:.2f}" if info['best_ask'] is not None else "N/A"
            spread_str = f"{info['spread']:.2f}" if info['spread'] is not None else "N/A"
            pnl_str = f"{info.get('episode_pnl', 0.0):,.2f}"
            quote_rew_str = f"{info.get('quoting_reward_step', 0.0):.6f}" # Get from info dict
            do_nothing_str = "DO_NOTHING" if info.get('do_nothing_triggered', False) else ""
            cancel_str = "CANCEL" if info.get('explicit_cancel_triggered', False) else ""
            action_flags = f"({cancel_str}{',' if cancel_str and do_nothing_str else ''}{do_nothing_str})" if cancel_str or do_nothing_str else ""
            error_flag = "ERROR" if info.get('step_error', False) else ""

            render_msg = (
                f"Step: {info['current_step']:<6} EpStep: {info['total_steps_elapsed']:<5} | "
                f"BBO: {bid_str}/{ask_str} (Sprd:{spread_str}) | MTM: {info['mtm']:<11,.2f} | EpPNL: {pnl_str:<9} | " # Added EpPNL label
                f"Inv: {info['net_inventory']:<+8.4f} | Cash: {info['cash']:<10,.2f} | "
                f"Ord(A/P): {info['active_orders_count']}/{info['pending_orders_count']} | "
                f"QuoteRw: {quote_rew_str} {action_flags} {error_flag}"
            )
            # Use logging.info for consistency if logger is configured, otherwise print
            if logging.getLogger().hasHandlers():
                logging.info(render_msg)
            else:
                print(render_msg)
        else:
            return super().render(mode=mode)

    def _liquidate_positions(self):
        """Closes out any open long/short position by executing market orders
           at the current BBO (or estimated BBO). Calculates Net PnL from liquidation
           after accounting for transaction costs.
           THIS METHOD IS NO LONGER CALLED AUTOMATICALLY AT EPISODE END.
           It can be called manually if needed.
           MODIFIED: Does NOT clear active/pending orders automatically anymore, only positions.
        """
        logging.warning("Manual liquidation called. This is NOT part of the standard episode flow anymore.")
        liquidation_pnl_net = 0.0 # Net PnL after costs
        total_liq_cost = 0.0

        tick_size = self.config['tick_size']
        midprice_fallback = self._last_valid_midprice if not math.isnan(self._last_valid_midprice) else 1.0
        best_bid_liq = self.best_bid if not math.isnan(self.best_bid) else midprice_fallback - tick_size / 2.0
        best_ask_liq = self.best_ask if not math.isnan(self.best_ask) else midprice_fallback + tick_size / 2.0

        best_bid_liq = max(tick_size / 10.0, best_bid_liq)
        best_ask_liq = max(best_bid_liq + tick_size / 10.0, best_ask_liq)

        logging.info(f"Using Manual Liquidation Prices: Bid={best_bid_liq:.2f}, Ask={best_ask_liq:.2f}")

        # --- Liquidate Long Position ---
        if self.long_position > 1e-9:
            qty = self.long_position
            price = best_bid_liq # Sell long at the best available bid
            value = qty * price
            cost_rate = self.config["transaction_cost_short"]
            cost = cost_rate * value
            pnl_gross = 0.0
            if self.long_avg_cost > 0:
                pnl_gross = (price - self.long_avg_cost) * qty
            else:
                logging.warning(f"Liquidating long position {qty:.4f} with zero average cost.")
            net_pnl = pnl_gross - cost
            self.cash += value - cost
            liquidation_pnl_net += net_pnl
            total_liq_cost += cost
            logging.info(f"MANUAL Liq LONG: {qty:.4f} @ {price:.2f}. Value={value:.2f}, Cost={cost:.4f}, PNL(Gross)={pnl_gross:.2f}, PNL(Net)={net_pnl:.2f}")
            self.long_position = 0.0
            self.long_avg_cost = 0.0

        # --- Liquidate Short Position ---
        if self.short_position > 1e-9:
            qty = self.short_position
            price = best_ask_liq # Buy short at the best available ask
            value = qty * price
            cost_rate = self.config["transaction_cost_long"]
            cost = cost_rate * value
            pnl_gross = 0.0
            if self.short_avg_cost > 0:
                pnl_gross = (self.short_avg_cost - price) * qty
            else:
                logging.warning(f"Liquidating short position {qty:.4f} with zero average cost.")
            net_pnl = pnl_gross - cost
            self.cash -= (value + cost)
            liquidation_pnl_net += net_pnl
            total_liq_cost += cost
            logging.info(f"MANUAL Liq SHORT: {qty:.4f} @ {price:.2f}. Value={value:.2f}, Cost={cost:.4f}, PNL(Gross)={pnl_gross:.2f}, PNL(Net)={net_pnl:.2f}")
            self.short_position = 0.0
            self.short_avg_cost = 0.0

        # --- Orders are NOT cancelled here automatically ---
        # active_cancel_count = len(self.active_orders)
        # pending_cancel_count = len(self.pending_orders)
        # if active_cancel_count > 0 or pending_cancel_count > 0:
        #      logging.info(f"Cancelling {active_cancel_count} active and {pending_cancel_count} pending orders during liquidation.")
        #      self.active_orders = []
        #      self.pending_orders = []

        # Ensure inventory state is clean after position clearing
        self.inventory = 0.0 # Should be zero after position liquidation
        logging.info(f"Manual Liquidation complete. Total Liq PNL(Net): {liquidation_pnl_net:.2f}. Total Liq Cost: {total_liq_cost:.4f}. Final Cash: {self.cash:.2f}")
        return liquidation_pnl_net # Return the net PnL (after costs) from the liquidation


    def close(self):
        """Performs any necessary cleanup when the environment is closed."""
        logging.info("Closing HFT Environment.")
        # Clear potentially large data structures
        self.order_book_history = []
        self.active_orders = []
        self.pending_orders = []
        # Reset other state variables to default
        self.long_position = 0.0; self.short_position = 0.0
        self.long_avg_cost = 0.0; self.short_avg_cost = 0.0
        self.cash = 0.0; self.inventory = 0.0
        self.order_id_counter = 0; self.current_step = 0
        self.total_steps_elapsed_in_episode = 0
        self._is_first_reset = True # Ensure next use starts fresh
        pass

# Example Usage (if run as main script)
if __name__ == '__main__':
    # --- Configuration ---
    config = {
        "csv_path": "path/to/your/lob_data.csv",  # <<< --- MUST PROVIDE A VALID PATH HERE
        "initial_capital": 100000.0,
        "order_book_levels": 5,
        "max_order_volume": 1.0,
        "latency_steps_long": 1, # Latency for buy orders (in steps)
        "latency_steps_short": 1, # Latency for sell orders (in steps)
        "tick_size": 0.01,
        "lot_size": 0.01,
        "max_active_orders": 10, # Increased max orders slightly for testing carry-over
        "inventory_penalty": 0.001, # Penalty factor for holding inventory
        "transaction_cost_long": 0.0001, # 0.01% cost for buys
        "transaction_cost_short": 0.0001, # 0.01% cost for sells
        "max_inventory": 10.0, # Max absolute inventory allowed
        "invalid_order_penalty": 0.0, # Penalty for placing invalid orders (e.g., blocked by inv)
        "activity_bonus": 0.0, # Bonus per unit volume executed (maker fills)
        "taker_penalty": 0.0005, # Penalty per unit volume for taker orders (applied if taker @ activation)
        "price_offset_ticks": 5, # Action [-1, 1] maps to [-5, +5] ticks offset from BBO reference
        "allowed_aggressiveness_ticks": 2, # Max ticks *beyond* opposite BBO allowed (e.g., buy can be Ask+2ticks)
        "quoting_reward_enabled": True,
        "quoting_reward_amount": 0.00001, # Reward per step for qualifying quotes (small value)
        "quoting_reward_max_ticks": 2, # Max distance (in ticks) from BBO to qualify for reward
        "explicit_cancel_enabled": True,
        "explicit_cancel_threshold": 0.5, # Action[4] > 0.5 triggers cancellation
        "explicit_cancel_penalty": 0.0, # Penalty applied once if explicit cancel is triggered
        "explicit_cancel_clears_pending": True, # Does explicit cancel also clear pending orders?
        "do_nothing_threshold": 0.8, # Action[5] > 0.8 triggers do-nothing override
        "max_steps": 2000, # Max steps to load from data (reduced for faster dummy creation/test)
        "episode_length": 150, # Max steps per episode before truncation (reduced for testing resets)
        # Observation scaling factors
        "obs_price_norm_scale": 100.0, # Divide normalized prices (in ticks) by this factor
        "obs_qty_norm_scale": 1.0,    # Multiply normalized quantities by this factor
    }

    # --- Create Dummy Data if File Doesn't Exist ---
    dummy_file_path = "dummy_lob_data_continuous.csv" # Use different name
    # Use absolute path for checking existence if needed, depends on execution context
    # if not os.path.isabs(config["csv_path"]):
    #     config["csv_path"] = os.path.join(os.getcwd(), config["csv_path"]) # Example: Assume relative to CWD

    if not os.path.exists(config["csv_path"]):
        print(f"Warning: Provided CSV path '{config['csv_path']}' not found. Creating dummy data at '{dummy_file_path}'.")
        config["csv_path"] = dummy_file_path # Update config to use dummy path

        # Create simple dummy data
        num_rows = config["max_steps"] + 200 # Ensure enough data
        timestamps = pd.to_datetime(np.arange(num_rows), unit='s', origin='2023-01-01')
        mid_price = 100.0
        volatility = 0.05 # Some price movement
        drift = 0.0001 # Slight upward drift
        prices = mid_price + np.cumsum(np.random.normal(drift, volatility, num_rows))
        prices = np.maximum(prices, config['tick_size'] * 10) # Ensure price stays reasonably positive

        data = {"timestamp": timestamps}
        for i in range(1, config['order_book_levels'] + 1):
            # Dynamic spread around the moving price
            spread = config['tick_size'] * np.random.uniform(1, 5) # Random spread in ticks
            # Ensure bid < ask
            bid_i = prices - (spread / 2) - (i - 1) * config['tick_size'] * np.random.uniform(0.8, 1.2)
            ask_i = prices + (spread / 2) + (i - 1) * config['tick_size'] * np.random.uniform(0.8, 1.2)
            bid_i = np.maximum(config['tick_size'], bid_i) # Min price is tick_size
            ask_i = np.maximum(bid_i + config['tick_size'], ask_i) # Ensure ask > bid

            data[f'bid{i}'] = np.round(bid_i / config['tick_size']) * config['tick_size']
            data[f'ask{i}'] = np.round(ask_i / config['tick_size']) * config['tick_size']

            # Quantities decreasing away from BBO
            base_qty = config['max_order_volume'] * np.random.uniform(0.5, 3.0)
            data[f'bidqty{i}'] = np.maximum(0.0, np.random.normal(base_qty / (i**0.8), base_qty * 0.2)).round(3)
            data[f'askqty{i}'] = np.maximum(0.0, np.random.normal(base_qty / (i**0.8), base_qty * 0.2)).round(3)


        df_dummy = pd.DataFrame(data)

        # Final check: ensure best bid < best ask strictly
        for idx in range(len(df_dummy)):
            if df_dummy.loc[idx, f'bid1'] >= df_dummy.loc[idx, f'ask1']:
                 mid = (df_dummy.loc[idx, f'bid1'] + df_dummy.loc[idx, f'ask1']) / 2.0
                 df_dummy.loc[idx, f'bid1'] = np.round((mid - config['tick_size']/2.0) / config['tick_size']) * config['tick_size']
                 df_dummy.loc[idx, f'ask1'] = np.round((mid + config['tick_size']/2.0) / config['tick_size']) * config['tick_size']
                 if df_dummy.loc[idx, f'bid1'] >= df_dummy.loc[idx, f'ask1']:
                     df_dummy.loc[idx, f'ask1'] = df_dummy.loc[idx, f'bid1'] + config['tick_size']

        for i in range(1, config['order_book_levels']):
            df_dummy[f'bid{i+1}'] = np.minimum(df_dummy[f'bid{i+1}'], df_dummy[f'bid{i}'] - config['tick_size'])
            df_dummy[f'ask{i+1}'] = np.maximum(df_dummy[f'ask{i+1}'], df_dummy[f'ask{i}'] + config['tick_size'])
            df_dummy[f'bid{i+1}'] = np.maximum(config['tick_size'], df_dummy[f'bid{i+1}'])

        df_dummy.to_csv(dummy_file_path, index=False)
        print(f"Dummy data saved to {dummy_file_path}")

    # --- Environment Initialization ---
    try:
        env = HFTEnv(config)
        print("--- Running Test with Carry-Forward State ---")

        num_episodes = 3
        max_total_steps = config["max_steps"] - 1 # Stop before running out of data

        global_step_count = 0
        for episode in range(num_episodes):
            print(f"\n--- Starting Episode {episode + 1}/{num_episodes} ---")
            obs, info = env.reset()
            print(f"Reset Complete. Data Step: {info['current_step']}, Cash: {info['cash']:.2f}, Inv: {info['net_inventory']:.4f}, ActOrd: {info['active_orders_count']}, PendOrd: {info['pending_orders_count']}")
            # print("Initial Observation Shape:", obs.shape)
            # print("Initial Info:", info)

            terminated = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while not terminated and not truncated and global_step_count < max_total_steps:
                action = env.action_space.sample() # Random action
                # Example: Sometimes try to build a position
                if global_step_count % 50 < 10 : # Try to buy for 10 steps
                    action = np.array([0.1, -0.5, 0.8, -0.8, -0.8, -0.8], dtype=np.float32) # Place a buy more likely
                elif global_step_count % 50 >= 10 and global_step_count % 50 < 20: # Try to sell
                    action = np.array([-0.5, 0.1, -0.8, 0.8, -0.8, -0.8], dtype=np.float32) # Place a sell more likely

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                global_step_count += 1

                if episode_steps % 50 == 0: # Render every 50 steps within episode
                    env.render()
                    # Optional: Check taker flags on active orders
                    # active_flags = [(o['id'], o.get('is_taker_at_activation')) for o in env.active_orders]
                    # print(f"Active Order Taker Flags: {active_flags}")

                # Check if global step limit reached
                if global_step_count >= max_total_steps:
                    print(f"Global step limit ({max_total_steps}) reached.")
                    terminated = True # Force termination if data limit hit globally

            # --- End of Episode ---
            print(f"\n--- Episode {episode + 1} Ended ---")
            print(f"Reason: {'Termination' if terminated else 'Truncation' if truncated else 'Global Step Limit'}")
            print(f"Steps in Episode: {episode_steps} (Total steps elapsed: {info['total_steps_elapsed']})")
            print(f"Global Data Step: {info['current_step']}")
            env.render() # Render final state of the episode
            print(f"Episode Reward: {episode_reward:.4f}")
            print(f"Final MTM: {info['mtm']:.2f}")
            print(f"Final EpPNL: {info['episode_pnl']:.2f}")
            print(f"Final State before next reset: Inv={info['net_inventory']:.4f}, ActOrd={info['active_orders_count']}, PendOrd={info['pending_orders_count']}")

            if global_step_count >= max_total_steps:
                print("\nGlobal step limit reached. Stopping test.")
                break # Exit episode loop

        env.close()
        print("\n--- Environment Closed ---")

    except ValueError as e:
        print(f"Error during environment setup or execution: {e}")
        import traceback
        traceback.print_exc()
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up dummy file if it was created
        if config["csv_path"] == dummy_file_path and os.path.exists(dummy_file_path):
            try:
                os.remove(dummy_file_path)
                print(f"Removed dummy data file: {dummy_file_path}")
            except OSError as e:
                print(f"Error removing dummy data file: {e}")
