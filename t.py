import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HFTEnv(gym.Env):
    """
    A High-Frequency Trading Market Making Environment allowing only LONG positions.
    Uses change in Mark-to-Market (MTM) value as the primary reward signal.
    Includes fix for unexecutable sell orders and initialization error.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.config = config
        self._validate_config()

        # Load and validate order book data
        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config["max_steps"], len(self.order_book_history))

        # Define action and observation spaces
        # Action: [signed_volume (-1 sell, +1 buy), price_offset (-1 to +1), cancel_fraction (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        # Observation space adjusted for long-only
        self.observation_space = self._create_observation_space()

        # Initialize running statistics for observation normalization
        self.running_stats = {'mean': None, 'var': None, 'count': 0}
        self.decay = 0.999  # Exponential decay factor for running stats

        # Initialize state variables (will be properly set in reset)
        self.current_step = 0
        self.cash = 0.0
        self.long_position = 0.0
        self.long_avg_cost = 0.0
        self.inventory = 0.0
        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.bids = None
        self.asks = None
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.midprice = 0.0
        self.spread = 0.0
        self.order_queue = deque()
        self.last_executed_volume = 0.0
        self.previous_mtm = 0.0 # For MTM reward calculation

        self.reset()

    def _validate_config(self):
        required_keys = [
            "csv_path", "initial_capital", "max_steps", "order_book_levels",
            "price_offset_ticks", "max_order_volume", "latency_steps", "tick_size",
            "lot_size", "max_active_orders", "inventory_penalty", "transaction_cost",
            "max_inventory"
        ]
        if "invalid_order_penalty" not in self.config:
            self.config["invalid_order_penalty"] = 1.0
        if "activity_bonus" not in self.config:
             self.config["activity_bonus"] = 0.0

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_order_book_data(self):
        logging.info(f"Loading order book data from: {self.config['csv_path']}")
        df = pd.read_csv(self.config["csv_path"])
        required_columns = ["timestamp"]
        for i in range(1, self.config["order_book_levels"] + 1):
            required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV data: {missing_cols}")

        history = []
        skipped_rows = 0
        for index, row in df.iterrows():
            try:
                bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)

                # Basic data validity checks
                if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any():
                    skipped_rows += 1
                    continue
                if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                    skipped_rows += 1
                    continue
                if not np.all(np.diff(bids[:, 0]) <= 1e-6):
                     skipped_rows += 1
                     continue
                if not np.all(np.diff(asks[:, 0]) >= -1e-6):
                     skipped_rows += 1
                     continue
                if bids[0, 0] >= asks[0, 0] - self.config["tick_size"]/2 :
                     skipped_rows += 1
                     continue

                history.append({"bids": bids, "asks": asks})
            except KeyError as e:
                raise ValueError(f"Missing expected column: {e} at index {index}")
            except Exception as e:
                 raise ValueError(f"Error processing row {index}: {e}")

        if skipped_rows > 0:
             logging.warning(f"Skipped {skipped_rows} rows due to data validation issues.")
        if not history:
             raise ValueError("No valid order book data could be loaded after filtering.")
        logging.info(f"Successfully loaded {len(history)} valid order book states.")
        return history

    def _create_observation_space(self):
        obs_size = (
            7 +  # mtm_norm, inv_norm, active_orders_norm, long_pos_norm, long_avg_cost_norm, bid_dev_norm, ask_dev_norm
            2 * self.config["max_active_orders"] +  # order prices (norm offset), volumes (norm, signed)
            4 * self.config["order_book_levels"] +  # bid/ask prices (norm offset)/volumes (norm)
            1  # spread_norm
        )
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Reset running statistics for normalization
        self.running_stats = {'mean': None, 'var': None, 'count': 0}

        # Reset portfolio state
        self.long_position = 0.0
        self.long_avg_cost = 0.0
        self.inventory = 0.0
        self.cash = self.config["initial_capital"]

        # Reset environment state
        self.current_step = 0
        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.execution_history = []
        self.last_executed_volume = 0.0 # Initialize here

        # Load initial market state
        if not self.order_book_history:
             raise RuntimeError("Order book history is empty. Cannot reset environment.")
        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        try:
             self._update_market_state() # Calculate initial midprice etc.
        except ValueError as e:
             raise RuntimeError(f"Failed to initialize market state at reset: {e}")

        # Initialize MTM state for reward calculation
        # Calculated *after* market state is updated
        self.previous_mtm = self.cash + self.inventory * self.midprice

        # Initialize order queue for latency
        self.order_queue = deque([None] * self.config["latency_steps"])

        logging.info("Environment reset.")
        # Return initial observation and info
        # Need to calculate observation *after* all state is initialized
        return self._get_observation(), self._get_info()

    def _update_market_state(self):
        """Updates market state variables based on current order book data."""
        if not (isinstance(self.bids, np.ndarray) and self.bids.ndim == 2 and self.bids.shape[0] > 0):
             raise ValueError("Invalid bids data during market state update")
        if not (isinstance(self.asks, np.ndarray) and self.asks.ndim == 2 and self.asks.shape[0] > 0):
             raise ValueError("Invalid asks data during market state update")

        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        if self.best_bid >= self.best_ask - self.config["tick_size"]/2:
             logging.warning(f"Step {self.current_step}: Market spread crossed or zero (Bid: {self.best_bid}, Ask: {self.best_ask}). Using simple average as midprice.")
             self.midprice = (self.best_bid + self.best_ask) / 2
             self.spread = self.best_ask - self.best_bid # Can be <= 0
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid

        if self.midprice <= 0:
             logging.error(f"Step {self.current_step}: Calculated midprice is zero or negative ({self.midprice}). Using best ask as fallback.")
             self.midprice = self.best_ask if self.best_ask > 0 else 1.0 # Fallback

    def step(self, action):
        # Store MTM from the end of the previous step (start of current step)
        mtm_previous = self.previous_mtm

        self.step_count += 1
        self.last_executed_volume = 0.0 # Reset executed volume for this step

        # 1. Parse Action
        try:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            signed_volume, price_offset, cancel_fraction = action
            is_buy = signed_volume >= 0.0
            volume_scaled = abs(signed_volume)
        except (TypeError, ValueError) as e:
             logging.error(f"Invalid action received: {action}. Error: {e}. Applying penalty.")
             # Return state before action attempt, but advance time conceptually
             self._update_step_state() # Still need to advance market data
             obs = self._get_observation() # Obs based on new market data
             reward = -5.0 # Significant penalty
             terminated = False # Or check termination conditions based on current state?
             truncated = self.step_count >= self.max_steps
             info = self._get_info()
             info["error"] = "Invalid action format"
             # Update previous MTM for next step based on current state
             self.previous_mtm = self.cash + self.inventory * self.midprice
             return obs, reward, terminated, truncated, info

        # 2. Handle Order Cancellation
        self._handle_order_cancellation(cancel_fraction)

        # 3. Place New Order (if volume > 0) -> generates order_penalty
        order_penalty = 0.0
        if volume_scaled > 1e-6:
             order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)

        # 4. Process Latency Queue
        self._process_latency_queue()

        # 5. Execute Orders (Main matching logic)
        # Note: _execute_orders updates self.cash, self.long_position, self.long_avg_cost
        #       Transaction costs are deducted from cash within _process_fill.
        #       realized_pnl is returned but *not* directly used in MTM reward.
        _, transaction_costs_this_step = self._execute_orders() # Capture costs for info if needed

        # 6. Update Inventory (LONG ONLY)
        self.inventory = self.long_position

        # 7. Check Termination Conditions
        terminated = (
            self.cash < 0 or
            self.inventory > self.config["max_inventory"] or
            self.current_step >= len(self.order_book_history) - 1
        )
        truncated = self.step_count >= self.max_steps

        # Check for critical errors
        if self.inventory < -1e-9:
             logging.error(f"CRITICAL: Negative inventory detected ({self.inventory}). Terminating.")
             terminated = True
             order_penalty -= 100 # Severe penalty in addition to MTM change

        # --- MTM Calculation and Reward Logic ---
        mtm_current = 0.0
        liquidation_occurred = False

        # 8. Liquidate Positions at Episode End (if needed)
        #    Liquidate based on market prices *before* advancing time.
        if terminated or truncated:
            liquidation_occurred = True
            pre_advance_best_bid = self.best_bid # Price to liquidate longs
            # _liquidate_positions updates self.cash and zeroes inventory
            _ = self._liquidate_positions(pre_advance_best_bid) # Realized PnL from liq is not the reward
            # MTM after liquidation is simply the final cash balance
            mtm_current = self.cash
            # Do NOT advance market state further if episode ended
        else:
            # 9. Advance Market State for Next Step (if episode continues)
            self._update_step_state() # Updates self.bids, asks, midprice, etc.

            # 10. Calculate MTM at the end of the current step
            #     Use the *new* midprice after market state update
            mtm_current = self.cash + self.inventory * self.midprice

        # 11. Calculate MTM Change (Core Reward Component)
        mtm_pnl = mtm_current - mtm_previous

        # 12. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty() # Based on end-of-step inventory
        activity_bonus = self.config.get("activity_bonus", 0.0) * self.last_executed_volume

        # 13. Calculate Final Reward for the Step
        #     MTM change already reflects realized PnL and transaction costs (via cash changes).
        #     We subtract explicit penalties and add explicit bonuses.
        reward = mtm_pnl - risk_penalty + order_penalty + activity_bonus

        # 14. Update `previous_mtm` for the next step's calculation
        self.previous_mtm = mtm_current

        # 15. Get Next Observation and Info (based on final state of the step)
        obs = self._get_observation()
        info = self._get_info()
        if liquidation_occurred:
             info["liquidation_occurred"] = True


        return obs, reward, terminated, truncated, info


    def _handle_order_cancellation(self, cancel_fraction):
        """Cancels a fraction of the oldest active orders."""
        if not (0.0 <= cancel_fraction <= 1.0):
             cancel_fraction = np.clip(cancel_fraction, 0.0, 1.0)

        if self.active_orders and cancel_fraction > 0:
            num_to_cancel = int(np.floor(cancel_fraction * len(self.active_orders)))
            if num_to_cancel > 0:
                self.active_orders = self.active_orders[num_to_cancel:]

    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        """Places a new order into the latency queue, applying penalties."""
        penalty = 0.0
        if len(self.active_orders) >= self.config["max_active_orders"]:
            return penalty # Cannot place, no penalty

        price_offset_val = price_offset * self.config["price_offset_ticks"] * self.config["tick_size"]
        limit_price = self.midprice + price_offset_val
        limit_price = self._round_to_tick(limit_price)

        # Basic validity check penalties
        allowed_buy_limit = self.best_ask + (self.config["price_offset_ticks"] * self.config["tick_size"])
        allowed_sell_limit = self.best_bid - (self.config["price_offset_ticks"] * self.config["tick_size"])
        if is_buy and limit_price > allowed_buy_limit:
             penalty -= self.config.get("invalid_order_penalty", 1.0)
        if not is_buy and limit_price < allowed_sell_limit:
             penalty -= self.config.get("invalid_order_penalty", 1.0)
        # Additional check: Don't place buy above best ask or sell below best bid? (optional stricter penalty)
        # if is_buy and limit_price > self.best_ask: penalty -= ...
        # if not is_buy and limit_price < self.best_bid: penalty -= ...


        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0.0, np.floor(volume / self.config["lot_size"]) * self.config["lot_size"])

        if volume < self.config["lot_size"]:
            return penalty # Return penalty from price check, but don't place order

        # Create and queue order
        order = {
            "id": self.order_id_counter, "price": limit_price, "volume": volume,
            "is_buy": is_buy, "timestamp": self.current_step
        }
        self.order_id_counter += 1
        self.order_queue.append(order)
        return penalty

    def _round_to_tick(self, price):
        """Rounds a price to the nearest valid tick."""
        if self.config["tick_size"] <= 0: return price
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_latency_queue(self):
        """Moves an order from the latency queue to active orders."""
        if len(self.order_queue) > 0:
            order_to_activate = self.order_queue.popleft()
            self.order_queue.append(None) # Maintain queue size

            if order_to_activate is not None:
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(order_to_activate)
                # else: Order rejected silently or add penalty?

    def _execute_orders(self):
        """Matches active orders against the current order book. LONG ONLY logic with fix."""
        total_realized_pnl_this_step = 0.0 # Sum of PnL from actual fills
        total_transaction_costs = 0.0
        executed_volume_this_step = 0.0
        new_active_orders = []

        for order in self.active_orders:
            # FIX: Cancel unexecutable sells immediately
            if not order["is_buy"] and self.long_position <= 1e-9:
                continue # Skip, do not add to new_active_orders

            remaining_volume = order["volume"]
            if remaining_volume <= 1e-9: continue

            levels_to_match = self.asks if order["is_buy"] else self.bids

            for level_price, level_qty in levels_to_match:
                if remaining_volume <= 1e-9: break
                if level_qty <= 1e-9: continue

                can_fill_at_level = (order["is_buy"] and order["price"] >= level_price - 1e-9) or \
                                    (not order["is_buy"] and order["price"] <= level_price + 1e-9)

                if can_fill_at_level:
                    fill_qty = min(remaining_volume, level_qty)

                    # Apply constraints
                    if not order["is_buy"]: # Selling
                        fill_qty = min(fill_qty, self.long_position)
                    elif order["is_buy"]: # Buying
                        allowed_buy = max(0.0, self.config["max_inventory"] - self.long_position)
                        fill_qty = min(fill_qty, allowed_buy)

                    # Process fill if valid quantity
                    if fill_qty > 1e-9:
                        executed_value = fill_qty * level_price
                        transaction_cost = self.config["transaction_cost"] * executed_value
                        total_transaction_costs += transaction_cost

                        # _process_fill updates position, avg_cost, cash, and returns realized PnL for this fill
                        pnl_from_fill = self._process_fill(fill_qty, level_price, order["is_buy"], transaction_cost)
                        total_realized_pnl_this_step += pnl_from_fill

                        remaining_volume -= fill_qty
                        executed_volume_this_step += fill_qty
                        self.inventory = max(0.0, self.long_position) # Ensure inventory syncs

                    if remaining_volume <= 1e-9: break # Break inner loop if order filled

            # Keep order if partially filled (and wasn't cancelled at start)
            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)

        self.active_orders = new_active_orders
        self.last_executed_volume = executed_volume_this_step
        # Return realized PnL and costs (useful for info, but not primary reward)
        return total_realized_pnl_this_step, total_transaction_costs

    def _process_fill(self, fill_qty, price, is_buy, transaction_cost):
        """
        Updates position, cash, and calculates realized PnL for a single fill.
        Deducts transaction costs from cash.
        """
        realized_pnl = 0.0

        if is_buy:
            # Update cash (decrease)
            self.cash -= (price * fill_qty + transaction_cost)
            # Update position
            new_total_qty = self.long_position + fill_qty
            if new_total_qty > 1e-9:
                self.long_avg_cost = ((self.long_avg_cost * self.long_position + price * fill_qty) / new_total_qty)
            else:
                 self.long_avg_cost = 0.0
            self.long_position = new_total_qty
            # No PnL realized on buy
        else: # is_sell
            if self.long_position <= 1e-9 or fill_qty <= 1e-9:
                 logging.warning(f"Attempted to process sell fill with zero inventory/qty. Skipping.")
                 return 0.0 # No PnL, no state change

            actual_sell_qty = min(fill_qty, self.long_position)

            # Update cash (increase)
            self.cash += (price * actual_sell_qty - transaction_cost)

            # Calculate realized PnL
            if self.long_avg_cost > 0:
                 realized_pnl = (price - self.long_avg_cost) * actual_sell_qty
            else:
                 logging.warning(f"Selling qty {actual_sell_qty} but long_avg_cost is {self.long_avg_cost}. PnL calc uses cost=0.")
                 realized_pnl = price * actual_sell_qty

            # Update position
            self.long_position -= actual_sell_qty
            if self.long_position <= 1e-9:
                self.long_position = 0.0
                self.long_avg_cost = 0.0

        # Update overall inventory metric
        self.inventory = self.long_position
        return realized_pnl # Return the PnL realized from this specific fill

    def _calculate_risk_penalty(self):
        """Calculates inventory risk penalty (quadratic)."""
        if self.config["max_inventory"] <= 0: return 0.0
        normalized_inventory = self.inventory / self.config["max_inventory"]
        penalty = self.config["inventory_penalty"] * (normalized_inventory ** 2)
        return penalty

    def _update_step_state(self):
        """Advances the environment time by one step and updates market data."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            next_state = self.order_book_history[self.current_step]
            self.bids = next_state["bids"]
            self.asks = next_state["asks"]
            try:
                self._update_market_state() # Update BBO, midprice, spread
            except ValueError as e:
                logging.error(f"Error updating market state at step {self.current_step}: {e}. Keeping previous midprice/spread.")
                # Midprice/spread from previous step will persist if update fails
        # else: State remains at final step if already there

    def _get_raw_observation(self):
        """Constructs the raw observation vector. LONG ONLY."""
        # Use safe defaults for normalization denominators
        safe_midprice = max(self.midprice, 1e-6)
        safe_max_inv = max(self.config["max_inventory"], 1.0)
        safe_max_orders = max(self.config["max_active_orders"], 1.0)
        safe_max_vol = max(self.config["max_order_volume"], 1.0)
        safe_init_cap = max(self.config["initial_capital"], 1.0)
        qty_normalization_factor = max(safe_max_vol * self.config["order_book_levels"], 1.0)

        # Calculate current MTM for observation
        current_mtm = self.cash + self.inventory * self.midprice

        market_features = [
            current_mtm / safe_init_cap,                            # MTM normalized by initial capital
            self.inventory / safe_max_inv,                          # Inventory normalized
            len(self.active_orders) / safe_max_orders,              # Active orders normalized
            self.long_position / safe_max_inv,                      # Long position normalized
            (self.long_avg_cost / safe_midprice) if self.long_position > 1e-9 else 0.0, # Avg cost relative to midprice
            (self.best_bid - self.midprice) / safe_midprice,        # Bid deviation normalized
            (self.best_ask - self.midprice) / safe_midprice,        # Ask deviation normalized
        ]

        order_prices = []
        order_volumes_signed = []
        active_orders_to_process = self.active_orders[:self.config["max_active_orders"]]
        for order in active_orders_to_process:
             price_offset_norm = (order["price"] - self.midprice) / safe_midprice
             norm_volume = order["volume"] / safe_max_vol
             order_prices.append(price_offset_norm)
             order_volumes_signed.append(norm_volume * (1 if order['is_buy'] else -1))

        padding_needed = self.config["max_active_orders"] - len(active_orders_to_process)
        order_prices += [0.0] * padding_needed
        order_volumes_signed += [0.0] * padding_needed

        book_features = []
        if self.bids is not None and self.asks is not None: # Check if book data exists
            for level in range(self.config["order_book_levels"]):
                bid_price_offset = (self.bids[level, 0] - self.midprice) / safe_midprice
                bid_qty_normalized = self.bids[level, 1] / qty_normalization_factor
                ask_price_offset = (self.asks[level, 0] - self.midprice) / safe_midprice
                ask_qty_normalized = self.asks[level, 1] / qty_normalization_factor
                book_features += [
                    bid_price_offset, np.clip(bid_qty_normalized, 0, 5),
                    ask_price_offset, np.clip(ask_qty_normalized, 0, 5)
                ]
        else: # Provide zeros if book data is missing (should only happen on error)
             book_features = [0.0] * (4 * self.config["order_book_levels"])

        spread_feature = [(self.spread / safe_midprice) if self.spread >=0 else 0.0]

        observation = np.concatenate([
            market_features, order_prices, order_volumes_signed,
            book_features, spread_feature
        ], dtype=np.float32)

        if observation.shape[0] != self.observation_space.shape[0]:
             raise ValueError(f"Observation shape mismatch: Got {observation.shape[0]}, Expected {self.observation_space.shape[0]}")

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
             logging.error(f"NaN or Inf detected in raw observation at step {self.current_step}. Clamping.")
             observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _get_observation(self):
        """Gets the raw observation and normalizes it using running statistics."""
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        normalized_obs = self._normalize_observation(raw_obs)
        clipped_obs = np.clip(normalized_obs, -5.0, 5.0) # Clip normalized obs

        if np.any(np.isnan(clipped_obs)) or np.any(np.isinf(clipped_obs)):
             logging.error(f"NaN or Inf detected AFTER normalization/clipping at step {self.current_step}. Returning zeros.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        return clipped_obs

    def _update_running_stats(self, obs):
        """Updates running mean and variance using EMA."""
        if self.running_stats['mean'] is None:
            self.running_stats['mean'] = obs
            self.running_stats['var'] = np.zeros_like(obs)
        else:
            delta_mean = obs - self.running_stats['mean']
            self.running_stats['mean'] += (1 - self.decay) * delta_mean
            # Update variance using updated mean
            delta_var = (obs - self.running_stats['mean'])**2
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta_var
        self.running_stats['count'] += 1

    def _normalize_observation(self, obs):
        """Normalizes observation using the running mean and variance."""
        if self.running_stats['mean'] is None or self.running_stats['count'] < 2:
            return obs # Return raw if not enough stats
        mean = self.running_stats['mean']
        std_dev = np.sqrt(self.running_stats['var']) + 1e-8 # Epsilon for stability
        normalized_obs = (obs - mean) / std_dev
        return normalized_obs

    def _get_info(self):
        """Returns dictionary with auxiliary information. Reflects end-of-step state."""
        # Calculate MTM based on the *final* state of the step
        mtm = self.cash + self.inventory * self.midprice
        return {
            "current_step": self.current_step,
            "mtm": mtm,
            "cash": self.cash,
            "inventory": self.inventory,
            "long_position": self.long_position,
            "long_avg_cost": self.long_avg_cost,
            "active_orders": len(self.active_orders),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midprice": self.midprice,
            "spread": self.spread,
            "episode_pnl": mtm - self.config["initial_capital"],
            "last_exec_volume": self.last_executed_volume
            # "previous_mtm": self.previous_mtm # Can be useful for debugging reward
        }

    def render(self, mode="human"):
        """Renders the environment state."""
        if mode == "human":
            info = self._get_info() # Get current info reflecting end of step
            render_msg = (
                f"--- Step {info['current_step']:<5}/{self.max_steps} (SimStep: {self.step_count}) --- \n"
                f"  Market: Bid={info['best_bid']:.2f} | Ask={info['best_ask']:.2f} | Mid={info['midprice']:.2f} | Sprd={info['spread']:.2f}\n"
                f"  Portfolio: MTM=${info['mtm']:>10,.2f} | Cash=${info['cash']:>10,.2f} | Ep PnL=${info['episode_pnl']:>+8,.2f}\n"
                f"  Position: Long={info['long_position']:<6.3f} @ ${info['long_avg_cost']:<7.2f} (Inv={info['inventory']:.3f})\n"
                f"  Orders: {info['active_orders']:<2} active | Last Exec Vol: {info['last_exec_volume']:.3f}\n"
                f"{'='*65}" # Adjusted width slightly
            )
            print(render_msg)
        else:
             super(HFTEnv, self).render(mode=mode)

    def _liquidate_positions(self, liquidation_price):
        """
        Liquidates remaining long position at the provided price.
        Updates cash, zeroes position/inventory. Returns realized PnL from the liquidation.
        """
        realized_liquidation_pnl = 0.0
        if self.long_position > 1e-9:
            qty_to_liquidate = self.long_position
            logging.info(f"Liquidating {qty_to_liquidate:.3f} units at ${liquidation_price:.2f} (Avg Cost: ${self.long_avg_cost:.2f})")

            executed_value = liquidation_price * qty_to_liquidate
            transaction_cost = self.config["transaction_cost"] * executed_value

            # Calculate realized PnL for this liquidation event
            if self.long_avg_cost > 0:
                 realized_liquidation_pnl = (liquidation_price - self.long_avg_cost) * qty_to_liquidate - transaction_cost
            else:
                 realized_liquidation_pnl = executed_value - transaction_cost # Assume cost was 0

            # Update cash balance
            self.cash += (executed_value - transaction_cost)

            # Reset position
            self.long_position = 0.0
            self.inventory = 0.0
            self.long_avg_cost = 0.0

        # Clear any remaining active orders and queue as the episode is ending
        if self.active_orders:
             self.active_orders = []
        self.order_queue = deque([None] * self.config["latency_steps"])

        # Return the PnL realized specifically during this liquidation
        return realized_liquidation_pnl

    def close(self):
        """Clean up any resources."""
        logging.info("Closing HFT Environment.")
        # No explicit resources to close in this version
        pass

# --- Example Configuration and Usage ---
config = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed.csv", # <<< --- UPDATE THIS PATH --- >>>
    "initial_capital": 20000.0,
    "max_steps": 5000, # Steps limit for truncation
    "order_book_levels": 9,
    "price_offset_ticks": 40,
    "max_order_volume": 2.5,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.005,
    "max_active_orders": 10,
    "inventory_penalty": 0.01, # Added a small penalty example
    "transaction_cost": 0.0005, # 0.05%
    "max_inventory": 8.0,
    "invalid_order_penalty": 0.5,
    "activity_bonus": 0.0 # Typically keep low or zero unless specifically needed
}

if __name__ == "__main__":
    print(f"Initializing HFTEnv with config:\n{config}")
    try:
        env = HFTEnv(config)
        obs, info = env.reset()

        print("\nEnvironment Initialized.")
        print("Observation Space Shape:", env.observation_space.shape)
        print("Action Space Shape:", env.action_space.shape)
        print("Initial Info:", info)
        print("-" * 65)

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0 # Use internal env.step_count now

        render_interval = 500 # Render every N steps
        print(f"Running simulation for max {env.max_steps} data points (max {config['max_steps']} steps)...")
        print(f"Reward is based on change in MTM per step minus penalties.")

        while not terminated and not truncated:
            # Sample a random action for testing
            action = env.action_space.sample()

            # Apply the action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # step_count is managed internally by env.step_count

            if info['current_step'] == 1 or info['current_step'] % render_interval == 0 or terminated or truncated:
                 print(f"\n>>> Data Point {info['current_step']} / Sim Step {env.step_count} <<<")
                 env.render() # Render provides detailed state
                 print(f"  Action Taken: [Vol: {action[0]:+.2f}, PxOff: {action[1]:+.2f}, Cancel: {action[2]:.2f}]")
                 print(f"  Step Reward (MTM Change - Pens + Bonus): {reward:+.4f}")
                 print(f"  Total Reward: {total_reward:+.4f}")
                 if info.get("liquidation_occurred", False):
                     print("  ** Liquidation occurred at episode end **")


        print(f"\n{'='*65}")
        print(f"Episode finished after {env.step_count} steps (Data point {info['current_step']}).")
        print(f"Termination reason: {'Negative Cash/Max Inventory/Data End' if terminated else 'Max Steps Reached' if truncated else 'Unknown'}")
        print(f"Final Info: {info}")
        print(f"Final Total Reward: {total_reward:.4f}")
        print(f"Final MTM: {info['mtm']:.2f}")
        print(f"Final Episode PnL: {info['episode_pnl']:.2f}")
        print(f"{'='*65}")

        env.close()

    except FileNotFoundError:
         print(f"\n\nERROR: CSV file not found at '{config['csv_path']}'.")
         print("Please update the 'csv_path' in the config dictionary.")
    except ValueError as e:
         print(f"\n\nERROR during environment setup or execution: {e}")
         import traceback
         traceback.print_exc()
    except Exception as e:
        print(f"\n\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

