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
    Includes fix to prevent accumulation of unexecutable sell orders.
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
        self.decay = 0.999  # Exponential decay factor

        self.reset()

    def _validate_config(self):
        required_keys = [
            "csv_path", "initial_capital", "max_steps", "order_book_levels",
            "price_offset_ticks", "max_order_volume", "latency_steps", "tick_size",
            "lot_size", "max_active_orders", "inventory_penalty", "transaction_cost",
            "max_inventory" # This now refers to max LONG inventory
        ]
        if "invalid_order_penalty" not in self.config:
            self.config["invalid_order_penalty"] = 1.0 # Default penalty
        if "activity_bonus" not in self.config:
             self.config["activity_bonus"] = 0.0 # Default activity bonus

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
                    # logging.warning(f"Invalid prices (<=0) found at index {index}. Skipping row.")
                    skipped_rows += 1
                    continue
                if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                    # logging.warning(f"Negative quantities found at index {index}. Skipping row.")
                    skipped_rows += 1
                    continue
                # Check ordering (allow equal prices within bids/asks, but BBO must not cross)
                if not np.all(np.diff(bids[:, 0]) <= 1e-6): # Allow for float inaccuracies
                     # logging.warning(f"Bid prices not descending at index {index}. Skipping row.")
                     skipped_rows += 1
                     continue
                if not np.all(np.diff(asks[:, 0]) >= -1e-6): # Allow for float inaccuracies
                     # logging.warning(f"Ask prices not ascending at index {index}. Skipping row.")
                     skipped_rows += 1
                     continue
                if bids[0, 0] >= asks[0, 0] - self.config["tick_size"]/2 : # Check spread validity strictly
                     # logging.warning(f"Bid-ask spread crossed or zero at index {index} (Bid: {bids[0,0]}, Ask: {asks[0,0]}). Skipping row.")
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
        # LONG ONLY: Removed short_position and short_avg_cost (2 features)
        obs_size = (
            7 +  # mtm_norm, inv_norm, active_orders_norm, long_pos_norm, long_avg_cost_norm, bid_dev_norm, ask_dev_norm
            2 * self.config["max_active_orders"] +  # order prices (norm offset), volumes (norm, signed)
            4 * self.config["order_book_levels"] +  # bid/ask prices (norm offset)/volumes (norm)
            1  # spread_norm
        )
        # Using -inf/inf bounds initially, normalization and clipping handle ranges later
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Reset running statistics
        self.running_stats = {'mean': None, 'var': None, 'count': 0}

        # Ensure any previous positions are cleared
        # No need to simulate liquidation PnL during reset, just reset state
        self.long_position = 0
        self.long_avg_cost = 0.0
        self.inventory = 0
        self.last_executed_volume = 0.0 # <<< --- ADD THIS INITIALIZATION ---
        self.current_step = 0
        self.cash = self.config["initial_capital"]

        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.execution_history = [] # Optional: track fills
        self.last_executed_volume = 0.0 # <<< --- ADD THIS INITIALIZATION ---

        # Load initial market state
        if not self.order_book_history:
             raise RuntimeError("Order book history is empty. Cannot reset environment.")
        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        try:
             self._update_market_state() # Calculate initial midprice etc.
        except ValueError as e:
             raise RuntimeError(f"Failed to initialize market state at reset: {e}")


        # Initialize order queue for latency
        self.order_queue = deque([None] * self.config["latency_steps"]) # Use None as placeholder


        logging.info("Environment reset.")
        return self._get_observation(), self._get_info() # Now _get_info will find the attribute

    def _update_market_state(self):
        """Updates market state variables based on current order book data."""
        if not (isinstance(self.bids, np.ndarray) and self.bids.ndim == 2 and self.bids.shape[0] > 0):
             raise ValueError("Invalid bids data during market state update")
        if not (isinstance(self.asks, np.ndarray) and self.asks.ndim == 2 and self.asks.shape[0] > 0):
             raise ValueError("Invalid asks data during market state update")

        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        if self.best_bid >= self.best_ask - self.config["tick_size"]/2:
             # Handle crossed or zero spread - Use midpoint as fallback? Log warning.
             logging.warning(f"Step {self.current_step}: Market spread crossed or zero (Bid: {self.best_bid}, Ask: {self.best_ask}). Using simple average as midprice.")
             self.midprice = (self.best_bid + self.best_ask) / 2
             self.spread = self.best_ask - self.best_bid # Can be <= 0
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid

        if self.midprice <= 0:
             # Fallback if midprice calculation results in zero or negative (highly unlikely with checks)
             logging.error(f"Step {self.current_step}: Calculated midprice is zero or negative ({self.midprice}). Using previous or default.")
             # Need a robust fallback, maybe use last valid midprice or raise error
             # For now, let's try to use best ask as a proxy if midprice fails
             self.midprice = self.best_ask if self.best_ask > 0 else 1.0


    def step(self, action):
        self.step_count += 1
        self.last_executed_volume = 0.0 # Reset executed volume for this step

        # 1. Parse Action
        try:
            # Ensure action components are within expected ranges if necessary
            action = np.clip(action, self.action_space.low, self.action_space.high)
            signed_volume, price_offset, cancel_fraction = action
            is_buy = signed_volume >= 0.0 # Treat 0 volume as neutral (no order placement intent)
            volume_scaled = abs(signed_volume)
        except (TypeError, ValueError) as e:
             logging.error(f"Invalid action received: {action}. Error: {e}. Applying penalty.")
             obs = self._get_observation()
             reward = -5.0 # Significant penalty for invalid action format
             terminated = False
             truncated = self.step_count >= self.max_steps
             info = self._get_info()
             info["error"] = "Invalid action format"
             return obs, reward, terminated, truncated, info

        # 2. Handle Order Cancellation
        self._handle_order_cancellation(cancel_fraction)

        # 3. Place New Order (if volume > 0)
        order_penalty = 0.0
        if volume_scaled > 1e-6: # Check if volume is meaningfully positive
             order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)

        # 4. Process Latency Queue
        self._process_latency_queue()

        # 5. Execute Orders (Main matching logic)
        realized_pnl, transaction_costs = self._execute_orders() # Returns PnL and costs separately

        # 6. Update Inventory (LONG ONLY)
        self.inventory = self.long_position # Inventory is just the long position

        # 7. Check Termination Conditions
        terminated = (
            self.cash < 0 or
            self.inventory > self.config["max_inventory"] or # Check only upper bound
            self.current_step >= len(self.order_book_history) - 1 # End of data
        )
        # Check for potentially invalid state like negative inventory (shouldn't happen)
        if self.inventory < -1e-9: # Allow for small float inaccuracies
             logging.error(f"CRITICAL: Negative inventory detected ({self.inventory}). Terminating.")
             terminated = True
             realized_pnl -= 100 # Severe penalty

        truncated = self.step_count >= self.max_steps

        # 8. Liquidate Positions at Episode End
        liquidation_pnl = 0.0
        if terminated or truncated:
            # Use the current market state for liquidation before advancing
            liquidation_pnl = self._liquidate_positions()
            logging.info(f"End of episode liquidation PnL: {liquidation_pnl:.2f}")


        # 9. Calculate Mark-to-Market Value
        mtm = self.cash + self.inventory * self.midprice # Inventory is always non-negative

        # 10. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty()
        activity_bonus = self.config.get("activity_bonus", 0.0) * self.last_executed_volume

        # 11. Calculate Reward
        # Reward = PnL from fills + PnL from liquidation - costs - penalties + bonuses
        reward = (realized_pnl + liquidation_pnl - transaction_costs -
                  risk_penalty + order_penalty + activity_bonus)


        # 12. Update Market State for Next Step
        self._update_step_state() # Advance time, update market data

        # 13. Get Next Observation and Info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _handle_order_cancellation(self, cancel_fraction):
        """Cancels a fraction of the oldest active orders."""
        if not (0.0 <= cancel_fraction <= 1.0):
             # logging.warning(f"Invalid cancel_fraction: {cancel_fraction}. Clamping to [0, 1].")
             cancel_fraction = np.clip(cancel_fraction, 0.0, 1.0)

        if self.active_orders and cancel_fraction > 0:
            num_to_cancel = int(np.floor(cancel_fraction * len(self.active_orders))) # Use floor to be conservative
            if num_to_cancel > 0:
                # Cancel oldest orders (FIFO)
                cancelled_orders = self.active_orders[:num_to_cancel]
                self.active_orders = self.active_orders[num_to_cancel:]
                # logging.debug(f"Cancelled {num_to_cancel} orders: {[o['id'] for o in cancelled_orders]}")

    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        """Places a new order into the latency queue, applying penalties for invalid placements."""
        penalty = 0.0
        if len(self.active_orders) >= self.config["max_active_orders"]:
            # logging.debug("Max active orders reached, cannot place new order.")
            return penalty # No penalty, just cannot place

        # Calculate Limit Price based on midprice
        price_offset_val = price_offset * self.config["price_offset_ticks"] * self.config["tick_size"]
        limit_price = self.midprice + price_offset_val
        limit_price = self._round_to_tick(limit_price)

        # --- Penalty Check (Optional Refinement) ---
        # More sophisticated penalty: e.g., penalize placing buy orders above best ask or sell orders below best bid?
        # This encourages making, not taking liquidity immediately, or placing within reasonable bounds.
        # Example: Penalize if buy limit > best ask or sell limit < best bid
        # if is_buy and limit_price > self.best_ask:
        #      penalty -= self.config.get("invalid_order_penalty", 1.0) * abs(limit_price - self.best_ask) # Penalty proportional to crossing
        # if not is_buy and limit_price < self.best_bid:
        #      penalty -= self.config.get("invalid_order_penalty", 1.0) * abs(self.best_bid - limit_price)
        # Using the simpler "too far" check for now:
        allowed_buy_limit = self.best_ask + (self.config["price_offset_ticks"] * self.config["tick_size"])
        allowed_sell_limit = self.best_bid - (self.config["price_offset_ticks"] * self.config["tick_size"])
        if is_buy and limit_price > allowed_buy_limit:
             penalty -= self.config.get("invalid_order_penalty", 1.0)
        if not is_buy and limit_price < allowed_sell_limit:
             penalty -= self.config.get("invalid_order_penalty", 1.0)
        # --- End Penalty Check ---

        # Calculate Volume
        volume = volume_scaled * self.config["max_order_volume"]
        # Ensure volume respects lot size and is positive
        volume = max(0.0, np.floor(volume / self.config["lot_size"]) * self.config["lot_size"])

        if volume < self.config["lot_size"]: # Check if volume is at least one lot
            # logging.debug(f"Order volume {volume} is less than lot size {self.config['lot_size']}, not placing.")
            return penalty # Return any penalty from price check, but don't place order

        # Create order dictionary
        order = {
            "id": self.order_id_counter,
            "price": limit_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp": self.current_step # Timestamp when the action was decided
        }
        self.order_id_counter += 1
        self.order_queue.append(order) # Add to latency queue
        # logging.debug(f"Placed order {order['id']} into latency queue: {'Buy' if is_buy else 'Sell'} {volume:.3f} @ {limit_price:.2f}")
        return penalty

    def _round_to_tick(self, price):
        """Rounds a price to the nearest valid tick."""
        if self.config["tick_size"] <= 0: return price
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_latency_queue(self):
        """Moves an order from the latency queue to active orders if applicable."""
        if len(self.order_queue) > 0:
            # Get the oldest item (could be None or an order)
            order_to_activate = self.order_queue.popleft()
            # Add a new None placeholder to the end to maintain latency window size
            self.order_queue.append(None)

            if order_to_activate is not None:
                # Check if we can add it to active orders
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(order_to_activate)
                    # logging.debug(f"Order {order_to_activate['id']} moved from queue to active orders.")
                else:
                    # logging.debug(f"Order {order_to_activate['id']} rejected from queue (max_active_orders reached).")
                    # Optionally track rejected orders or apply a penalty
                    pass

    def _execute_orders(self):
        """Matches active orders against the current order book. LONG ONLY logic with fix."""
        realized_pnl = 0.0
        total_transaction_costs = 0.0
        executed_volume_this_step = 0.0
        new_active_orders = [] # Orders that remain partially or fully unfilled

        for order in self.active_orders:
            # *** FIX IMPLEMENTED HERE ***
            # If it's a sell order and we have no long position, cancel it immediately.
            # Use a small tolerance for floating point comparisons.
            if not order["is_buy"] and self.long_position <= 1e-9:
                # logging.debug(f"Cancelling unexecutable sell order {order['id']} (inventory is zero).")
                continue # Skip this order entirely, do not add to new_active_orders

            # --- Existing logic proceeds from here ---
            remaining_volume = order["volume"]
            if remaining_volume <= 1e-9: continue # Skip if effectively zero volume remaining

            levels_to_match = self.asks if order["is_buy"] else self.bids

            for level_price, level_qty in levels_to_match:
                if remaining_volume <= 1e-9: break # Order filled

                # Ensure level qty is positive before attempting match
                if level_qty <= 1e-9: continue

                can_fill_at_level = False
                 # Use tolerance for price comparison
                if order["is_buy"] and order["price"] >= level_price - 1e-9:
                    can_fill_at_level = True
                elif not order["is_buy"] and order["price"] <= level_price + 1e-9:
                     can_fill_at_level = True

                if can_fill_at_level:
                    fill_qty = min(remaining_volume, level_qty)

                    # --- Constraint Checks ---
                    # 1. SELL constraint (Long Only): Cannot sell more than available inventory.
                    if not order["is_buy"]:
                        max_sellable = self.long_position
                        fill_qty = min(fill_qty, max_sellable)

                    # 2. BUY constraint: Cannot exceed max inventory.
                    if order["is_buy"]:
                         potential_inventory = self.long_position + fill_qty
                         if potential_inventory > self.config["max_inventory"]:
                              allowed_buy = max(0.0, self.config["max_inventory"] - self.long_position)
                              fill_qty = min(fill_qty, allowed_buy)

                    # Ensure fill_qty is still positive after constraints
                    if fill_qty > 1e-9: # Use tolerance
                        # Process the fill
                        executed_value = fill_qty * level_price # Fill at book level price
                        transaction_cost = self.config["transaction_cost"] * executed_value
                        total_transaction_costs += transaction_cost

                        pnl_from_fill = self._process_fill(fill_qty, level_price, order["is_buy"])
                        realized_pnl += pnl_from_fill

                        # Update cash
                        if order["is_buy"]:
                            self.cash -= (executed_value + transaction_cost)
                        else:
                            self.cash += (executed_value - transaction_cost)

                        remaining_volume -= fill_qty
                        executed_volume_this_step += fill_qty
                        # Ensure inventory doesn't go negative due to float issues
                        self.inventory = max(0.0, self.long_position)

                        # logging.debug(f"Order {order['id']} filled: {fill_qty:.3f} @ {level_price:.2f}. PnL: {pnl_from_fill:.2f}. Cash: {self.cash:.2f}. Inventory: {self.inventory:.3f}")

                    # Need to break inner loop if remaining_volume became zero after fill
                    if remaining_volume <= 1e-9:
                         break


            # If order still has volume remaining (and wasn't cancelled at the start), add back.
            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)
            # else:
                # logging.debug(f"Order {order['id']} fully filled or cancelled.")


        self.active_orders = new_active_orders
        self.last_executed_volume = executed_volume_this_step
        return realized_pnl, total_transaction_costs

    def _process_fill(self, fill_qty, price, is_buy):
        """Updates position and calculates PnL for a fill. LONG ONLY logic."""
        realized_pnl = 0.0

        if is_buy:
            # Add to long position
            new_total_qty = self.long_position + fill_qty
            if new_total_qty > 1e-9: # Avoid division by zero if total qty is near zero
                self.long_avg_cost = (
                    (self.long_avg_cost * self.long_position + price * fill_qty) / new_total_qty
                )
            else:
                 self.long_avg_cost = 0.0 # Reset if position becomes effectively zero
            self.long_position = new_total_qty
            # No PnL realized when opening or adding to long position
        else: # is_sell
            # This sell MUST be reducing an existing long position (enforced before calling/checked inside)
            if self.long_position <= 1e-9 or fill_qty <= 1e-9:
                 logging.warning(f"Attempted to process sell fill with zero inventory ({self.long_position}) or zero qty ({fill_qty}). Skipping.")
                 return 0.0

            actual_sell_qty = min(fill_qty, self.long_position) # Ensure we don't sell more than we have

            # Realize PnL: (sell_price - avg_buy_cost) * quantity
            if self.long_avg_cost > 0: # Ensure avg cost is valid
                 realized_pnl = (price - self.long_avg_cost) * actual_sell_qty
            else:
                 # This case implies selling something bought at cost 0? Log warning.
                 logging.warning(f"Processing sell fill for qty {actual_sell_qty} but long_avg_cost is {self.long_avg_cost}. PnL might be inaccurate.")
                 realized_pnl = price * actual_sell_qty # PnL is just the proceeds

            self.long_position -= actual_sell_qty

            # If position is closed (or very close to zero), reset avg cost
            if self.long_position <= 1e-9:
                self.long_position = 0.0 # Clean up small floats
                self.long_avg_cost = 0.0

        # Update overall inventory metric (redundant but clear)
        self.inventory = self.long_position
        return realized_pnl


    def _calculate_risk_penalty(self):
        """Calculates inventory risk penalty (quadratic)."""
        if self.config["max_inventory"] <= 0: return 0.0
        # Inventory is always non-negative here
        normalized_inventory = self.inventory / self.config["max_inventory"]
        # Use midprice for scaling? Optional. If midprice is unstable, might be better without it.
        # penalty = self.config["inventory_penalty"] * (normalized_inventory ** 2) * self.midprice
        penalty = self.config["inventory_penalty"] * (normalized_inventory ** 2) # Simpler version
        return penalty

    def _update_step_state(self):
        """Advances the environment time by one step."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            next_state = self.order_book_history[self.current_step]
            self.bids = next_state["bids"]
            self.asks = next_state["asks"]
            try:
                self._update_market_state() # Update BBO, midprice, spread
            except ValueError as e:
                logging.error(f"Error updating market state at step {self.current_step}: {e}. Using previous state values.")
                # Revert to previous bids/asks? Or just keep old midprice/spread?
                # Keeping old midprice/spread might be simplest if BBO update failed.
                pass # Midprice/spread from previous step will be used
        # else: No need for else, state remains at final step


    def _get_raw_observation(self):
        """Constructs the raw observation vector. LONG ONLY."""
        safe_midprice = self.midprice if self.midprice > 1e-6 else 1.0
        safe_max_inv = self.config["max_inventory"] if self.config["max_inventory"] > 0 else 1.0
        safe_max_orders = self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 1.0
        safe_max_vol = self.config["max_order_volume"] if self.config["max_order_volume"] > 0 else 1.0
        safe_init_cap = self.config["initial_capital"] if self.config["initial_capital"] > 0 else 1.0


        market_features = [
            (self.cash + self.inventory * self.midprice) / safe_init_cap, # MTM normalized
            self.inventory / safe_max_inv,                             # Inventory normalized
            len(self.active_orders) / safe_max_orders,                 # Active orders normalized
            self.long_position / safe_max_inv,                         # Long position normalized
            self.long_avg_cost / safe_midprice if self.long_position > 1e-9 else 0, # Avg cost relative
            (self.best_bid - self.midprice) / safe_midprice,           # Bid deviation norm
            (self.best_ask - self.midprice) / safe_midprice,           # Ask deviation norm
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
        # Use max_inventory or max_order_volume for qty normalization? max_inventory might be too large. Let's try max_order_volume * levels
        qty_normalization_factor = safe_max_vol * self.config["order_book_levels"]

        for level in range(self.config["order_book_levels"]):
            bid_price_offset = (self.bids[level, 0] - self.midprice) / safe_midprice
            bid_qty_normalized = self.bids[level, 1] / qty_normalization_factor
            ask_price_offset = (self.asks[level, 0] - self.midprice) / safe_midprice
            ask_qty_normalized = self.asks[level, 1] / qty_normalization_factor
            book_features += [
                bid_price_offset, np.clip(bid_qty_normalized, 0, 5), # Clip normalized qty
                ask_price_offset, np.clip(ask_qty_normalized, 0, 5)  # Clip normalized qty
            ]

        spread_feature = [(self.spread / safe_midprice) if self.spread >=0 else 0.0] # Normalized spread, ensure non-negative

        observation = np.concatenate([
            market_features,
            order_prices,
            order_volumes_signed,
            book_features,
            spread_feature
        ], dtype=np.float32)

        if observation.shape[0] != self.observation_space.shape[0]:
             raise ValueError(f"Observation shape mismatch: Got {observation.shape[0]}, Expected {self.observation_space.shape[0]}")

        # Final check for NaN/Inf before returning raw
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
             logging.error(f"NaN or Inf detected in raw observation generation at step {self.current_step}. Clamping.")
             observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation

    def _get_observation(self):
        """Gets the raw observation and normalizes it using running statistics."""
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        normalized_obs = self._normalize_observation(raw_obs)

        # Clip normalized obs to reasonable bounds, e.g., [-5, 5] std devs
        clipped_obs = np.clip(normalized_obs, -5.0, 5.0)

        # Final check post-normalization/clipping
        if np.any(np.isnan(clipped_obs)) or np.any(np.isinf(clipped_obs)):
             logging.error(f"NaN or Inf detected AFTER normalization/clipping at step {self.current_step}. Returning zero vector.")
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        return clipped_obs


    def _update_running_stats(self, obs):
        """Updates running mean and variance using EMA."""
        if self.running_stats['mean'] is None:
            self.running_stats['mean'] = obs
            self.running_stats['var'] = np.zeros_like(obs)
        else:
            # Update mean
            delta_mean = obs - self.running_stats['mean']
            self.running_stats['mean'] += (1 - self.decay) * delta_mean
            # Update variance (using new mean)
            delta_var = (obs - self.running_stats['mean'])**2 # Use updated mean for variance calc? Or old mean? Using updated mean here.
            # Alternative: delta_var = delta_mean * (obs - self.running_stats['mean']) # Welford's component
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta_var

        self.running_stats['count'] += 1


    def _normalize_observation(self, obs):
        """Normalizes observation using the running mean and variance."""
        if self.running_stats['mean'] is None or self.running_stats['count'] < 2:
            return obs

        mean = self.running_stats['mean']
        # Use variance for normalization, add epsilon
        std_dev = np.sqrt(self.running_stats['var']) + 1e-8

        normalized_obs = (obs - mean) / std_dev
        return normalized_obs

    def _get_info(self):
        """Returns dictionary with auxiliary information. LONG ONLY."""
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
        }

    def render(self, mode="human"):
        """Renders the environment state."""
        if mode == "human":
            info = self._get_info()
            render_msg = (
                f"--- Step {info['current_step']:<5}/{self.max_steps} --- \n"
                f"  Market: Bid={info['best_bid']:.2f} | Ask={info['best_ask']:.2f} | Mid={info['midprice']:.2f} | Sprd={info['spread']:.2f}\n"
                f"  Portfolio: MTM=${info['mtm']:>10,.2f} | Cash=${info['cash']:>10,.2f} | PnL=${info['episode_pnl']:>+8,.2f}\n"
                f"  Position: Long={info['long_position']:<6.3f} @ ${info['long_avg_cost']:<7.2f} (Inv={info['inventory']:.3f})\n"
                f"  Orders: {info['active_orders']:<2} active | Last Exec Vol: {info['last_exec_volume']:.3f}\n"
                f"{'='*60}"
            )
            print(render_msg)
        else:
             super(HFTEnv, self).render(mode=mode)


    def _liquidate_positions(self):
        """Liquidates any remaining long position at the current best bid."""
        liquidation_pnl = 0.0
        if self.long_position > 1e-9: # Use tolerance
            logging.info(f"Liquidating {self.long_position:.3f} units at best bid ${self.best_bid:.2f} (Avg Cost: ${self.long_avg_cost:.2f})")

            executed_value = self.best_bid * self.long_position
            transaction_cost = self.config["transaction_cost"] * executed_value

            # Calculate PnL relative to average cost
            if self.long_avg_cost > 0:
                 pnl_from_liq = (self.best_bid - self.long_avg_cost) * self.long_position
            else:
                 pnl_from_liq = executed_value # Assume cost was 0 if avg_cost is invalid

            liquidation_pnl = pnl_from_liq - transaction_cost

            self.cash += (executed_value - transaction_cost)
            self.long_position = 0.0
            self.inventory = 0.0
            self.long_avg_cost = 0.0
        # else: No liquidation needed

        # Clear any remaining active orders as the episode is ending
        if self.active_orders:
             # logging.debug(f"Clearing {len(self.active_orders)} active orders at episode end.")
             self.active_orders = []
        self.order_queue = deque([None] * self.config["latency_steps"]) # Reset queue

        return liquidation_pnl

    def close(self):
        """Clean up any resources."""
        logging.info("Closing HFT Environment.")
        pass

# --- Example Configuration and Usage ---
config = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_large.csv", # <<< --- UPDATE THIS PATH --- >>>
    "initial_capital": 20000.0,
    "max_steps": 5000, # Reduced steps for quicker testing
    "order_book_levels": 9,
    "price_offset_ticks": 40,
    "max_order_volume": 2.5,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.005,
    "max_active_orders": 10, # Reduced for testing clarity
    "inventory_penalty": 0.0,
    "transaction_cost": 0.000, # 0.05%
    "max_inventory": 8,
    "invalid_order_penalty": 0.5,
    "activity_bonus": 0.00
}

if __name__ == "__main__":
    print(f"Initializing HFTEnv with config:\n{config}")
    try:
        env = HFTEnv(config)
        obs, info = env.reset()

        print("\nEnvironment Initialized.")
        print("Observation Space Shape:", env.observation_space.shape)
        print("Action Space Shape:", env.action_space.shape)
        # print("Initial Observation (clipped):", obs)
        print("Initial Info:", info)
        print("-" * 60)

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        render_interval = 200 # Render every N steps
        print(f"Running simulation for max {env.max_steps} steps (rendering every {render_interval} steps)...")

        while not terminated and not truncated:
            # Sample a random action
            action = env.action_space.sample()

            # Simple strategy for testing: try to buy low, sell high slightly
            # if info['inventory'] < env.config['max_inventory'] / 2:
            #     action[0] = 0.5 # Try to buy
            #     action[1] = -0.1 # Place buy slightly below mid
            # else:
            #     action[0] = -0.5 # Try to sell
            #     action[1] = 0.1 # Place sell slightly above mid
            # action[2] = 0.05 # Cancel fraction


            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count == 1 or step_count % render_interval == 0 or terminated or truncated:
                 print(f"\n>>> Step {step_count} <<<")
                 env.render()
                 print(f"  Action Taken: [Vol: {action[0]:+.2f}, PxOff: {action[1]:+.2f}, Cancel: {action[2]:.2f}]")
                 print(f"  Reward: {reward:+.4f} | Total Reward: {total_reward:+.4f}")
                 # print(f"  Obs (sample): {obs[:5]}...") # Print first few obs values


        print(f"\n{'='*60}")
        print(f"Episode finished after {step_count} steps.")
        print(f"Termination: {terminated}, Truncation: {truncated}")
        print(f"Final Info: {info}")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"{'='*60}")

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