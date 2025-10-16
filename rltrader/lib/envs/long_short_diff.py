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
    A High-Frequency Trading Environment for Gymnasium.

    This environment simulates trading in a limit order book (LOB) with latency,
    transaction costs, inventory risk penalties, and separate settings for
    long and short positions/orders.

    Args:
        config (dict): Configuration dictionary containing environment parameters.
    """
    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.config = config
        self._validate_config()

        # Load and validate order book data
        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config["max_steps"], len(self.order_book_history))

        # Define action and observation spaces
        # Action: [signed_volume (-1 buy, +1 sell), price_offset (-1 to +1), cancel_fraction (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = self._create_observation_space()

        # Initialize running statistics for observation normalization
        self.running_stats = {
            'mean': None,
            'var': None,
            'count': 0
        }
        self.decay = 0.999  # Exponential decay factor for running stats

        # Position tracking
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0

        self.reset()

    def _validate_config(self):
        """Validates the configuration dictionary."""
        required_keys = [
            "csv_path", "initial_capital", "max_steps", "order_book_levels",
            "price_offset_ticks", "max_order_volume",
            "latency_steps_long",  # NEW: Latency for buy orders
            "latency_steps_short", # NEW: Latency for sell orders
            "tick_size", "lot_size", "max_active_orders", "inventory_penalty",
            "transaction_cost_long", # NEW: Cost for buy orders
            "transaction_cost_short",# NEW: Cost for sell orders
            "max_inventory", "invalid_order_penalty", "activity_bonus"
        ]

        missing_keys = set(required_keys) - set(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Validate numeric types and ranges if necessary
        if self.config["latency_steps_long"] < 0 or self.config["latency_steps_short"] < 0:
             raise ValueError("Latency steps cannot be negative.")
        if self.config["transaction_cost_long"] < 0 or self.config["transaction_cost_short"] < 0:
             raise ValueError("Transaction costs cannot be negative.")

        logging.info("Configuration validated successfully.")

    def _load_order_book_data(self):
        """Loads and preprocesses order book data from a CSV file."""
        logging.info(f"Loading order book data from: {self.config['csv_path']}")
        df = pd.read_csv(self.config["csv_path"])
        required_columns = ["timestamp"]
        for i in range(1, self.config["order_book_levels"] + 1):
            required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV data: {missing_cols}")

        history = []
        row_count = 0
        for _, row in df.iterrows():
            try:
                bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)

                # Data validation
                if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any():
                    logging.warning(f"Invalid (non-positive) prices found at row index {row_count}. Skipping row.")
                    continue # Skip this row
                if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                    logging.warning(f"Negative quantities found at row index {row_count}. Skipping row.")
                    continue # Skip this row
                if not all(bids[i, 0] >= bids[i+1, 0] for i in range(len(bids)-1)) or \
                   not all(asks[i, 0] <= asks[i+1, 0] for i in range(len(asks)-1)) or \
                   (len(bids) > 0 and len(asks) > 0 and bids[0,0] >= asks[0,0]):
                   logging.warning(f"Order book structure violation (bid/ask order or crossing) at row index {row_count}. Skipping row.")
                   continue # Skip invalid LOB state

                history.append({"bids": bids, "asks": asks})
                row_count += 1
            except Exception as e:
                logging.error(f"Error processing row index {row_count}: {e}")
                raise # Re-raise after logging

        if not history:
            raise ValueError("No valid order book data rows loaded. Check CSV file and format.")
        logging.info(f"Successfully loaded {len(history)} valid order book snapshots.")
        return history


    def _create_observation_space(self):
        """Creates the observation space based on configuration."""
        obs_size = (
            9 +  # mtm, inventory, active_orders, long_position, short_position, long_avg_cost_norm, short_avg_cost_norm, bid_dev, ask_dev
            2 * self.config["max_active_orders"] +  # order prices (normalized), volumes (normalized)
            4 * self.config["order_book_levels"] +  # bid/ask prices (normalized), volumes (normalized)
            1  # spread (normalized)
        )
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        logging.info("Resetting environment.")

        # Reset running statistics for normalization
        self.running_stats = {'mean': None, 'var': None, 'count': 0}

        # Liquidate any existing positions if called mid-episode (though typically called at start)
        if hasattr(self, 'long_position') and (self.long_position > 0 or self.short_position > 0):
            logging.warning("Reset called with active positions. Liquidating...")
            self._liquidate_positions() # Ensure clean state

        self.current_step = 0
        self.cash = self.config["initial_capital"]
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.inventory = 0.0  # Calculated as long - short

        self.active_orders = []
        self.pending_orders = [] # NEW: List to store orders waiting due to latency (tuple: (target_step, order_dict))
        self.order_id_counter = 0
        self.step_count = 0
        self.last_executed_volume = 0.0 # For activity bonus calculation

        # Initialize market state
        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        self._update_market_state() # Calculate midprice, best bid/ask etc.

        observation = self._get_observation()
        info = self._get_info()
        logging.info(f"Reset complete. Initial MTM: {info['mtm']:.2f}")
        return observation, info

    def _update_market_state(self):
        """
        Updates internal market state variables (best bid/ask, midprice, spread)
        based on the current order book snapshot.
        """
        # Basic validation already done in _load_order_book_data
        if len(self.bids) == 0 or len(self.asks) == 0:
             # This should ideally not happen if data loading is correct, but handle defensively
             logging.warning(f"Step {self.current_step}: Empty bids or asks encountered. Using previous state or defaults.")
             # Keep previous best_bid, best_ask, midprice or set to NaN/default if first step
             if not hasattr(self, 'best_bid'):
                 self.best_bid = np.nan
                 self.best_ask = np.nan
                 self.midprice = np.nan
                 self.spread = np.nan
             return # Avoid calculation errors

        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        # Handle potential crossing books if data isn't perfectly clean (though filtered in loading)
        if self.best_bid >= self.best_ask:
            logging.warning(f"Step {self.current_step}: Order book crossed or locked (Best Bid: {self.best_bid}, Best Ask: {self.best_ask}). Adjusting midprice calculation.")
            # Use a small default spread or alternative logic if needed
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = 0.0 # Or a minimal tick size?
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid

    def step(self, action):
        """Executes one time step within the environment."""
        self.step_count += 1
        logging.debug(f"--- Step {self.current_step} ({self.step_count}/{self.max_steps}) ---")
        logging.debug(f"Action received: {action}")

        # 1. Parse Action
        signed_volume, price_offset, cancel_fraction = action
        is_buy = signed_volume > 0.0
        volume_scaled = abs(signed_volume)

        # 2. Handle Order Cancellation
        self._handle_order_cancellation(cancel_fraction)

        # 3. Place New Order (if volume > 0 and space available)
        order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)

        # 4. Process Latency: Move matured pending orders to active orders
        self._process_pending_orders()

        # 5. Execute Active Orders against current LOB
        realized_pnl_from_fills = self._execute_orders()

        # 6. Update Inventory
        self.inventory = self.long_position - self.short_position

        # 7. Check Termination Conditions
        terminated = (
            self.cash < 0 or
            abs(self.inventory) > self.config["max_inventory"] or
            self.current_step >= len(self.order_book_history) - 1 # End of data
        )
        truncated = self.step_count >= self.max_steps # Max steps reached

        # 8. Liquidate Positions at the End
        liquidation_pnl = 0.0
        if terminated or truncated:
            logging.info(f"Episode end condition met (Terminated: {terminated}, Truncated: {truncated}). Liquidating positions.")
            liquidation_pnl = self._liquidate_positions()
            # Update MTM after liquidation for final info dict
            self.inventory = 0 # Ensure inventory is zero after liquidation

        # 9. Calculate Mark-to-Market (MTM)
        # Use midprice for unrealized value. If midprice is NaN, maybe use last valid or zero.
        current_inventory_value = self.inventory * self.midprice if not np.isnan(self.midprice) else 0
        mtm = self.cash + current_inventory_value

        # 10. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty()
        activity_bonus = self.config["activity_bonus"] * self.last_executed_volume

        # 11. Calculate Reward
        # Total PnL for the step = Realized PnL from fills + Unrealized PnL change (implicitly handled by MTM in some RL algos)
        # We use realized PnL + liquidation PnL (if applicable) adjusted by penalties/bonuses.
        reward = realized_pnl_from_fills + liquidation_pnl - risk_penalty + order_penalty + activity_bonus
        logging.debug(f"Step {self.current_step} - PNL(fills): {realized_pnl_from_fills:.4f}, PNL(liq): {liquidation_pnl:.4f}, RiskPen: {risk_penalty:.4f}, OrderPen: {order_penalty:.4f}, ActBonus: {activity_bonus:.4f}, Total Reward: {reward:.4f}")

        # 12. Advance Market Data to Next Step (if not terminated)
        if not (terminated or truncated):
             self._update_step_state()
        else:
             # If terminated/truncated, we might not have a next state's midprice for final MTM.
             # Ensure self.midprice reflects the state *before* liquidation for the final info dict.
             pass # self.midprice is already set based on the current_step's data

        # 13. Get Observation and Info
        observation = self._get_observation()
        info = self._get_info() # Info should reflect state *after* actions but *before* market moves to next step if not terminated
        logging.debug(f"Step {self.current_step} End - MTM: {info['mtm']:.2f}, Inv: {self.inventory:.4f}, Cash: {self.cash:.2f}")

        return observation, reward, terminated, truncated, info

    def _handle_order_cancellation(self, cancel_fraction):
        """Cancels a fraction of the oldest active orders."""
        if not self.active_orders:
            return

        num_to_cancel = int(np.clip(cancel_fraction, 0.0, 1.0) * len(self.active_orders))
        if num_to_cancel > 0:
            logging.debug(f"Cancelling {num_to_cancel} oldest orders.")
            # Remove from the beginning (oldest)
            self.active_orders = self.active_orders[num_to_cancel:]

    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        """Places a new order, applying penalties for invalid placements."""
        penalty = 0.0
        # Check if max active orders limit reached
        if len(self.active_orders) + len(self.pending_orders) >= self.config["max_active_orders"]:
            logging.debug("Max active/pending orders reached. Cannot place new order.")
            # Apply penalty? Optional. Current setup doesn't penalize this.
            return penalty # No penalty for now, just can't place

        # Calculate Limit Price
        price_offset_abs = price_offset * self.config["price_offset_ticks"] * self.config["tick_size"]
        # Reference price is midprice
        limit_price = self.midprice + price_offset_abs # Offset applied relative to midprice
        limit_price = self._round_to_tick(limit_price)

        # Check if order price is valid relative to BBO (Apply Penalty if outside reasonable bounds)
        # These bounds are somewhat arbitrary but prevent placing orders way too far
        max_allowable_buy_price = self.best_ask + self.config["price_offset_ticks"] * self.config["tick_size"]
        min_allowable_sell_price = self.best_bid - self.config["price_offset_ticks"] * self.config["tick_size"]

        if is_buy and limit_price > max_allowable_buy_price:
            logging.debug(f"Buy order price {limit_price} too aggressive (>{max_allowable_buy_price}). Applying penalty.")
            penalty -= self.config["invalid_order_penalty"]
            # Optionally: Reject the order entirely, or clip the price
            # return penalty # Reject order if penalty applied
            limit_price = self._round_to_tick(max_allowable_buy_price) # Clip price instead of rejecting

        if not is_buy and limit_price < min_allowable_sell_price:
            logging.debug(f"Sell order price {limit_price} too aggressive (<{min_allowable_sell_price}). Applying penalty.")
            penalty -= self.config["invalid_order_penalty"]
            # Optionally: Reject the order entirely, or clip the price
            # return penalty # Reject order if penalty applied
            limit_price = self._round_to_tick(min_allowable_sell_price) # Clip price instead of rejecting

        # Calculate Volume
        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0.0, round(volume / self.config["lot_size"]) * self.config["lot_size"])
        if volume <= 1e-9: # Use tolerance for float comparison
            logging.debug("Order volume is effectively zero. Not placing order.")
            return penalty # No order placed, return existing penalty

        # Determine Latency and Target Step
        latency = self.config["latency_steps_long"] if is_buy else self.config["latency_steps_short"]
        target_step = self.current_step + latency

        # Create and queue the order
        order = {
            "id": self.order_id_counter,
            "price": limit_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp_placed": self.current_step, # Track when it was submitted
            "target_step": target_step # NEW: Track when it should become active
        }
        self.order_id_counter += 1
        self.pending_orders.append(order) # Add to pending list
        logging.debug(f"Placed Order {order['id']}: {'Buy' if is_buy else 'Sell'} {volume:.4f} @ {limit_price:.2f}. Target Step: {target_step}")

        return penalty

    def _round_to_tick(self, price):
        """Rounds a price to the nearest tick size."""
        if np.isnan(price) or self.config["tick_size"] <= 0:
             return price # Cannot round NaN or with invalid tick size
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_pending_orders(self):
        """Moves orders from the pending list to active orders if their latency period is over."""
        still_pending = []
        activated_count = 0
        for order in self.pending_orders:
            if self.current_step >= order["target_step"]:
                # Check again if we have space for active orders right before activation
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(order)
                    activated_count += 1
                else:
                    # If max active orders is reached *now*, the order effectively "fails"
                    # or could be kept pending, but simpler to discard for now.
                    logging.warning(f"Order {order['id']} reached target step {order['target_step']} at current step {self.current_step}, but max active orders ({self.config['max_active_orders']}) limit reached. Order discarded.")
                    # Alternatively, keep it in still_pending to retry next step:
                    # still_pending.append(order)
            else:
                still_pending.append(order) # Keep in pending list

        if activated_count > 0:
             logging.debug(f"Activated {activated_count} orders from pending list.")
        self.pending_orders = still_pending


    def _execute_orders(self):
        """Matches active orders against the current order book levels."""
        realized_pnl = 0.0
        total_transaction_costs = 0.0
        self.last_executed_volume = 0.0 # Reset executed volume for this step
        new_active_orders = [] # Orders that remain partially or fully unfilled

        # Sort orders? Maybe by price-time priority? For simplicity, process as is.
        # A more realistic simulation might prioritize more aggressive prices.

        for order in self.active_orders:
            remaining_volume = order["volume"]
            is_buy = order["is_buy"]
            limit_price = order["price"]

            # Determine relevant book side and transaction cost rate
            levels = self.asks if is_buy else self.bids
            cost_rate = self.config["transaction_cost_long"] if is_buy else self.config["transaction_cost_short"]

            for level_idx in range(len(levels)):
                level_price, level_qty = levels[level_idx]

                if remaining_volume <= 1e-9: # Effectively zero remaining
                    break

                # Check if price condition is met
                can_fill_at_level = (is_buy and limit_price >= level_price) or \
                                    (not is_buy and limit_price <= level_price)

                if can_fill_at_level:
                    fill_qty_possible = min(remaining_volume, level_qty)

                    # Check inventory limits *before* processing the fill
                    current_net_inventory = self.long_position - self.short_position
                    inventory_change = fill_qty_possible if is_buy else -fill_qty_possible
                    potential_new_inventory = current_net_inventory + inventory_change

                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        # Calculate how much can be filled without exceeding the limit
                        if inventory_change > 0: # Trying to increase inventory (buy or cover short)
                            allowed_increase = self.config["max_inventory"] - current_net_inventory
                            fill_qty = max(0.0, min(fill_qty_possible, allowed_increase))
                        else: # Trying to decrease inventory (sell or cover long)
                            allowed_decrease = self.config["max_inventory"] + current_net_inventory # current_inv is negative if short
                            fill_qty = max(0.0, min(fill_qty_possible, allowed_decrease))

                        logging.debug(f"Inventory limit potentially breached. Adjusted fill from {fill_qty_possible:.4f} to {fill_qty:.4f}")
                    else:
                        fill_qty = fill_qty_possible

                    if fill_qty <= 1e-9: # No fill possible due to limits or level empty
                        continue # Check next level

                    # Process the fill
                    fill_price = level_price # Filled at the level's price
                    executed_value = fill_qty * fill_price
                    transaction_cost = cost_rate * executed_value

                    # Update cash (subtract cost here)
                    self.cash -= transaction_cost
                    total_transaction_costs += transaction_cost
                    if is_buy:
                        self.cash -= executed_value
                    else:
                        self.cash += executed_value

                    # Update positions and calculate realized PnL from this fill
                    pnl_from_fill = self._process_fill(fill_qty, fill_price, is_buy)
                    realized_pnl += pnl_from_fill

                    # Update tracking variables
                    remaining_volume -= fill_qty
                    self.last_executed_volume += fill_qty
                    levels[level_idx, 1] -= fill_qty # Reduce quantity on the matched LOB level (in-memory modification for this step)

                    logging.debug(f"Order {order['id']} filled: {fill_qty:.4f} @ {fill_price:.2f}. PNL: {pnl_from_fill:.4f}. Cost: {transaction_cost:.4f}. Cash: {self.cash:.2f}")

            # If order is not fully filled, add the remainder back to the list
            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)
                logging.debug(f"Order {order['id']} partially filled. Remaining vol: {remaining_volume:.4f}")
            else:
                 logging.debug(f"Order {order['id']} fully filled.")


        self.active_orders = new_active_orders
        # The returned PnL already accounts for transaction costs applied during fill processing
        return realized_pnl # Note: Transaction costs were subtracted from cash directly

    def _process_fill(self, fill_qty, price, is_buy):
        """Updates positions (long/short) based on a fill and calculates realized PnL."""
        realized_pnl = 0.0

        if is_buy:
            # Buying: Can either cover a short or add to a long position
            if self.short_position > 0:
                cover_qty = min(fill_qty, self.short_position)
                # PnL from covering short = (Avg Short Cost - Fill Price) * Qty Covered
                if self.short_avg_cost > 0: # Avoid PnL calc if avg cost is zero
                    realized_pnl += (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                # If short position becomes zero, reset avg cost
                if self.short_position <= 1e-9:
                    self.short_position = 0.0
                    self.short_avg_cost = 0.0

                remaining_fill = fill_qty - cover_qty
                if remaining_fill > 1e-9:
                    # Add remaining qty to long position
                    self._add_to_long(remaining_fill, price)
            else:
                # No short position, add directly to long
                self._add_to_long(fill_qty, price)
        else:
            # Selling: Can either close a long or add to a short position
            if self.long_position > 0:
                close_qty = min(fill_qty, self.long_position)
                # PnL from closing long = (Fill Price - Avg Long Cost) * Qty Closed
                if self.long_avg_cost > 0: # Avoid PnL calc if avg cost is zero
                     realized_pnl += (price - self.long_avg_cost) * close_qty
                self.long_position -= close_qty
                 # If long position becomes zero, reset avg cost
                if self.long_position <= 1e-9:
                    self.long_position = 0.0
                    self.long_avg_cost = 0.0

                remaining_fill = fill_qty - close_qty
                if remaining_fill > 1e-9:
                    # Add remaining qty to short position
                    self._add_to_short(remaining_fill, price)
            else:
                # No long position, add directly to short
                self._add_to_short(fill_qty, price)

        return realized_pnl

    def _add_to_long(self, qty, price):
        """Helper to add quantity to the long position and update average cost."""
        if qty <= 1e-9: return
        new_total_qty = self.long_position + qty
        if self.long_position <= 1e-9: # If starting new long position
             self.long_avg_cost = price
        else:
             self.long_avg_cost = (
                 (self.long_avg_cost * self.long_position) + (price * qty)
             ) / new_total_qty
        self.long_position = new_total_qty


    def _add_to_short(self, qty, price):
        """Helper to add quantity to the short position and update average cost."""
        if qty <= 1e-9: return
        new_total_qty = self.short_position + qty
        if self.short_position <= 1e-9: # If starting new short position
            self.short_avg_cost = price
        else:
            self.short_avg_cost = (
                (self.short_avg_cost * self.short_position) + (price * qty)
            ) / new_total_qty
        self.short_position = new_total_qty


    def _calculate_risk_penalty(self):
        """Calculates a penalty based on the current net inventory."""
        if self.config["max_inventory"] <= 0:
            return 0.0 # Avoid division by zero if max_inventory is not set properly

        # Use net inventory (long - short)
        net_inventory = self.long_position - self.short_position
        normalized_inventory = net_inventory / self.config["max_inventory"]

        # Quadratic penalty, scaled by midprice to make it value-based
        penalty = (
            self.config["inventory_penalty"] *
            (normalized_inventory ** 2) *
            abs(self.midprice) # Use abs in case midprice is somehow negative
        )
        # Penalty should be positive (a cost)
        return abs(penalty)


    def _update_step_state(self) -> None:
        """Advances the environment time by one step, updating the order book."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            # Update LOB from historical data
            current_lob_data = self.order_book_history[self.current_step]
            self.bids = current_lob_data["bids"].copy() # Use copy to avoid modifying history
            self.asks = current_lob_data["asks"].copy() # Use copy
            self._update_market_state() # Recalculate best bid/ask, midprice, spread
        else:
            # This case should be handled by the terminated flag already
             logging.warning("Tried to update step state beyond available data.")


    def _get_raw_observation(self):
        """Constructs the raw observation vector before normalization."""
        # Use safe defaults if midprice is NaN
        midprice_safe = self.midprice if not np.isnan(self.midprice) and self.midprice != 0 else 1.0 # Avoid division by zero
        best_bid_safe = self.best_bid if not np.isnan(self.best_bid) else midprice_safe
        best_ask_safe = self.best_ask if not np.isnan(self.best_ask) else midprice_safe
        spread_safe = best_ask_safe - best_bid_safe if best_ask_safe > best_bid_safe else 0.0

        # MTM Calculation - Use midprice_safe
        inventory_value = (self.long_position - self.short_position) * midprice_safe
        mtm = self.cash + inventory_value

        # Normalize costs relative to current midprice
        long_avg_cost_norm = (self.long_avg_cost / midprice_safe) - 1.0 if self.long_position > 0 else 0.0
        short_avg_cost_norm = (self.short_avg_cost / midprice_safe) - 1.0 if self.short_position > 0 else 0.0

        market_features = [
            mtm / self.config["initial_capital"], # Normalize MTM by initial capital
            (self.long_position - self.short_position) / self.config["max_inventory"] if self.config["max_inventory"] > 0 else 0.0, # Normalized net inventory
            len(self.active_orders) / self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 0.0, # Normalized count of active orders
            self.long_position / self.config["max_inventory"] if self.config["max_inventory"] > 0 else 0.0, # Normalized long position
            self.short_position / self.config["max_inventory"] if self.config["max_inventory"] > 0 else 0.0, # Normalized short position
            long_avg_cost_norm,
            short_avg_cost_norm,
            (best_bid_safe - midprice_safe) / midprice_safe, # Normalized bid deviation
            (best_ask_safe - midprice_safe) / midprice_safe  # Normalized ask deviation
        ]

        # Active Order Features (normalized)
        order_prices_norm = []
        order_volumes_norm = []
        # Consider both active and pending orders? Or just active? Let's stick to active for now.
        orders_to_show = sorted(self.active_orders, key=lambda x: x['price'])
        orders_to_show = orders_to_show[:self.config["max_active_orders"]] # Limit

        max_vol_safe = self.config["max_order_volume"] if self.config["max_order_volume"] > 0 else 1.0
        for order in orders_to_show:
            order_prices_norm.append((order["price"] - midprice_safe) / midprice_safe)
            order_volumes_norm.append(order["volume"] / max_vol_safe * (1 if order['is_buy'] else -1)) # Signed normalized volume

        # Pad with zeros
        order_prices_norm += [0.0] * (self.config["max_active_orders"] - len(order_prices_norm))
        order_volumes_norm += [0.0] * (self.config["max_active_orders"] - len(order_volumes_norm))

        # Order Book Features (normalized)
        book_features = []
        for level in range(self.config["order_book_levels"]):
            bid_price_norm = (self.bids[level, 0] - midprice_safe) / midprice_safe if len(self.bids) > level else 0.0
            bid_qty_norm = self.bids[level, 1] / max_vol_safe if len(self.bids) > level else 0.0
            ask_price_norm = (self.asks[level, 0] - midprice_safe) / midprice_safe if len(self.asks) > level else 0.0
            ask_qty_norm = self.asks[level, 1] / max_vol_safe if len(self.asks) > level else 0.0
            book_features.extend([bid_price_norm, bid_qty_norm, ask_price_norm, ask_qty_norm])

        # Final Feature: Normalized Spread
        spread_norm = spread_safe / midprice_safe

        # Concatenate all features
        raw_obs_list = market_features + order_prices_norm + order_volumes_norm + book_features + [spread_norm]

        # Ensure all values are finite, replace NaN/inf with 0
        raw_obs_array = np.array(raw_obs_list, dtype=np.float32)
        raw_obs_array[~np.isfinite(raw_obs_array)] = 0.0

        return raw_obs_array


    def _get_observation(self):
        """Gets the current observation, normalizes it using running stats."""
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        normalized_obs = self._normalize_observation(raw_obs)
        # Clip observations to a reasonable range if needed? e.g., [-5, 5]
        # normalized_obs = np.clip(normalized_obs, -5.0, 5.0)
        return normalized_obs

    def _update_running_stats(self, obs):
        """Updates the running mean and variance using exponential moving average."""
        if self.running_stats['mean'] is None:
            self.running_stats['mean'] = obs
            self.running_stats['var'] = np.zeros_like(obs, dtype=np.float32)
            self.running_stats['count'] = 1
        else:
            self.running_stats['count'] += 1
            # Welford's algorithm update (more stable) or EMA
            # Using EMA for simplicity here:
            old_mean = self.running_stats['mean']
            new_mean = old_mean * self.decay + obs * (1 - self.decay)
            self.running_stats['mean'] = new_mean

            # Update variance using EMA
            delta_sq = (obs - old_mean) * (obs - new_mean) # Variance update needs care with EMA
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta_sq
            # Ensure variance is non-negative
            self.running_stats['var'] = np.maximum(self.running_stats['var'], 0.0)


    def _normalize_observation(self, obs):
        """Normalizes the observation using the running mean and variance."""
        if self.running_stats['mean'] is None or self.running_stats['count'] < 2: # Need at least 2 points for variance
            return obs # Return raw observation if not enough stats

        mean = self.running_stats['mean']
        variance = self.running_stats['var']
        std_dev = np.sqrt(variance)

        # Normalize: (obs - mean) / (std_dev + epsilon)
        normalized_obs = (obs - mean) / (std_dev + 1e-8)
        return normalized_obs

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        # Calculate MTM based on the *current* state's midprice
        inventory_value = (self.long_position - self.short_position) * self.midprice if not np.isnan(self.midprice) else 0
        mtm = self.cash + inventory_value

        return {
            "mtm": mtm,
            "cash": self.cash,
            "net_inventory": self.long_position - self.short_position,
            "long_position": self.long_position,
            "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost,
            "short_avg_cost": self.short_avg_cost,
            "active_orders_count": len(self.active_orders),
            "pending_orders_count": len(self.pending_orders),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.midprice,
            "current_step": self.current_step,
            "total_steps_elapsed": self.step_count,
            "episode_pnl": mtm - self.config["initial_capital"] # Total PnL relative to start
        }

    def render(self, mode="human"):
        """Renders the environment state (prints to console)."""
        if mode == "human":
            info = self._get_info()
            render_msg = (
                f"\n--- Step: {info['current_step']} / Total Elapsed: {info['total_steps_elapsed']} ---\n"
                f"  Market: Bid={info['best_bid']:.2f} | Ask={info['best_ask']:.2f} | Mid={info['mid_price']:.2f}\n"
                f"  Portfolio: MTM=${info['mtm']:,.2f} | Cash=${info['cash']:,.2f} | PnL=${info['episode_pnl']:,.2f}\n"
                f"  Position: Net={info['net_inventory']:.4f} (L:{info['long_position']:.4f}@{info['long_avg_cost']:.2f}, S:{info['short_position']:.4f}@{info['short_avg_cost']:.2f})\n"
                f"  Orders: Active={info['active_orders_count']} | Pending={info['pending_orders_count']}\n"
                f"{'='*40}"
            )
            # Use logging to print, respects logging level
            logging.info(render_msg)
        else:
            super().render(mode=mode) # Or raise NotImplementedError for other modes

    def _liquidate_positions(self):
        """Liquidates all open positions at current best market prices and cancels orders."""
        logging.info("Liquidating positions...")
        liquidation_pnl = 0.0
        total_liq_cost = 0.0

        # Use current best bid/ask for liquidation
        # These should be available from the last _update_market_state call
        best_bid_liq = self.best_bid if not np.isnan(self.best_bid) else self.midprice # Fallback
        best_ask_liq = self.best_ask if not np.isnan(self.best_ask) else self.midprice # Fallback
        if np.isnan(best_bid_liq) or np.isnan(best_ask_liq):
             logging.error("Cannot liquidate - market prices unavailable (NaN). Setting PnL to 0.")
             return 0.0 # Cannot calculate PnL without prices

        # Liquidate long position by selling at the best bid
        if self.long_position > 0:
            qty = self.long_position
            price = best_bid_liq
            value = qty * price
            # Selling uses the SHORT transaction cost rate
            cost = self.config["transaction_cost_short"] * value
            pnl = (price - self.long_avg_cost) * qty
            self.cash += value - cost
            liquidation_pnl += pnl - cost
            total_liq_cost += cost
            logging.info(f"Liquidated LONG: {qty:.4f} @ {price:.2f}. PNL: {pnl-cost:.2f} (Cost: {cost:.2f})")
            self.long_position = 0.0
            self.long_avg_cost = 0.0

        # Cover short position by buying at the best ask
        if self.short_position > 0:
            qty = self.short_position
            price = best_ask_liq
            value = qty * price
            # Buying uses the LONG transaction cost rate
            cost = self.config["transaction_cost_long"] * value
            pnl = (self.short_avg_cost - price) * qty
            self.cash -= value + cost # Buying costs cash, plus transaction cost
            liquidation_pnl += pnl - cost
            total_liq_cost += cost
            logging.info(f"Liquidated SHORT: {qty:.4f} @ {price:.2f}. PNL: {pnl-cost:.2f} (Cost: {cost:.2f})")
            self.short_position = 0.0
            self.short_avg_cost = 0.0

        # Clear all orders
        self.active_orders = []
        self.pending_orders = []

        self.inventory = 0.0 # Ensure inventory is zero
        logging.info(f"Liquidation complete. Total Liq PNL: {liquidation_pnl:.2f}. Final Cash: {self.cash:.2f}")
        return liquidation_pnl

    def close(self):
        """Clean up any resources (if any were used)."""
        logging.info("Closing HFT Environment.")
        # No external resources to close in this version
        pass


# Example Configuration with separate latency and costs
config = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_small.csv", # UPDATE THIS PATH
    "initial_capital": 20000.0,       # Starting cash
    "max_steps": 10000,               # Max steps per episode
    "order_book_levels": 9,           # Number of LOB levels to use
    "price_offset_ticks": 40,         # Max price offset from midprice in ticks for placing orders
    "max_order_volume": 2.5,          # Max volume for a single order (scaled by action)
    "tick_size": 0.01,                # Smallest price increment
    "lot_size": 0.001,                # Smallest volume increment
    "max_active_orders": 15,          # Max number of orders waiting in the book
    "max_inventory": 8.0,             # Max absolute net inventory allowed

    # --- NEW/UPDATED Parameters ---
    "latency_steps_long": 2,          # Latency (in steps) for BUY orders to become active
    "latency_steps_short": 1,         # Latency (in steps) for SELL orders to become active
    "transaction_cost_long": 0.0005,  # Percentage cost for BUY executions (e.g., 0.05%)
    "transaction_cost_short": 0.0005, # Percentage cost for SELL executions (e.g., 0.05%)
    "inventory_penalty": 0.01,        # Penalty factor for holding inventory (quadratic)
    "invalid_order_penalty": 0.5,     # Penalty for placing orders too far from BBO
    "activity_bonus": 0.001,          # Bonus per unit of volume executed
    # --- End NEW/UPDATED ---
}

# Example Usage (Basic Test)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured
    try:
        env = HFTEnv(config)
        obs, info = env.reset()
        print("Environment Initialized.")
        print("Initial Observation Shape:", obs.shape)
        print("Initial Info:", info)

        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # Take random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            if step_count % 500 == 0: # Render every 500 steps
                 env.render()
                 # Optional: Print raw obs stats for debugging normalization
                 # raw_obs = env._get_raw_observation()
                 # print(f"Raw Obs Stats (Step {step_count}): Mean={np.mean(raw_obs):.3f}, Std={np.std(raw_obs):.3f}, Min={np.min(raw_obs):.3f}, Max={np.max(raw_obs):.3f}")
                 # print(f"Norm Obs Stats (Step {step_count}): Mean={np.mean(obs):.3f}, Std={np.std(obs):.3f}, Min={np.min(obs):.3f}, Max={np.max(obs):.3f}")


        print("\n--- Episode Finished ---")
        print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
        print(f"Total Steps: {step_count}")
        print(f"Final Info: {info}")
        print(f"Total Reward: {total_reward:.4f}")

        env.close()

    except FileNotFoundError:
        logging.error(f"FATAL ERROR: CSV file not found at path: {config['csv_path']}")
        logging.error("Please update the 'csv_path' in the config dictionary.")
    except ValueError as e:
        logging.error(f"FATAL ERROR during environment setup or execution: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)