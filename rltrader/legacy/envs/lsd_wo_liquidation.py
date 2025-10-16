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

    NOTE: Forced liquidation at the end of the episode has been removed.
          The episode ends with the final held position.
          Reset still liquidates to ensure clean starts.

    Args:
        config (dict): Configuration dictionary containing environment parameters.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10} # Added metadata

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
            "latency_steps_long",  # Latency for buy orders
            "latency_steps_short", # Latency for sell orders
            "tick_size", "lot_size", "max_active_orders", "inventory_penalty",
            "transaction_cost_long", # Cost for buy orders
            "transaction_cost_short",# Cost for sell orders
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
        try:
            df = pd.read_csv(self.config["csv_path"])
        except FileNotFoundError:
            logging.error(f"CSV file not found at {self.config['csv_path']}")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise

        required_columns = ["timestamp"]
        for i in range(1, self.config["order_book_levels"] + 1):
            required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]

        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV data: {missing_cols}")

        history = []
        row_count = 0
        for index, row in df.iterrows(): # Use index for better error reporting
            row_count = index # Keep track of original row index for logging
            try:
                bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)

                # Data validation
                if np.isnan(bids).any() or np.isnan(asks).any():
                    logging.warning(f"NaN values found in LOB data at row index {row_count}. Skipping row.")
                    continue
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
            except KeyError as e:
                 logging.error(f"Missing column key '{e}' processing row index {row_count}. Ensure all bid/ask/qty columns exist for levels 1 to {self.config['order_book_levels']}.")
                 raise # Re-raise after logging column specific error
            except Exception as e:
                logging.error(f"Error processing row index {row_count}: {e}")
                # Depending on severity, might want to 'continue' or 'raise'
                raise # Re-raise other unexpected errors

        if not history:
            raise ValueError("No valid order book data rows loaded. Check CSV file, format, and validation warnings.")
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

        # Liquidate any existing positions IF CALLED MID-EPISODE.
        # This ensures each episode starts clean.
        if hasattr(self, 'current_step') and self.current_step > 0 and (self.long_position > 1e-9 or self.short_position > 1e-9):
            logging.warning("Reset called mid-episode or with residual positions. Liquidating to ensure clean start.")
            self._liquidate_positions()

        self.current_step = 0
        self.cash = self.config["initial_capital"]
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.inventory = 0.0  # Calculated as long - short

        self.active_orders = []
        self.pending_orders = [] # List to store orders waiting due to latency (tuple: (target_step, order_dict))
        self.order_id_counter = 0
        self.step_count = 0
        self.last_executed_volume = 0.0 # For activity bonus calculation

        # Initialize market state
        if not self.order_book_history: # Should not happen if constructor succeeded
             raise RuntimeError("Order book history is empty during reset.")
        self.bids = self.order_book_history[0]["bids"].copy() # Use copy
        self.asks = self.order_book_history[0]["asks"].copy() # Use copy
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
        if self.bids is None or self.asks is None or len(self.bids) == 0 or len(self.asks) == 0:
             logging.warning(f"Step {self.current_step}: Empty or invalid bids/asks encountered during update. Using previous state or defaults.")
             # Keep previous values if they exist, otherwise set to NaN
             if not hasattr(self, 'best_bid'): self.best_bid = np.nan
             if not hasattr(self, 'best_ask'): self.best_ask = np.nan
             if not hasattr(self, 'midprice'): self.midprice = np.nan
             if not hasattr(self, 'spread'): self.spread = np.nan
             return

        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]

        if np.isnan(self.best_bid) or np.isnan(self.best_ask):
             logging.warning(f"Step {self.current_step}: NaN best bid/ask detected. Setting midprice/spread to NaN.")
             self.midprice = np.nan
             self.spread = np.nan
        elif self.best_bid >= self.best_ask:
            logging.warning(f"Step {self.current_step}: Order book crossed or locked (Best Bid: {self.best_bid}, Best Ask: {self.best_ask}). Adjusting midprice calculation.")
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = max(0.0, self.best_ask - self.best_bid) # Spread is 0 or negative
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid

    def step(self, action):
        """Executes one time step within the environment."""
        self.step_count += 1
        logging.debug(f"--- Step {self.current_step} ({self.step_count}/{self.max_steps}) ---")
        logging.debug(f"Action received: {action}")

        # 0. Check if data exists for current step (precautionary)
        if self.current_step >= len(self.order_book_history):
             logging.error(f"Attempted to step beyond available data (Step {self.current_step}). This should have been caught by termination/truncation.")
             # Handle this gracefully - perhaps return last state? Or raise error?
             # Let's treat it as truncated.
             last_obs = self._get_observation()
             last_info = self._get_info()
             # Return 0 reward, terminated=False, truncated=True
             return last_obs, 0.0, False, True, last_info

        # 1. Parse Action
        try:
            signed_volume, price_offset, cancel_fraction = action
        except (TypeError, ValueError) as e:
            logging.error(f"Invalid action format received: {action}. Error: {e}. Using zero action.")
            signed_volume, price_offset, cancel_fraction = 0.0, 0.0, 0.0
            # Optionally apply a penalty for invalid actions
            # penalty_for_invalid_action = -1.0 # Example penalty

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

        # 7. Check Termination Conditions (based on state *after* execution)
        terminated = (
            self.cash < 0 or
            abs(self.inventory) > self.config["max_inventory"]
            # Removed end-of-data check here, handled by truncation
        )
        truncated = self.step_count >= self.max_steps or \
                    self.current_step >= len(self.order_book_history) - 1 # Truncate if we are at the last data point

        # --- REMOVED LIQUIDATION BLOCK ---
        # No forced liquidation here. Episode ends with current position.

        # 8. Calculate Mark-to-Market (MTM) - Based on current midprice
        current_inventory_value = self.inventory * self.midprice if not np.isnan(self.midprice) else 0
        mtm = self.cash + current_inventory_value

        # 9. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty()
        activity_bonus = self.config["activity_bonus"] * self.last_executed_volume

        # 10. Calculate Reward
        # Reward = Realized PnL from fills +/- penalties/bonuses for this step.
        # Does NOT include unrealized PnL change or end-of-episode liquidation.
        reward = realized_pnl_from_fills - risk_penalty + order_penalty + activity_bonus
        # reward += penalty_for_invalid_action # Add if implementing penalty for bad actions
        logging.debug(f"Step {self.current_step} - PNL(fills): {realized_pnl_from_fills:.4f}, RiskPen: {risk_penalty:.4f}, OrderPen: {order_penalty:.4f}, ActBonus: {activity_bonus:.4f}, Total Reward: {reward:.4f}")

        # 11. Advance Market Data to Next Step (if not terminated/truncated)
        if not (terminated or truncated):
             self._update_step_state()
        # else: # If terminated or truncated, market state remains as it was at the end

        # 12. Get Observation and Info for the *next* state (or final state if ended)
        observation = self._get_observation()
        info = self._get_info() # Info reflects state *after* actions and market update (if any)
        logging.debug(f"Step {self.current_step} End - MTM: {info['mtm']:.2f}, Inv: {info['net_inventory']:.4f}, Cash: {self.cash:.2f}")

        # 13. Return step results
        # Ensure terminated and truncated flags are boolean
        terminated = bool(terminated)
        truncated = bool(truncated)
        return observation, reward, terminated, truncated, info

    def _handle_order_cancellation(self, cancel_fraction):
        """Cancels a fraction of the oldest active orders."""
        if not self.active_orders:
            return

        # Ensure cancel_fraction is valid
        cancel_fraction = np.clip(cancel_fraction, 0.0, 1.0)

        num_to_cancel = int(cancel_fraction * len(self.active_orders))
        if num_to_cancel > 0:
            logging.debug(f"Cancelling {num_to_cancel} oldest orders out of {len(self.active_orders)}.")
            # Cancel from the beginning (oldest orders assuming FIFO append)
            cancelled_orders = self.active_orders[:num_to_cancel]
            self.active_orders = self.active_orders[num_to_cancel:]
            # Log details of cancelled orders if needed
            # for order in cancelled_orders:
            #     logging.debug(f"Cancelled Order ID: {order['id']}")

    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        """Places a new order, applying penalties for invalid placements."""
        penalty = 0.0
        # Check if max active orders limit reached (including pending)
        if len(self.active_orders) + len(self.pending_orders) >= self.config["max_active_orders"]:
            logging.debug("Max active/pending orders reached ({}/{self.config['max_active_orders']}). Cannot place new order.")
            # Consider adding a penalty for trying to place when full?
            # penalty -= self.config.get("order_queue_full_penalty", 0.1) # Example
            return penalty # Cannot place order

        # Check if midprice is valid for calculation
        if np.isnan(self.midprice):
            logging.warning("Cannot place order: Midprice is NaN.")
            penalty -= self.config["invalid_order_penalty"] # Penalize attempt with invalid market state
            return penalty

        # Calculate Limit Price
        price_offset_abs = price_offset * self.config["price_offset_ticks"] * self.config["tick_size"]
        limit_price = self.midprice + price_offset_abs
        limit_price = self._round_to_tick(limit_price)

        # Basic price validity check
        if limit_price <= 0:
            logging.warning(f"Order price calculated to be non-positive ({limit_price}). Applying penalty, rejecting order.")
            penalty -= self.config["invalid_order_penalty"]
            return penalty # Reject order

        # Check if order price is valid relative to BBO (Apply Penalty if too aggressive)
        if not np.isnan(self.best_ask) and is_buy and limit_price > self.best_ask:
             logging.debug(f"Buy order price {limit_price:.2f} is aggressive (crosses best ask {self.best_ask:.2f}). Applying penalty.")
             penalty -= self.config["invalid_order_penalty"]
             # Option 1: Reject order
             # return penalty
             # Option 2: Clip price to best ask (or slightly better)
             limit_price = self._round_to_tick(self.best_ask) # Clip to best ask

        if not np.isnan(self.best_bid) and not is_buy and limit_price < self.best_bid:
             logging.debug(f"Sell order price {limit_price:.2f} is aggressive (crosses best bid {self.best_bid:.2f}). Applying penalty.")
             penalty -= self.config["invalid_order_penalty"]
             # Option 1: Reject order
             # return penalty
             # Option 2: Clip price to best bid (or slightly worse)
             limit_price = self._round_to_tick(self.best_bid) # Clip to best bid

        # Calculate Volume
        volume = volume_scaled * self.config["max_order_volume"]
        # Round to nearest lot size, ensuring minimum is lot_size if volume > 0
        volume = round(volume / self.config["lot_size"]) * self.config["lot_size"]
        if volume < self.config["lot_size"] and volume_scaled > 1e-6 : # If scaled vol > 0 but rounded vol < lot_size
            volume = self.config["lot_size"] # Set to minimum lot size
        elif volume <= 1e-9: # Use tolerance for float comparison
            logging.debug("Order volume is zero or negligible. Not placing order.")
            return penalty # No order placed, return any penalty accrued so far

        # Check if placing this order would immediately violate max inventory
        # This is a simplified check, as it doesn't account for other orders filling
        current_net_inventory = self.long_position - self.short_position
        potential_inventory_change = volume if is_buy else -volume
        if abs(current_net_inventory + potential_inventory_change) > self.config["max_inventory"]:
            logging.warning(f"Placing order {'Buy' if is_buy else 'Sell'} {volume:.4f} would potentially exceed max inventory ({self.config['max_inventory']}). Applying penalty and rejecting.")
            penalty -= self.config["invalid_order_penalty"] # Penalize attempt
            return penalty

        # Determine Latency and Target Step
        latency = self.config["latency_steps_long"] if is_buy else self.config["latency_steps_short"]
        target_step = self.current_step + latency + 1 # Order becomes active *at the start* of target_step+1

        # Create and queue the order
        order = {
            "id": self.order_id_counter,
            "price": limit_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp_placed": self.current_step,
            "target_step": target_step # Step when it should become active
        }
        self.order_id_counter += 1
        self.pending_orders.append(order)
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
        # Use self.current_step + 1 because target_step indicates the step *after* which it becomes active
        effective_step_for_activation = self.current_step + 1

        for order in self.pending_orders:
            if effective_step_for_activation >= order["target_step"]:
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(order)
                    activated_count += 1
                    logging.debug(f"Order {order['id']} activated at step {self.current_step} (Target: {order['target_step']}).")
                else:
                    logging.warning(f"Order {order['id']} reached target step {order['target_step']} at current step {self.current_step}, but max active orders ({self.config['max_active_orders']}) limit reached. Order discarded.")
                    # Optionally keep pending: still_pending.append(order)
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

        # Make copies of LOB levels to modify during execution for this step only
        current_bids = self.bids.copy() if self.bids is not None else np.array([])
        current_asks = self.asks.copy() if self.asks is not None else np.array([])

        # Sort orders? Optional: by aggressiveness (buy high, sell low) then time?
        # For simplicity, process in current order (FIFO based on activation time)
        # sorted_orders = sorted(self.active_orders, key=lambda o: (-o['price'] if o['is_buy'] else o['price']))

        for order in self.active_orders: # Use original order list
            remaining_volume = order["volume"]
            is_buy = order["is_buy"]
            limit_price = order["price"]

            # Determine relevant book side and transaction cost rate
            levels = current_asks if is_buy else current_bids
            cost_rate = self.config["transaction_cost_long"] if is_buy else self.config["transaction_cost_short"]

            if levels.size == 0: # No liquidity on the required side
                 logging.debug(f"Order {order['id']} ({'Buy' if is_buy else 'Sell'} @ {limit_price:.2f}) - No liquidity available on {'ask' if is_buy else 'bid'} side.")
                 new_active_orders.append(order) # Keep order active
                 continue

            for level_idx in range(len(levels)):
                if remaining_volume <= 1e-9: # Effectively zero remaining
                    break

                level_price, level_qty = levels[level_idx]

                # Check if quantity available at this level
                if level_qty <= 1e-9:
                     continue # Skip empty level

                # Check if price condition is met
                can_fill_at_level = (is_buy and limit_price >= level_price) or \
                                    (not is_buy and limit_price <= level_price)

                if can_fill_at_level:
                    fill_price = level_price # Fill at the level's price

                    # How much can we fill? Limited by order volume, level qty, and inventory limits.
                    fill_qty_possible_at_level = min(remaining_volume, level_qty)

                    # Check inventory limits *before* processing the fill
                    current_net_inventory = self.long_position - self.short_position
                    inventory_change = fill_qty_possible_at_level if is_buy else -fill_qty_possible_at_level
                    potential_new_inventory = current_net_inventory + inventory_change

                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        # Calculate max fill qty allowed by inventory limit
                        if inventory_change > 0: # Trying to increase inventory
                            allowed_increase = self.config["max_inventory"] - current_net_inventory
                            fill_qty = max(0.0, min(fill_qty_possible_at_level, allowed_increase))
                        else: # Trying to decrease inventory (inventory_change < 0)
                            # target = -max_inventory -> allowed_decrease = current - target = current - (-max_inv) = current + max_inv
                            allowed_decrease = self.config["max_inventory"] + current_net_inventory # max amount inv can decrease
                            fill_qty = max(0.0, min(fill_qty_possible_at_level, allowed_decrease))

                        logging.debug(f"Inventory limit potentially breached by order {order['id']}. Adjusted fill from {fill_qty_possible_at_level:.4f} to {fill_qty:.4f}")
                        if fill_qty <= 1e-9:
                             # Cannot fill even minimal amount due to inventory limit, stop trying for this order at this level/further levels?
                             # Or just continue to next level? Let's continue for now.
                             continue
                    else:
                        fill_qty = fill_qty_possible_at_level

                    if fill_qty <= 1e-9: # Double check after potential adjustment
                        continue

                    # --- Process the fill ---
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
                    levels[level_idx, 1] -= fill_qty # Reduce quantity on the matched LOB level (modified copy)

                    logging.debug(f"Order {order['id']} filled: {fill_qty:.4f} @ {fill_price:.2f}. PNL: {pnl_from_fill:.4f}. Cost: {transaction_cost:.4f}. Cash: {self.cash:.2f}")

                    # If fill_qty was limited by inventory, stop filling this order further in this step?
                    # This prevents accidentally filling more against the next level if the limit was hit exactly.
                    if fill_qty < fill_qty_possible_at_level:
                         logging.debug(f"Order {order['id']} hit inventory limit during fill. Stopping fills for this order this step.")
                         break # Stop processing this order against further levels

            # If order is not fully filled, add the remainder back to the list
            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)
                logging.debug(f"Order {order['id']} partially filled or unfilled. Remaining vol: {remaining_volume:.4f}")
            else:
                 logging.debug(f"Order {order['id']} fully filled.")


        self.active_orders = new_active_orders
        # PnL returned already accounts for transaction costs applied during fill processing
        return realized_pnl

    def _process_fill(self, fill_qty, price, is_buy):
        """Updates positions (long/short) based on a fill and calculates realized PnL."""
        realized_pnl = 0.0

        if is_buy:
            # Buying: Can either cover a short or add to a long position
            if self.short_position > 1e-9: # Use tolerance
                cover_qty = min(fill_qty, self.short_position)
                if self.short_avg_cost > 0: # Avoid PnL calc if avg cost is zero/NaN
                    realized_pnl += (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                # If short position becomes zero, reset avg cost
                if self.short_position <= 1e-9:
                    self.short_position = 0.0
                    self.short_avg_cost = 0.0

                remaining_fill = fill_qty - cover_qty
                if remaining_fill > 1e-9:
                    self._add_to_long(remaining_fill, price)
            else:
                self._add_to_long(fill_qty, price)
        else:
            # Selling: Can either close a long or add to a short position
            if self.long_position > 1e-9: # Use tolerance
                close_qty = min(fill_qty, self.long_position)
                if self.long_avg_cost > 0: # Avoid PnL calc if avg cost is zero/NaN
                     realized_pnl += (price - self.long_avg_cost) * close_qty
                self.long_position -= close_qty
                if self.long_position <= 1e-9:
                    self.long_position = 0.0
                    self.long_avg_cost = 0.0

                remaining_fill = fill_qty - close_qty
                if remaining_fill > 1e-9:
                    self._add_to_short(remaining_fill, price)
            else:
                self._add_to_short(fill_qty, price)

        # Update net inventory after processing fill
        self.inventory = self.long_position - self.short_position
        return realized_pnl

    def _add_to_long(self, qty, price):
        """Helper to add quantity to the long position and update average cost."""
        if qty <= 1e-9: return
        new_total_qty = self.long_position + qty
        if self.long_position <= 1e-9: # If starting new long position
             self.long_avg_cost = price
        else:
             # Weighted average cost
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
            # Weighted average cost
            self.short_avg_cost = (
                (self.short_avg_cost * self.short_position) + (price * qty)
            ) / new_total_qty
        self.short_position = new_total_qty


    def _calculate_risk_penalty(self):
        """Calculates a penalty based on the current net inventory."""
        if self.config["max_inventory"] <= 0:
            return 0.0 # Avoid division by zero

        net_inventory = self.long_position - self.short_position
        # Check for NaN inventory which might occur if positions become NaN (shouldn't happen)
        if np.isnan(net_inventory):
             logging.error("Inventory became NaN during risk penalty calculation.")
             return self.config["inventory_penalty"] * 10 # Apply a large penalty if state is corrupt

        normalized_inventory = net_inventory / self.config["max_inventory"]

        # Use a safe midprice for scaling
        midprice_safe = self.midprice if not np.isnan(self.midprice) and self.midprice > 0 else 1.0

        # Quadratic penalty, scaled by midprice to make it value-based
        penalty = (
            self.config["inventory_penalty"] *
            (normalized_inventory ** 2) *
            midprice_safe # Scale penalty by price level
        )
        return abs(penalty) # Penalty is a positive cost


    def _update_step_state(self) -> None:
        """Advances the environment time by one step, updating the order book."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            current_lob_data = self.order_book_history[self.current_step]
            # Ensure data is valid before assigning
            if isinstance(current_lob_data.get("bids"), np.ndarray) and isinstance(current_lob_data.get("asks"), np.ndarray):
                self.bids = current_lob_data["bids"].copy()
                self.asks = current_lob_data["asks"].copy()
                self._update_market_state() # Recalculate best bid/ask, midprice, spread
            else:
                 logging.error(f"Invalid LOB data format encountered at step {self.current_step}. Expected numpy arrays.")
                 # Keep previous market state or handle error appropriately
                 self._update_market_state() # Try updating with potentially old self.bids/self.asks
        else:
             # This case is now handled by truncation logic in step()
             logging.debug("Reached end of data, state update skipped.")


    def _get_raw_observation(self):
        """Constructs the raw observation vector before normalization."""
        # Use safe defaults if market state is invalid
        midprice_safe = 1.0
        best_bid_safe = np.nan
        best_ask_safe = np.nan
        spread_safe = np.nan

        if not np.isnan(self.midprice) and self.midprice > 0:
            midprice_safe = self.midprice
            best_bid_safe = self.best_bid if not np.isnan(self.best_bid) else midprice_safe
            best_ask_safe = self.best_ask if not np.isnan(self.best_ask) else midprice_safe
            spread_safe = self.spread if not np.isnan(self.spread) else (best_ask_safe - best_bid_safe)
        elif hasattr(self, 'midprice'): # Keep previous safe values if midprice becomes nan
             # This relies on the previous step having valid data
             # A better fallback might be needed if data starts bad
             pass # Keep the previously set safe values if they exist

        # Ensure safe values are reasonable numbers
        midprice_safe = midprice_safe if np.isfinite(midprice_safe) and midprice_safe > 0 else 1.0
        best_bid_safe = best_bid_safe if np.isfinite(best_bid_safe) else midprice_safe
        best_ask_safe = best_ask_safe if np.isfinite(best_ask_safe) else midprice_safe
        spread_safe = spread_safe if np.isfinite(spread_safe) and spread_safe >= 0 else 0.0


        # MTM Calculation
        net_inventory = self.long_position - self.short_position
        inventory_value = net_inventory * midprice_safe
        mtm = self.cash + inventory_value

        # Normalize costs relative to current midprice
        long_avg_cost_norm = (self.long_avg_cost / midprice_safe) - 1.0 if self.long_position > 1e-9 and self.long_avg_cost > 0 else 0.0
        short_avg_cost_norm = (self.short_avg_cost / midprice_safe) - 1.0 if self.short_position > 1e-9 and self.short_avg_cost > 0 else 0.0

        # Max inventory safety check
        max_inv_safe = self.config["max_inventory"] if self.config["max_inventory"] > 0 else 1.0
        max_orders_safe = self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 1.0
        max_vol_safe = self.config["max_order_volume"] if self.config["max_order_volume"] > 0 else 1.0


        market_features = [
            mtm / self.config["initial_capital"], # Normalize MTM by initial capital
            net_inventory / max_inv_safe, # Normalized net inventory
            len(self.active_orders) / max_orders_safe, # Normalized count of active orders
            self.long_position / max_inv_safe, # Normalized long position
            self.short_position / max_inv_safe, # Normalized short position
            long_avg_cost_norm,
            short_avg_cost_norm,
            (best_bid_safe - midprice_safe) / midprice_safe, # Normalized bid deviation
            (best_ask_safe - midprice_safe) / midprice_safe  # Normalized ask deviation
        ]

        # Active Order Features (normalized)
        order_prices_norm = []
        order_volumes_norm = []
        # Sort orders consistently (e.g., by price) for stable observation
        # Include pending orders? Maybe - depends if agent should react to them. Let's stick to active.
        orders_to_show = sorted(self.active_orders, key=lambda x: (x['price'], x['id'])) # Sort by price, then ID
        orders_to_show = orders_to_show[:self.config["max_active_orders"]] # Limit

        for order in orders_to_show:
            order_prices_norm.append((order["price"] - midprice_safe) / midprice_safe)
            # Signed normalized volume
            order_volumes_norm.append(order["volume"] / max_vol_safe * (1 if order['is_buy'] else -1))

        # Pad with zeros
        pad_orders = self.config["max_active_orders"] - len(orders_to_show)
        order_prices_norm += [0.0] * pad_orders
        order_volumes_norm += [0.0] * pad_orders

        # Order Book Features (normalized)
        book_features = []
        # Ensure bids/asks are valid numpy arrays before accessing
        valid_bids = isinstance(self.bids, np.ndarray) and self.bids.ndim == 2 and self.bids.shape[1] == 2
        valid_asks = isinstance(self.asks, np.ndarray) and self.asks.ndim == 2 and self.asks.shape[1] == 2

        for level in range(self.config["order_book_levels"]):
            bid_price_norm = (self.bids[level, 0] - midprice_safe) / midprice_safe if valid_bids and len(self.bids) > level else 0.0
            bid_qty_norm = self.bids[level, 1] / max_vol_safe if valid_bids and len(self.bids) > level else 0.0
            ask_price_norm = (self.asks[level, 0] - midprice_safe) / midprice_safe if valid_asks and len(self.asks) > level else 0.0
            ask_qty_norm = self.asks[level, 1] / max_vol_safe if valid_asks and len(self.asks) > level else 0.0
            book_features.extend([bid_price_norm, bid_qty_norm, ask_price_norm, ask_qty_norm])

        # Final Feature: Normalized Spread
        spread_norm = spread_safe / midprice_safe if midprice_safe > 0 else 0.0

        # Concatenate all features
        raw_obs_list = market_features + order_prices_norm + order_volumes_norm + book_features + [spread_norm]

        # Ensure all values are finite, replace NaN/inf with 0
        raw_obs_array = np.array(raw_obs_list, dtype=np.float32)
        if not np.all(np.isfinite(raw_obs_array)):
            logging.warning(f"Non-finite values detected in raw observation at step {self.current_step}. Replacing with 0.")
            raw_obs_array[~np.isfinite(raw_obs_array)] = 0.0

        # Check if shape matches observation space
        expected_shape = self.observation_space.shape
        if raw_obs_array.shape != expected_shape:
             logging.error(f"Observation shape mismatch! Expected {expected_shape}, got {raw_obs_array.shape}. Check feature calculation.")
             # Attempt to reshape or pad/truncate, or raise error
             # For safety, returning zeros if shape is wrong
             # return np.zeros(expected_shape, dtype=np.float32)
             raise ValueError(f"Observation shape mismatch! Expected {expected_shape}, got {raw_obs_array.shape}.")


        return raw_obs_array


    def _get_observation(self):
        """Gets the current observation, normalizes it using running stats."""
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        normalized_obs = self._normalize_observation(raw_obs)
        # Clip observations to prevent extreme values after normalization?
        # normalized_obs = np.clip(normalized_obs, -10.0, 10.0) # Example clip
        return normalized_obs

    def _update_running_stats(self, obs):
        """Updates the running mean and variance using exponential moving average."""
        # Ensure obs is a flat numpy array
        obs = np.asarray(obs, dtype=np.float32).flatten()

        if self.running_stats['mean'] is None:
             # Initialize stats only if obs has the correct shape
            if obs.shape == self.observation_space.shape:
                self.running_stats['mean'] = obs.copy()
                self.running_stats['var'] = np.zeros_like(obs, dtype=np.float32)
                self.running_stats['count'] = 1
                logging.debug("Initialized running stats for normalization.")
            else:
                 logging.warning(f"Skipping running stats update: Observation shape {obs.shape} mismatch with expected {self.observation_space.shape}")
                 return # Don't initialize if shape is wrong
        else:
            # Ensure shapes match before updating
            if obs.shape != self.running_stats['mean'].shape:
                logging.warning(f"Skipping running stats update: Observation shape {obs.shape} mismatch with running mean shape {self.running_stats['mean'].shape}")
                return

            self.running_stats['count'] += 1
            # EMA update (less prone to drift than simple moving average for non-stationary data)
            old_mean = self.running_stats['mean']
            new_mean = old_mean * self.decay + obs * (1 - self.decay)
            self.running_stats['mean'] = new_mean

            # Update variance using EMA on squared differences
            # Note: EMA variance estimate can be biased, Welford is better but more complex
            delta_sq = (obs - old_mean) * (obs - new_mean) # Approximation suitable for EMA
            new_var = self.decay * self.running_stats['var'] + (1 - self.decay) * delta_sq
            # Ensure variance is non-negative due to potential floating point issues
            self.running_stats['var'] = np.maximum(new_var, 0.0)


    def _normalize_observation(self, obs):
        """Normalizes the observation using the running mean and variance."""
         # Ensure obs is a flat numpy array
        obs = np.asarray(obs, dtype=np.float32).flatten()

        # Check if stats are initialized and valid
        if self.running_stats['mean'] is None or self.running_stats['count'] < 2:
            logging.debug("Normalization skipped: Running stats not sufficiently initialized.")
            return obs # Return raw observation

        # Ensure shapes match
        if obs.shape != self.running_stats['mean'].shape:
            logging.warning(f"Skipping normalization: Observation shape {obs.shape} mismatch with running mean shape {self.running_stats['mean'].shape}")
            return obs # Return raw observation if shapes don't match

        mean = self.running_stats['mean']
        variance = self.running_stats['var']
        # Avoid division by zero or near-zero std dev
        std_dev = np.sqrt(variance + 1e-8)

        # Normalize: (obs - mean) / std_dev
        normalized_obs = (obs - mean) / std_dev

        # Check for NaNs/Infs after normalization (can happen if std_dev was zero despite epsilon)
        if not np.all(np.isfinite(normalized_obs)):
            logging.warning("Non-finite values detected after normalization. Replacing with 0.")
            normalized_obs[~np.isfinite(normalized_obs)] = 0.0

        return normalized_obs

    def _get_info(self):
        """Returns auxiliary information about the current state."""
        # Use safe midprice for MTM calculation in info
        midprice_safe = self.midprice if not np.isnan(self.midprice) else 0.0
        inventory_value = (self.long_position - self.short_position) * midprice_safe
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
            "best_bid": self.best_bid if hasattr(self, 'best_bid') else np.nan,
            "best_ask": self.best_ask if hasattr(self, 'best_ask') else np.nan,
            "mid_price": self.midprice if hasattr(self, 'midprice') else np.nan,
            "current_step": self.current_step,
            "total_steps_elapsed": self.step_count,
            "episode_pnl": mtm - self.config["initial_capital"] # Total PnL relative to start (including unrealized)
        }

    def render(self, mode="human"):
        """Renders the environment state (prints to console)."""
        if mode == "human":
            info = self._get_info()
            render_msg = (
                #f"\n--- Step: {info['current_step']} / Total Elapsed: {info['total_steps_elapsed']} ---" # Original
                f"\n--- LOB Step: {info['current_step']} | Episode Step: {info['total_steps_elapsed']}/{self.max_steps} ---" # Clearer distinction
                f"\n  Market: Bid={info['best_bid']:.2f} | Ask={info['best_ask']:.2f} | Mid={info['mid_price']:.2f}"
                f"\n  Portfolio: MTM=${info['mtm']:,.2f} | Cash=${info['cash']:,.2f} | Ep PnL=${info['episode_pnl']:,.2f}"
                f"\n  Position: Net={info['net_inventory']:.4f} (L:{info['long_position']:.4f}@{info['long_avg_cost']:.2f}, S:{info['short_position']:.4f}@{info['short_avg_cost']:.2f})"
                f"\n  Orders: Active={info['active_orders_count']} | Pending={info['pending_orders_count']}"
                #f"\n{'='*40}" # Original
                f"\n========================================"
            )
            # Print directly for immediate feedback during interactive use/debugging
            # Or use logging.info(render_msg) to adhere strictly to logging setup
            print(render_msg)

        # elif mode == "rgb_array": # Example for future extension
        #     # Implement logic to create an image representation of the state
        #     # return np.zeros((100, 100, 3), dtype=np.uint8) # Placeholder
        #     raise NotImplementedError("rgb_array rendering not implemented.")

        else:
             # For compatibility with Gymnasium render modes if needed later
             # super().render(mode=mode) # This might raise error if base class doesn't implement
             logging.warning(f"Rendering mode '{mode}' not explicitly supported. Defaulting to no output.")
             pass

    def _liquidate_positions(self):
        """
        Liquidates all open positions at current best market prices.
        Used primarily by reset() to ensure a clean slate for new episodes.
        Returns the PnL realized from this liquidation.
        """
        logging.info("Liquidating positions for reset...")
        liquidation_pnl = 0.0
        total_liq_cost = 0.0

        # Use current best bid/ask. Handle potential NaN values from market state.
        best_bid_liq = self.best_bid if hasattr(self, 'best_bid') and not np.isnan(self.best_bid) else np.nan
        best_ask_liq = self.best_ask if hasattr(self, 'best_ask') and not np.isnan(self.best_ask) else np.nan

        # If BBO is unavailable, try using midprice as a last resort (less realistic)
        if np.isnan(best_bid_liq) or np.isnan(best_ask_liq):
            midprice_liq = self.midprice if hasattr(self, 'midprice') and not np.isnan(self.midprice) else np.nan
            if not np.isnan(midprice_liq):
                logging.warning(f"BBO unavailable for liquidation. Using midprice ({midprice_liq:.2f}) as fallback.")
                # Assume some spread around midprice for liquidation
                best_bid_liq = midprice_liq - self.config.get("liquidation_spread_fallback_ticks", 5) * self.config["tick_size"]
                best_ask_liq = midprice_liq + self.config.get("liquidation_spread_fallback_ticks", 5) * self.config["tick_size"]
                best_bid_liq = max(0.01, best_bid_liq) # Ensure > 0
                best_ask_liq = max(best_bid_liq + self.config["tick_size"], best_ask_liq) # Ensure ask > bid
            else:
                 logging.error("Cannot liquidate - market prices (BBO and Midprice) unavailable (NaN). Setting PnL to 0, positions might not be fully cleared.")
                 # Force clear positions even without PnL calculation? Risky.
                 # self.long_position = 0.0; self.short_position = 0.0; etc.
                 return 0.0 # Cannot calculate PnL

        # Liquidate long position by selling at the best bid
        if self.long_position > 1e-9:
            qty = self.long_position
            price = best_bid_liq
            value = qty * price
            cost = self.config["transaction_cost_short"] * value # Selling uses short cost rate
            pnl = (price - self.long_avg_cost) * qty if self.long_avg_cost > 0 else (-cost) # PnL vs avg cost
            self.cash += value - cost
            liquidation_pnl += pnl # PnL before cost was 'pnl_gross', this is net pnl contribution
            total_liq_cost += cost
            logging.info(f"Reset Liquidation - Sold LONG: {qty:.4f} @ {price:.2f}. PNL: {pnl:.2f} (Cost: {cost:.2f})")
            self.long_position = 0.0
            self.long_avg_cost = 0.0

        # Cover short position by buying at the best ask
        if self.short_position > 1e-9:
            qty = self.short_position
            price = best_ask_liq
            value = qty * price
            cost = self.config["transaction_cost_long"] * value # Buying uses long cost rate
            pnl = (self.short_avg_cost - price) * qty if self.short_avg_cost > 0 else (-cost) # PnL vs avg cost
            self.cash -= value + cost # Buying costs cash, plus transaction cost
            liquidation_pnl += pnl # Net PnL contribution
            total_liq_cost += cost
            logging.info(f"Reset Liquidation - Covered SHORT: {qty:.4f} @ {price:.2f}. PNL: {pnl:.2f} (Cost: {cost:.2f})")
            self.short_position = 0.0
            self.short_avg_cost = 0.0

        # Clear all orders (active and pending) during reset liquidation
        if self.active_orders or self.pending_orders:
             logging.info(f"Clearing {len(self.active_orders)} active and {len(self.pending_orders)} pending orders during reset.")
             self.active_orders = []
             self.pending_orders = []

        self.inventory = 0.0 # Ensure inventory is zero
        logging.info(f"Liquidation for reset complete. Total Liq PNL: {liquidation_pnl:.2f}. Final Cash: {self.cash:.2f}")
        return liquidation_pnl # Return the PnL generated during this specific liquidation event

    def close(self):
        """Clean up any resources (if any were used)."""
        logging.info("Closing HFT Environment.")
        # No external resources (like network connections or files) opened persistently by the env itself
        pass


# --- Example Configuration and Usage ---
config = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_small.csv", # UPDATE THIS PATH
    "initial_capital": 20000.0,
    "max_steps": 10000,
    "order_book_levels": 9,
    "price_offset_ticks": 40,
    "max_order_volume": 2.5,
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_active_orders": 15,
    "max_inventory": 8.0,
    "latency_steps_long": 2,
    "latency_steps_short": 1,
    "transaction_cost_long": 0.0005,
    "transaction_cost_short": 0.0005,
    "inventory_penalty": 0.01,
    "invalid_order_penalty": 0.5,
    "activity_bonus": 0.001,
}

if __name__ == "__main__":
    # Use INFO level to see reset/liquidation messages, DEBUG for step-by-step details
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True) # force=True to override potential defaults

    try:
        env = HFTEnv(config)
        obs, info = env.reset()
        print("\nEnvironment Initialized.")
        print("Initial Observation Shape:", obs.shape)
        # print("Initial Observation Sample:", obs[:15]) # Print first few features
        print("Initial Info:", info)

        terminated = False
        truncated = False
        total_reward = 0
        episode_step_count = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # Take random actions

            # --- Action Override Example (Market Making) ---
            # Place buy 1 tick below best bid, sell 1 tick above best ask
            # mid = env.midprice
            # best_bid = env.best_bid
            # best_ask = env.best_ask
            # if not np.isnan(mid) and not np.isnan(best_bid) and not np.isnan(best_ask):
            #     buy_price = best_bid - env.config['tick_size']
            #     sell_price = best_ask + env.config['tick_size']
            #     buy_offset = (buy_price - mid) / (env.config['price_offset_ticks'] * env.config['tick_size'])
            #     sell_offset = (sell_price - mid) / (env.config['price_offset_ticks'] * env.config['tick_size'])
            #     # Simple logic: alternate placing buy/sell or place both if possible
            #     if episode_step_count % 2 == 0: # Place buy
            #          action = np.array([0.5, np.clip(buy_offset,-1,1) , 0.0], dtype=np.float32) # Buy half max vol
            #     else: # Place sell
            #          action = np.array([-0.5, np.clip(sell_offset,-1,1), 0.0], dtype=np.float32) # Sell half max vol
            # else:
            #     action = env.action_space.sample() # Fallback to random if market invalid
            # ---------------------------------------------

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_step_count += 1

            # Render periodically
            if episode_step_count % 500 == 0:
                 env.render()
                 # Debugging normalization stats
                 # if env.running_stats['mean'] is not None:
                 #     mean_norm = np.mean(env.running_stats['mean'])
                 #     std_norm = np.mean(np.sqrt(env.running_stats['var'] + 1e-8))
                 #     print(f"Running Stats (Mean of means/stds): Mean={mean_norm:.3f}, Std={std_norm:.3f}")
                 #     print(f"Current Norm Obs Stats: Mean={np.mean(obs):.3f}, Std={np.std(obs):.3f}, Min={np.min(obs):.3f}, Max={np.max(obs):.3f}")


        print("\n--- Episode Finished ---")
        print(f"Reason: {'Terminated' if terminated else 'Truncated'}")
        print(f"Total Episode Steps: {episode_step_count}")
        print(f"Final LOB Step Index: {info['current_step']}")
        print(f"Final Info: {info}") # Note: MTM/PnL includes unrealized value of final position
        print(f"Cumulative Reward: {total_reward:.4f}")

        env.close()

    except FileNotFoundError:
        logging.error(f"FATAL ERROR: CSV file not found at path specified in config: {config['csv_path']}")
        logging.error("Please update the 'csv_path' in the config dictionary.")
    except ValueError as e:
        logging.error(f"FATAL ERROR during environment setup or execution: {e}")
        # Print traceback for detailed debugging
        import traceback
        traceback.print_exc()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)