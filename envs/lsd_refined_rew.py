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
        # NOTE: Corrected signed_volume interpretation: positive is BUY, negative is SELL to match logic
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]), # [-max_vol (sell), min_offset, min_cancel]
            high=np.array([1.0, 1.0, 1.0]),  # [+max_vol (buy), max_offset, max_cancel]
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
            "latency_steps_long",
            "latency_steps_short",
            "tick_size", "lot_size", "max_active_orders", "inventory_penalty",
            "transaction_cost_long",
            "transaction_cost_short",
            "max_inventory", "invalid_order_penalty", "activity_bonus"
        ]

        missing_keys = set(required_keys) - set(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        if self.config["latency_steps_long"] < 0 or self.config["latency_steps_short"] < 0:
             raise ValueError("Latency steps cannot be negative.")
        if self.config["transaction_cost_long"] < 0 or self.config["transaction_cost_short"] < 0:
             raise ValueError("Transaction costs cannot be negative.")
        if self.config["tick_size"] <= 0:
             raise ValueError("Tick size must be positive.")
        if self.config["lot_size"] <= 0:
             raise ValueError("Lot size must be positive.")
        if self.config["max_inventory"] <= 0:
             raise ValueError("Max inventory must be positive.")
        if self.config["max_active_orders"] <= 0:
             raise ValueError("Max active orders must be positive.")

        logging.info("Configuration validated successfully.")

    def _load_order_book_data(self):
        """Loads and preprocesses order book data from a CSV file."""
        logging.info(f"Loading order book data from: {self.config['csv_path']}")
        try:
            df = pd.read_csv(self.config["csv_path"])
        except FileNotFoundError:
            logging.error(f"CSV file not found at path: {self.config['csv_path']}")
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
        for index, row in df.iterrows():
            row_count = index # Use index for clearer reporting
            try:
                bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]]
                              for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)

                # Data validation
                if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any() or np.isnan(bids[:, 0]).any() or np.isnan(asks[:, 0]).any():
                    logging.warning(f"Invalid (non-positive or NaN) prices found at row index {row_count}. Skipping row.")
                    continue
                if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any() or np.isnan(bids[:, 1]).any() or np.isnan(asks[:, 1]).any():
                    logging.warning(f"Invalid (negative or NaN) quantities found at row index {row_count}. Skipping row.")
                    continue
                if not all(bids[i, 0] >= bids[i+1, 0] for i in range(len(bids)-1) if not np.isnan(bids[i,0]) and not np.isnan(bids[i+1,0])) or \
                   not all(asks[i, 0] <= asks[i+1, 0] for i in range(len(asks)-1) if not np.isnan(asks[i,0]) and not np.isnan(asks[i+1,0])) or \
                   (len(bids) > 0 and len(asks) > 0 and not np.isnan(bids[0,0]) and not np.isnan(asks[0,0]) and bids[0,0] >= asks[0,0]):
                   logging.warning(f"Order book structure violation (bid/ask order or crossing) at row index {row_count}. B0:{bids[0,0]}, A0:{asks[0,0]}. Skipping row.")
                   continue

                history.append({"bids": bids, "asks": asks})
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
            9 +  # mtm_norm, inventory_norm, active_orders_norm, long_pos_norm, short_pos_norm, long_avg_cost_norm, short_avg_cost_norm, bid_dev_norm, ask_dev_norm
            2 * self.config["max_active_orders"] +  # order prices (normalized), volumes (normalized, signed)
            4 * self.config["order_book_levels"] +  # bid/ask prices (normalized), volumes (normalized)
            1  # spread (normalized)
        )
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        logging.info("Resetting environment.")

        self.running_stats = {'mean': None, 'var': None, 'count': 0}

        if hasattr(self, 'long_position') and (self.long_position > 1e-9 or self.short_position > 1e-9):
            logging.warning("Reset called with active positions. Liquidating...")
            self._liquidate_positions() # Ensure clean state

        self.current_step = 0
        self.cash = self.config["initial_capital"]
        self.long_position = 0.0
        self.short_position = 0.0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        self.inventory = 0.0

        self.active_orders = []
        self.pending_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.last_executed_volume = 0.0

        if not self.order_book_history:
             raise RuntimeError("Cannot reset environment without loaded order book history.")

        self.bids = self.order_book_history[0]["bids"].copy()
        self.asks = self.order_book_history[0]["asks"].copy()
        self._update_market_state()

        observation = self._get_observation()
        info = self._get_info()
        logging.info(f"Reset complete. Initial MTM: {info['mtm']:.2f}")
        return observation, info

    def _update_market_state(self):
        """
        Updates internal market state variables based on the current order book.
        Handles potential empty levels or invalid states gracefully.
        """
        if len(self.bids) == 0 or np.isnan(self.bids[0, 0]):
            self.best_bid = np.nan
        else:
            self.best_bid = self.bids[0, 0]

        if len(self.asks) == 0 or np.isnan(self.asks[0, 0]):
            self.best_ask = np.nan
        else:
            self.best_ask = self.asks[0, 0]

        if np.isnan(self.best_bid) or np.isnan(self.best_ask):
            logging.warning(f"Step {self.current_step}: Best bid or ask is NaN. Midprice/Spread calculation skipped.")
            self.midprice = np.nan
            self.spread = np.nan
            # Attempt to use previous valid midprice if available? For now, keep as NaN.
            if not hasattr(self, 'midprice'): # Ensure attribute exists even if NaN
                self.midprice = np.nan
                self.spread = np.nan
            return

        if self.best_bid >= self.best_ask:
            logging.warning(f"Step {self.current_step}: Order book crossed or locked (Best Bid: {self.best_bid}, Best Ask: {self.best_ask}). Using average, spread is 0.")
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = 0.0
        else:
            self.midprice = (self.best_bid + self.best_ask) / 2
            self.spread = self.best_ask - self.best_bid

    def step(self, action):
        """Executes one time step within the environment."""
        self.step_count += 1
        logging.debug(f"--- Step {self.current_step} ({self.step_count}/{self.max_steps}) ---")
        logging.debug(f"Action received: {action}")

        # --- Pre-Step State Logging ---
        pre_step_info = self._get_info()
        logging.debug(f"Pre-Step State - MTM: {pre_step_info['mtm']:.2f}, Inv: {pre_step_info['net_inventory']:.4f}, Cash: {pre_step_info['cash']:.2f}, BBO: {pre_step_info['best_bid']:.2f}/{pre_step_info['best_ask']:.2f}")


        # 1. Parse Action
        # Action: [signed_volume (-1 sell, +1 buy), price_offset (-1 to +1), cancel_fraction (0 to 1)]
        signed_volume_action, price_offset, cancel_fraction = action
        is_buy = signed_volume_action > 0.0
        volume_scaled = abs(signed_volume_action) # Scale [0, 1]

        # 2. Handle Order Cancellation
        self._handle_order_cancellation(cancel_fraction)

        # 3. Place New Order (if volume > 0 and space available)
        order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)

        # 4. Process Latency: Move matured pending orders to active orders
        self._process_pending_orders()

        # 5. Execute Active Orders against current LOB
        # --- THIS NOW RETURNS NET REALIZED PNL ---
        net_realized_pnl_from_fills = self._execute_orders()

        # 6. Update Inventory (Calculated Property)
        self.inventory = self.long_position - self.short_position

        # 7. Check Termination Conditions
        terminated = (
            self.cash < 0 or
            abs(self.inventory) > self.config["max_inventory"] or
            self.current_step >= len(self.order_book_history) - 1
        )
        truncated = self.step_count >= self.max_steps

        # 8. Liquidate Positions at the End
        liquidation_pnl = 0.0
        if terminated or truncated:
            logging.info(f"Episode end condition met at step {self.current_step} (Terminated: {terminated}, Truncated: {truncated}). Liquidating positions.")
            # Use market state *before* advancing for liquidation
            liquidation_pnl = self._liquidate_positions()
            self.inventory = 0 # Ensure inventory is zero post-liquidation

        # 9. Calculate Mark-to-Market (MTM) - based on current state's midprice
        current_inventory_value = self.inventory * self.midprice if not np.isnan(self.midprice) else 0
        mtm = self.cash + current_inventory_value

        # 10. Calculate Penalties and Bonuses
        risk_penalty = self._calculate_risk_penalty()
        activity_bonus = self.config["activity_bonus"] * self.last_executed_volume

        # 11. Calculate Reward
        # Uses NET PnL from fills + liquidation PnL (if applicable) + penalties/bonuses
        reward = net_realized_pnl_from_fills + liquidation_pnl - risk_penalty + order_penalty + activity_bonus
        # --- UPDATED LOGGING MESSAGE ---
        logging.debug(f"Step {self.current_step} - Net PNL(fills): {net_realized_pnl_from_fills:.4f}, PNL(liq): {liquidation_pnl:.4f}, RiskPen: {risk_penalty:.4f}, OrderPen: {order_penalty:.4f}, ActBonus: {activity_bonus:.4f}, Total Reward: {reward:.4f}")

        # 12. Advance Market Data to Next Step (if not ended)
        if not (terminated or truncated):
             self._update_step_state()
        # Else: market state remains as it was for the final info/observation

        # 13. Get Observation and Info
        observation = self._get_observation() # Based on potentially new market state
        info = self._get_info() # Based on state *after* actions but before market advance if not terminated/truncated
        logging.debug(f"Step {self.current_step} End - MTM: {info['mtm']:.2f}, Inv: {info['net_inventory']:.4f}, Cash: {info['cash']:.2f}")

        return observation, reward, terminated, truncated, info

    def _handle_order_cancellation(self, cancel_fraction):
        """Cancels a fraction of the oldest active orders."""
        if not self.active_orders:
            return

        num_to_cancel = int(np.clip(cancel_fraction, 0.0, 1.0) * len(self.active_orders))
        if num_to_cancel > 0:
            logging.debug(f"Cancelling {num_to_cancel} oldest active orders.")
            # Remove from the beginning (oldest)
            cancelled_orders = self.active_orders[:num_to_cancel]
            self.active_orders = self.active_orders[num_to_cancel:]
            # Log details of cancelled orders if needed
            # for order in cancelled_orders:
            #     logging.debug(f"Cancelled Order ID: {order['id']}")


    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        """Places a new order, applying penalties for invalid placements."""
        penalty = 0.0
        if len(self.active_orders) + len(self.pending_orders) >= self.config["max_active_orders"]:
            logging.debug("Max active/pending orders reached. Cannot place new order.")
            return penalty

        if np.isnan(self.midprice):
             logging.warning("Cannot place order: Midprice is NaN.")
             return penalty - self.config["invalid_order_penalty"] # Penalize attempting action with invalid state

        # Calculate Limit Price relative to midprice
        price_offset_abs = price_offset * self.config["price_offset_ticks"] * self.config["tick_size"]
        limit_price = self.midprice + price_offset_abs
        limit_price = self._round_to_tick(limit_price)

        # Check if order price is valid relative to BBO (prevent crossing/locking immediately)
        # Allow placing *at* BBO, but not beyond if BBO exists
        if is_buy and not np.isnan(self.best_ask) and limit_price > self.best_ask:
            logging.debug(f"Buy order price {limit_price:.2f} too aggressive (>{self.best_ask:.2f}). Applying penalty and clipping.")
            penalty -= self.config["invalid_order_penalty"]
            limit_price = self.best_ask # Clip to best ask
            limit_price = self._round_to_tick(limit_price) # Re-round after clipping

        if not is_buy and not np.isnan(self.best_bid) and limit_price < self.best_bid:
            logging.debug(f"Sell order price {limit_price:.2f} too aggressive (<{self.best_bid:.2f}). Applying penalty and clipping.")
            penalty -= self.config["invalid_order_penalty"]
            limit_price = self.best_bid # Clip to best bid
            limit_price = self._round_to_tick(limit_price) # Re-round after clipping

        # Calculate Volume
        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0.0, round(volume / self.config["lot_size"]) * self.config["lot_size"])
        if volume <= 1e-9:
            logging.debug("Order volume is effectively zero. Not placing order.")
            return penalty

        # Determine Latency and Target Step
        latency = self.config["latency_steps_long"] if is_buy else self.config["latency_steps_short"]
        target_step = self.current_step + latency + 1 # +1 because order arrives *before* the start of target_step+1 market state

        # Create and queue the order
        order = {
            "id": self.order_id_counter,
            "price": limit_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp_placed": self.current_step,
            "target_step": target_step
        }
        self.order_id_counter += 1
        self.pending_orders.append(order)
        logging.debug(f"Placed Order {order['id']}: {'Buy' if is_buy else 'Sell'} {volume:.4f} @ {limit_price:.2f}. Target Step: {target_step}")

        return penalty

    def _round_to_tick(self, price):
        """Rounds a price to the nearest tick size."""
        if np.isnan(price) or self.config["tick_size"] <= 0:
             return price
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_pending_orders(self):
        """Moves orders from the pending list to active orders if their latency period is over."""
        still_pending = []
        activated_count = 0
        # Use current_step + 1 because an order placed at step 't' with latency 'L'
        # should become active based on the market state at step 't + L + 1'.
        effective_step = self.current_step + 1
        for order in self.pending_orders:
            if effective_step >= order["target_step"]:
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(order)
                    activated_count += 1
                    logging.debug(f"Order {order['id']} activated at step {self.current_step} (Target: {order['target_step']}).")
                else:
                    logging.warning(f"Order {order['id']} reached target step {order['target_step']} at current step {self.current_step}, but max active orders ({self.config['max_active_orders']}) limit reached. Order discarded.")
                    # Optionally keep pending: still_pending.append(order)
            else:
                still_pending.append(order) # Keep in pending list

        # Sort active orders? Price-time priority is complex. Let's sort by price aggressiveness.
        # Buys: Higher price first. Sells: Lower price first.
        # This is a simplification of time priority.
        self.active_orders.sort(key=lambda x: x['price'], reverse=x['is_buy'])

        self.pending_orders = still_pending


    def _execute_orders(self):
        """
        Matches active orders against the current order book levels.
        Calculates and returns the NET realized PnL for the step.
        """
        # --- MODIFICATION START ---
        gross_realized_pnl_step = 0.0
        total_transaction_costs_step = 0.0
        # --- MODIFICATION END ---
        self.last_executed_volume = 0.0
        new_active_orders = []

        # Process orders one by one (already sorted by price aggressiveness in _process_pending_orders)
        for order in self.active_orders:
            remaining_volume = order["volume"]
            is_buy = order["is_buy"]
            limit_price = order["price"]

            # Determine relevant book side and cost rate
            levels_to_match = self.asks if is_buy else self.bids # Match buys against asks, sells against bids
            cost_rate = self.config["transaction_cost_long"] if is_buy else self.config["transaction_cost_short"]

            for level_idx in range(len(levels_to_match)):
                level_price, level_qty = levels_to_match[level_idx]

                # Skip if level is invalid or empty
                if np.isnan(level_price) or np.isnan(level_qty) or level_qty <= 1e-9:
                    continue

                if remaining_volume <= 1e-9:
                    break

                # Check if price condition is met for this level
                can_fill_at_level = (is_buy and limit_price >= level_price) or \
                                    (not is_buy and limit_price <= level_price)

                if can_fill_at_level:
                    fill_qty_possible = min(remaining_volume, level_qty)

                    # Check inventory limits *before* calculating fill
                    current_net_inventory = self.long_position - self.short_position
                    inventory_change = fill_qty_possible if is_buy else -fill_qty_possible
                    potential_new_inventory = current_net_inventory + inventory_change

                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        if inventory_change > 0: # Trying to increase inventory
                             # Max allowed increase is max_inv - current_inv
                             allowed_increase = self.config["max_inventory"] - current_net_inventory
                             fill_qty = max(0.0, min(fill_qty_possible, allowed_increase))
                        else: # Trying to decrease inventory (inventory_change < 0)
                             # Max allowed decrease is max_inv + current_inv (since current_inv could be negative)
                             allowed_decrease = self.config["max_inventory"] + current_net_inventory
                             fill_qty = max(0.0, min(fill_qty_possible, allowed_decrease))
                        logging.debug(f"Inventory limit potentially breached. Order {order['id']}. Current: {current_net_inventory:.4f}, Change: {inventory_change:.4f}. Adjusted fill from {fill_qty_possible:.4f} to {fill_qty:.4f}")

                    else:
                        fill_qty = fill_qty_possible

                    if fill_qty <= 1e-9: # No fill possible due to limits or level empty
                        continue # Check next level

                    # Process the fill
                    fill_price = level_price # Filled at the level's price
                    executed_value = fill_qty * fill_price
                    transaction_cost = cost_rate * executed_value

                    # Update cash (state update)
                    self.cash -= transaction_cost # Deduct cost regardless of buy/sell
                    if is_buy:
                        self.cash -= executed_value
                    else:
                        self.cash += executed_value

                    # --- MODIFICATION START ---
                    # Accumulate step cost for net PnL calculation
                    total_transaction_costs_step += transaction_cost
                    # --- MODIFICATION END ---


                    # Update positions and calculate GROSS realized PnL from this fill
                    # _process_fill calculates PnL based on avg cost vs fill price
                    pnl_from_fill = self._process_fill(fill_qty, fill_price, is_buy)

                    # --- MODIFICATION START ---
                    # Accumulate GROSS pnl for the step
                    gross_realized_pnl_step += pnl_from_fill
                    # --- MODIFICATION END ---


                    # Update tracking variables
                    remaining_volume -= fill_qty
                    self.last_executed_volume += fill_qty
                    levels_to_match[level_idx, 1] -= fill_qty # Reduce quantity on the matched LOB level (in-memory mod)

                    logging.debug(f"Order {order['id']} filled: {fill_qty:.4f} @ {fill_price:.2f}. Gross PNL: {pnl_from_fill:.4f}. Cost: {transaction_cost:.4f}. Cash: {self.cash:.2f}")


            # If order is not fully filled, add the remainder back
            if remaining_volume > 1e-9:
                order["volume"] = remaining_volume
                new_active_orders.append(order)
                # No debug log here, it's expected orders might persist
            else:
                 logging.debug(f"Order {order['id']} fully filled.")


        self.active_orders = new_active_orders

        # --- MODIFICATION START ---
        # Calculate and return NET realized PnL for the step
        net_realized_pnl_step = gross_realized_pnl_step - total_transaction_costs_step
        logging.debug(f"Step {self.current_step} Executions Summary - Gross PNL(fills): {gross_realized_pnl_step:.4f}, Step Costs: {total_transaction_costs_step:.4f}, Net PNL(fills): {net_realized_pnl_step:.4f}")
        return net_realized_pnl_step # Return the NET PnL
        # --- MODIFICATION END ---


    def _process_fill(self, fill_qty, price, is_buy):
        """Updates positions (long/short) based on a fill and calculates GROSS realized PnL."""
        realized_pnl = 0.0

        if is_buy:
            # Buying: Covers short first, then adds to long
            if self.short_position > 0:
                cover_qty = min(fill_qty, self.short_position)
                if self.short_avg_cost > 0: # Avoid PnL calc if avg cost is zero/invalid
                    realized_pnl += (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                if self.short_position <= 1e-9:
                    self.short_position = 0.0
                    self.short_avg_cost = 0.0

                remaining_fill = fill_qty - cover_qty
                if remaining_fill > 1e-9:
                    self._add_to_long(remaining_fill, price)
            else:
                self._add_to_long(fill_qty, price)
        else:
            # Selling: Closes long first, then adds to short
            if self.long_position > 0:
                close_qty = min(fill_qty, self.long_position)
                if self.long_avg_cost > 0: # Avoid PnL calc if avg cost is zero/invalid
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

        return realized_pnl

    def _add_to_long(self, qty, price):
        """Helper to add quantity to the long position and update average cost."""
        if qty <= 1e-9: return
        new_total_cost = (self.long_avg_cost * self.long_position) + (price * qty)
        self.long_position += qty
        self.long_avg_cost = new_total_cost / self.long_position if self.long_position > 1e-9 else 0.0


    def _add_to_short(self, qty, price):
        """Helper to add quantity to the short position and update average cost."""
        if qty <= 1e-9: return
        new_total_proceeds = (self.short_avg_cost * self.short_position) + (price * qty)
        self.short_position += qty
        self.short_avg_cost = new_total_proceeds / self.short_position if self.short_position > 1e-9 else 0.0


    def _calculate_risk_penalty(self):
        """Calculates a penalty based on the current net inventory."""
        if self.config["max_inventory"] <= 0 or np.isnan(self.midprice):
            return 0.0 # No penalty if max_inventory not set or no valid price

        net_inventory = self.long_position - self.short_position
        # Normalize relative to max allowed inventory
        normalized_inventory = net_inventory / self.config["max_inventory"]

        # Quadratic penalty, scaled by inventory value (using midprice)
        inventory_value = abs(net_inventory * self.midprice)
        penalty_factor = self.config["inventory_penalty"]

        # Penalty = Factor * (NormInv)^2 * InventoryValue ? Or just Factor * (NormInv)^2?
        # Let's use a simpler quadratic penalty on normalized inventory, scaled by a fixed factor.
        # Scaling by midprice might make it too volatile or dependent on price level.
        penalty = penalty_factor * (normalized_inventory ** 2)

        # Ensure penalty is positive (it's a cost)
        return abs(penalty)


    def _update_step_state(self) -> None:
        """Advances the environment time by one step, updating the order book."""
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            current_lob_data = self.order_book_history[self.current_step]
            # IMPORTANT: Use copy() to prevent modifying the history data
            self.bids = current_lob_data["bids"].copy()
            self.asks = current_lob_data["asks"].copy()
            self._update_market_state() # Recalculate best bid/ask, midprice, spread
        else:
             # This state should be caught by the 'terminated' flag in the step function
             logging.warning("Attempted to update step state beyond available data.")


    def _get_raw_observation(self):
        """Constructs the raw observation vector before normalization."""
        # Use safe defaults if market state is invalid
        midprice_safe = self.midprice if not np.isnan(self.midprice) and self.midprice > 1e-9 else 1.0
        best_bid_safe = self.best_bid if not np.isnan(self.best_bid) else midprice_safe * 0.999 # Approx
        best_ask_safe = self.best_ask if not np.isnan(self.best_ask) else midprice_safe * 1.001 # Approx
        spread_safe = best_ask_safe - best_bid_safe if best_ask_safe > best_bid_safe else 0.0

        # MTM Calculation
        net_inventory = self.long_position - self.short_position
        inventory_value = net_inventory * midprice_safe
        mtm = self.cash + inventory_value

        # Normalize position costs relative to current midprice (as % deviation)
        long_avg_cost_norm = (self.long_avg_cost / midprice_safe) - 1.0 if self.long_position > 1e-9 and self.long_avg_cost > 1e-9 else 0.0
        short_avg_cost_norm = (self.short_avg_cost / midprice_safe) - 1.0 if self.short_position > 1e-9 and self.short_avg_cost > 1e-9 else 0.0

        # Normalize positions relative to max inventory
        max_inv_safe = self.config["max_inventory"] if self.config["max_inventory"] > 0 else 1.0
        net_inventory_norm = net_inventory / max_inv_safe
        long_pos_norm = self.long_position / max_inv_safe
        short_pos_norm = self.short_position / max_inv_safe

        # Normalize active orders count
        max_orders_safe = self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 1.0
        active_orders_norm = len(self.active_orders) / max_orders_safe

        # Market Features
        market_features = [
            mtm / self.config["initial_capital"], # MTM relative to initial capital
            net_inventory_norm,
            active_orders_norm,
            long_pos_norm,
            short_pos_norm,
            long_avg_cost_norm,
            short_avg_cost_norm,
            (best_bid_safe / midprice_safe) - 1.0, # Normalized bid deviation from mid
            (best_ask_safe / midprice_safe) - 1.0  # Normalized ask deviation from mid
        ]

        # Active Order Features (normalized price deviation and signed normalized volume)
        order_prices_norm = []
        order_volumes_norm = []
        orders_to_show = self.active_orders[:self.config["max_active_orders"]] # Already sorted by price agg.

        max_vol_safe = self.config["max_order_volume"] if self.config["max_order_volume"] > 0 else 1.0
        for order in orders_to_show:
            order_prices_norm.append((order["price"] / midprice_safe) - 1.0)
            order_volumes_norm.append(order["volume"] / max_vol_safe * (1 if order['is_buy'] else -1))

        # Pad with zeros
        pad_len = self.config["max_active_orders"] - len(order_prices_norm)
        order_prices_norm += [0.0] * pad_len
        order_volumes_norm += [0.0] * pad_len

        # Order Book Features (normalized price deviation and normalized volume)
        book_features = []
        total_book_volume = 1.0 # Fallback normalizer
        bid_vols = self.bids[:self.config["order_book_levels"], 1]
        ask_vols = self.asks[:self.config["order_book_levels"], 1]
        valid_vols = np.concatenate([bid_vols[~np.isnan(bid_vols)], ask_vols[~np.isnan(ask_vols)]])
        if len(valid_vols) > 0:
            total_book_volume = np.sum(valid_vols)
        if total_book_volume < 1e-9: total_book_volume = 1.0 # Avoid division by zero

        for level in range(self.config["order_book_levels"]):
            bid_price_norm = (self.bids[level, 0] / midprice_safe) - 1.0 if len(self.bids) > level and not np.isnan(self.bids[level, 0]) else 0.0
            bid_qty_norm = self.bids[level, 1] / total_book_volume if len(self.bids) > level and not np.isnan(self.bids[level, 1]) else 0.0
            ask_price_norm = (self.asks[level, 0] / midprice_safe) - 1.0 if len(self.asks) > level and not np.isnan(self.asks[level, 0]) else 0.0
            ask_qty_norm = self.asks[level, 1] / total_book_volume if len(self.asks) > level and not np.isnan(self.asks[level, 1]) else 0.0
            book_features.extend([bid_price_norm, bid_qty_norm, ask_price_norm, ask_qty_norm])

        # Final Feature: Normalized Spread
        spread_norm = spread_safe / midprice_safe if midprice_safe > 1e-9 else 0.0

        # Concatenate all features
        raw_obs_list = market_features + order_prices_norm + order_volumes_norm + book_features + [spread_norm]

        # Ensure all values are finite floats
        raw_obs_array = np.array(raw_obs_list, dtype=np.float32)
        raw_obs_array[~np.isfinite(raw_obs_array)] = 0.0 # Replace NaN/inf with 0

        # Check shape consistency
        expected_len = self.observation_space.shape[0]
        if len(raw_obs_array) != expected_len:
             logging.error(f"Observation length mismatch! Expected {expected_len}, Got {len(raw_obs_array)}")
             # Attempt to pad/truncate if possible, or raise error
             if len(raw_obs_array) < expected_len:
                 raw_obs_array = np.pad(raw_obs_array, (0, expected_len - len(raw_obs_array)))
             else:
                 raw_obs_array = raw_obs_array[:expected_len]
             # Alternatively: raise ValueError("Observation length mismatch!")


        return raw_obs_array


    def _get_observation(self):
        """Gets the current observation, normalizes it using running stats."""
        raw_obs = self._get_raw_observation()

        # Handle case where observation space might be Box(-inf, inf, ...) causing issues with stats
        if np.isinf(self.observation_space.low).any() or np.isinf(self.observation_space.high).any():
             # Option 1: Don't normalize if space is unbounded (simplest)
             # return raw_obs
             # Option 2: Normalize but be aware of potential instability
             pass # Proceed with normalization below

        self._update_running_stats(raw_obs)
        normalized_obs = self._normalize_observation(raw_obs)

        # Optional: Clip observations to a reasonable finite range after normalization
        normalized_obs = np.clip(normalized_obs, -10.0, 10.0) # Example range

        return normalized_obs

    def _update_running_stats(self, obs):
        """Updates the running mean and variance using EMA."""
        if self.running_stats['mean'] is None:
            # Initialize with the first observation
            self.running_stats['mean'] = obs.copy()
            self.running_stats['var'] = np.zeros_like(obs, dtype=np.float32)
            self.running_stats['count'] = 1
        else:
            self.running_stats['count'] += 1
            # Use EMA for mean and variance update
            old_mean = self.running_stats['mean']
            new_mean = self.decay * old_mean + (1 - self.decay) * obs
            self.running_stats['mean'] = new_mean

            # Variance EMA update (more stable way)
            delta_sq = (obs - old_mean) * (obs - new_mean) # Approx based on EMA of mean
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta_sq
            self.running_stats['var'] = np.maximum(self.running_stats['var'], 1e-8) # Prevent variance collapse


    def _normalize_observation(self, obs):
        """Normalizes the observation using the running mean and variance."""
        # Only normalize if stats are sufficiently warmed up
        if self.running_stats['mean'] is None or self.running_stats['count'] < 5: # Threshold for stable stats
            return obs # Return raw observation initially

        mean = self.running_stats['mean']
        variance = self.running_stats['var']
        # Add small epsilon for numerical stability
        std_dev = np.sqrt(variance + 1e-8)

        normalized_obs = (obs - mean) / std_dev
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
            "best_bid": self.best_bid if not np.isnan(self.best_bid) else None,
            "best_ask": self.best_ask if not np.isnan(self.best_ask) else None,
            "mid_price": self.midprice if not np.isnan(self.midprice) else None,
            "current_step": self.current_step,
            "total_steps_elapsed": self.step_count,
            "episode_pnl": mtm - self.config["initial_capital"] # Total PnL relative to start
        }

    def render(self, mode="human"):
        """Renders the environment state (prints to console)."""
        if mode == "human":
            info = self._get_info()
            # Format costs nicely
            long_cost_str = f"{info['long_avg_cost']:.2f}" if info['long_position'] > 1e-9 else "N/A"
            short_cost_str = f"{info['short_avg_cost']:.2f}" if info['short_position'] > 1e-9 else "N/A"
            # Format BBO nicely
            bid_str = f"{info['best_bid']:.2f}" if info['best_bid'] is not None else "NaN"
            ask_str = f"{info['best_ask']:.2f}" if info['best_ask'] is not None else "NaN"
            mid_str = f"{info['mid_price']:.2f}" if info['mid_price'] is not None else "NaN"

            render_msg = (
                f"\n--- Step: {info['current_step']} / Total Elapsed: {info['total_steps_elapsed']} ---\n"
                f"  Market: Bid={bid_str} | Ask={ask_str} | Mid={mid_str}\n"
                f"  Portfolio: MTM=${info['mtm']:,.2f} | Cash=${info['cash']:,.2f} | Ep. PnL=${info['episode_pnl']:,.2f}\n"
                f"  Position: Net={info['net_inventory']:,.4f} (L:{info['long_position']:.4f}@{long_cost_str}, S:{info['short_position']:.4f}@{short_cost_str})\n"
                f"  Orders: Active={info['active_orders_count']} | Pending={info['pending_orders_count']}\n"
            )

            # Optionally add details of active orders
            if info['active_orders_count'] > 0:
                 render_msg += "  Active Orders (Top 5): "
                 details = []
                 for o in self.active_orders[:5]:
                     side = "BUY" if o['is_buy'] else "SELL"
                     details.append(f"[{o['id']}:{side} {o['volume']:.3f}@{o['price']:.2f}]")
                 render_msg += ", ".join(details) + "\n"

            render_msg += f"{'='*40}"
            # Use logging INFO level for standard rendering output
            logging.info(render_msg)
        else:
            # Delegate non-human modes to the superclass (which might raise NotImplementedError)
            super().render(mode=mode)

    def _liquidate_positions(self):
        """Liquidates all open positions at current best market prices and cancels orders."""
        logging.info("Attempting liquidation...")
        liquidation_pnl = 0.0
        total_liq_cost = 0.0

        # Use current BBO for liquidation price. Handle NaN cases.
        # If liquidating long, sell at bid. If liquidating short, buy at ask.
        liq_sell_price = self.best_bid
        liq_buy_price = self.best_ask

        # Check if prices are valid for liquidation
        can_liq_long = self.long_position > 1e-9 and not np.isnan(liq_sell_price)
        can_liq_short = self.short_position > 1e-9 and not np.isnan(liq_buy_price)

        if self.long_position > 1e-9 and not can_liq_long:
             logging.error(f"Cannot liquidate LONG position ({self.long_position:.4f}) - Invalid sell price (Best Bid: {liq_sell_price}). Setting PnL contribution to 0.")
        if self.short_position > 1e-9 and not can_liq_short:
             logging.error(f"Cannot liquidate SHORT position ({self.short_position:.4f}) - Invalid buy price (Best Ask: {liq_buy_price}). Setting PnL contribution to 0.")


        # Liquidate long position (Sell at Bid)
        if can_liq_long:
            qty = self.long_position
            price = liq_sell_price
            value = qty * price
            cost = self.config["transaction_cost_short"] * value # Use SHORT cost rate for selling
            gross_pnl = (price - self.long_avg_cost) * qty if self.long_avg_cost > 0 else 0 # PnL before cost
            net_pnl = gross_pnl - cost

            self.cash += value - cost # Add value, subtract cost
            liquidation_pnl += net_pnl
            total_liq_cost += cost
            logging.info(f"Liquidated LONG: {qty:.4f} @ {price:.2f}. Gross PNL: {gross_pnl:.2f}, Cost: {cost:.2f}, Net PNL: {net_pnl:.2f}")
            self.long_position = 0.0
            self.long_avg_cost = 0.0

        # Cover short position (Buy at Ask)
        if can_liq_short:
            qty = self.short_position
            price = liq_buy_price
            value = qty * price
            cost = self.config["transaction_cost_long"] * value # Use LONG cost rate for buying
            gross_pnl = (self.short_avg_cost - price) * qty if self.short_avg_cost > 0 else 0 # PnL before cost
            net_pnl = gross_pnl - cost

            self.cash -= value + cost # Subtract value AND cost
            liquidation_pnl += net_pnl
            total_liq_cost += cost
            logging.info(f"Liquidated SHORT: {qty:.4f} @ {price:.2f}. Gross PNL: {gross_pnl:.2f}, Cost: {cost:.2f}, Net PNL: {net_pnl:.2f}")
            self.short_position = 0.0
            self.short_avg_cost = 0.0

        # Clear all orders
        if self.active_orders or self.pending_orders:
             logging.debug(f"Clearing {len(self.active_orders)} active and {len(self.pending_orders)} pending orders during liquidation.")
             self.active_orders = []
             self.pending_orders = []

        self.inventory = 0.0 # Ensure inventory state is zero
        logging.info(f"Liquidation complete. Total Net Liq PNL: {liquidation_pnl:.2f}. Total Liq Costs: {total_liq_cost:.2f}. Final Cash: {self.cash:.2f}")
        return liquidation_pnl # Return NET PnL from liquidation

    def close(self):
        """Clean up any resources (if any were used)."""
        logging.info("Closing HFT Environment.")
        # No external resources to close in this version
        pass


# --- Example Configuration ---
config = {
    "csv_path": "orderbook_trimmed_small.csv", # <<< --- UPDATE THIS PATH --- >>>
    "initial_capital": 20000.0,
    "max_steps": 10000,
    "order_book_levels": 5, # Reduced for smaller example data
    "price_offset_ticks": 10, # Max offset relative to midprice
    "max_order_volume": 1.0, # Max volume per order action
    "tick_size": 0.01,
    "lot_size": 0.01, # Smallest tradeable unit
    "max_active_orders": 10,
    "max_inventory": 5.0, # Max net position size

    "latency_steps_long": 2,  # Buy order latency
    "latency_steps_short": 1, # Sell order latency
    "transaction_cost_long": 0.0005,  # 0.05% cost for buys
    "transaction_cost_short": 0.0005, # 0.05% cost for sells
    "inventory_penalty": 0.001, # Penalty factor for holding inventory
    "invalid_order_penalty": 0.1,  # Penalty for bad order placement
    "activity_bonus": 0.000,       # Bonus per unit volume traded (set to 0 usually)
}

# --- Example Usage ---
if __name__ == "__main__":
    # Set logging level (DEBUG provides more detail)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        env = HFTEnv(config)
        obs, info = env.reset()
        print("\n" + "="*30 + " Environment Initialized " + "="*30)
        print(f"Initial Observation Shape: {obs.shape}")
        print(f"Initial Observation (first 10): {obs[:10]}")
        print(f"Initial Info: {info}")
        print("="*80 + "\n")


        terminated = False
        truncated = False
        total_reward = 0.0
        step_count = 0
        max_run_steps = 2000 # Limit test run length

        while not terminated and not truncated and step_count < max_run_steps:
            action = env.action_space.sample() # Take random actions
            # Example: Force a buy action occasionally
            # if step_count % 10 == 0:
            #    action = np.array([0.5, -0.2, 0.0], dtype=np.float32) # Small buy near mid
            # Example: Force a sell action occasionally
            # elif step_count % 20 == 0:
            #    action = np.array([-0.3, 0.1, 0.1], dtype=np.float32) # Small sell near mid, cancel some

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if step_count % 200 == 0: # Render periodically
                 print(f"\n--- Checkpoint Step {step_count} ---")
                 env.render()
                 # print(f"Current Observation (first 10): {obs[:10]}")
                 # print(f"Running Mean (first 10): {env.running_stats['mean'][:10] if env.running_stats['mean'] is not None else 'N/A'}")
                 # print(f"Running StdDev (first 10): {np.sqrt(env.running_stats['var'][:10] + 1e-8) if env.running_stats['var'] is not None else 'N/A'}")
                 print(f"Reward this step: {reward:.6f}")
                 print(f"Total Reward so far: {total_reward:.4f}")


        print("\n" + "="*30 + " Episode Finished " + "="*30)
        print(f"Reason: {'Terminated' if terminated else ('Truncated' if truncated else 'Max Steps Reached')}")
        print(f"Total Steps: {step_count}")
        print(f"Final Info:")
        for key, value in info.items():
             if isinstance(value, float):
                 print(f"  {key}: {value:,.4f}")
             else:
                 print(f"  {key}: {value}")
        print(f"Total Accumulated Reward: {total_reward:.4f}")
        # Note: Total Reward should ideally approximate Final Ep. PnL, adjusted for penalties/bonuses.
        print("="*80 + "\n")


        env.close()

    except FileNotFoundError:
        logging.error(f"FATAL ERROR: CSV file not found at path specified in config: '{config['csv_path']}'")
        logging.error("Please update the 'csv_path' in the config dictionary to a valid file.")
    except ValueError as e:
        logging.error(f"FATAL ERROR during environment setup or execution: {e}")
        logging.exception("ValueError Traceback:") # Provides more context for ValueErrors
    except KeyError as e:
        logging.error(f"FATAL ERROR: Missing key in data processing or config: {e}")
        logging.exception("KeyError Traceback:")
    except Exception as e:
        logging.error(f"An unexpected FATAL error occurred: {e}")
        logging.exception("Unexpected Error Traceback:")