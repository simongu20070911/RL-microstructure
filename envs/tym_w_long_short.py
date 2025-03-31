import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging

class HFTEnv(gym.Env):
    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.config = config
        self._validate_config()

        # Load and validate order book data
        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config["max_steps"], len(self.order_book_history))
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),  # signed volume, price offset, cancel fraction
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = self._create_observation_space()

        # Initialize running statistics
        self.running_stats = {
            'mean': None,
            'var': None,
            'count': 0
        }
        self.decay = 0.999  # Exponential decay factor

        self.long_position = 0
        self.short_position = 0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0

        self.reset()

    def _validate_config(self):
        required_keys = [
            "csv_path", "initial_capital", "max_steps", "order_book_levels",
            "price_offset_ticks", "max_order_volume", "latency_steps", "tick_size",
            "lot_size", "max_active_orders", "inventory_penalty", "transaction_cost",
            "max_inventory"
        ]
        # Optionally, if not provided, you could set a default penalty for invalid orders
        if "invalid_order_penalty" not in self.config:
            self.config["invalid_order_penalty"] = 1.0

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_order_book_data(self):
        df = pd.read_csv(self.config["csv_path"])
        required_columns = ["timestamp"]
        for i in range(1, self.config["order_book_levels"] + 1):
            required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]
            
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV data: {missing_cols}")

        history = []
        for _, row in df.iterrows():
            bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]] 
                          for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
            asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]] 
                          for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
            
            if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any():
                raise ValueError("Invalid prices in order book data")
            if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                raise ValueError("Negative quantities in order book data")
                
            history.append({"bids": bids, "asks": asks})
            
        return history

    def _create_observation_space(self):
        obs_size = (
            9 +  # mtm, inventory, active_orders, long_position, short_position, long_avg_cost, short_avg_cost, bid_dev, ask_dev
            2 * self.config["max_active_orders"] +  # order prices, volumes
            4 * self.config["order_book_levels"] +  # bid/ask prices/volumes
            1  # spread/midprice
        )
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        
        # Reset running statistics
        self.running_stats = {
            'mean': None,
            'var': None,
            'count': 0
        }
        
        # Ensure any previous positions are cleared
        if hasattr(self, 'long_position') and (self.long_position > 0 or self.short_position > 0):
            self._liquidate_positions()
        
        self.current_step = 0
        self.cash = self.config["initial_capital"]  # In quote currency (e.g., USDT)
        self.inventory = 0  # Will be calculated as long - short
        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.execution_history = []

        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        self._update_market_state()

        # Initialize order queue without maxlen
        self.order_queue = deque()
        # Initialize with None values for the latency period
        for _ in range(self.config["latency_steps"]):
            self.order_queue.append(None)

        self.long_position = 0
        self.short_position = 0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0

        return self._get_observation(), self._get_info()

    def _update_market_state(self):
        """
        Update market state variables and validate order book integrity.
        Raises ValueError if order book prices are not properly ordered.
        """
        # Validate bid/ask ordering
        if not all(self.bids[i, 0] >= self.bids[i+1, 0] for i in range(len(self.bids)-1)):
            raise ValueError("Bid prices must be in descending order")
        if not all(self.asks[i, 0] <= self.asks[i+1, 0] for i in range(len(self.asks)-1)):
            raise ValueError("Ask prices must be in ascending order")
        
        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]
        self.midprice = (self.best_bid + self.best_ask) / 2
        self.spread = self.best_ask - self.best_bid

    def step(self, action):
        self.step_count += 1
        
        # Parse action components
        signed_volume, price_offset, cancel_fraction = action
        is_buy = signed_volume > 0.0
        volume_scaled = abs(signed_volume)
        
        
        self._handle_order_cancellation(cancel_fraction)
        # Modified: Capture penalty from placing an order outside the allowed zone
        order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)
        
        self._process_latency_queue()
        realized_pnl = self._execute_orders()
        
        terminated = (
            self.cash < 0 or 
            abs(self.inventory) > self.config["max_inventory"] or
            self.current_step >= len(self.order_book_history) - 1
        )
        truncated = self.step_count >= self.max_steps

        # Add position liquidation at episode end
        if terminated or truncated:
            liquidation_pnl = self._liquidate_positions()
            realized_pnl += liquidation_pnl

        mtm = self.cash + self.inventory * self.midprice
        risk_penalty = self._calculate_risk_penalty()
        
        # Modified: Add an activity bonus to encourage trading
        # The bonus is proportional to the total executed volume in this step.
        activity_bonus = self.config.get("activity_bonus", 0.0) * getattr(self, "last_executed_volume", 0.0)
        reward = realized_pnl - risk_penalty + order_penalty + activity_bonus
        
        self._update_step_state()
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_order_cancellation(self, cancel_fraction):
        if self.active_orders:
            num_to_cancel = int(cancel_fraction * len(self.active_orders))
            self.active_orders = self.active_orders[num_to_cancel:]

    def _place_new_order(self, is_buy, price_offset, volume_scaled):
        penalty = 0.0
        if len(self.active_orders) >= self.config["max_active_orders"]:
            return penalty

        # Use midprice as reference price and adjust direction logic
        price_offset_ticks = price_offset * self.config["price_offset_ticks"]
        limit_price = self.midprice + (price_offset_ticks * self.config["tick_size"])
        limit_price = self._round_to_tick(limit_price)
        
        # Modified: Check if order is outside the allowed clipped zone and apply penalty
        if is_buy and limit_price > self.best_ask + (self.config["price_offset_ticks"] * self.config["tick_size"]):
            penalty -= self.config.get("invalid_order_penalty", 1.0)
            #assert False, "Invalid order placed"
            return penalty
        if not is_buy and limit_price < self.best_bid - (self.config["price_offset_ticks"] * self.config["tick_size"]):
            penalty -= self.config.get("invalid_order_penalty", 1.0)
            #assert False, "Invalid order placed"
            return penalty

        # Calculate volume
        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0, round(volume / self.config["lot_size"]) * self.config["lot_size"])
        if volume <= 0:
            return penalty

        # Create order
        order = {
            "id": self.order_id_counter,
            "price": limit_price,
            "volume": volume,
            "is_buy": is_buy,
            "timestamp": self.current_step
        }
        self.order_id_counter += 1
        self.order_queue.append(order)
        return penalty

    def _round_to_tick(self, price):
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_latency_queue(self):
        """Process orders in the latency queue with max_active_orders check"""
        if len(self.order_queue) > 0:
            oldest_order = self.order_queue.popleft()
            if oldest_order is not None:
                # Only add if under max_active_orders
                if len(self.active_orders) < self.config["max_active_orders"]:
                    self.active_orders.append(oldest_order)
                else:
                    # Optionally track rejected orders
                    pass

    def _execute_orders(self):
        realized_pnl = 0.0
        transaction_costs = 0.0
        total_executed_volume = 0.0  # Track executed volume to reward activity
        new_active_orders = []

        for order in self.active_orders:
            remaining = order["volume"]
            levels = self.asks if order["is_buy"] else self.bids

            for level_price, level_qty in levels:
                if remaining <= 0:
                    break

                if (order["is_buy"] and order["price"] >= level_price) or \
                   (not order["is_buy"] and order["price"] <= level_price):
                    
                    fill = min(remaining, level_qty)
                    total_executed_volume += fill  # accumulate executed volume

                    # Check against max_inventory first
                    potential_new_inventory = self.long_position - self.short_position + \
                                              (fill if order["is_buy"] else -fill)
                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        max_allowed = self.config["max_inventory"] - abs(self.inventory)
                        fill = min(fill, max_allowed)
                        if fill <= 0:
                            continue

                    # Calculate executed_value after fill adjustment
                    executed_value = fill * level_price
                    transaction_cost = self.config["transaction_cost"] * executed_value
                    transaction_costs += transaction_cost

                    pnl = self._process_fill(fill, level_price, order["is_buy"])
                    realized_pnl += pnl

                    self.inventory = self.long_position - self.short_position
                    self.cash = self.cash - executed_value if order["is_buy"] else self.cash + executed_value
                    self.cash -= transaction_cost
                    remaining -= fill

            # Handle unfilled portion
            if remaining > 0:
                new_active_orders.append({**order, "volume": remaining})

        self.active_orders = new_active_orders
        # Store executed volume to use in reward calculation (activity bonus)
        self.last_executed_volume = total_executed_volume
        return realized_pnl - transaction_costs  # Include transaction costs in PnL

    def _process_fill(self, fill_qty, price, is_buy):
        """Process a fill with clear position management"""
        if is_buy:
            # First cover any existing short
            if self.short_position > 0:
                cover_qty = min(fill_qty, self.short_position)
                pnl = (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                remaining = fill_qty - cover_qty
                
                # Then add to long if anything remains
                if remaining > 0:
                    self._add_to_long(remaining, price)
                
                return pnl
            else:
                self._add_to_long(fill_qty, price)
                return 0.0
        else:
            # First cover any existing long
            if self.long_position > 0:
                close_qty = min(fill_qty, self.long_position)
                pnl = (price - self.long_avg_cost) * close_qty
                self.long_position -= close_qty
                remaining = fill_qty - close_qty
                
                # Then add to short if anything remains
                if remaining > 0:
                    self._add_to_short(remaining, price)
                
                return pnl
            else:
                self._add_to_short(fill_qty, price)
                return 0.0

    def _add_to_long(self, qty, price):
        """Add quantity to long position"""
        total_position = self.long_position + qty
        self.long_avg_cost = (
            (self.long_avg_cost * self.long_position + price * qty) / total_position
        )
        self.long_position = total_position

    def _add_to_short(self, qty, price):
        """Add quantity to short position"""
        total_position = self.short_position + qty
        self.short_avg_cost = (
            (self.short_avg_cost * self.short_position + price * qty) / total_position
        )
        self.short_position = total_position

    def _calculate_risk_penalty(self):
        """Quadratic risk penalty based on inventory"""
        normalized_inventory = self.inventory / self.config["max_inventory"]
        return (
            self.config["inventory_penalty"] * 
            (normalized_inventory ** 2) * 
            self.midprice
        )

    def _update_step_state(self) -> None:
        """
        Update the environment state for the next timestep.
        Updates current_step, bids, asks, and market state if not at the end of data.
        """
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            self.bids = self.order_book_history[self.current_step]["bids"]
            self.asks = self.order_book_history[self.current_step]["asks"]
            self._update_market_state()

    def _get_raw_observation(self):
        market_features = [
            self.cash + self.inventory * self.midprice,  # MTM
            self.inventory,  # Net inventory
            len(self.active_orders),
            self.long_position,
            self.short_position,
            self.long_avg_cost / self.midprice if self.long_position > 0 else 0,
            self.short_avg_cost / self.midprice if self.short_position > 0 else 0,
            (self.best_bid - self.midprice) / self.midprice if self.midprice != 0 else 0,
            (self.best_ask - self.midprice) / self.midprice if self.midprice != 0 else 0
        ]

        # Add safety slice to ensure we never exceed max_active_orders
        limited_orders = self.active_orders[:self.config["max_active_orders"]]
        
        order_prices = []
        order_volumes = []
        for order in limited_orders:
            order_prices.append((order["price"] - self.midprice) / self.midprice)
            order_volumes.append(order["volume"] / self.config["max_order_volume"])
        
        # Pad with zeros up to max_active_orders
        order_prices += [0] * (self.config["max_active_orders"] - len(order_prices))
        order_volumes += [0] * (self.config["max_active_orders"] - len(order_volumes))

        book_features = []
        for level in range(self.config["order_book_levels"]):
            book_features += [
                (self.bids[level, 0] - self.midprice) / self.midprice,
                self.bids[level, 1] / self.config["max_order_volume"],
                (self.asks[level, 0] - self.midprice) / self.midprice,
                self.asks[level, 1] / self.config["max_order_volume"]
            ]

        return np.concatenate([
            market_features,
            order_prices,
            order_volumes,
            book_features,
            [self.spread / self.midprice]  # single spread feature
        ], dtype=np.float32)

    def _get_observation(self):
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        return self._normalize_observation(raw_obs)

    def _update_running_stats(self, obs):
        """Update running mean and variance using exponential moving averages"""
        if self.running_stats['mean'] is None:
            self.running_stats['mean'] = obs
            self.running_stats['var'] = np.zeros_like(obs)
            self.running_stats['count'] = 1
        else:
            old_mean = self.running_stats['mean']
            # Update mean first
            self.running_stats['mean'] = old_mean * self.decay + obs * (1 - self.decay)
            # Calculate delta using new mean for variance
            delta = obs - self.running_stats['mean']
            # Update variance with proper decay
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta * delta
            self.running_stats['count'] += 1

    def _normalize_observation(self, obs):
        """Normalize observation using running statistics"""
        if self.running_stats['mean'] is None:
            return obs
        return (obs - self.running_stats['mean']) / (
            np.sqrt(self.running_stats['var']) + 1e-8
        )

    def _get_info(self):
        return {
            "mtm": self.cash + self.inventory * self.midprice,  # Total account value
            "net_inventory": self.inventory,
            "long_position": self.long_position,
            "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost,
            "short_avg_cost": self.short_avg_cost,
            "active_orders": len(self.active_orders),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "cash_change": self.cash - self.config["initial_capital"],  # Change in cash
            "episode_pnl": (self.cash + self.inventory * self.midprice) - self.config["initial_capital"]  # True PnL including position value
        }

    def render(self, mode="human"):
        """Render the environment with distinctive formatting."""
        render_msg = (
            "\n================== ENVIRONMENT RENDER ==================\n"
            f"Step {self.current_step}/{self.max_steps} | "
            f"MTM: ${self.cash + self.inventory * self.midprice:,.2f} | "
            f"Inventory: {self.inventory:+d} | "
            f"Active Orders: {len(self.active_orders)} | "
            f"Best Bid: ${self.best_bid:.2f} | "
            f"Best Ask: ${self.best_ask:.2f}"
            "\n====================================================="
        )
        # Use a separate logger for rendering
        logging.info("\033[92m%s\033[0m", render_msg)  # Green color

    def _liquidate_positions(self):
        """Liquidate all positions using best available prices"""
        liquidation_pnl = 0.0
        transaction_costs = 0.0
        
        # Liquidate long positions at best bid (more realistic)
        if self.long_position > 0:
            executed_value = self.best_bid * self.long_position
            transaction_cost = self.config["transaction_cost"] * executed_value
            liquidation_pnl += (self.best_bid - self.long_avg_cost) * self.long_position - transaction_cost
            self.cash += executed_value - transaction_cost
            self.long_position = 0
            
        # Cover short positions at best ask (more realistic)
        if self.short_position > 0:
            executed_value = self.best_ask * self.short_position
            transaction_cost = self.config["transaction_cost"] * executed_value
            liquidation_pnl += (self.short_avg_cost - self.best_ask) * self.short_position - transaction_cost
            self.cash -= executed_value + transaction_cost
            self.short_position = 0
        
        self.inventory = 0
        self.active_orders = []
        self.order_queue.clear()
        
        return liquidation_pnl

# Updated example configuration with clear units and an added parameter for order penalty and activity bonus
config = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_small.csv",
    "initial_capital": 20000.0,
    "max_steps": 10000,
    "order_book_levels": 9,
    "price_offset_ticks": 40,
    "max_order_volume": 2.5,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_active_orders": 15,
    "inventory_penalty": 0.0,
    "transaction_cost": 0.0,
    "max_inventory": 8,
    "invalid_order_penalty": 1,
    "activity_bonus": 0.0
}

