from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

class HFTEnvWeb(gym.Env):
    def __init__(self, config):
        super(HFTEnvWeb, self).__init__()
        self.config = config
        self._validate_config()

        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config["max_steps"], len(self.order_book_history))
        
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.observation_space = self._create_observation_space()

        self.running_stats = {'mean': None, 'var': None, 'count': 0}
        self.decay = 0.999
        self.executions = []  # Track executions for web rendering
        self.reset()

    def _validate_config(self):
        required_keys = ["csv_path", "initial_capital", "max_steps", "order_book_levels", "price_offset_ticks", 
                         "max_order_volume", "latency_steps", "tick_size", "lot_size", "max_active_orders", 
                         "inventory_penalty", "transaction_cost", "max_inventory"]
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
            bids = np.array([[row[f"bid{i}"], row[f"bidqty{i}"]] for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
            asks = np.array([[row[f"ask{i}"], row[f"askqty{i}"]] for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
            if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any() or (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                raise ValueError("Invalid order book data")
            history.append({"bids": bids, "asks": asks})
        return history

    def _create_observation_space(self):
        obs_size = (9 + 2 * self.config["max_active_orders"] + 4 * self.config["order_book_levels"] + 1)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.running_stats = {'mean': None, 'var': None, 'count': 0}
        if hasattr(self, 'long_position') and (self.long_position > 0 or self.short_position > 0):
            self._liquidate_positions()
        
        self.current_step = 0
        self.cash = self.config["initial_capital"]
        self.inventory = 0
        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.executions = []
        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        self._update_market_state()
        self.order_queue = deque([None] * self.config["latency_steps"])
        self.long_position = 0
        self.short_position = 0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0
        return self._get_observation(), self._get_info()

    def _update_market_state(self):
        if not all(self.bids[i, 0] >= self.bids[i+1, 0] for i in range(len(self.bids)-1)) or \
           not all(self.asks[i, 0] <= self.asks[i+1, 0] for i in range(len(self.asks)-1)):
            raise ValueError("Order book prices not properly ordered")
        self.best_bid = self.bids[0, 0]
        self.best_ask = self.asks[0, 0]
        self.midprice = (self.best_bid + self.best_ask) / 2
        self.spread = self.best_ask - self.best_bid

    def step(self, action):
        self.step_count += 1
        signed_volume, price_offset, cancel_fraction = action
        is_buy = signed_volume > 0.0
        volume_scaled = abs(signed_volume)
        
        self._handle_order_cancellation(cancel_fraction)
        order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)
        self._process_latency_queue()
        realized_pnl = self._execute_orders()
        
        terminated = (self.cash < 0 or abs(self.inventory) > self.config["max_inventory"] or 
                      self.current_step >= len(self.order_book_history) - 1)
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            liquidation_pnl = self._liquidate_positions()
            realized_pnl += liquidation_pnl

        mtm = self.cash + self.inventory * self.midprice
        risk_penalty = self._calculate_risk_penalty()
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

        price_offset_ticks = price_offset * self.config["price_offset_ticks"]
        limit_price = self.midprice + (price_offset_ticks * self.config["tick_size"])
        limit_price = self._round_to_tick(limit_price)

        if is_buy and limit_price > self.best_ask + (self.config["price_offset_ticks"] * self.config["tick_size"]) or \
           not is_buy and limit_price < self.best_bid - (self.config["price_offset_ticks"] * self.config["tick_size"]):
            penalty -= self.config.get("invalid_order_penalty", 1.0)
            return penalty

        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0, round(volume / self.config["lot_size"]) * self.config["lot_size"])
        if volume <= 0:
            return penalty

        order = {"id": self.order_id_counter, "price": limit_price, "volume": volume, "is_buy": is_buy, "timestamp": self.current_step}
        self.order_id_counter += 1
        self.order_queue.append(order)
        return penalty

    def _round_to_tick(self, price):
        return round(price / self.config["tick_size"]) * self.config["tick_size"]

    def _process_latency_queue(self):
        if len(self.order_queue) > 0:
            oldest_order = self.order_queue.popleft()
            if oldest_order is not None:
                self.active_orders.append(oldest_order)

    def _execute_orders(self):
        realized_pnl = 0.0
        transaction_costs = 0.0
        total_executed_volume = 0.0
        new_active_orders = []
        self.executions = []  # Reset executions for this step

        for order in self.active_orders:
            remaining = order["volume"]
            levels = self.asks if order["is_buy"] else self.bids

            for level_price, level_qty in levels:
                if remaining <= 0:
                    break
                if (order["is_buy"] and order["price"] >= level_price) or (not order["is_buy"] and order["price"] <= level_price):
                    fill = min(remaining, level_qty)
                    potential_new_inventory = self.long_position - self.short_position + (fill if order["is_buy"] else -fill)
                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        max_allowed = self.config["max_inventory"] - abs(self.inventory)
                        fill = min(fill, max_allowed)
                        if fill <= 0:
                            continue

                    executed_value = fill * level_price
                    transaction_cost = self.config["transaction_cost"] * executed_value
                    transaction_costs += transaction_cost
                    total_executed_volume += fill

                    pnl = self._process_fill(fill, level_price, order["is_buy"])
                    realized_pnl += pnl
                    self.executions.append({"price": level_price, "volume": fill, "is_buy": order["is_buy"]})

                    self.inventory = self.long_position - self.short_position
                    self.cash = self.cash - executed_value if order["is_buy"] else self.cash + executed_value
                    self.cash -= transaction_cost
                    remaining -= fill

            if remaining > 0:
                new_active_orders.append({**order, "volume": remaining})

        self.active_orders = new_active_orders
        self.last_executed_volume = total_executed_volume
        return realized_pnl - transaction_costs

    def _process_fill(self, fill_qty, price, is_buy):
        if is_buy:
            if self.short_position > 0:
                cover_qty = min(fill_qty, self.short_position)
                pnl = (self.short_avg_cost - price) * cover_qty
                self.short_position -= cover_qty
                remaining = fill_qty - cover_qty
                if remaining > 0:
                    self._add_to_long(remaining, price)
                return pnl
            else:
                self._add_to_long(fill_qty, price)
                return 0.0
        else:
            if self.long_position > 0:
                close_qty = min(fill_qty, self.long_position)
                pnl = (price - self.long_avg_cost) * close_qty
                self.long_position -= close_qty
                remaining = fill_qty - close_qty
                if remaining > 0:
                    self._add_to_short(remaining, price)
                return pnl
            else:
                self._add_to_short(fill_qty, price)
                return 0.0

    def _add_to_long(self, qty, price):
        total_position = self.long_position + qty
        self.long_avg_cost = (self.long_avg_cost * self.long_position + price * qty) / total_position
        self.long_position = total_position

    def _add_to_short(self, qty, price):
        total_position = self.short_position + qty
        self.short_avg_cost = (self.short_avg_cost * self.short_position + price * qty) / total_position
        self.short_position = total_position

    def _calculate_risk_penalty(self):
        normalized_inventory = self.inventory / self.config["max_inventory"]
        return self.config["inventory_penalty"] * (normalized_inventory ** 2) * self.midprice

    def _update_step_state(self):
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            self.bids = self.order_book_history[self.current_step]["bids"]
            self.asks = self.order_book_history[self.current_step]["asks"]
            self._update_market_state()

    def _get_raw_observation(self):
        market_features = [
            self.cash + self.inventory * self.midprice, self.inventory, len(self.active_orders),
            self.long_position, self.short_position,
            self.long_avg_cost / self.midprice if self.long_position > 0 else 0,
            self.short_avg_cost / self.midprice if self.short_position > 0 else 0,
            (self.best_bid - self.midprice) / self.midprice if self.midprice != 0 else 0,
            (self.best_ask - self.midprice) / self.midprice if self.midprice != 0 else 0
        ]
        limited_orders = self.active_orders[:self.config["max_active_orders"]]
        order_prices = [((o["price"] - self.midprice) / self.midprice) for o in limited_orders] + [0] * (self.config["max_active_orders"] - len(limited_orders))
        order_volumes = [(o["volume"] / self.config["max_order_volume"]) for o in limited_orders] + [0] * (self.config["max_active_orders"] - len(limited_orders))
        book_features = [item for level in range(self.config["order_book_levels"]) for item in [
            (self.bids[level, 0] - self.midprice) / self.midprice, self.bids[level, 1] / self.config["max_order_volume"],
            (self.asks[level, 0] - self.midprice) / self.midprice, self.asks[level, 1] / self.config["max_order_volume"]
        ]]
        return np.concatenate([market_features, order_prices, order_volumes, book_features, [self.spread / self.midprice]], dtype=np.float32)

    def _get_observation(self):
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        return self._normalize_observation(raw_obs)

    def _update_running_stats(self, obs):
        if self.running_stats['mean'] is None:
            self.running_stats['mean'] = obs
            self.running_stats['var'] = np.zeros_like(obs)
            self.running_stats['count'] = 1
        else:
            old_mean = self.running_stats['mean']
            self.running_stats['mean'] = old_mean * self.decay + obs * (1 - self.decay)
            delta = obs - self.running_stats['mean']
            self.running_stats['var'] = self.decay * self.running_stats['var'] + (1 - self.decay) * delta * delta
            self.running_stats['count'] += 1

    def _normalize_observation(self, obs):
        if self.running_stats['mean'] is None:
            return obs
        return (obs - self.running_stats['mean']) / (np.sqrt(self.running_stats['var']) + 1e-8)

    def _get_info(self):
        return {
            "mtm": self.cash + self.inventory * self.midprice, "net_inventory": self.inventory,
            "long_position": self.long_position, "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost, "short_avg_cost": self.short_avg_cost,
            "active_orders": len(self.active_orders), "best_bid": self.best_bid, "best_ask": self.best_ask,
            "cash_change": self.cash - self.config["initial_capital"],
            "episode_pnl": (self.cash + self.inventory * self.midprice) - self.config["initial_capital"]
        }

    def _liquidate_positions(self):
        liquidation_pnl = 0.0
        if self.long_position > 0:
            executed_value = self.best_bid * self.long_position
            transaction_cost = self.config["transaction_cost"] * executed_value
            liquidation_pnl += (self.best_bid - self.long_avg_cost) * self.long_position - transaction_cost
            self.cash += executed_value - transaction_cost
            self.long_position = 0
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

    # Web Rendering Methods
    def get_order_book_data(self):
        return {"bids": [list(bid) for bid in self.bids], "asks": [list(ask) for ask in self.asks]}

    def get_state_data(self):
        return {"obs": list(self._get_observation()), "info": self._get_info()}

    def get_execution_data(self):
        return [{"price": e["price"], "volume": e["volume"], "type": "buy" if e["is_buy"] else "sell"} for e in self.executions]

    def render(self, mode="human"):
        logging.info(f"\033[92mStep {self.current_step}/{self.max_steps} | MTM: ${self.cash + self.inventory * self.midprice:,.2f} | "
                     f"Inventory: {self.inventory:+d} | Active Orders: {len(self.active_orders)} | "
                     f"Best Bid: ${self.best_bid:.2f} | Best Ask: ${self.best_ask:.2f}\033[0m")

# Configuration
config = {
    "csv_path": str((DATA_DIR / "orderbook_trimmed_small.csv").resolve()),
    "initial_capital": 20000.0,
    "max_steps": 10000,
    "order_book_levels": 9,
    "price_offset_ticks": 15,
    "max_order_volume": 100.0,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_active_orders": 15,
    "inventory_penalty": 0.0,
    "transaction_cost": 0.0,
    "max_inventory": 5,
    "invalid_order_penalty": 1,
    "activity_bonus": 0.0
}
