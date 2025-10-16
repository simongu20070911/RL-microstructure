from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging
from flask import Flask, request, jsonify, render_template_string
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ------------------------------
# HFT Environment Definition
# ------------------------------

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
        if "invalid_order_penalty" not in self.config:
            self.config["invalid_order_penalty"] = 1.0
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _load_order_book_data(self):
        try:
            df = pd.read_csv(self.config["csv_path"])
            logging.info(f"Loaded CSV file with {len(df)} rows")
            
            # Validate columns
            required_columns = ["timestamp"]
            for i in range(1, self.config["order_book_levels"] + 1):
                required_columns += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing columns in CSV data: {missing_cols}")
            
            history = []
            valid_rows = 0
            for idx, row in df.iterrows():
                try:
                    # Convert to numpy arrays and validate data
                    bids = np.array([[float(row[f"bid{i}"]), float(row[f"bidqty{i}"])] 
                                for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                    asks = np.array([[float(row[f"ask{i}"]), float(row[f"askqty{i}"])] 
                                for i in range(1, self.config["order_book_levels"] + 1)], dtype=np.float32)
                    
                    # Validate price levels
                    if not (np.all(bids[:-1, 0] >= bids[1:, 0]) and np.all(asks[:-1, 0] <= asks[1:, 0])):
                        logging.warning(f"Skipping row {idx}: Invalid price levels")
                        continue
                    
                    # Validate prices and quantities
                    if (bids[:, 0] <= 0).any() or (asks[:, 0] <= 0).any():
                        logging.warning(f"Skipping row {idx}: Invalid prices")
                        continue
                    if (bids[:, 1] < 0).any() or (asks[:, 1] < 0).any():
                        logging.warning(f"Skipping row {idx}: Negative quantities")
                        continue
                    
                    # Validate bid-ask relationship
                    if bids[0, 0] >= asks[0, 0]:
                        logging.warning(f"Skipping row {idx}: Crossed market")
                        continue
                        
                    history.append({"bids": bids, "asks": asks})
                    valid_rows += 1
                    
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error processing row {idx}: {e}")
                    continue
            
            logging.info(f"Successfully loaded {valid_rows} valid order book states")
            if not history:
                raise ValueError("No valid order book states found in CSV")
                
            return history
            
        except Exception as e:
            logging.error(f"Failed to load order book data: {e}")
            raise

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
        self.running_stats = {'mean': None, 'var': None, 'count': 0}
        if hasattr(self, 'long_position') and (self.long_position > 0 or self.short_position > 0):
            self._liquidate_positions()
        self.current_step = 0
        self.cash = self.config["initial_capital"]
        self.inventory = 0
        self.active_orders = []
        self.order_id_counter = 0
        self.step_count = 0
        self.execution_history = []

        self.bids = self.order_book_history[0]["bids"]
        self.asks = self.order_book_history[0]["asks"]
        self._update_market_state()

        self.order_queue = deque()
        for _ in range(self.config["latency_steps"]):
            self.order_queue.append(None)

        self.long_position = 0
        self.short_position = 0
        self.long_avg_cost = 0.0
        self.short_avg_cost = 0.0

        return self._get_observation(), self._get_info()

    def _update_market_state(self):
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
        signed_volume, price_offset, cancel_fraction = action
        is_buy = signed_volume > 0.0
        volume_scaled = abs(signed_volume)
        self._handle_order_cancellation(cancel_fraction)
        order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)
        self._process_latency_queue()
        realized_pnl = self._execute_orders()
        terminated = (
            self.cash < 0 or 
            abs(self.inventory) > self.config["max_inventory"] or
            self.current_step >= len(self.order_book_history) - 1
        )
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
        if is_buy and limit_price > self.best_ask + (self.config["price_offset_ticks"] * self.config["tick_size"]):
            penalty -= self.config.get("invalid_order_penalty", 1.0)
            return penalty
        if not is_buy and limit_price < self.best_bid - (self.config["price_offset_ticks"] * self.config["tick_size"]):
            penalty -= self.config.get("invalid_order_penalty", 1.0)
            return penalty
        volume = volume_scaled * self.config["max_order_volume"]
        volume = max(0, round(volume / self.config["lot_size"]) * self.config["lot_size"])
        if volume <= 0:
            return penalty
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
        if len(self.order_queue) > 0:
            oldest_order = self.order_queue.popleft()
            if oldest_order is not None:
                self.active_orders.append(oldest_order)

    def _execute_orders(self):
        realized_pnl = 0.0
        transaction_costs = 0.0
        total_executed_volume = 0.0
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
                    total_executed_volume += fill
                    potential_new_inventory = self.long_position - self.short_position + (fill if order["is_buy"] else -fill)
                    if abs(potential_new_inventory) > self.config["max_inventory"]:
                        max_allowed = self.config["max_inventory"] - abs(self.inventory)
                        fill = min(fill, max_allowed)
                        if fill <= 0:
                            continue
                    executed_value = fill * level_price
                    transaction_cost = self.config["transaction_cost"] * executed_value
                    transaction_costs += transaction_cost
                    pnl = self._process_fill(fill, level_price, order["is_buy"])
                    realized_pnl += pnl
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
        self.long_avg_cost = ((self.long_avg_cost * self.long_position + price * qty) / total_position)
        self.long_position = total_position

    def _add_to_short(self, qty, price):
        total_position = self.short_position + qty
        self.short_avg_cost = ((self.short_avg_cost * self.short_position + price * qty) / total_position)
        self.short_position = total_position

    def _calculate_risk_penalty(self):
        normalized_inventory = self.inventory / self.config["max_inventory"]
        return self.config["inventory_penalty"] * (normalized_inventory ** 2) * self.midprice

    def _update_step_state(self) -> None:
        if self.current_step < len(self.order_book_history) - 1:
            self.current_step += 1
            self.bids = self.order_book_history[self.current_step]["bids"]
            self.asks = self.order_book_history[self.current_step]["asks"]
            self._update_market_state()

    def _get_raw_observation(self):
        market_features = [
            self.cash + self.inventory * self.midprice,
            self.inventory,
            len(self.active_orders),
            self.long_position,
            self.short_position,
            self.long_avg_cost / self.midprice if self.long_position > 0 else 0,
            self.short_avg_cost / self.midprice if self.short_position > 0 else 0,
            (self.best_bid - self.midprice) / self.midprice if self.midprice != 0 else 0,
            (self.best_ask - self.midprice) / self.midprice if self.midprice != 0 else 0
        ]
        limited_orders = self.active_orders[:self.config["max_active_orders"]]
        order_prices = []
        order_volumes = []
        for order in limited_orders:
            order_prices.append((order["price"] - self.midprice) / self.midprice)
            order_volumes.append(order["volume"] / self.config["max_order_volume"])
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
            [self.spread / self.midprice if self.midprice != 0 else 0]
        ], dtype=np.float32)

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

    def _get_observation(self):
        raw_obs = self._get_raw_observation()
        self._update_running_stats(raw_obs)
        return self._normalize_observation(raw_obs)

    def _get_info(self):
        return {
            "mtm": self.cash + self.inventory * self.midprice,
            "net_inventory": self.inventory,
            "long_position": self.long_position,
            "short_position": self.short_position,
            "long_avg_cost": self.long_avg_cost,
            "short_avg_cost": self.short_avg_cost,
            "active_orders": len(self.active_orders),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "cash_change": self.cash - self.config["initial_capital"],
            "episode_pnl": (self.cash + self.inventory * self.midprice) - self.config["initial_capital"]
        }

    def render(self, mode="human"):
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
        logging.info("\033[92m%s\033[0m", render_msg)

    def _liquidate_positions(self):
        liquidation_pnl = 0.0
        transaction_costs = 0.0
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

    def get_state(self):
        """
        Returns the environment state in a JSON-serializable format (no float32 or np arrays).
        """
        # Helper function to recursively convert numpy types to Python types
        def convert_np(o):
            if isinstance(o, np.generic):
                return o.item()
            elif isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, list):
                return [convert_np(i) for i in o]
            elif isinstance(o, dict):
                return {k: convert_np(v) for k, v in o.items()}
            return o

        obs = self._get_raw_observation()  # raw observation
        state = {
            "current_step": self.current_step,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midprice": self.midprice,
            "spread": self.spread,
            "cash": self.cash,
            "inventory": self.inventory,
            "active_orders": self.active_orders,
            "order_book_bids": self.bids,
            "order_book_asks": self.asks,
            "info": self._get_info(),
            "observation": obs
        }
        return convert_np(state)

# ------------------------------
# End HFTEnv Definition
# ------------------------------

# Example configuration (adjust csv_path as needed)
config = {
    "csv_path": str((DATA_DIR / "orderbook_trimmed_small.csv").resolve()),
    "initial_capital": 2000.0,
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

try:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create and initialize environment
    env = HFTEnv(config)
    initial_obs, initial_info = env.reset()
    logging.info("Environment initialized successfully")
    logging.info(f"Initial state: Bid={env.best_bid:.2f}, Ask={env.best_ask:.2f}, Spread={env.spread:.2f}")
    
except Exception as e:
    logging.error(f"Failed to initialize environment: {e}")
    raise

# ------------------------------
# Flask Web Interface
# ------------------------------

app = Flask(__name__)

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HFTEnv Exchange Simulator</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body { padding: 20px; }
            .order-book { margin-top: 20px; }
            .order-book table { width: 100%; }
            .market-info { font-size: 1.2em; margin: 20px 0; }
            .bid-row { background-color: #e0f7e9; }
            .ask-row { background-color: #fde0e0; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>HFTEnv Exchange Simulator</h1>
            
            <!-- Market Information -->
            <div class="row market-info">
                <div class="col-md-3">
                    <strong>MTM:</strong> <span id="mtm">N/A</span>
                </div>
                <div class="col-md-3">
                    <strong>Position:</strong> <span id="position">N/A</span>
                </div>
                <div class="col-md-3">
                    <strong>PnL:</strong> <span id="pnl">N/A</span>
                </div>
                <div class="col-md-3">
                    <strong>Spread:</strong> <span id="spread">N/A</span>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <h3>Bids</h3>
                    <table class="table table-bordered" id="bids-table">
                        <thead><tr><th>Price</th><th>Quantity</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
                <div class="col-md-6">
                    <h3>Asks</h3>
                    <table class="table table-bordered" id="asks-table">
                        <thead><tr><th>Price</th><th>Quantity</th></tr></thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>

            <!-- Market Observations -->
            <div class="card mt-4">
                <div class="card-header">
                    <h3 class="mb-0">Market Observations</h3>
                </div>
                <div class="card-body">
                    <table class="table table-sm" id="observations-table">
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>

            <!-- Order Entry Form -->
            <div class="card mt-4">
                <div class="card-header">
                    <h3 class="mb-0">Place Order</h3>
                </div>
                <div class="card-body">
                    <form id="order-form" class="mb-4">
                        <div class="form-row">
                            <div class="form-group col-md-3">
                                <label for="side">Side</label>
                                <select class="form-control" id="side" name="side" required>
                                    <option value="bid">Buy</option>
                                    <option value="ask">Sell</option>
                                </select>
                            </div>
                            <div class="form-group col-md-3">
                                <label for="price">Price</label>
                                <input type="number" step="0.01" min="0" class="form-control" id="price" name="price" required>
                            </div>
                            <div class="form-group col-md-3">
                                <label for="quantity">Quantity</label>
                                <input type="number" step="0.01" min="0" class="form-control" id="quantity" name="quantity" required>
                            </div>
                            <div class="form-group col-md-3 align-self-end">
                                <button type="submit" class="btn btn-primary btn-block">Place Order</button>
                            </div>
                        </div>
                        <div id="order-error" class="error mt-2" style="display:none;"></div>
                    </form>
                </div>
            </div>
        </div>
        
        <script>
            async function fetchState() {
                try {
                    const response = await fetch('/env_state');
                    const data = await response.json();

                    // Debug: log the entire response
                    console.log('env_state data:', data);

                    updateUI(data);
                } catch (error) {
                    console.error('Failed to fetch state:', error);
                }
            }
            
            function updateUI(data) {
                // Update market info
                document.getElementById('mtm').innerText = `$${data.info.mtm.toFixed(2)}`;
                document.getElementById('position').innerText = data.info.net_inventory;
                document.getElementById('pnl').innerText = `$${data.info.episode_pnl.toFixed(2)}`;
                document.getElementById('spread').innerText = `$${data.spread.toFixed(2)}`;
                
                // Update order books
                updateOrderBook('bids-table', data.order_book_bids, true);
                updateOrderBook('asks-table', data.order_book_asks, false);
                
                // Update observations
                const obsTable = document.querySelector('#observations-table tbody');
                obsTable.innerHTML = '';
                
                // Define feature names for the first 9 items
                const featureNames = [
                    'MTM', 'Inventory', 'Active Orders', 'Long Position', 'Short Position',
                    'Long Avg Cost Ratio', 'Short Avg Cost Ratio', 'Bid Deviation', 'Ask Deviation'
                ];
                
                const obs = data.observation;
                // Display first 9 market features from obs
                for (let i = 0; i < 9; i++) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${featureNames[i]}</td>
                        <td>${obs[i].toFixed(4)}</td>
                    `;
                    obsTable.appendChild(row);
                }
                
                // The next 2*max_active_orders are order prices and volumes
                const maxActive = ${config["max_active_orders"]};
                const orderPricesStart = 9;
                const orderVolumesStart = 9 + maxActive;
                
                // Then 4*order_book_levels come after that
                const bookStart = orderVolumesStart + maxActive;
                const nLevels = ${config["order_book_levels"]};

                // Display each order book level's normalized price/volume for bids/asks
                let currentIndex = bookStart;
                for (let i = 0; i < nLevels; i++) {
                    // bids: [price deviation, volume], asks: [price deviation, volume]
                    const bidPrice = obs[currentIndex].toFixed(4);
                    const bidVol = obs[currentIndex + 1].toFixed(4);
                    const askPrice = obs[currentIndex + 2].toFixed(4);
                    const askVol = obs[currentIndex + 3].toFixed(4);

                    const bidRow = document.createElement('tr');
                    bidRow.innerHTML = `
                        <td>Level ${i+1} Bid (Price/Volume)</td>
                        <td>${bidPrice} / ${bidVol}</td>
                    `;
                    obsTable.appendChild(bidRow);

                    const askRow = document.createElement('tr');
                    askRow.innerHTML = `
                        <td>Level ${i+1} Ask (Price/Volume)</td>
                        <td>${askPrice} / ${askVol}</td>
                    `;
                    obsTable.appendChild(askRow);

                    currentIndex += 4;
                }

                // The last index is the normalized spread
                const spreadNorm = obs[currentIndex].toFixed(4);
                const spreadRow = document.createElement('tr');
                spreadRow.innerHTML = `
                    <td>Spread (Normalized)</td>
                    <td>${spreadNorm}</td>
                `;
                obsTable.appendChild(spreadRow);
            }
            
            function updateOrderBook(tableId, data, isBid) {
                const tbody = document.querySelector(`#${tableId} tbody`);
                tbody.innerHTML = '';
                data.forEach(level => {
                    const row = document.createElement('tr');
                    row.classList.add(isBid ? 'bid-row' : 'ask-row');
                    const price = parseFloat(level[0]).toFixed(2);
                    const qty   = parseFloat(level[1]).toFixed(2);
                    row.innerHTML = `<td>$${price}</td><td>${qty}</td>`;
                    tbody.appendChild(row);
                });
            }
            
            document.getElementById('order-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const errorDiv = document.getElementById('order-error');
                errorDiv.style.display = 'none';
                
                try {
                    const side = document.getElementById('side').value;
                    const price = parseFloat(document.getElementById('price').value);
                    const quantity = parseFloat(document.getElementById('quantity').value);
                    
                    if (price <= 0 || quantity <= 0) {
                        throw new Error('Price and quantity must be positive');
                    }
                    
                    const response = await fetch('/place_order', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({side, price, quantity})
                    });
                    
                    const result = await response.json();
                    if (result.status === 'success') {
                        document.getElementById('price').value = '';
                        document.getElementById('quantity').value = '';
                        fetchState();
                    } else {
                        errorDiv.textContent = result.message;
                        errorDiv.style.display = 'block';
                    }
                } catch (error) {
                    errorDiv.textContent = error.message;
                    errorDiv.style.display = 'block';
                }
            });
            
            // Update state every second
            setInterval(fetchState, 1000);
            fetchState();
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/env_state')
def env_state():
    try:
        # If we've reached the end, reset the environment
        if env.current_step >= len(env.order_book_history):
            env.reset()
            logging.info("Environment reset due to end of data")
            
        state = env.get_state()
        return jsonify(state)
    except Exception as e:
        logging.error(f"Error in env_state: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/place_order', methods=['POST'])
def place_order():
    try:
        data = request.get_json()
        if not data:
            raise ValueError("No data provided")

        side = data.get('side', '').lower()
        if side not in ['bid', 'ask']:
            raise ValueError("Invalid side specified")

        price = float(data.get('price', 0))
        quantity = float(data.get('quantity', 0))
        
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive")
            
        # Check if price is within valid range
        if (side == 'bid' and price >= env.best_ask) or (side == 'ask' and price <= env.best_bid):
            raise ValueError("Price would cause immediate execution")

        # Convert to environment action
        midprice = env.midprice
        # Guard against zero midprice
        if midprice == 0:
            raise ValueError("Midprice is zero; cannot place order offset")

        price_offset = (price - midprice) / (config["price_offset_ticks"] * config["tick_size"])
        # Clip offset to [-1, 1]
        price_offset = max(min(price_offset, 1.0), -1.0)
        
        volume_scaled = quantity / config["max_order_volume"]
        volume_scaled = max(min(volume_scaled, 1.0), 0.0)
        signed_volume = volume_scaled if side == "bid" else -volume_scaled
        
        action = np.array([signed_volume, price_offset, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        return jsonify({
            'status': 'success',
            'info': info
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

def run_flask():
    app.run(port=5000, debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    flask_thread.join()
