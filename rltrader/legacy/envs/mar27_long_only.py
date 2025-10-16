import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging
import locale # For formatting currency

# Configure logging - INFO level will show lifecycle events, warnings, and errors.
# Use DEBUG for detailed execution tracing (if uncommented in the code).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set locale for currency formatting (e.g., en_US.UTF-8 or your system's locale)
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # Example locale
except locale.Error:
    logging.warning("Locale 'en_US.UTF-8' not found. Using default locale for currency formatting.")
    locale.setlocale(locale.LC_ALL, '') # Use default system locale

class HFTEnv(gym.Env):
    """
    A High-Frequency Trading Environment for Gymnasium simulating a Limit Order Book.
    ... (rest of the docstring) ...
    """
    def __init__(self, config):
        super(HFTEnv, self).__init__()
        self.logger = logging.getLogger(__name__)

        logging.info("Initializing HFT Environment...")
        self.config = config
        self._validate_config()

        # Load and validate order book data
        self.order_book_history = self._load_order_book_data()
        self.max_steps = min(self.config["max_steps"], len(self.order_book_history))
        logging.info(f"Environment configured for max_steps: {self.max_steps}")

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.observation_space = self._create_observation_space()
        logging.info(f"Action Space: {self.action_space}")
        logging.info(f"Observation Space: {self.observation_space}")

        # Initialize running statistics for observation normalization
        self.running_stats = {
            'mean': None,
            'var': None,
            'count': 0
        }
        self.decay = self.config.get('obs_norm_decay', 0.999) # Allow config override

        # --- State and PnL Tracking Variables ---
        self.position = 0.0
        self.avg_cost = 0.0
        self.current_step = 0
        self.step_count = 0
        self.cash = 0.0
        self.active_orders = []
        self.order_id_counter = 0
        self.order_queue = deque()
        self.bids = np.array([])
        self.asks = np.array([])
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.midprice = 0.0
        self.spread = 0.0
        self.last_executed_volume = 0.0

        # --- Cumulative PnL Component Trackers (Initialize here, reset in reset()) ---
        self.cumulative_realized_pnl_fills = 0.0 # PnL from buy/sell fills (gross)
        self.cumulative_transaction_costs = 0.0 # Costs from fills and liquidation
        self.cumulative_risk_penalties = 0.0    # Inventory holding penalties
        self.cumulative_order_penalties = 0.0   # Invalid order placement penalties
        self.cumulative_activity_bonuses = 0.0  # Volume execution bonuses
        self.final_liquidation_pnl_net = 0.0    # Net PnL from final liquidation event

        logging.info("Environment Initialized. Call reset() to start.")
        # Note: reset() must be called externally before first step

    # ... (keep _validate_config, _load_order_book_data, _create_observation_space) ...

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Resets the environment to its initial state for a new episode."""
        logging.info("Resetting environment...")
        super().reset(seed=seed)

        # Reset running statistics for observation normalization
        self.running_stats = {'mean': None, 'var': None, 'count': 0}
        # logging.debug("Running statistics for normalization reset.") # DEBUG

        self.current_step = 0
        self.step_count = 0
        self.cash = self.config["initial_capital"]
        self.position = 0.0
        self.avg_cost = 0.0
        self.active_orders = []
        self.order_id_counter = 0
        self.last_executed_volume = 0.0

        # Reset cumulative PnL trackers
        self.cumulative_realized_pnl_fills = 0.0
        self.cumulative_transaction_costs = 0.0
        self.cumulative_risk_penalties = 0.0
        self.cumulative_order_penalties = 0.0
        self.cumulative_activity_bonuses = 0.0
        self.final_liquidation_pnl_net = 0.0

        # Load initial market state from the first data point
        if not self.order_book_history:
             logging.error("Order book history is empty during reset.")
             raise RuntimeError("Order book history is empty, cannot reset environment.")
        try:
            self.bids = self.order_book_history[0]["bids"]
            self.asks = self.order_book_history[0]["asks"]
            self._update_market_state()
        except Exception as e:
            logging.error(f"Error setting initial market state during reset: {e}")
            raise

        # Initialize order queue
        self.order_queue = deque()
        for _ in range(self.config["latency_steps"]):
            self.order_queue.append(None)
        # logging.debug(f"Latency queue initialized with {self.config['latency_steps']} placeholders.") # DEBUG

        logging.info(f"Environment reset complete. Initial state: Cash={self.cash:.2f}, Position={self.position:.4f}, Midprice={self.midprice:.2f}")

        try:
             observation = self._get_observation()
             info = self._get_info()
        except Exception as e:
             logging.error(f"Error generating initial observation/info during reset: {e}")
             raise

        if observation.shape != self.observation_space.shape:
            logging.error(f"Observation shape mismatch after reset: Expected {self.observation_space.shape}, got {observation.shape}. Check _get_raw_observation().")
            raise ValueError(f"Observation shape mismatch after reset.")

        return observation, info

    # ... (keep _update_market_state) ...

    # 3 & 4. Modify `_execute_orders` to return gross PnL and costs separately
    def _execute_orders(self):
        """
        Matches active orders against the current order book state.
        Handles partial fills, inventory constraints.
        Updates position, average cost, and cash.
        Returns a tuple: (step_realized_pnl_gross, step_transaction_costs) for the step.
        """
        step_realized_pnl_gross = 0.0 # PnL from sells *before* costs
        step_transaction_costs = 0.0
        step_total_executed_volume = 0.0
        remaining_active_orders = []

        # logging.debug(f"--- Executing Orders (Step {self.current_step}) ---") # DEBUG
        # ... (rest of the initial logging lines) ...

        if not self.active_orders:
            # logging.debug("No active orders to execute.") # DEBUG
            return 0.0, 0.0 # Return zero PnL and zero cost

        for order in self.active_orders:
            order_remaining_volume = order["volume"]
            # logging.debug(f"Processing Order ID {order['id']}: {'BUY' if order['is_buy'] else 'SELL'} {order_remaining_volume:.4f} @ {order['price']:.4f}") # DEBUG

            if order_remaining_volume <= 1e-9:
                continue

            levels_to_match = self.asks if order["is_buy"] else self.bids
            side_filled_label = "ask" if order["is_buy"] else "bid"

            for level_idx, (level_price, level_qty) in enumerate(levels_to_match):

                if order_remaining_volume <= 1e-9: break
                if level_qty <= 1e-9: continue

                price_match = (order["is_buy"] and order["price"] >= level_price - 1e-9) or \
                              (not order["is_buy"] and order["price"] <= level_price + 1e-9)

                if price_match:
                    potential_fill = min(order_remaining_volume, level_qty)
                    fill_qty = 0
                    if order["is_buy"]:
                        inventory_headroom = self.config["max_inventory"] - self.position
                        fill_qty = max(0, min(potential_fill, inventory_headroom))
                    else: # Selling
                        fill_qty = max(0, min(potential_fill, self.position))

                    fill_qty = np.floor(fill_qty / self.config["lot_size"]) * self.config["lot_size"]

                    if fill_qty <= 1e-9:
                        # logging.debug(f" Level {level_idx} ({side_filled_label}@{level_price:.4f}): Fill=0 (...)") # DEBUG
                        continue

                    # --- Process the Fill ---
                    # logging.debug(f" Level {level_idx} ({side_filled_label}@{level_price:.4f}): Executing {fill_qty:.4f} / {level_qty:.4f}") # DEBUG

                    # 1. Calculate Realized Gross PnL (only for sells) & Update Position/AvgCost
                    pnl_from_this_fill_gross = self._process_fill(fill_qty, level_price, order["is_buy"])
                    step_realized_pnl_gross += pnl_from_this_fill_gross # Accumulate gross PnL

                    # 2. Calculate Transaction Costs
                    executed_value = fill_qty * level_price
                    cost = self.config["transaction_cost"] * executed_value
                    step_transaction_costs += cost # Accumulate costs for the step

                    # 3. Update Cash
                    self.cash -= cost # Cost applies to both buys and sells
                    if order["is_buy"]:
                        self.cash -= executed_value
                    else: # Sell
                        self.cash += executed_value

                    # 4. Update Volumes
                    order_remaining_volume -= fill_qty
                    step_total_executed_volume += fill_qty

                    # logging.debug(f"  Fill Result: GrossPnL={pnl_from_this_fill_gross:.4f}, TCost={cost:.4f}, NewCash={self.cash:.2f}, NewPos={self.position:.4f}, NewAvgCost={self.avg_cost:.4f}") # DEBUG

                # else: # Price doesn't match
                #     logging.debug(f" Level {level_idx} ({side_filled_label}@{level_price:.4f}): Price mismatch ...") # DEBUG
                #     # break # Optimization

            # Update order or discard
            if order_remaining_volume > 1e-9:
                order["volume"] = order_remaining_volume
                remaining_active_orders.append(order)
                # logging.debug(f"Order {order['id']} partially filled. Remaining Vol: {order_remaining_volume:.4f}") # DEBUG
            # else:
                # logging.debug(f"Order {order['id']} fully filled.") # DEBUG


        self.active_orders = remaining_active_orders
        self.last_executed_volume = step_total_executed_volume

        # logging.debug(f"--- Execution Finished --- Step Gross PnL: {step_realized_pnl_gross:.4f}, Step TCost: {step_transaction_costs:.4f}, Exec Vol: {step_total_executed_volume:.4f}") # DEBUG
        # logging.debug(f"After Exec: Cash={self.cash:.2f}, Pos={self.position:.4f}, AvgCost={self.avg_cost:.4f}, #Active={len(self.active_orders)}") # DEBUG

        # Return Gross PnL and Costs separately
        return step_realized_pnl_gross, step_transaction_costs

    # ... (keep _process_fill, _calculate_risk_penalty) ...

    # 3 & 4. Modify `_liquidate_positions` to return net PnL and costs separately
    def _liquidate_positions(self):
        """
        Liquidates the existing long position at the current best bid price.
        Clears active orders and latency queue. Logs details.
        Returns a tuple: (liquidation_pnl_net, liquidation_cost).
        """
        liquidation_pnl_net = 0.0 # Net PnL after costs
        liquidation_cost = 0.0    # Cost specific to this liquidation

        if self.position > 1e-9:
            logging.info(f"Liquidating position of {self.position:.4f}...")
            liquidation_price = self.best_bid

            if not np.isfinite(liquidation_price) or liquidation_price <= 0:
                 logging.error(f"Invalid liquidation price (Best Bid = {liquidation_price}). Assuming zero value.")
                 pnl_gross = (0 - self.avg_cost) * self.position
                 liquidation_cost = 0 # No transaction cost if no value
                 liquidation_pnl_net = pnl_gross # PnL is just the loss of the cost basis
                 self.cash += 0
            else:
                 executed_value = liquidation_price * self.position
                 liquidation_cost = self.config["transaction_cost"] * executed_value # Calculate cost
                 pnl_gross = (liquidation_price - self.avg_cost) * self.position
                 liquidation_pnl_net = pnl_gross - liquidation_cost # Net PnL after cost

                 self.cash += executed_value - liquidation_cost # Update cash

                 logging.info(
                     f"  Liquidation details: Sold {self.position:.4f} @ Best Bid ${liquidation_price:.4f}. "
                     f"Gross PnL=${pnl_gross:.2f}, T-Cost=${liquidation_cost:.2f} => Net PnL=${liquidation_pnl_net:.2f}"
                 )
                 logging.info(f"  Cash after liquidation: ${self.cash:.2f}")

            # Reset position state
            self.position = 0.0
            self.avg_cost = 0.0

        else:
            logging.info("No position to liquidate.")

        # Clear orders and queue (no change needed here)
        if self.active_orders:
            logging.info(f"Clearing {len(self.active_orders)} remaining active orders during liquidation.")
            self.active_orders = []
        if any(item is not None for item in self.order_queue):
            logging.info("Clearing latency queue during liquidation.")
            self.order_queue.clear()
            for _ in range(self.config["latency_steps"]):
                self.order_queue.append(None)

        # Return Net PnL and the specific Cost for this event
        return liquidation_pnl_net, liquidation_cost


    # 3. Modify `step` to use new return values and update cumulative trackers
    def step(self, action):
        """Executes one time step within the environment."""
        # logging.debug(f"--- Entering Step {self.step_count + 1} (Data Step {self.current_step}) ---") # DEBUG
        self.step_count += 1
        self.last_executed_volume = 0.0

        # 1. Parse and Validate Action
        # ... (no changes needed here) ...
        try:
            if not isinstance(action, np.ndarray) or action.shape != (3,):
                raise ValueError(f"Action must be a numpy array of shape (3,). Received: {action}")
            signed_volume, price_offset, cancel_fraction = action.astype(np.float32) # Ensure float32
            signed_volume = np.clip(signed_volume, -1.0, 1.0)
            price_offset = np.clip(price_offset, -1.0, 1.0)
            cancel_fraction = np.clip(cancel_fraction, 0.0, 1.0)
            is_buy = signed_volume > 0.0
            volume_scaled = abs(signed_volume)
            # logging.debug(f"Action Received: SignedVol={signed_volume:.3f}, PriceOffset={price_offset:.3f}, CancelFrac={cancel_fraction:.3f}") # DEBUG
        except (TypeError, ValueError) as e:
            logging.error(f"Invalid action received: {action}. Error: {e}. Taking no action this step.")
            signed_volume, price_offset, cancel_fraction = 0.0, 0.0, 0.0
            is_buy, volume_scaled = False, 0.0
            # No penalty for format error, consistent with original

        # 2. Handle Order Cancellation
        try:
            self._handle_order_cancellation(cancel_fraction)
        except Exception as e:
            logging.error(f"Error during order cancellation: {e}")

        # 3. Place New Order
        order_penalty = 0.0
        try:
            order_penalty = self._place_new_order(is_buy, price_offset, volume_scaled)
            self.cumulative_order_penalties += order_penalty # Update cumulative tracker
        except Exception as e:
            logging.error(f"Error during placing new order: {e}")
            order_penalty -= 1.0 # Penalize for error
            self.cumulative_order_penalties += order_penalty # Add error penalty

        # 4. Process Latency Queue
        try:
            self._process_latency_queue()
        except Exception as e:
            logging.error(f"Error processing latency queue: {e}")

        # 5. Execute Active Orders against Current Market
        realized_pnl_step_net = 0.0 # Net PnL for *this step* from fills
        try:
            # Get gross PnL and costs for the step
            step_realized_pnl_gross, step_transaction_costs = self._execute_orders()

            # Update cumulative trackers
            self.cumulative_realized_pnl_fills += step_realized_pnl_gross
            self.cumulative_transaction_costs += step_transaction_costs

            # Calculate net PnL for *this step's fills* for the reward calculation
            realized_pnl_step_net = step_realized_pnl_gross - step_transaction_costs

        except Exception as e:
            logging.error(f"Critical error during order execution: {e}")
            terminated = True # Force termination
            truncated = False
            reward = -100 # Large penalty
            observation = self._get_observation() # Get current observation
            info = self._get_info()
            info["error"] = f"Order execution failure: {e}"
            logging.critical("Terminating episode due to order execution failure.")
            return observation, reward, terminated, truncated, info


        # 6. Check Termination & Truncation Conditions
        terminated = False
        truncated = False
        termination_reason = None
        # ... (termination checks remain the same) ...
        if self.cash < 0:
            terminated = True
            termination_reason = "Negative Cash"
        elif self.position > self.config["max_inventory"] + 1e-9:
            terminated = True
            termination_reason = f"Max Inventory Exceeded ({self.position:.4f} > {self.config['max_inventory']})"
        elif self.current_step >= len(self.order_book_history) - 1:
            terminated = True
            termination_reason = "End of Data Reached"
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "Max Steps Reached"


        # 7. Liquidate Position at Episode End
        # This happens *within* the step if terminated/truncated
        if terminated or truncated:
            try:
                if terminated:
                     logging.warning(f"Episode terminated at data step {self.current_step} / env step {self.step_count}. Reason: {termination_reason}.")
                elif truncated:
                     logging.info(f"Episode truncated at data step {self.current_step} / env step {self.step_count}. Reason: {termination_reason}.")

                # Get net PnL and cost from liquidation
                liquidation_pnl_net, liquidation_cost = self._liquidate_positions()

                # Store the net PnL *specifically* from liquidation for reporting
                self.final_liquidation_pnl_net = liquidation_pnl_net

                # Add the liquidation cost to the cumulative costs
                self.cumulative_transaction_costs += liquidation_cost

                # Add the net PnL from liquidation to this final step's net realized PnL
                realized_pnl_step_net += liquidation_pnl_net

            except Exception as e:
                 logging.error(f"Error during final liquidation: {e}")
                 realized_pnl_step_net -= 10 # Penalize


        # 8. Calculate Mark-to-Market Value and Risk Penalty
        mtm = self.cash + self.position * self.midprice # MTM is calculated *before* potential state update
        risk_penalty = self._calculate_risk_penalty()
        self.cumulative_risk_penalties += risk_penalty # Update cumulative tracker


        # 9. Calculate Reward for the Step
        activity_bonus = self.config.get("activity_bonus", 0.0) * self.last_executed_volume
        self.cumulative_activity_bonuses += activity_bonus # Update cumulative tracker

        # Reward = Net PnL (fills + liquidation if any) - Risk Penalty + Order Penalty + Activity Bonus
        reward = realized_pnl_step_net - risk_penalty + order_penalty + activity_bonus
        # logging.debug(f"Reward Calculation: StepNetPnL={realized_pnl_step_net:.4f}, RiskPen={risk_penalty:.4f}, OrderPen={order_penalty:.4f}, ActBonus={activity_bonus:.4f} -> Reward={reward:.4f}") # DEBUG


        # 10. Update Market State for the *Next* Step (if not ended)
        if not (terminated or truncated): # Avoid update if already ended
             try:
                 self._update_step_state()
             except Exception as e:
                  logging.error(f"Failed to update market state for next step: {e}. Terminating.")
                  terminated = True
                  if not termination_reason: termination_reason = "Market state update failure"
                  reward -= 100 # Penalize heavily

        # 11. Get Next Observation and Info dictionary
        try:
            observation = self._get_observation()
            info = self._get_info() # Info now includes cumulative trackers implicitly
            if termination_reason:
                 info["termination_reason"] = termination_reason
                 # Add final state info keys needed for summary print
                 info["final_mtm"] = mtm
                 info["final_cash"] = self.cash
                 info["final_position"] = self.position # Should be 0 after liquidation

        except Exception as e:
             logging.error(f"Error generating observation/info after step: {e}. Returning last valid state if possible.")
             terminated = True
             if not termination_reason: termination_reason = "Observation generation failure"
             observation = np.zeros(self.observation_space.shape, dtype=np.float32)
             info = self._get_info() # Try again or return minimal info
             info["error"] = f"Observation generation failure: {e}"
             reward -= 100

        # Final validation of observation shape
        if observation.shape != self.observation_space.shape:
             logging.error(f"Post-step observation shape mismatch: Expected {self.observation_space.shape}, got {observation.shape}. Check _get_raw_observation().")
             raise ValueError(f"Post-step observation shape mismatch.")

        # logging.debug(f"--- Exiting Step --- MTM={info.get('mtm', 'N/A'):.2f}, Reward={reward:.4f}, Term={terminated}, Trunc={truncated}") # DEBUG
        return observation, reward, terminated, truncated, info

    # ... (keep _handle_order_cancellation, _place_new_order, _round_to_tick, _process_latency_queue) ...
    # ... (keep _update_step_state, _get_raw_observation, _get_observation, _update_running_stats, _normalize_observation) ...

    def _get_info(self):
        """Returns auxiliary information about the current environment state."""
        # MTM calculation moved to step() as it's needed before potential state update
        # Recalculate here just for the info dict if needed, or use value from step if possible
        mtm = self.cash + self.position * self.midprice
        initial_capital = self.config["initial_capital"]
        pnl_pct = ((mtm / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0

        info = {
            "current_step": self.current_step,
            "env_step_count": self.step_count,
            "mtm": mtm,
            "position": self.position,
            "avg_cost": self.avg_cost if self.position > 1e-9 else 0.0,
            "cash": self.cash,
            "active_orders_count": len(self.active_orders),
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "midprice": self.midprice,
            "spread": self.spread,
            "episode_pnl_abs": mtm - initial_capital,
            "episode_pnl_pct": pnl_pct,
            "last_executed_volume": self.last_executed_volume,

            # Include cumulative trackers in info (optional, but useful)
            # "cumulative_fills_pnl": self.cumulative_realized_pnl_fills,
            # "cumulative_costs": self.cumulative_transaction_costs,
            # "cumulative_risk_penalty": self.cumulative_risk_penalties,
            # "cumulative_order_penalty": self.cumulative_order_penalties,
            # "cumulative_activity_bonus": self.cumulative_activity_bonuses,
            # "final_liquidation_pnl": self.final_liquidation_pnl_net,
        }
        return info

    # Keep render() as is - it doesn't need the cumulative PnL breakdown
    def render(self, mode="human"):
        """Renders the current state of the environment using logging."""
        if mode == "human":
            info = self._get_info() # Get current state info

            # --- Main Status Line ---
            time_pct = (self.current_step / len(self.order_book_history) * 100) if len(self.order_book_history) > 0 else 0
            # Use locale.currency for formatting
            render_msg = (
                f"\n=== Step {info['env_step_count']} | Data Step {info['current_step']} ({time_pct:.1f}%) ===\n"
                f"  Portfolio: Cash={locale.currency(info['cash'], grouping=True):>15s} | Pos={info['position']:>9.4f} @ {locale.currency(info['avg_cost'], grouping=True):<10s} | MTM={locale.currency(info['mtm'], grouping=True):>16s}\n"
                f"  Performance: Ep. PnL={locale.currency(info['episode_pnl_abs'], grouping=True):>+13s} ({info['episode_pnl_pct']:>+6.2f}%) | Last Exec Vol={info['last_executed_volume']:.4f}\n"
                f"  Market:    Bid={locale.currency(info['best_bid'], symbol=False):<8s} | Ask={locale.currency(info['best_ask'], symbol=False):<8s} | Mid={locale.currency(info['midprice'], symbol=False):<8s} | Spread={locale.currency(info['spread'], symbol=False):<6s}\n"
            )

            # --- Active Orders Details ---
            active_order_lines = ["  Active Orders:"]
            if info['active_orders_count'] == 0:
                 active_order_lines.append("    (None)")
            else:
                 max_orders_to_render = 5
                 for i, order in enumerate(self.active_orders):
                     if i >= max_orders_to_render:
                         active_order_lines.append("    ...")
                         break
                     side = "BUY" if order['is_buy'] else "SELL"
                     active_order_lines.append(f"    - ID {order['id']:<4}: {side:<4} {order['volume']:>8.4f} @ {locale.currency(order['price'], symbol=False)}")

            # Combine messages and log with color
            full_render_msg = render_msg + "\n".join(active_order_lines)
            logging.warning("\033[96m%s\033[0m", full_render_msg) # Cyan color

        elif mode == "ansi":
             info = self._get_info()
             return (
                 f"Step: {info['env_step_count']}, DataStep: {info['current_step']}, MTM: {info['mtm']:.2f}, "
                 f"Pos: {info['position']:.4f}, Cash: {info['cash']:.2f}, Orders: {info['active_orders_count']}, "
                 f"Mid: {info['midprice']:.2f}"
             )
        else:
            return super(HFTEnv, self).render(mode=mode)

    def close(self):
        """Clean up any resources associated with the environment."""
        logging.info("HFT Environment closed.")


# ============================
# Example Configuration & Usage
# ============================

# Example Configuration (No Short Selling Focused)
config_example = {
    "csv_path": "/home/gaen/Documents/RL/orderbook_trimmed_small.csv", # <--- ADJUST PATH
    "initial_capital": 20000.0,
    "max_steps": 1000,
    "order_book_levels": 5,
    "price_offset_ticks": 10,
    "max_order_volume": 1.0,
    "latency_steps": 1,
    "tick_size": 0.01,
    "lot_size": 0.001,
    "max_active_orders": 10,
    "inventory_penalty": 0.02,
    "transaction_cost": 0.00075,
    "max_inventory": 5.0,
    "invalid_order_penalty": 0.5,
    "activity_bonus": 0.001,
    "obs_norm_decay": 0.999,
    "obs_clip_limit": 5.0
}

# --- Helper Function for Verbose PnL Print ---
def print_verbose_pnl_summary(env_instance, initial_capital, final_info):
    """Prints a detailed breakdown of PnL components for the completed episode."""
    final_mtm = final_info.get('final_mtm', final_info.get('mtm')) # Get final MTM
    total_pnl_mtm = final_mtm - initial_capital

    # Access cumulative values directly from the env instance
    fills_pnl = env_instance.cumulative_realized_pnl_fills
    costs = env_instance.cumulative_transaction_costs # Includes fills and liquidation costs
    liq_pnl = env_instance.final_liquidation_pnl_net # Net PnL from liquidation
    risk_pen = env_instance.cumulative_risk_penalties
    order_pen = env_instance.cumulative_order_penalties
    activity_bon = env_instance.cumulative_activity_bonuses

    # Calculate total PnL by summing components
    # Note: Liquidation PnL (liq_pnl) is already NET of its costs.
    #       `costs` includes costs from both fills and liquidation.
    #       `fills_pnl` is GROSS PnL from fills.
    # So, Total = Gross Fills - Total Costs + Net Liq PnL - Risk Pen + Order Pen + Activity Bonus
    # Alternatively, use MTM change as the ground truth. Let's calculate the sum for verification.
    total_pnl_calculated = fills_pnl - costs + liq_pnl - risk_pen + order_pen + activity_bon

    # --- Formatting ---
    header = "\n--- EPISODE PNL SUMMARY ---"
    mtm_line = f"Final Mark-to-Market:      {locale.currency(final_mtm, grouping=True):>15s}"
    init_cap_line = f"Initial Capital:           {locale.currency(initial_capital, grouping=True):>15s}"
    total_pnl_line = f"Total PnL (MTM based):     {locale.currency(total_pnl_mtm, grouping=True):>15s} ({total_pnl_mtm/initial_capital*100:.2f}%)"
    separator = "-" * (len(header) + 20) # Adjust width as needed
    components_header = "\nComponent Breakdown:"
    fills_line =   f"  Gross PnL (Fills):       {locale.currency(fills_pnl, grouping=True):>15s}"
    liq_line =     f"  Net PnL (Liquidation):   {locale.currency(liq_pnl, grouping=True):>15s}"
    costs_line =   f"  Transaction Costs (-):   {locale.currency(costs, grouping=True):>15s}"
    risk_line =    f"  Risk Penalties (-):      {locale.currency(risk_pen, grouping=True):>15s}"
    order_line =   f"  Order Penalties (+/-):   {locale.currency(order_pen, grouping=True):>15s}" # Can be negative
    bonus_line =   f"  Activity Bonuses (+):    {locale.currency(activity_bon, grouping=True):>15s}"
    calc_pnl_line =f"Calculated Total PnL:      {locale.currency(total_pnl_calculated, grouping=True):>15s}"
    check_line =   f"Check (Calculated vs MTM): {'OK' if abs(total_pnl_calculated - total_pnl_mtm) < 1e-5 else 'MISMATCH!'}"


    # --- Print the Summary ---
    print(header)
    print(mtm_line)
    print(init_cap_line)
    print(total_pnl_line)
    print(separator)
    print(components_header)
    print(fills_line)
    print(liq_line)
    print(costs_line)
    print(risk_line)
    print(order_line)
    print(bonus_line)
    print(separator)
    print(calc_pnl_line)
    print(check_line)
    print("-" * (len(header) + 20))


# 5. Modify Example Usage Script
if __name__ == "__main__":
    try:
        print("\n--- Creating HFT Environment ---")
        env = HFTEnv(config_example)
        print("Environment created.")

        print("\n--- Testing Environment Reset ---")
        observation, info = env.reset()
        print("Reset successful.")
        print(f"Initial Observation shape: {observation.shape}")
        print("Initial Info:", info)
        env.render()

        print("\n--- Running Episode with Random Actions ---")
        terminated = False
        truncated = False
        episode_reward_sum = 0 # Use a different name to avoid confusion with step reward
        step_count = 0
        last_info = info # Store initial info

        while not terminated and not truncated:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            episode_reward_sum += reward
            step_count += 1
            last_info = info # Keep track of the latest info dict

            if step_count % 250 == 0: # Render less frequently
                 print(f"\n--- Mid-Episode Snapshot (Step {step_count}) ---")
                 env.render()
                 print(f"  Step Reward: {reward:.4f} | Cumulative Reward: {episode_reward_sum:.4f}")


        print("\n--- Episode Finished ---")
        print(f"Total Steps Taken: {step_count}")
        print(f"Final Cumulative Step Rewards: {episode_reward_sum:.4f}") # Sum of step rewards
        if "termination_reason" in last_info:
            print(f"Termination Reason: {last_info['termination_reason']}")

        # --- Call the Verbose PnL Summary ---
        print_verbose_pnl_summary(env, config_example["initial_capital"], last_info)

        # Final state details from info (optional, already in verbose summary)
        # print(f"Final MTM: {last_info.get('final_mtm', last_info.get('mtm')):.2f}")
        # print(f"Final Position: {last_info.get('final_position', last_info.get('position')):.4f}")
        # print(f"Final Cash: {last_info.get('final_cash', last_info.get('cash')):.2f}")

        env.close()
        print("\nEnvironment closed.")

    except FileNotFoundError:
         print(f"\n\033[91mERROR: CSV file not found at '{config_example['csv_path']}'. Please update the path in `config_example`.\033[0m")
    except ValueError as e:
         print(f"\n\033[91mERROR during environment setup or execution: {e}\033[0m")
         import traceback
         traceback.print_exc()
    except Exception as e:
         print(f"\n\033[91mAn unexpected error occurred: {e}\033[0m")
         import traceback
         traceback.print_exc()