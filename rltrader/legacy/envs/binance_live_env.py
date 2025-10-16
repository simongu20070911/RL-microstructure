# binance_live_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import pandas as pd
import logging
import time
import threading
import math

# Use the official WebSocket client
from binance.websocket.spot.websocket_client import SpotWebsocketClient as WebsocketClient 
from binance.error import ClientError, ServerError # For handling API errors from wrapper

# Import the updated wrapper (ensure binance_api_wrapper.py is in the same directory)
try:
    from binance_api_wrapper import BinanceAPIWrapper, API_KEY, API_SECRET 
except ImportError:
    print("Error: Could not import BinanceAPIWrapper.")
    print("Ensure 'binance_api_wrapper.py' is in the same directory as 'binance_live_env.py'.")
    exit(1)

# Configure logging
logger = logging.getLogger("BinanceLiveEnv")
# Prevent adding handlers multiple times if reloaded in interactive sessions
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Set to DEBUG for more verbose logs

# Default configuration 
DEFAULT_CONFIG = {
    # --- API & Connection ---
    "trading_pair": "ETHFDUSD", # Default trading pair
    "api_key": None, # Wrapper handles loading from .env or uses passed value/fallback
    "api_secret": None, # Wrapper handles loading from .env or uses passed value/fallback
    "base_url": "https://testnet.binance.vision", # Binance Testnet URL
    "websocket_base_url": "wss://testnet.binance.vision/ws", # Official WS URL for Spot Testnet
    
    # --- Environment Settings ---
    "initial_portfolio_value_guess": 20000.0, # Used for normalization/scaling, actual value fetched
    "max_steps": 10000, # Maximum steps per episode
    "order_book_levels": 10, # Corresponds to @depth10 stream (must be 5, 10, or 20 for standard streams)
    "max_active_orders": 15, # Agent's limit on concurrent open orders
    "max_inventory_ratio": 0.5, # Max inventory allowed, as fraction of initial portfolio value / entry price
    
    # --- Action Scaling ---
    "price_offset_ticks": 40, # Max offset from midprice in number of ticks
    "max_order_volume_ratio": 0.05, # Max order volume relative to available free balance
    
    # --- RL Parameters ---
    "inventory_penalty": 0.005, # Quadratic penalty factor for holding inventory
    "transaction_cost": 0.001, # Assumed taker fee (0.1% on testnet/often real)
    "invalid_action_penalty": 0.5, # Penalty for actions causing API errors or validation failures
    "hold_penalty": 0.001, # Small penalty per step if no action is taken
    "reward_scaling": 1.0, # Optional factor to scale the final reward signal
    
    # --- Observation ---
    "normalize_obs": True, # Whether to normalize observations using running stats
    "obs_norm_decay": 0.999, # Decay factor for running normalization stats (Welford algorithm doesn't use this)

    # --- Timing & Sync ---
    "websocket_update_ms": 100, # Corresponds to @100ms stream (must be 100 or 1000 for standard @depth)
    "step_time_allowance_factor": 1.5, # Multiplier for WS interval to define step timeout
    "api_call_cooldown": 0.1, # Minimum seconds between consecutive API calls
}

class BinanceLiveEnv(gym.Env):
    """
    A Gymnasium environment for live trading on Binance Spot (Testnet)
    using the official binance-connector library.

    Action Space:
        Box(low=[-1, -1, 0], high=[1, 1, 1], shape=(3,), dtype=float32)
        - action[0]: Signed volume fraction (-1: max sell, +1: max buy)
        - action[1]: Price offset fraction (-1: max offset below mid, +1: max offset above mid)
        - action[2]: Cancel fraction (0: none, 1: cancel all oldest)

    Observation Space:
        Box(low=-5.0, high=5.0, shape=(obs_size,), dtype=float32) # Clamped normalized space
        Features include: normalized balances, normalized order count,
        normalized order book levels (price offsets & quantities), normalized spread.
    """
    metadata = {"render_modes": ["human", "log"], "render_fps": 10} # Approx based on 100ms data

    def __init__(self, config=None):
        super(BinanceLiveEnv, self).__init__()
        logger.info("Initializing BinanceLiveEnv (using binance-connector)...")

        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        self.trading_pair = self.config["trading_pair"]
        self.websocket_update_interval_sec = self.config["websocket_update_ms"] / 1000.0
        self.step_timeout = self.websocket_update_interval_sec * self.config["step_time_allowance_factor"]
        self.max_steps = self.config["max_steps"]
        self.last_api_call_time = 0
        
        # --- Initialize API Wrapper ---
        try:
            self.api = BinanceAPIWrapper(
                api_key=self.config.get("api_key"), # Let wrapper handle None/env vars
                api_secret=self.config.get("api_secret"), 
                base_url=self.config["base_url"],
                trading_pair=self.trading_pair
            )
            # Store fetched rules
            self.tick_size = self.api.tick_size
            self.lot_size = self.api.lot_size
            self.min_qty = self.api.min_qty
            self.max_qty = self.api.max_qty
            self.min_notional = self.api.min_notional
            self.base_asset = self.api.base_asset
            self.quote_asset = self.api.quote_asset
            self.price_precision = self.api.price_precision
            self.qty_precision = self.api.qty_precision
            logger.info(f"API Wrapper initialized for {self.trading_pair} ({self.base_asset}/{self.quote_asset})")
            
            # Validate config *after* fetching API details
            self._validate_config() 
            
        except Exception as e:
            logger.error(f"Fatal: Failed to initialize Binance API Wrapper: {e}", exc_info=True)
            raise 
            
        # --- Define Action Space ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # --- Define Observation Space ---
        obs_components = [
            2, # free quote balance, free base balance (normalized)
            1, # num active orders / max_active_orders
            4 * self.config["order_book_levels"], # bid/ask price offsets & quantities (normalized)
            1  # spread / midprice (normalized)
        ]
        obs_size = sum(obs_components)
        # Using a fixed range like [-5, 5] for normalized & clipped observations
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_size,), dtype=np.float32 
        ) 
        logger.info(f"Observation space size: {obs_size}")
        
        # --- State Variables ---
        self.current_step = 0
        self.total_steps_elapsed = 0
        self.episode_start_portfolio_value = 0.0 # Set in reset
        self.last_portfolio_value = 0.0 # Set in reset
        
        # Market Data (updated by WebSocket)
        self.current_order_book = {"bids": [], "asks": [], "last_update_id": 0}
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.midprice = 0.0
        self.spread = 0.0
        self._order_book_lock = threading.Lock()
        self._new_data_event = threading.Event() # Used to sync step with WS updates
        self._last_ws_update_time = 0

        # Account State (updated by API calls)
        self.base_balance_free = 0.0 
        self.quote_balance_free = 0.0
        self.base_balance_locked = 0.0
        self.quote_balance_locked = 0.0
        self.active_orders = [] # List of dicts from API: {'orderId', 'price', 'origQty', 'side'}
        self._account_lock = threading.Lock() 

        # Max inventory (refined in reset)
        self.max_inventory = (self.config["initial_portfolio_value_guess"] * self.config["max_inventory_ratio"])
        
        # Running statistics for normalization (Welford's algorithm)
        self.running_stats = {'mean': None, 'std': None, 'count': 0}
        # self.decay = self.config["obs_norm_decay"] # Not used by Welford

        # --- WebSocket Client ---
        self.ws_client = None
        self._ws_running = threading.Event() # Tracks if WS *should* be running
        self._start_websocket()

        # Wait for the first order book update
        logger.info("Waiting for initial WebSocket order book data...")
        if not self._new_data_event.wait(timeout=30): 
             logger.error("Fatal: Timed out waiting for initial WebSocket data.")
             self.close() 
             raise RuntimeError("Failed to receive initial WebSocket data.")
        logger.info("Initial WebSocket data received.")

        logger.info("BinanceLiveEnv initialized successfully.")
        # Initial reset is typically called externally after __init__

    def _validate_config(self):
        """Validate config keys, some depend on API info fetched during init."""
        required = ["trading_pair", "max_steps", "order_book_levels",
                    "price_offset_ticks", "max_order_volume_ratio", "max_active_orders",
                    "inventory_penalty", "transaction_cost", "max_inventory_ratio"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key in BinanceLiveEnv: {key}")
        
        valid_depth_levels = [5, 10, 20]
        if self.config["order_book_levels"] not in valid_depth_levels:
             logger.warning(f"order_book_levels ({self.config['order_book_levels']}) not in standard {valid_depth_levels}. Ensure stream name is correct or use a standard level.")
             # Consider raising error or forcing a valid level depending on strictness needed

        if self.config["websocket_update_ms"] not in [100, 1000]:
            logger.warning(f"websocket_update_ms ({self.config['websocket_update_ms']}) should typically be 100 or 1000 for standard @depth streams.")
        
        if self.config["max_active_orders"] > 50:
             logger.warning("max_active_orders > 50 might impact performance or hit API limits eventually.")
             
        logger.info("Configuration validated.")

    # --- WebSocket Handling ---
    def _ws_message_handler(self, message):
        """Callback function to process incoming WebSocket depth messages."""
        try:
            # Identify message type (handles single or multiplexed streams from binance-connector)
            stream_name = ""
            data = None
            event_type = None

            if isinstance(message, dict) and 'stream' in message and 'data' in message: # Multiplexed
                 stream_name = message['stream']
                 data = message['data']
                 event_type = data.get('e')
            elif isinstance(message, dict) and 'e' in message: # Single stream
                 data = message
                 event_type = data.get('e')
                 # Attempt to reconstruct stream name prefix for checking
                 symbol = data.get('s', '').lower()
                 if symbol: stream_name = f"{symbol}@depth..."
            else:
                 # logger.debug(f"Ignored WS message (unexpected format): {message}")
                 return # Ignore non-standard messages
                 
            # Check if it's the depth stream we care about
            expected_stream_part = f"{self.trading_pair.lower()}@depth"
            if expected_stream_part not in stream_name:
                 # logger.debug(f"Ignored WS message from unexpected stream: {stream_name}")
                 return

            if event_type == 'depthUpdate':
                with self._order_book_lock:
                    # TODO: Implement optional sequence number checking (data['U'] and data['u']) 
                    #       against self.current_order_book['last_update_id'] to detect missed messages.
                    #       If gap detected, might need to fetch snapshot via REST API.

                    # Update bids and asks
                    self.current_order_book['bids'] = [[float(p), float(q)] for p, q in data['b']]
                    self.current_order_book['asks'] = [[float(p), float(q)] for p, q in data['a']]
                    self.current_order_book['last_update_id'] = data['u']
                    
                    # Update derived market state
                    if self.current_order_book['bids'] and self.current_order_book['asks']:
                        self.best_bid = self.current_order_book['bids'][0][0]
                        self.best_ask = self.current_order_book['asks'][0][0]
                        # Prevent division by zero if bid/ask are somehow identical (though unlikely)
                        if self.best_ask > self.best_bid:
                             self.midprice = (self.best_bid + self.best_ask) / 2.0
                             self.spread = self.best_ask - self.best_bid
                        else:
                             logger.warning(f"Best bid {self.best_bid} >= Best ask {self.best_ask}. Setting midprice=ask, spread=0.")
                             self.midprice = self.best_ask 
                             self.spread = 0.0
                    else:
                        logger.warning("Order book update received with empty bids or asks. Resetting market state.")
                        self.best_bid = 0.0; self.best_ask = 0.0; self.midprice = 0.0; self.spread = 0.0
                
                # Signal that new data is ready for the step function
                self._last_ws_update_time = time.time()
                self._new_data_event.set() 

            # Can handle other event types like 'snapshot' if needed

        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}\nMessage: {message}", exc_info=True)

    def _get_websocket_stream_name(self):
        """Constructs the websocket stream name based on config."""
        stream_level = self.config['order_book_levels']
        # Use standard levels required by binance-connector for @depth stream
        if stream_level not in [5, 10, 20]:
             logger.warning(f"Configured order_book_levels={stream_level} is not standard [5, 10, 20]. Using 10 for stream name.")
             stream_level = 10 
        ms = self.config['websocket_update_ms']
        if ms not in [100, 1000]:
             logger.warning(f"Configured websocket_update_ms={ms} is not standard [100, 1000]. Using 100ms for stream name.")
             ms = 100
        return f"{self.trading_pair.lower()}@depth{stream_level}@{ms}ms"

    def _start_websocket(self):
        if self.ws_client: 
            logger.warning("WebSocket client object exists. Ensuring previous is stopped.")
            self._stop_websocket() 

        logger.info("Starting WebSocket client (using binance-connector)...")
        # The client handles its own background threads for message processing
        self.ws_client = WebsocketClient(on_message=self._ws_message_handler, 
                                         url=self.config["websocket_base_url"]) 
        self.ws_client.start() # Starts the processing loop in background thread(s)
        self._ws_running.set() 
        
        stream_name = self._get_websocket_stream_name()
        logger.info(f"Subscribing to WebSocket stream: {stream_name}")
        
        # Subscribe after starting
        try:
             self.ws_client.subscribe(stream=stream_name, id=1) 
        except Exception as e:
             logger.error(f"Error subscribing to WebSocket stream '{stream_name}': {e}", exc_info=True)
             self._stop_websocket() # Clean up if subscription fails
             raise ConnectionError(f"Failed to subscribe to WebSocket stream: {e}")

        self._new_data_event.clear() # Clear event after starting subscription
        logger.info("WebSocket client started and subscribed.")
        
    def _stop_websocket(self):
        if self.ws_client:
            logger.info("Stopping WebSocket client...")
            try:
                stream_name = self._get_websocket_stream_name()
                # Attempt to unsubscribe gracefully, ignore errors if already stopped
                try:
                     self.ws_client.unsubscribe(stream=stream_name, id=1)
                except Exception as unsub_e:
                     logger.warning(f"Issue during WS unsubscribe (may already be stopped): {unsub_e}")

                self.ws_client.stop() # Signals the client to shut down
                logger.info("WebSocket client stop signal sent.")
            except Exception as e:
                logger.error(f"Error stopping WebSocket client: {e}", exc_info=True)
            finally:
                 self.ws_client = None # Clear the client object
                 self._ws_running.clear()
                 logger.info("WebSocket client resources should be released.")
        else:
            logger.debug("WebSocket client not running or already stopped.")

    # --- API Interaction Helpers ---
    def _ensure_api_cooldown(self):
        """Sleeps if necessary to enforce API call cooldown."""
        now = time.time()
        elapsed = now - self.last_api_call_time
        required_wait = self.config["api_call_cooldown"]
        if elapsed < required_wait:
            sleep_time = required_wait - elapsed
            # logger.debug(f"API Cooldown: Sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)
        self.last_api_call_time = time.time() # Update time *after* potential sleep

    def _update_account_state(self, fetch_orders=True):
        """Fetches account balances and optionally open orders via API wrapper."""
        with self._account_lock: # Ensure thread safety for account balance updates
            try:
                # --- Update Balances ---
                self._ensure_api_cooldown()
                account_info = self.api.get_account_info(verbose=False)
                if not account_info:
                    logger.error("Failed to fetch account balance update (API returned None).")
                    return False # Indicate critical failure

                balances = {b['asset']: {'free': float(b['free']), 'locked': float(b['locked'])} 
                            for b in account_info.get('balances', [])}
                self.base_balance_free = balances.get(self.base_asset, {}).get('free', 0.0)
                self.quote_balance_free = balances.get(self.quote_asset, {}).get('free', 0.0)
                self.base_balance_locked = balances.get(self.base_asset, {}).get('locked', 0.0)
                self.quote_balance_locked = balances.get(self.quote_asset, {}).get('locked', 0.0)
                # logger.debug(f"Balances Updated: {self.base_asset} F:{self.base_balance_free:.4f} L:{self.base_balance_locked:.4f}, {self.quote_asset} F:{self.quote_balance_free:.2f} L:{self.quote_balance_locked:.2f}")

                # --- Update Open Orders ---
                if fetch_orders:
                    self._ensure_api_cooldown()
                    open_orders_raw = self.api.get_open_orders(symbol=self.trading_pair, verbose=False)
                    if open_orders_raw is None: # API returns [] if no orders, None on error
                        logger.error("Failed to fetch open orders update (API returned None).")
                        # Don't necessarily fail the whole update if only orders fail? Maybe use previous list?
                        # For now, let's return False if orders fail too, as it's important state.
                        return False 
                    
                    # Store relevant info, converting types as needed
                    self.active_orders = [
                            {'orderId': int(o['orderId']), 
                            'price': float(o['price']), 
                            'origQty': float(o['origQty']), 
                            'side': o['side']} 
                            for o in open_orders_raw if o['symbol'] == self.trading_pair # Filter just in case
                    ]
                    # logger.debug(f"Fetched {len(self.active_orders)} open orders.")
                
                return True # Indicate success

            except (ClientError, ServerError) as e: 
                 logger.error(f"API Error updating account state: Code={e.error_code} Msg={e.error_message}", exc_info=False)
                 return False
            except Exception as e:
                logger.error(f"Unexpected error updating account state: {e}", exc_info=True)
                return False

    # --- Gym Env Core Methods ---
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed) # Handles seeding for action space sampling etc.
        logger.info(f"Resetting environment (Total steps elapsed: {self.total_steps_elapsed})")
        
        # Ensure WebSocket is running or restart it
        if not self.ws_client or not self._ws_running.is_set():
             logger.warning("WebSocket not running at reset, attempting restart.")
             self._start_websocket() # This will handle stopping previous if needed
             if not self._new_data_event.wait(timeout=30): # Wait for data after restart
                 logger.error("Fatal: Timed out waiting for WebSocket data after restart.")
                 raise RuntimeError("Failed to receive WebSocket data after restart.")
             logger.info("WebSocket restarted and data received.")

        # Cancel any existing orders from previous episode/run
        logger.info("Cancelling any existing open orders...")
        try:
            self._ensure_api_cooldown()
            cancelled = self.api.cancel_all_open_orders(symbol=self.trading_pair, verbose=False)
            if cancelled is None:
                logger.warning("Could not confirm cancellation of all orders (API error). Proceeding cautiously.")
            else:
                logger.info(f"Cancelled {len(cancelled)} open order(s).")
            self.active_orders = [] # Clear local list immediately
        except (ClientError, ServerError) as e:
             logger.error(f"API Error cancelling orders during reset: {e}", exc_info=False)
             # Decide if fatal? For now, log and continue.
        except Exception as e:
            logger.error(f"Unexpected error cancelling orders during reset: {e}", exc_info=True)

        # Fetch initial account state (balances, ensure orders are clear)
        logger.info("Fetching initial account state...")
        if not self._update_account_state(fetch_orders=True): 
            logger.error("Fatal: Failed to fetch initial account state during reset.")
            raise RuntimeError("Could not fetch initial account state in reset.")
            
        # Wait for a fresh order book update *after* setup actions
        self._new_data_event.clear()
        logger.info("Waiting for fresh order book data post-reset actions...")
        if not self._new_data_event.wait(timeout=self.step_timeout * 2): # Allow slightly longer wait
            logger.error("Fatal: Timed out waiting for fresh order book data during reset.")
            raise RuntimeError("Timed out waiting for order book in reset.")

        # Calculate initial portfolio value and set derived limits
        with self._order_book_lock, self._account_lock:
             if self.midprice > 0:
                  current_portfolio_value = self.quote_balance_free + self.base_balance_free * self.midprice
                  self.episode_start_portfolio_value = current_portfolio_value
                  self.last_portfolio_value = current_portfolio_value
                  # Update max_inventory based on actual starting value and current price
                  self.max_inventory = (self.episode_start_portfolio_value * self.config["max_inventory_ratio"]) / self.midprice
                  logger.info(f"Reset complete. Start Value: ${self.episode_start_portfolio_value:,.2f}, Max Inventory: {self.max_inventory:.{self.qty_precision}f} {self.base_asset}")
             else:
                  logger.error("Fatal: Midprice is zero or invalid during reset, cannot calculate portfolio value or max inventory.")
                  # Attempt to fetch balances/orders again? Or just fail?
                  raise RuntimeError("Midprice is zero or invalid during reset.")

        self.current_step = 0
        
        # Reset normalization statistics for the new episode
        self.running_stats = {'mean': None, 'std': None, 'count': 0}
        
        logger.info("Environment reset successfully.")
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.current_step += 1
        self.total_steps_elapsed += 1
        step_start_time = time.time()
        
        # --- Wait for next market data update ---
        self._new_data_event.clear() # Clear the flag before waiting
        wait_success = self._new_data_event.wait(timeout=self.step_timeout)
        if not wait_success:
            logger.warning(f"Step {self.current_step}: Timed out waiting for WebSocket update ({self.step_timeout}s). Proceeding with potentially stale market data.")
            # If using normalization, stale data could skew stats. Consider consequences.
            
        # --- Process Action within locks for consistency ---
        with self._order_book_lock, self._account_lock:
            # Store state at start of step for reward calculation
            portfolio_value_before_action = self.quote_balance_free + self.base_balance_free * self.midprice
            inventory_before_action = self.base_balance_free # Use free balance for fill estimation later
            
            # --- Parse Action ---
            # Clip action to valid range just in case
            action = np.clip(action, self.action_space.low, self.action_space.high)
            signed_volume_frac, price_offset_frac, cancel_frac = action
            
            invalid_action_penalty_incurred = 0.0
            placed_order_info = None # Store details if placement is successful

            # 1. Handle Cancellations
            num_to_cancel = math.floor(cancel_frac * len(self.active_orders)) 
            if num_to_cancel > 0 and self.active_orders:
                # Cancel oldest orders first
                orders_to_cancel = sorted(self.active_orders, key=lambda o: o['orderId'])[:num_to_cancel]
                logger.info(f"Step {self.current_step}: Attempting to cancel {len(orders_to_cancel)} orders.")
                cancelled_ids = set()
                for order in orders_to_cancel:
                     try:
                         self._ensure_api_cooldown()
                         cancel_result = self.api.cancel_order(orderId=order['orderId'], symbol=self.trading_pair, verbose=False)
                         
                         if cancel_result is None: # Indicates API/wrapper error during call
                              logger.warning(f"Failed to cancel order {order['orderId']} (API error or wrapper failure).")
                              # Don't add to cancelled_ids
                         elif isinstance(cancel_result, dict) and "warning" in cancel_result: # Order likely inactive
                              logger.info(f"Order {order['orderId']} likely inactive (Warn: {cancel_result['warning']}). Treating as removed.")
                              cancelled_ids.add(order['orderId'])
                         else: # Assume successful API response dict
                              logger.info(f"Cancellation request for order {order['orderId']} successful.")
                              cancelled_ids.add(order['orderId'])
                     except (ClientError, ServerError) as e: 
                          logger.error(f"API error cancelling order {order['orderId']}: Code={e.error_code} Msg={e.error_message}", exc_info=False)
                     except Exception as e:
                          logger.error(f"Unexpected error cancelling order {order['orderId']}: {e}", exc_info=True)
                
                # Update local list immediately based on successful cancellations/inactive warnings
                self.active_orders = [o for o in self.active_orders if o['orderId'] not in cancelled_ids]

            # 2. Handle New Order Placement
            # Use a small threshold to avoid tiny, likely invalid orders
            if abs(signed_volume_frac) > 1e-5: 
                if len(self.active_orders) >= self.config["max_active_orders"]:
                    logger.warning(f"Step {self.current_step}: Max active orders ({self.config['max_active_orders']}) reached. Cannot place new order.")
                    invalid_action_penalty_incurred += self.config["invalid_action_penalty"]
                elif self.midprice <= 0: # Need valid midprice for calculation
                    logger.error(f"Step {self.current_step}: Cannot place order, midprice is invalid ({self.midprice:.{self.price_precision}f}).")
                    invalid_action_penalty_incurred += self.config["invalid_action_penalty"]
                else:
                    is_buy = signed_volume_frac > 0
                    volume_frac = abs(signed_volume_frac)
                    side = "BUY" if is_buy else "SELL"
                    
                    # Calculate Price
                    price_offset_abs = price_offset_frac * self.config["price_offset_ticks"] * self.tick_size
                    # Calculate raw target price relative to current midprice
                    limit_price_raw = self.midprice + price_offset_abs 
                    # Let the wrapper handle final rounding/adjustment based on tick size

                    # Calculate Quantity (Target based on available funds and ratio)
                    if is_buy:
                        # Max qty based on available quote balance
                        max_buy_qty_funds = self.quote_balance_free / limit_price_raw if limit_price_raw > 0 else 0
                        # Target qty based on config ratio of available quote
                        target_buy_qty_ratio = (self.quote_balance_free * self.config["max_order_volume_ratio"]) / limit_price_raw if limit_price_raw > 0 else 0
                        # Base the fraction on the smaller of the two potential limits
                        base_qty_target = min(max_buy_qty_funds, target_buy_qty_ratio) * volume_frac
                    else: # Selling
                        # Max qty is simply available free base balance
                        max_sell_qty_funds = self.base_balance_free
                        # Target qty based on config ratio of available base
                        target_sell_qty_ratio = self.base_balance_free * self.config["max_order_volume_ratio"]
                        # Base the fraction on the smaller of the two
                        base_qty_target = min(max_sell_qty_funds, target_sell_qty_ratio) * volume_frac

                    # Attempt to place the order - Wrapper handles formatting and validation (min/max qty, notional)
                    if base_qty_target > 0: # Only proceed if calculated target > 0
                        try:
                            self._ensure_api_cooldown()
                            order_response = self.api.place_limit_order(
                                side=side,
                                quantity=base_qty_target,   # Pass the calculated target quantity
                                price=limit_price_raw,     # Pass the calculated raw price
                                symbol=self.trading_pair,
                                verbose=False # Keep logs cleaner during steps
                            )
                            
                            if order_response and 'orderId' in order_response:
                                placed_order_info = order_response # Store response
                                # Use details from response for logging and local state
                                placed_qty = float(placed_order_info.get('origQty', 0)) # Qty after API rounding
                                placed_price = float(placed_order_info.get('price', 0)) # Price after API rounding
                                logger.info(f"Step {self.current_step}: Placed {side} order {placed_order_info['orderId']} for {placed_qty:.{self.qty_precision}f} @ {placed_price:.{self.price_precision}f}")
                                
                                # Add to local active orders immediately for faster state update perception
                                self.active_orders.append({
                                    'orderId': int(placed_order_info['orderId']), 
                                    'price': placed_price, 
                                    'origQty': placed_qty, 
                                    'side': side
                                })
                                # Optimistically update locked balances (will be corrected by API poll)
                                if is_buy:
                                    # Estimate cost based on placed price/qty
                                    estimated_cost = placed_qty * placed_price
                                    self.quote_balance_locked += estimated_cost
                                    # Reduce free balance cautiously, maybe clamp at 0
                                    self.quote_balance_free = max(0, self.quote_balance_free - estimated_cost) 
                                else: # Selling
                                    self.base_balance_locked += placed_qty
                                    self.base_balance_free = max(0, self.base_balance_free - placed_qty)
                            else:
                                # Placement failed (returned None from wrapper due to validation or API error)
                                logger.error(f"Step {self.current_step}: Failed to place {side} order request (Qty: {base_qty_target:.8f}, Price: {limit_price_raw:.8f}). Check wrapper logs/API error.")
                                invalid_action_penalty_incurred += self.config["invalid_action_penalty"]
                                
                        except (ClientError, ServerError) as e: 
                             logger.error(f"Step {self.current_step}: API error during order placement: {e}", exc_info=False)
                             invalid_action_penalty_incurred += self.config["invalid_action_penalty"]
                        except Exception as e:
                             logger.error(f"Step {self.current_step}: Unexpected error placing order: {e}", exc_info=True)
                             invalid_action_penalty_incurred += self.config["invalid_action_penalty"]
                    else:
                         logger.debug(f"Step {self.current_step}: Calculated order quantity {base_qty_target:.8f} was zero or negative. No order placed.")

            # --- Update Account State (Post-Action) ---
            # Crucial for accurate reward calculation based on actual fills/balances
            # logger.debug(f"Step {self.current_step}: Updating account state post-action...")
            update_success = self._update_account_state(fetch_orders=True) 
            if not update_success:
                 logger.error(f"Step {self.current_step}: CRITICAL - Failed to update account state after actions. State/Reward might be inaccurate.")
                 # Consider termination or alternative handling if state updates fail

            # --- Calculate Reward ---
            # Use account state *after* the API update
            portfolio_value_after_action = self.quote_balance_free + self.base_balance_free * self.midprice 
            # Ensure midprice used here is the one locked at the start of the 'with' block if consistency needed,
            # or use potentially updated midprice if available (depends on desired reward signal timing)
            
            # 1. MTM PnL: Change in portfolio value
            mtm_pnl = portfolio_value_after_action - portfolio_value_before_action
            
            # 2. Transaction Costs Estimation: Based on change in FREE base balance
            inventory_after_action = self.base_balance_free
            inventory_change = inventory_after_action - inventory_before_action # Change in free inventory reflects fills
            
            # Estimate fill price using midprice (less accurate without user data stream)
            estimated_filled_value = abs(inventory_change) * self.midprice if self.midprice > 0 else 0
            transaction_costs = estimated_filled_value * self.config["transaction_cost"] if abs(inventory_change) > 1e-9 else 0.0
            
            # 3. Inventory Penalty: Based on TOTAL inventory (free + locked) after actions
            current_inventory_total = self.base_balance_free + self.base_balance_locked 
            inventory_penalty = self._calculate_risk_penalty(current_inventory_total)
            
            # 4. Hold Penalty: If no significant action was taken
            hold_penalty = 0.0
            if abs(signed_volume_frac) < 1e-5 and cancel_frac < 0.1: # Thresholds for 'no action'
                 hold_penalty = self.config["hold_penalty"]
                 
            # 5. Combine Reward Components
            reward = (mtm_pnl - transaction_costs - inventory_penalty - invalid_action_penalty_incurred - hold_penalty)
            # Apply reward scaling if needed
            reward *= self.config["reward_scaling"]
            
            # Update last portfolio value for next step's MTM calculation
            self.last_portfolio_value = portfolio_value_after_action

        # --- Check Termination & Truncation ---
        # Check outside the lock to allow other threads (like WS) to proceed
        terminated = False
        # Terminate if portfolio value drops below a threshold (e.g., 50% loss)
        if portfolio_value_after_action < self.episode_start_portfolio_value * 0.5:
            logger.warning(f"Step {self.current_step}: Episode terminated due to significant portfolio value loss (below 50% of start).")
            terminated = True
            
        truncated = self.current_step >= self.max_steps
        if truncated:
            logger.info(f"Step {self.current_step}: Episode truncated after reaching max_steps ({self.max_steps}).")
        
        # Liquidate positions if episode ended
        if terminated or truncated:
            logger.info(f"Episode ended. Terminated={terminated}, Truncated={truncated}. Liquidating remaining positions...")
            liquidation_pnl, liquidation_costs = self._liquidate_positions()
            logger.info(f"Liquidation PnL (estimated): {liquidation_pnl:.{self.price_precision}f}, Costs (estimated): {liquidation_costs:.{self.price_precision}f}")
            # Add liquidation result to the final step's reward (use estimates)
            # Adjust reward: add PnL, subtract costs. Also subtract final inventory penalty if applicable.
            final_inventory_penalty = self._calculate_risk_penalty(current_inventory_total) # Penalty *before* liquidation
            reward += (liquidation_pnl - liquidation_costs - final_inventory_penalty) * self.config["reward_scaling"] 

        # --- Prepare Observation & Info ---
        observation = self._get_observation() # Get potentially normalized observation
        info = self._get_info() # Get current state dictionary
        # Add step-specific reward components to info for debugging/logging
        info["step_reward"] = reward 
        info["mtm_pnl"] = mtm_pnl
        info["transaction_costs_est"] = transaction_costs
        info["inventory_penalty"] = inventory_penalty
        info["invalid_action_penalty"] = invalid_action_penalty_incurred
        info["hold_penalty"] = hold_penalty
        info["inventory_change"] = inventory_change

        # Log step duration and potential delays
        step_duration = time.time() - step_start_time
        # logger.debug(f"Step {self.current_step} completed in {step_duration:.4f}s. Reward: {reward:.4f}")
        if step_duration > self.websocket_update_interval_sec * 1.1: # If step takes >10% longer than WS interval
             logger.warning(f"Step {self.current_step} took {step_duration:.3f}s, potentially lagging behind WebSocket interval ({self.websocket_update_interval_sec:.3f}s).")

        return observation, reward, terminated, truncated, info

    # --- Helper Methods ---
    def _calculate_risk_penalty(self, inventory):
        """Quadratic risk penalty based on total inventory value relative to max allowed."""
        if self.max_inventory <= 1e-9 or self.midprice <= 0: 
             return 0.0 # Avoid division by zero or meaningless penalty if market/config is invalid
        
        # Normalize inventory relative to the max allowed inventory quantity
        normalized_inventory = inventory / self.max_inventory 
        
        # Penalty scales quadratically with normalized inventory size
        # Also scale by the current value of the inventory being held
        inventory_value = abs(inventory) * self.midprice
        penalty = (self.config["inventory_penalty"] * (normalized_inventory ** 2) * inventory_value)
        return penalty
        
    def _get_raw_observation(self):
        """Constructs the raw observation vector before normalization."""
        with self._order_book_lock, self._account_lock:
            # Use values locked at the beginning of the construction
            mid_p = self.midprice
            if mid_p <= 0: # Handle invalid midprice
                logger.warning("Midprice is zero or negative in _get_raw_observation. Using 1.0 as fallback for normalization.")
                mid_p = 1.0 # Avoid division by zero

            # Normalize balances relative to initial portfolio value (provides consistent scale)
            start_value = self.episode_start_portfolio_value
            if start_value <= 0: start_value = self.config["initial_portfolio_value_guess"] # Use guess if start value is invalid
            if start_value <= 0: start_value = 1.0 # Final fallback to avoid division by zero

            norm_quote_free = self.quote_balance_free / start_value
            # Normalize base balance by its approximate value in quote currency
            norm_base_free = (self.base_balance_free * mid_p) / start_value 
            
            # Normalize order count
            norm_num_orders = len(self.active_orders) / self.config["max_active_orders"] if self.config["max_active_orders"] > 0 else 0

            obs_list = [norm_quote_free, norm_base_free, norm_num_orders]

            # Order Book Features: Price offsets relative to midprice, Quantities relative to max_inventory
            max_levels = self.config["order_book_levels"]
            max_inv = self.max_inventory if self.max_inventory > 1e-9 else 1.0 # Avoid division by zero

            for i in range(max_levels):
                # Bids
                if i < len(self.current_order_book['bids']):
                    bid_p, bid_q = self.current_order_book['bids'][i]
                    price_offset = (bid_p - mid_p) / mid_p # Relative offset
                    norm_qty = bid_q / max_inv
                    obs_list.extend([price_offset, norm_qty])
                else:
                    obs_list.extend([0.0, 0.0]) # Pad with zeros if level doesn't exist

                # Asks
                if i < len(self.current_order_book['asks']):
                    ask_p, ask_q = self.current_order_book['asks'][i]
                    price_offset = (ask_p - mid_p) / mid_p # Relative offset
                    norm_qty = ask_q / max_inv
                    obs_list.extend([price_offset, norm_qty])
                else:
                    obs_list.extend([0.0, 0.0]) # Pad

            # Spread feature: Spread relative to midprice
            norm_spread = self.spread / mid_p if mid_p > 0 else 0
            obs_list.append(norm_spread)

            return np.array(obs_list, dtype=np.float32)

    def _update_running_stats(self, obs):
        """Update running mean and variance using Welford's online algorithm."""
        # Initialize on first call
        if self.running_stats['mean'] is None:
             self.running_stats['mean'] = np.zeros_like(obs, dtype=np.float64)
             self.running_stats['std'] = np.ones_like(obs, dtype=np.float64) # Initialize std dev to 1
             self.running_stats['M2'] = np.zeros_like(obs, dtype=np.float64) # Sum of squares of differences
             self.running_stats['count'] = 0

        # Welford's algorithm update steps
        self.running_stats['count'] += 1
        count = self.running_stats['count']
        delta = obs - self.running_stats['mean']
        self.running_stats['mean'] += delta / count
        delta2 = obs - self.running_stats['mean'] # New delta using the updated mean
        self.running_stats['M2'] += delta * delta2
        
        # Update standard deviation (using sample variance)
        if count < 2:
             self.running_stats['std'] = np.ones_like(obs, dtype=np.float64) # Keep as 1 until count >= 2
        else:
             variance = self.running_stats['M2'] / (count - 1) 
             # Ensure variance isn't negative due to floating point errors
             variance = np.maximum(variance, 0) 
             self.running_stats['std'] = np.sqrt(variance)

    def _normalize_observation(self, obs):
        """Normalize observation using running mean and standard deviation."""
        # Skip normalization if disabled, or if stats haven't been updated enough (count < 2)
        if not self.config["normalize_obs"] or self.running_stats.get('count', 0) < 2:
            return obs
            
        mean = self.running_stats['mean']
        std = self.running_stats['std']
        
        # Prevent division by zero or very small std dev (epsilon)
        std_safe = np.maximum(std, 1e-8) 
        
        normalized_obs = (obs - mean) / std_safe
        
        # Clip observation to the defined space boundaries after normalization
        normalized_obs = np.clip(normalized_obs, self.observation_space.low, self.observation_space.high)
        
        return normalized_obs.astype(np.float32) # Ensure correct dtype

    def _get_observation(self):
        """Get the raw observation, update running stats, and return normalized observation."""
        raw_obs = self._get_raw_observation()
        if self.config["normalize_obs"]:
             self._update_running_stats(raw_obs)
        return self._normalize_observation(raw_obs)

    def _get_info(self):
        """Returns dictionary with useful information about the environment state."""
        # Use locks to ensure consistent reading of state variables
        with self._order_book_lock, self._account_lock:
             current_portfolio_value = self.quote_balance_free + self.base_balance_free * self.midprice if self.midprice > 0 else self.quote_balance_free
             total_base = self.base_balance_free + self.base_balance_locked
             total_quote = self.quote_balance_free + self.quote_balance_locked
             
             info = {
                 "step": self.current_step,
                 "total_steps_elapsed": self.total_steps_elapsed,
                 "timestamp_ms": int(time.time() * 1000),
                 "portfolio_value_usd": current_portfolio_value, # Approx value in quote currency
                 "episode_pnl_usd": current_portfolio_value - self.episode_start_portfolio_value,
                 "episode_pnl_percent": ((current_portfolio_value / self.episode_start_portfolio_value) - 1) * 100 if self.episode_start_portfolio_value > 1e-6 else 0,
                 "base_asset": self.base_asset,
                 "quote_asset": self.quote_asset,
                 "base_balance_free": self.base_balance_free,
                 "base_balance_locked": self.base_balance_locked,
                 "base_balance_total": total_base,
                 "quote_balance_free": self.quote_balance_free,
                 "quote_balance_locked": self.quote_balance_locked,
                 "quote_balance_total": total_quote,
                 "active_orders_count": len(self.active_orders),
                 "best_bid": self.best_bid,
                 "best_ask": self.best_ask,
                 "midprice": self.midprice,
                 "spread": self.spread,
                 "max_inventory": self.max_inventory,
                 # Step-specific reward info added in step() method
             }
        return info

    def render(self, mode="human"):
        """Render the environment state (console output)."""
        info = self._get_info() # Get the latest info dictionary
        
        pnl_sign = "+" if info['episode_pnl_usd'] >= 0 else ""
        # Use price precision for market prices, quantity precision for balances
        price_fmt = f".{self.price_precision}f"
        qty_fmt = f".{self.qty_precision}f"
        
        render_str = (
            f"--- Step: {info['step']:<5} | Total Steps: {info['total_steps_elapsed']:<7} ---\n"
            #f"Time: {pd.Timestamp.now()} | Last WS: {self._last_ws_update_time:.2f}\n"
            f"Market: {self.trading_pair:<10} | Bid: {info['best_bid']:{price_fmt}} | Ask: {info['best_ask']:{price_fmt}} | Mid: {info['midprice']:{price_fmt}} | Spread: {info['spread']:{price_fmt}}\n"
            f"Portfolio Value: ${info['portfolio_value_usd']:<,.2f} | Ep PnL: {pnl_sign}${info['episode_pnl_usd']:<,.2f} ({info['episode_pnl_percent']:.2f}%)\n"
            f"Balances: {self.base_asset}: {info['base_balance_free']:{qty_fmt}} (L: {info['base_balance_locked']:{qty_fmt}}) | {self.quote_asset}: ${info['quote_balance_free']:<,.2f} (L: ${info['quote_balance_locked']:<,.2f})\n"
            f"Inventory: {info['base_balance_total']:{qty_fmt}} / {info['max_inventory']:{qty_fmt}} {self.base_asset}\n"
            f"Active Orders: {info['active_orders_count']}/{self.config['max_active_orders']}\n"
        )
        # Append reward info if available (usually added in step return info)
        if "step_reward" in info: 
             reward_detail = (
                 f"Step Reward: {info['step_reward']:.4f} "
                 f"(MTM: {info.get('mtm_pnl', 0):.3f}, "
                 f"Costs: {-info.get('transaction_costs_est', 0):.3f}, "
                 f"InvPen: {-info.get('inventory_penalty', 0):.3f}, "
                 f"ActPen: {-info.get('invalid_action_penalty', 0):.3f})"
             )
             render_str += reward_detail + "\n"
             
        if mode == "human":
            print(render_str) # Print to console
        elif mode == "log":
            # Log relevant parts, maybe less verbose than console print
            log_summary = (
                 f"Step:{info['step']} | PVal:{info['portfolio_value_usd']:.2f} | PnL:{info['episode_pnl_usd']:.2f} "
                 f"| Inv:{info['base_balance_total']:{qty_fmt}} | Orders:{info['active_orders_count']} "
                 f"| Bid:{info['best_bid']:{price_fmt}} | Ask:{info['best_ask']:{price_fmt}} "
                 f"| Reward:{info.get('step_reward', 'N/A'):.4f}"
            )
            logger.info(log_summary)
        
        # Could potentially return an image/plot array for other modes

    def _liquidate_positions(self):
        """Liquidates all FREE base asset inventory using market orders."""
        logger.info("Attempting to liquidate positions...")
        # Return estimated PnL and Costs from liquidation itself
        liquidation_pnl_est = 0.0
        liquidation_costs_est = 0.0
        
        # Ensure we have the latest balance before deciding quantity
        update_ok = self._update_account_state(fetch_orders=False) # Don't necessarily need orders here
        if not update_ok:
             logger.error("Failed to update account state before liquidation. Aborting liquidation.")
             return 0.0, 0.0 # Return zero impact if state is unreliable
        
        with self._account_lock:
            qty_to_liquidate = self.base_balance_free # Only liquidate what's freely available
            
            # Check if liquidation quantity is significant enough
            if abs(qty_to_liquidate) < self.min_qty:
                 logger.info(f"Inventory to liquidate ({qty_to_liquidate:{qty_fmt}}) is less than min_qty ({self.min_qty}). No liquidation action needed.")
                 return 0.0, 0.0

            side = "SELL" if qty_to_liquidate > 0 else "BUY"
            abs_qty = abs(qty_to_liquidate)
            
            logger.info(f"Liquidating {abs_qty:{qty_fmt}} {self.base_asset} via MARKET {side} order...")
            
            try:
                self._ensure_api_cooldown()
                # Use the wrapper's market order function
                # Pass the absolute quantity, wrapper handles formatting/validation
                response = self.api.place_market_order(side=side, quantity=abs_qty, symbol=self.trading_pair, verbose=False)
                
                if response and response.get('status') == 'FILLED':
                    fills = response.get('fills', [])
                    filled_qty = sum(float(f['qty']) for f in fills)
                    filled_value = sum(float(f['price']) * float(f['qty']) for f in fills)
                    
                    # Estimate commission based on standard fee (more accurate would be User Data Stream)
                    commission_est = filled_value * self.config["transaction_cost"]
                    
                    avg_price = filled_value / filled_qty if filled_qty > 0 else 0
                    
                    # Estimate PnL based on average fill price vs current midprice (very rough)
                    # A better PnL requires tracking entry costs, which this env doesn't do.
                    # We focus on the cash change caused by the liquidation.
                    if side == "SELL":
                         cash_change = filled_value - commission_est
                         liquidation_pnl_est = cash_change # Simplified: Treat cash received as PnL contribution
                    else: # BUY to cover short (assumption)
                         cash_change = - (filled_value + commission_est)
                         liquidation_pnl_est = cash_change # Simplified: Treat cash spent as PnL contribution
                    
                    liquidation_costs_est = commission_est
                    
                    logger.info(
                        f"Liquidation successful: {side} {filled_qty:{qty_fmt}} @ avg price {avg_price:{price_fmt}}. "
                        f"Value: {filled_value:.2f}, Est. Commission: {commission_est:.4f}"
                    )
                    
                    # Update balances locally optimistically (will be corrected later)
                    # Careful with floating point arithmetic here
                    if side == "SELL":
                         self.base_balance_free -= filled_qty
                         self.quote_balance_free += filled_value - commission_est
                    else: # BUY 
                         self.base_balance_free += filled_qty
                         self.quote_balance_free -= (filled_value + commission_est)
                         
                    # Clamp balances at 0 if they go slightly negative due to precision issues
                    self.base_balance_free = max(0, self.base_balance_free)
                    self.quote_balance_free = max(0, self.quote_balance_free)

                else:
                    # Market order might not fill instantly or could fail
                    logger.error(f"Liquidation market order failed or did not fill immediately. Status: {response.get('status') if response else 'Unknown'}. Manual intervention may be needed.")
                    # Return 0 impact if liquidation fails
                    return 0.0, 0.0
                    
            except (ClientError, ServerError) as e:
                 logger.error(f"API Error during liquidation order placement: {e}", exc_info=False)
                 return 0.0, 0.0 # Return 0 impact
            except Exception as e:
                 logger.error(f"Unexpected error during liquidation order: {e}", exc_info=True)
                 return 0.0, 0.0 # Return 0 impact

        # Final cleanup: Cancel any remaining limit orders after liquidation attempt
        try:
            logger.info("Cancelling any remaining open orders after liquidation attempt...")
            self._ensure_api_cooldown()
            cancelled = self.api.cancel_all_open_orders(symbol=self.trading_pair, verbose=False)
            if cancelled: logger.info(f"Cancelled {len(cancelled)} remaining orders.")
            self.active_orders = [] # Clear local list
        except Exception as e:
             logger.error(f"Error cancelling orders post-liquidation: {e}", exc_info=True)
             
        logger.info("Liquidation process finished.")
        return liquidation_pnl_est, liquidation_costs_est


    def close(self):
        """Clean up resources when the environment is no longer needed."""
        logger.info("Closing BinanceLiveEnv.")
        self._stop_websocket() # Ensure WebSocket is stopped and threads released
        
        # API client resources (like HTTP sessions) are typically managed by the underlying library,
        # but explicit cleanup if available could be added to the wrapper if needed.
        # e.g., if self.api had a close method: self.api.close()
        
        logger.info("Environment closed.")


# --- Example Usage Block ---
if __name__ == "__main__":
    # Configure root logger for visibility if running script directly
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("--- Binance Live Env Example (using binance-connector) ---")
    
    # IMPORTANT: Ensure you have a .env file in the same directory with:
    # BINANCE_API_KEY="YOUR_TESTNET_API_KEY"
    # BINANCE_API_SECRET="YOUR_TESTNET_API_SECRET"
    # Or pass keys directly in env_config (less secure)
    
    # Example Configuration (adjust to your testnet setup)
    env_config = {
        "trading_pair": "BTCUSDT", # Use a liquid pair available on testnet
        "initial_portfolio_value_guess": 50000, # Rough estimate of your testnet account value in USDT
        "max_steps": 50, # Keep episode short for testing
        "api_call_cooldown": 0.15, # Be slightly conservative with API calls
        "websocket_update_ms": 100, # Use 100ms stream
        "order_book_levels": 10, # Matches @depth10 stream
        "max_inventory_ratio": 0.1, # Limit inventory risk during testing
        "normalize_obs": True, # Use observation normalization
        "invalid_action_penalty": 0.1, # Lower penalty for simple testing
        "inventory_penalty": 0.001, # Small inventory penalty
    }
    
    env = None # Define env outside try for the finally block
    try:
        # Create the environment instance
        env = BinanceLiveEnv(config=env_config)
        
        logger.info("Starting basic test loop with random actions...")
        # Reset the environment to get initial state
        obs, info = env.reset()
        
        print("\n" + "="*20 + " Initial State " + "="*20)
        env.render(mode="human")
        print(f"Initial Observation (shape {obs.shape}):\n{obs}\n" + "="*55)
        
        terminated = False
        truncated = False
        total_reward = 0.0
        
        # Run the loop for max_steps or until terminated/truncated
        for i in range(env_config["max_steps"]):
            if terminated or truncated:
                logger.info(f"Episode ended early at step {i}. Terminated={terminated}, Truncated={truncated}")
                break
                
            # --- Agent's Action Selection ---
            # Replace this with your actual RL agent's action selection logic
            # For testing, we sample a random action from the action space
            action = env.action_space.sample()
            # -------------------------------
            
            print(f"\n--- Step {i+1}/{env_config['max_steps']} ---")
            print(f"Action: [VolumeFrac: {action[0]:.3f}, PriceOffsetFrac: {action[1]:.3f}, CancelFrac: {action[2]:.3f}]")
            
            # Execute the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Render the current state and results
            print(f"Reward: {reward:.4f} | Total Ep Reward: {total_reward:.4f}")
            env.render(mode="human")
            # Optional: Print observation for debugging
            # print(f"Observation (shape {obs.shape}):\n{obs}") 
            
            # Add a small delay to simulate agent thinking time and make output readable
            time.sleep(0.2) 
            
        logger.info(f"Test loop finished after {i+1} steps.")
        
    except KeyboardInterrupt:
         logger.warning("Keyboard interrupt received. Exiting test loop.")
    except Exception as e:
        # Log any errors that occur during the run
        logger.error(f"An critical error occurred during the example run: {e}", exc_info=True)
        
    finally:
        # --- IMPORTANT: Cleanup ---
        # Ensure the environment resources (like WebSocket) are closed properly
        if env is not None:
            logger.info("Closing environment resources...")
            env.close()
            logger.info("Environment resources released.")
        else:
             logger.info("Environment object was not created successfully.")