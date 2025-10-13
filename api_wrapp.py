# binance_api_wrapper.py

import os
import logging
import time
import math
from binance.spot import Spot as SpotClient # Official client
from binance.error import ClientError, ServerError
from dotenv import load_dotenv

# Configure logging
# Use a specific logger for the wrapper
wrapper_logger = logging.getLogger("BinanceAPIWrapper")
# Avoid adding multiple handlers if reloaded in interactive sessions
if not wrapper_logger.handlers:
    wrapper_handler = logging.StreamHandler()
    wrapper_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    wrapper_handler.setFormatter(wrapper_formatter)
    wrapper_logger.addHandler(wrapper_handler)
    wrapper_logger.setLevel(logging.INFO) # Set to INFO or DEBUG as needed

# --- Load API Keys Securely ---
# Looks for a .env file in the current directory or parent directories
load_dotenv() 
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Fallback values if .env file or keys are missing (NOT RECOMMENDED FOR PRODUCTION)
# Replace these with your actual TESTNET keys ONLY for temporary local testing if needed.
# It's much safer to use the .env file.
FALLBACK_API_KEY = "YOUR_FALLBACK_TESTNET_API_KEY_HERE" # Replace if absolutely necessary
FALLBACK_API_SECRET = "YOUR_FALLBACK_TESTNET_API_SECRET_HERE" # Replace if absolutely necessary

if not API_KEY:
    wrapper_logger.warning("API Key not found in .env file. Using hardcoded fallback (unsafe). Ensure .env exists and is configured.")
    API_KEY = FALLBACK_API_KEY
if not API_SECRET:
    wrapper_logger.warning("API Secret not found in .env file. Using hardcoded fallback (unsafe). Ensure .env exists and is configured.")
    API_SECRET = FALLBACK_API_SECRET

# Default Base URL for Binance Testnet
BASE_URL = "https://testnet.binance.vision"

class BinanceAPIWrapper:
    """
    A wrapper class for Binance Spot API interactions using binance-connector-python,
    providing methods needed for a live trading environment. Includes validation
    based on fetched symbol rules.
    """
    def __init__(self, api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL, trading_pair="ETHFDUSD"):
        """
        Initializes the API wrapper.

        Args:
            api_key (str, optional): Binance API key. Defaults to value from .env or fallback.
            api_secret (str, optional): Binance API secret. Defaults to value from .env or fallback.
            base_url (str, optional): Base URL for the API (e.g., Testnet or Production). Defaults to TESTNET_BASE_URL.
            trading_pair (str, optional): Default trading pair for operations (e.g., 'BTCUSDT'). Defaults to "ETHFDUSD".
        
        Raises:
            ValueError: If API key or secret is missing or invalid.
            ConnectionError: If initial connectivity test fails.
            ClientError/ServerError: If fetching initial symbol info fails due to API issues.
            Exception: For other unexpected initialization errors.
        """
        wrapper_logger.info(f"Initializing BinanceAPIWrapper for {trading_pair} on {base_url}")
        
        if not api_key or not api_secret or api_key == FALLBACK_API_KEY or api_secret == FALLBACK_API_SECRET:
             wrapper_logger.error("CRITICAL: API Key or Secret is missing or using fallback values. Please provide valid credentials via .env file or arguments.")
             # Depending on requirements, you might raise an error immediately or allow proceeding with caution.
             # Raising immediately is safer:
             raise ValueError("API Key and Secret must be provided and should not be fallback values.")

        # Ensure keys are strings, as expected by the library
        api_key = str(api_key)
        api_secret = str(api_secret)

        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.trading_pair = trading_pair.upper() # Ensure trading pair is uppercase

        try:
            # Initialize the official Spot client from binance-connector
            self.client = SpotClient(key=self.api_key, secret=self.api_secret, base_url=self.base_url)
            wrapper_logger.info("SpotClient object created successfully.")
            
            # --- Test Connectivity ---
            if not self.test_connectivity():
                 # Test connectivity failure is critical for a live trading bot
                 raise ConnectionError("Failed initial connectivity test with Binance API. Check network and API endpoint.")

            # --- Fetch and Store Symbol Information ---
            self.symbol_info = self._fetch_symbol_info(self.trading_pair)
            if not self.symbol_info:
                 # If the primary trading pair info can't be fetched, it's a critical setup issue
                 raise ValueError(f"Could not fetch or find symbol info for the primary trading pair '{self.trading_pair}'. Check if the pair exists on the exchange/endpoint.")

            # Extract relevant filters and rules from symbol_info
            self.tick_size = float(self._get_filter_value('PRICE_FILTER', 'tickSize', default=0.0))
            self.lot_size = float(self._get_filter_value('LOT_SIZE', 'stepSize', default=0.0))
            self.min_qty = float(self._get_filter_value('LOT_SIZE', 'minQty', default=0.0))
            self.max_qty = float(self._get_filter_value('LOT_SIZE', 'maxQty', default=float('inf'))) # Default to infinity if not specified

            # Handle variations in the minNotional filter name/type across exchanges/endpoints
            min_notional_val = self._get_filter_value('MIN_NOTIONAL', 'minNotional') # Newer filter name
            if min_notional_val is None:
                min_notional_val = self._get_filter_value('NOTIONAL', 'minNotional') # Common on Testnet
                if min_notional_val is not None:
                    wrapper_logger.warning("Using 'NOTIONAL' filter for minNotional.")
            if min_notional_val is None:
                min_notional_val = self._get_filter_value('MARKET_LOT_SIZE', 'minNotional') # Another possible filter
                if min_notional_val is not None:
                     wrapper_logger.warning("Using 'MARKET_LOT_SIZE' filter for minNotional.")
            
            # Assign a default if still not found, with a warning
            self.min_notional = float(min_notional_val) if min_notional_val is not None else 10.0 # Default guess (e.g., 10 USDT)
            if min_notional_val is None:
                 wrapper_logger.warning(f"minNotional filter not found definitively. Using default guess: {self.min_notional}")

            # Store base and quote assets
            self.base_asset = self.symbol_info.get('baseAsset', '')
            self.quote_asset = self.symbol_info.get('quoteAsset', '')
            if not self.base_asset or not self.quote_asset:
                 wrapper_logger.warning("Could not extract base or quote asset from symbol info.")

            # Calculate precision based on step sizes
            self.price_precision = self._get_precision_from_step(self.tick_size)
            self.qty_precision = self._get_precision_from_step(self.lot_size)

            # Log the extracted rules
            wrapper_logger.info(f"Successfully initialized BinanceAPIWrapper for {self.trading_pair} ({self.base_asset}/{self.quote_asset})")
            wrapper_logger.info(f"Trading Rules | Tick Size: {self.tick_size} (Precision: {self.price_precision})")
            wrapper_logger.info(f"Trading Rules | Lot Size (Step): {self.lot_size} (Precision: {self.qty_precision})")
            wrapper_logger.info(f"Trading Rules | Min Qty: {self.min_qty}, Max Qty: {self.max_qty if self.max_qty != float('inf') else 'Infinity'}")
            wrapper_logger.info(f"Trading Rules | Min Notional: {self.min_notional}")

        except (ClientError, ServerError) as e:
            # Catch specific API errors during initialization (e.g., bad keys, permission issues)
            wrapper_logger.error(f"Binance API Error during initialization: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            raise # Re-raise to indicate critical failure
        except ValueError as ve:
             wrapper_logger.error(f"Value Error during initialization: {ve}", exc_info=True)
             raise
        except ConnectionError as ce:
             wrapper_logger.error(f"Connection Error during initialization: {ce}", exc_info=True)
             raise
        except Exception as e:
             # Catch any other unexpected errors
             wrapper_logger.error(f"An unexpected error occurred during initialization: {e}", exc_info=True)
             raise

    def _get_filter_value(self, filter_type, filter_key, default=None):
        """
        Helper function to safely extract a specific value from the symbol_info filters list.

        Args:
            filter_type (str): The 'filterType' to look for (e.g., 'PRICE_FILTER').
            filter_key (str): The key within the filter dictionary to extract (e.g., 'tickSize').
            default (any, optional): The value to return if the filter or key is not found. Defaults to None.

        Returns:
            any: The extracted value, or the default if not found.
        """
        if not self.symbol_info or 'filters' not in self.symbol_info:
            wrapper_logger.warning(f"Symbol info or filters list missing when searching for {filter_type}/{filter_key}.")
            return default
        
        filters = self.symbol_info.get('filters', [])
        for f in filters:
            if f.get('filterType') == filter_type:
                # Found the correct filter type, try to get the key
                value = f.get(filter_key)
                if value is not None:
                    return value
                else:
                    # Filter type found, but the specific key is missing
                    wrapper_logger.warning(f"Filter type '{filter_type}' found, but key '{filter_key}' is missing.")
                    return default
                    
        # Filter type itself was not found in the list
        # wrapper_logger.debug(f"Filter type '{filter_type}' not found for symbol {self.trading_pair}.")
        return default
        
    def _get_precision_from_step(self, step_size):
        """
        Calculate precision (number of decimal places) required based on a step size.
        Handles edge cases like step_size of 0 or 1.

        Args:
            step_size (float): The step size (e.g., tick_size or lot_size).

        Returns:
            int: The number of decimal places required.
        """
        if step_size <= 0: 
             wrapper_logger.warning(f"Received step_size <= 0 ({step_size}). Cannot determine precision, returning 0.")
             return 0
        if step_size == 1: 
             return 0 # No decimal places needed if step is 1

        # Convert to string with sufficient precision, remove trailing zeros, then count decimals
        try:
            # Use decimal formatting to avoid floating point inaccuracies inherent in str(float)
            step_str = "{:.16f}".format(float(step_size)).rstrip('0') 
            if '.' in step_str:
                return len(step_str.split('.')[-1])
            else:
                return 0 # No decimal part found (e.g., step_size was integer-like like 10.0)
        except Exception as e:
             wrapper_logger.error(f"Error calculating precision for step size {step_size}: {e}", exc_info=True)
             return 8 # Return a reasonable default precision on error

    def test_connectivity(self):
        """
        Tests connectivity to the Binance API using ping and server time checks.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        wrapper_logger.info("Testing API connectivity...")
        try:
            # 1. Ping the server
            self.client.ping()
            wrapper_logger.debug("API ping successful.")
            
            # 2. Get server time (also implicitly checks authentication if keys are needed)
            server_time_response = self.client.time()
            server_time_ms = server_time_response.get('serverTime')
            if server_time_ms:
                 wrapper_logger.info(f"Successfully connected. Server time: {server_time_ms} ({pd.Timestamp(server_time_ms, unit='ms')})")
                 return True
            else:
                 wrapper_logger.error("Connectivity test failed: Server time not found in response.")
                 return False
                 
        except (ClientError, ServerError) as e:
            # Handle specific API errors (e.g., invalid keys, network issues endpoint mapping)
            wrapper_logger.error(f"Connectivity test failed (API Error): Code={e.error_code}, Message={e.error_message}")
            return False
        except Exception as e:
            # Handle other potential errors (e.g., network timeout before API error)
            wrapper_logger.error(f"Connectivity test failed (Unexpected Error): {e}", exc_info=True)
            return False
            
    def _fetch_symbol_info(self, symbol):
        """
        Fetches and returns exchange information (including filters) for a specific symbol.

        Args:
            symbol (str): The trading symbol (e.g., 'BTCUSDT').

        Returns:
            dict: A dictionary containing the symbol's information, or None if not found or on error.
        
        Raises:
            ClientError/ServerError: If the API call itself fails.
            Exception: For other unexpected errors during the process.
        """
        wrapper_logger.info(f"Fetching exchange information for symbol: {symbol}...")
        try:
            # Use the exchange_info method, specifying the symbol
            exchange_info_response = self.client.exchange_info(symbol=symbol) 
            
            # Check if the response structure is as expected
            if exchange_info_response and 'symbols' in exchange_info_response:
                symbols_list = exchange_info_response['symbols']
                if symbols_list and isinstance(symbols_list, list) and len(symbols_list) > 0:
                    # Assuming the API returns a list containing only the requested symbol's info
                    symbol_data = symbols_list[0]
                    wrapper_logger.info(f"Successfully fetched info for {symbol}.")
                    return symbol_data
                else:
                    wrapper_logger.error(f"Symbol '{symbol}' not found in exchange info response (symbols list empty or invalid).")
                    return None
            else:
                wrapper_logger.error(f"Invalid or unexpected structure in exchange info response for {symbol}: {exchange_info_response}")
                return None
                
        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"API Error fetching exchange info for {symbol}: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            raise # Re-raise API errors as they are often critical for setup
        except Exception as e:
             wrapper_logger.error(f"Unexpected error fetching exchange info for {symbol}: {e}", exc_info=True)
             raise # Re-raise other unexpected errors

    def get_account_info(self, verbose=False):
        """
        Retrieves the current account information, including asset balances.

        Args:
            verbose (bool, optional): If True, logs detailed account information. Defaults to False.

        Returns:
            dict: A dictionary containing account information, or None if an error occurs.
        """
        if verbose:
            wrapper_logger.info("Fetching account information...")
        try:
            # Call the account method from the client
            account_info = self.client.account()
            
            # Optional detailed logging
            if verbose:
                wrapper_logger.info("\n" + "="*15 + " Account Information " + "="*15)
                wrapper_logger.info(f"Maker Commission: {account_info.get('makerCommission')}")
                wrapper_logger.info(f"Taker Commission: {account_info.get('takerCommission')}")
                wrapper_logger.info(f"Buyer Commission: {account_info.get('buyerCommission')}")
                wrapper_logger.info(f"Seller Commission: {account_info.get('sellerCommission')}")
                wrapper_logger.info(f"Can Trade: {account_info.get('canTrade')}")
                wrapper_logger.info(f"Can Withdraw: {account_info.get('canWithdraw')}")
                wrapper_logger.info(f"Can Deposit: {account_info.get('canDeposit')}")
                wrapper_logger.info(f"Update Time: {account_info.get('updateTime')} ({pd.Timestamp(account_info.get('updateTime'), unit='ms') if account_info.get('updateTime') else 'N/A'})")
                wrapper_logger.info(f"Account Type: {account_info.get('accountType')}")
                
                wrapper_logger.info("\n--- Balances (Non-Zero) ---")
                balances = account_info.get('balances', [])
                found_non_zero = False
                for balance in balances:
                    try:
                        free = float(balance.get('free', 0))
                        locked = float(balance.get('locked', 0))
                        # Only display assets with a non-zero balance (free or locked)
                        if free > 1e-9 or locked > 1e-9: # Use small threshold for float comparison
                            wrapper_logger.info(f"  Asset: {balance['asset']:<6} | Free: {free:<18.8f} | Locked: {locked:<18.8f}")
                            found_non_zero = True
                    except (ValueError, TypeError) as parse_err:
                         wrapper_logger.warning(f"Could not parse balance for asset {balance.get('asset', 'UNKNOWN')}: {parse_err}")
                if not found_non_zero:
                     wrapper_logger.info("  No non-zero balances found.")
                wrapper_logger.info("="*49)

            return account_info # Return the full dictionary
            
        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"API Error getting account info: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            return None # Return None to indicate failure
        except Exception as e:
             wrapper_logger.error(f"Unexpected error getting account info: {e}", exc_info=True)
             return None

    def get_server_time(self):
        """
        Gets the current Binance server time in milliseconds.

        Returns:
            int: Server timestamp in milliseconds, or None if an error occurs.
        """
        try:
            response = self.client.time()
            return response['serverTime']
        except (ClientError, ServerError, Exception) as e:
            wrapper_logger.error(f"Error getting server time: {e}", exc_info=True)
            return None

    def get_order_book(self, symbol=None, limit=10):
        """
        Fetches the order book (depth) for the specified symbol.

        Args:
            symbol (str, optional): The trading symbol. Defaults to the wrapper's default trading_pair.
            limit (int, optional): The number of depth levels to retrieve (e.g., 5, 10, 20, 50...). Defaults to 10. 
                                   Check API docs for valid limits.

        Returns:
            dict: A dictionary containing 'bids' and 'asks' lists [[price_str, qty_str], ...], plus 'lastUpdateId',
                  or None if an error occurs. Prices and quantities are returned as strings by the library.
                  This implementation converts them to floats.
        """
        symbol_to_fetch = symbol or self.trading_pair
        if not symbol_to_fetch:
             wrapper_logger.error("Cannot get order book: symbol not specified and no default trading pair set.")
             return None
        
        wrapper_logger.debug(f"Fetching order book for {symbol_to_fetch} with limit {limit}...")
        try:
            # Call the depth method from the client
            depth_data_raw = self.client.depth(symbol=symbol_to_fetch, limit=limit)
            
            # Process the raw data: convert price/qty strings to floats for easier use
            processed_depth = {
                'lastUpdateId': depth_data_raw.get('lastUpdateId'),
                'bids': [],
                'asks': []
            }
            
            if 'bids' in depth_data_raw:
                processed_depth['bids'] = [[float(price), float(qty)] for price, qty in depth_data_raw['bids']]
            if 'asks' in depth_data_raw:
                 processed_depth['asks'] = [[float(price), float(qty)] for price, qty in depth_data_raw['asks']]

            wrapper_logger.debug(f"Successfully fetched and processed order book for {symbol_to_fetch}.")
            return processed_depth

        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"API Error getting order book for {symbol_to_fetch}: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            return None
        except (ValueError, TypeError) as parse_err:
             wrapper_logger.error(f"Error parsing order book data for {symbol_to_fetch}: {parse_err}", exc_info=True)
             return None # Return None if data conversion fails
        except Exception as e:
             wrapper_logger.error(f"Unexpected error getting order book for {symbol_to_fetch}: {e}", exc_info=True)
             return None

    # --- Order Placement & Management ---
    
    def _apply_filters_and_format(self, side, quantity, price=None):
        """
        Internal helper to validate and format quantity and price based on symbol rules.
        Adjusts values to comply with tick size, lot size, min/max qty, and min notional.

        Args:
            side (str): "BUY" or "SELL".
            quantity (float): The desired order quantity.
            price (float, optional): The desired order price (for LIMIT orders). Required if not None.

        Returns:
            tuple: (formatted_quantity_str, formatted_price_str, is_valid)
                   Returns formatted strings ready for API call and a boolean indicating validity.
                   If invalid, returns (None, None, False). Price string is None for Market orders.
        """
        try:
            quantity = float(quantity)
            if price is not None:
                price = float(price)

            # 1. Adjust Quantity by Lot Size (Step Size) - Use floor to be conservative
            if self.lot_size > 0:
                # Calculate number of steps, floor it, then multiply back
                num_steps = math.floor(quantity / self.lot_size)
                quantity_adjusted = num_steps * self.lot_size
                if quantity_adjusted < quantity - 1e-12: # Check if adjustment happened
                     wrapper_logger.debug(f"Adjusted quantity {quantity} to {quantity_adjusted} based on lot size {self.lot_size}")
                quantity = quantity_adjusted
            else:
                # If lot size is 0 or invalid, proceed with original quantity but warn
                wrapper_logger.warning(f"Lot size is {self.lot_size}. Proceeding without lot size adjustment for quantity {quantity}.")

            # 2. Check Min/Max Quantity AFTER lot size adjustment
            if quantity < self.min_qty - 1e-12: # Allow for tiny float inaccuracies
                wrapper_logger.error(f"Validation Error: Adjusted quantity {quantity:.{self.qty_precision}f} is below minQty ({self.min_qty}).")
                return None, None, False
            if quantity > self.max_qty + 1e-12:
                wrapper_logger.warning(f"Validation Warning: Adjusted quantity {quantity:.{self.qty_precision}f} exceeds maxQty ({self.max_qty}). Clamping to maxQty.")
                quantity = self.max_qty # Clamp quantity to maximum allowed

            # 3. Adjust Price by Tick Size (for LIMIT orders)
            price_adjusted_str = None
            price_adjusted = price # Keep track of adjusted price for notional check
            if price is not None:
                if self.tick_size > 0:
                    # Adjust price DOWN for BUY orders, UP for SELL orders to ensure placeable price
                    if side == "BUY":
                        num_ticks = math.floor(price / self.tick_size)
                    else: # SELL
                        num_ticks = math.ceil(price / self.tick_size)
                    price_adjusted = num_ticks * self.tick_size
                    if abs(price_adjusted - price) > 1e-12: # Check if adjustment happened
                        wrapper_logger.debug(f"Adjusted price {price} to {price_adjusted:.{self.price_precision}f} based on tick size {self.tick_size} for {side} order.")
                else:
                    # If tick size is 0 or invalid, proceed with original price but warn
                    wrapper_logger.warning(f"Tick size is {self.tick_size}. Proceeding without tick size adjustment for price {price}.")
                    price_adjusted = price

                # Format price string AFTER adjustment
                price_adjusted_str = "{:.{prec}f}".format(price_adjusted, prec=self.price_precision)


            # 4. Check Min Notional (Value = Price * Quantity) - Primarily for LIMIT orders, approximate for MARKET
            # Use the adjusted price for the check if it's a LIMIT order
            check_price = price_adjusted if price is not None else None
            if check_price is not None: # Only check notional if price is available (LIMIT orders)
                 notional_value = quantity * check_price
                 if notional_value < self.min_notional - 1e-9: # Allow small tolerance
                     wrapper_logger.error(f"Validation Error: Order notional value ({notional_value:.8f}) is below minNotional ({self.min_notional:.8f}). Adjusted Qty: {quantity}, Adjusted Price: {check_price}")
                     return None, None, False
            # else: For MARKET orders, notional check is harder pre-flight, API might reject.

            # 5. Format Quantity String AFTER all checks and adjustments
            quantity_str = "{:.{prec}f}".format(quantity, prec=self.qty_precision)

            return quantity_str, price_adjusted_str, True

        except (ValueError, TypeError) as e:
             wrapper_logger.error(f"Error during value formatting/validation (Qty: {quantity}, Price: {price}): {e}", exc_info=True)
             return None, None, False

    def place_limit_order(self, side, quantity, price, symbol=None, time_in_force="GTC", verbose=False):
        """
        Places a LIMIT order after validating against symbol rules and formatting parameters.

        Args:
            side (str): "BUY" or "SELL".
            quantity (float): The desired order quantity.
            price (float): The desired limit price.
            symbol (str, optional): The trading symbol. Defaults to the wrapper's default trading_pair.
            time_in_force (str, optional): GTC (Good-Til-Canceled), IOC (Immediate-Or-Cancel), FOK (Fill-Or-Kill). Defaults to "GTC".
            verbose (bool, optional): If True, logs detailed information before and after placement. Defaults to False.

        Returns:
            dict: The API response dictionary if successful, None if validation fails or API error occurs.
        """
        symbol_to_use = symbol or self.trading_pair
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            wrapper_logger.error(f"Invalid order side '{side}'. Must be 'BUY' or 'SELL'.")
            return None
        if not symbol_to_use:
             wrapper_logger.error("Cannot place order: symbol not specified and no default trading pair set.")
             return None

        # Validate and format parameters using the helper function
        quantity_str, price_str, is_valid = self._apply_filters_and_format(side, quantity, price)

        if not is_valid:
            wrapper_logger.error("Order placement aborted due to validation failure.")
            return None # Abort if validation failed

        if verbose:
            wrapper_logger.info(f"Attempting to place LIMIT {side} order:")
            wrapper_logger.info(f"  Symbol: {symbol_to_use}")
            wrapper_logger.info(f"  Quantity (formatted): {quantity_str} (Original: {quantity})")
            wrapper_logger.info(f"  Price (formatted): {price_str} (Original: {price})")
            wrapper_logger.info(f"  TimeInForce: {time_in_force}")

        try:
            start_time = time.time()
            # Call the new_order method from the client
            response = self.client.new_order(
                symbol=symbol_to_use,
                side=side,
                type="LIMIT",
                timeInForce=time_in_force,
                quantity=quantity_str,
                price=price_str
            )
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000
            
            # Log success with details from the response
            order_id = response.get('orderId', 'N/A')
            status = response.get('status', 'N/A')
            wrapper_logger.info(f"✅ LIMIT Order request successful ({elapsed_ms:.1f} ms). OrderId: {order_id}, Status: {status}")
            if verbose:
                wrapper_logger.info(f"📦 API Response: {response}")
            
            return response

        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"❌ API Error placing LIMIT {side} order ({quantity_str} {symbol_to_use} @ {price_str}): Code={e.error_code}, Msg={e.error_message}", exc_info=False)
            return None # Return None on API error
        except Exception as e:
            # Catch any other unexpected errors during the API call
            wrapper_logger.error(f"❌ Unexpected error placing LIMIT order: {e}", exc_info=True)
            return None

    def place_market_order(self, side, quantity, symbol=None, is_quote_order=False, verbose=False):
        """
        Places a MARKET order using either base quantity or quote quantity.
        Performs basic validation before sending.

        Args:
            side (str): "BUY" or "SELL".
            quantity (float): The desired order quantity (either base asset or quote asset).
            symbol (str, optional): The trading symbol. Defaults to the wrapper's default trading_pair.
            is_quote_order (bool, optional): If True, 'quantity' represents the amount of the quote asset to spend (BUY) or receive (SELL). 
                                            If False (default), 'quantity' represents the amount of the base asset to buy or sell.
            verbose (bool, optional): If True, logs detailed information. Defaults to False.

        Returns:
            dict: The API response dictionary if successful, None if validation fails or API error occurs. 
                  Note: Market orders might fill partially or fully; check the 'fills' in the response.
        """
        symbol_to_use = symbol or self.trading_pair
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            wrapper_logger.error(f"Invalid order side '{side}'. Must be 'BUY' or 'SELL'.")
            return None
        if not symbol_to_use:
             wrapper_logger.error("Cannot place order: symbol not specified and no default trading pair set.")
             return None
             
        params = {'symbol': symbol_to_use, 'side': side, 'type': 'MARKET'}
        log_qty_description = ""
        original_quantity = quantity # Keep for logging

        try:
            quantity = float(quantity) # Ensure quantity is float

            if not is_quote_order:
                # --- Base Quantity Order ---
                # Validate and format using a simplified version of the helper (no price needed)
                quantity_str, _, is_valid = self._apply_filters_and_format(side, quantity, price=None) # Pass price=None

                if not is_valid:
                    wrapper_logger.error("Market order (base qty) aborted due to validation failure.")
                    return None
                
                params['quantity'] = quantity_str
                log_qty_description = f"{quantity_str} {self.base_asset} (Original: {original_quantity})"
                
            else:
                # --- Quote Quantity Order ---
                # Basic check against minNotional (less precise for market orders)
                if quantity < self.min_notional:
                    wrapper_logger.warning(f"Market order quoteQty {quantity} is below minNotional {self.min_notional}. Order may be rejected by API.")
                
                # Quote quantity precision is typically fixed (e.g., 8 for USDT) - hardcoding common default
                # A more robust solution would fetch quote asset precision if needed, but often it's standard.
                quote_asset_precision = 8 
                quantity_str = "{:.{prec}f}".format(quantity, prec=quote_asset_precision)
                params['quoteOrderQty'] = quantity_str
                log_qty_description = f"{quantity_str} {self.quote_asset} (Original: {original_quantity})"

            if verbose:
                wrapper_logger.info(f"Attempting to place MARKET {side} order:")
                wrapper_logger.info(f"  Symbol: {symbol_to_use}")
                wrapper_logger.info(f"  Quantity: {log_qty_description}")
                if is_quote_order: wrapper_logger.info("  Type: Quote Order Quantity")
                else: wrapper_logger.info("  Type: Base Order Quantity")

            start_time = time.time()
            # Call the new_order method with constructed parameters
            response = self.client.new_order(**params)
            end_time = time.time()
            elapsed_ms = (end_time - start_time) * 1000

            order_id = response.get('orderId', 'N/A')
            status = response.get('status', 'N/A')
            
            # Log details, especially for filled market orders
            if status == 'FILLED':
                 fills = response.get('fills', [])
                 filled_qty = sum(float(f['qty']) for f in fills)
                 filled_quote_qty = sum(float(f['price']) * float(f['qty']) for f in fills)
                 avg_price = filled_quote_qty / filled_qty if filled_qty > 0 else 0
                 wrapper_logger.info(
                     f"✅ MARKET Order Filled ({elapsed_ms:.1f} ms). OrderId: {order_id}. "
                     f"Filled Qty: {filled_qty:.{self.qty_precision}f} {self.base_asset}, "
                     f"Avg Price: {avg_price:.{self.price_precision}f}"
                 )
            else:
                 wrapper_logger.warning(
                     f"⚠️ MARKET Order request status: {status} ({elapsed_ms:.1f} ms). OrderId: {order_id}. May not have filled."
                 )

            if verbose:
                 wrapper_logger.info(f"📦 API Response: {response}")
            return response

        except (ValueError, TypeError) as format_err:
             wrapper_logger.error(f"Error processing market order quantity ({original_quantity}): {format_err}", exc_info=True)
             return None
        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"❌ API Error placing MARKET {side} order ({log_qty_description} on {symbol_to_use}): Code={e.error_code}, Msg={e.error_message}", exc_info=False)
            return None
        except Exception as e:
             wrapper_logger.error(f"❌ Unexpected error placing MARKET order: {e}", exc_info=True)
             return None

    def get_open_orders(self, symbol=None, verbose=False):
        """
        Retrieves all currently open orders for a specific symbol or the default trading pair.

        Args:
            symbol (str, optional): The trading symbol. If None, uses the wrapper's default trading_pair.
            verbose (bool, optional): If True, logs detailed information about fetched orders. Defaults to False.

        Returns:
            list: A list of dictionaries, each representing an open order, or None if an error occurs. 
                  Returns an empty list ([]) if there are no open orders.
        """
        symbol_to_fetch = symbol or self.trading_pair
        # If no symbol provided and no default, API might fetch for all pairs (could be slow/large response)
        # It's safer to require a symbol or default.
        if not symbol_to_fetch:
             wrapper_logger.warning("Getting open orders: Symbol not specified and no default set. API might fetch for all symbols.")
             # Or raise error: raise ValueError("Symbol must be specified or default trading_pair set.")

        log_symbol_display = symbol_to_fetch if symbol_to_fetch else "all symbols"
        
        if verbose:
            wrapper_logger.info(f"Fetching open orders for {log_symbol_display}...")

        try:
            # Construct parameters - only include symbol if specified
            params = {}
            if symbol_to_fetch:
                 params['symbol'] = symbol_to_fetch
                 
            # Call the get_open_orders method
            open_orders_list = self.client.get_open_orders(**params)
            
            # Log details if verbose
            if verbose:
                if open_orders_list:
                    wrapper_logger.info(f"📦 Found {len(open_orders_list)} open orders for {log_symbol_display}:")
                    for order in open_orders_list:
                         wrapper_logger.info(
                             f"  - ID:{order.get('orderId')} | {order.get('symbol')} | {order.get('side')} | Type:{order.get('type')} | "
                             f"Qty:{order.get('origQty')} | Price:{order.get('price')} | Status:{order.get('status')}"
                         )
                else:
                    wrapper_logger.info(f"📦 No open orders found for {log_symbol_display}.")
            
            return open_orders_list # Return the list (can be empty)

        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"❌ API Error getting open orders for {log_symbol_display}: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            return None # Return None to indicate failure
        except Exception as e:
            wrapper_logger.error(f"❌ Unexpected error getting open orders: {e}", exc_info=True)
            return None

    def cancel_order(self, orderId, symbol=None, verbose=False):
        """
        Cancels an active order identified by its order ID and symbol.

        Args:
            orderId (int or str): The ID of the order to cancel.
            symbol (str, optional): The trading symbol of the order. Defaults to the wrapper's default trading_pair.
            verbose (bool, optional): If True, logs detailed information. Defaults to False.

        Returns:
            dict: The API response dictionary confirming cancellation, 
                  or a dictionary with a 'warning' if the order was likely already inactive,
                  or None if a critical error occurs.
        """
        symbol_to_use = symbol or self.trading_pair
        if not symbol_to_use:
             wrapper_logger.error("Cannot cancel order: symbol not specified and no default trading pair set.")
             return None
        if not orderId:
             wrapper_logger.error("Cannot cancel order: orderId is required.")
             return None
             
        if verbose:
            wrapper_logger.info(f"Attempting to cancel order {orderId} for {symbol_to_use}...")

        try:
            # Call the cancel_order method
            response = self.client.cancel_order(symbol=symbol_to_use, orderId=orderId)
            
            # Log success
            cancelled_id = response.get('orderId', 'N/A')
            status = response.get('status', 'N/A')
            wrapper_logger.info(f"✅ Order cancellation request successful for OrderId: {cancelled_id} (Status: {status}) on {symbol_to_use}.")
            if verbose:
                wrapper_logger.info(f"📦 API Response: {response}")
            
            return response

        except ClientError as e:
            # Handle specific error code for "Unknown order" (-2011)
            if e.error_code == -2011: 
                 # This often means the order was already filled or cancelled
                 wrapper_logger.warning(f"⚠️ Attempted to cancel order {orderId} on {symbol_to_use}, but it was likely already inactive (API Error {e.error_code}: {e.error_message}).")
                 # Return a specific dictionary indicating this non-critical situation
                 return {"warning": "Order likely already filled or cancelled.", 
                         "symbol": symbol_to_use, 
                         "orderId": orderId, 
                         "original_error_code": e.error_code}
            else:
                 # Log other client errors as failures
                 wrapper_logger.error(f"❌ ClientError cancelling order {orderId} on {symbol_to_use}: Code={e.error_code}, Message={e.error_message}", exc_info=True)
                 return None # Indicate failure
                 
        except ServerError as e:
             # Log server errors as failures
             wrapper_logger.error(f"❌ ServerError cancelling order {orderId} on {symbol_to_use}: {e}", exc_info=True)
             return None
             
        except Exception as e:
            # Log any other unexpected errors
            wrapper_logger.error(f"❌ Unexpected error cancelling order {orderId} on {symbol_to_use}: {e}", exc_info=True)
            return None

    def cancel_all_open_orders(self, symbol=None, verbose=False):
        """
        Cancels ALL open orders for a specific symbol. Use with caution!

        Args:
            symbol (str, optional): The trading symbol for which to cancel all orders. 
                                   Defaults to the wrapper's default trading_pair.
            verbose (bool, optional): If True, logs detailed information. Defaults to False.

        Returns:
            list: A list of dictionaries, each representing a cancelled order, 
                  or None if an error occurs. Returns an empty list ([]) if no orders were open to cancel.
        """
        symbol_to_use = symbol or self.trading_pair
        if not symbol_to_use:
             wrapper_logger.error("Cannot cancel all orders: symbol not specified and no default trading pair set.")
             return None

        if verbose:
            wrapper_logger.info(f"Attempting to cancel ALL open orders for {symbol_to_use}...")
            
        # Confirmation step might be wise here in a real application
        # input(f"Press Enter to confirm cancellation of ALL orders for {symbol_to_use}, or Ctrl+C to abort...")

        try:
            # Call the cancel_open_orders method (plural)
            # Note: The library method is cancel_open_orders
            response_list = self.client.cancel_open_orders(symbol=symbol_to_use) 
            
            count = len(response_list)
            if count > 0:
                wrapper_logger.info(f"✅ Successfully sent cancellation request for {count} open order(s) on {symbol_to_use}.")
            else:
                 wrapper_logger.info(f"✅ No open orders found to cancel for {symbol_to_use}.")

            if verbose:
                wrapper_logger.info(f"📦 API Response (list of cancelled orders): {response_list}")
                
            return response_list # Return the list of cancelled orders

        except (ClientError, ServerError) as e:
            wrapper_logger.error(f"❌ API Error cancelling all open orders for {symbol_to_use}: Code={e.error_code}, Message={e.error_message}", exc_info=True)
            return None
        except Exception as e:
            wrapper_logger.error(f"❌ Unexpected error cancelling all open orders for {symbol_to_use}: {e}", exc_info=True)
            return None

# --- Example Usage Block ---
# This block runs only when the script is executed directly (python binance_api_wrapper.py)
# It demonstrates how to use the wrapper methods.
if __name__ == "__main__":
    # Configure root logger for visibility if running script directly
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    wrapper_logger.info("--- Binance API Wrapper Example (using binance-connector) ---")
    
    # IMPORTANT: Ensure you have a .env file in the same directory with:
    # BINANCE_API_KEY="YOUR_TESTNET_API_KEY"
    # BINANCE_API_SECRET="YOUR_TESTNET_API_SECRET"
    # Or that the fallback keys above are set (unsafe).
    
    test_symbol = "BTCUSDT" # Use a common pair on testnet for the example

    try:
        # --- Initialize Wrapper ---
        # Create an instance of the wrapper for the desired trading pair
        wrapper = BinanceAPIWrapper(trading_pair=test_symbol) 
        
        # --- Get Account Info ---
        print("\n--- Fetching Account Info ---")
        wrapper.get_account_info(verbose=True) # Use verbose=True for detailed output
        
        # --- Get Order Book ---
        print(f"\n--- Fetching Order Book for {test_symbol} (Top 5 Levels) ---")
        depth = wrapper.get_order_book(symbol=test_symbol, limit=5)
        if depth:
             print("Bids (Price, Quantity):")
             for price, qty in depth.get('bids', []): print(f"  {price:.2f}, {qty:.5f}")
             print("Asks (Price, Quantity):")
             for price, qty in depth.get('asks', []): print(f"  {price:.2f}, {qty:.5f}")
             print(f"Last Update ID: {depth.get('lastUpdateId')}")
        else:
             print(f"Could not fetch order book for {test_symbol}.")
             
        # --- Place a Test Limit Order (far from market to likely not fill) ---
        print(f"\n--- Placing a Test Limit Buy Order ---")
        if depth and depth.get('bids'):
            try:
                # Place order significantly below the best bid to avoid immediate fill
                best_bid = depth['bids'][0][0]
                far_buy_price = best_bid * 0.80 # Target 20% below best bid
                
                # Determine a valid quantity meeting minQty and minNotional
                test_qty = wrapper.min_qty 
                if test_qty * far_buy_price < wrapper.min_notional:
                     # If min_qty is too small for min_notional, calculate required qty
                     required_qty = (wrapper.min_notional / far_buy_price) * 1.01 # Add 1% buffer
                     # Adjust required_qty to the next valid lot size step
                     if wrapper.lot_size > 0:
                          required_qty = math.ceil(required_qty / wrapper.lot_size) * wrapper.lot_size
                     test_qty = max(required_qty, wrapper.min_qty) # Ensure it's still >= min_qty
                     print(f"Adjusted test quantity to ~{test_qty:.{wrapper.qty_precision}f} to meet min_notional at target price.")
                
                print(f"Attempting to place buy order for ~{test_qty:.{wrapper.qty_precision}f} {test_symbol} @ {far_buy_price:.{wrapper.price_precision}f}")
                
                # Place the order using the wrapper method
                buy_order_response = wrapper.place_limit_order(
                    side="BUY", 
                    quantity=test_qty, 
                    price=far_buy_price, 
                    verbose=True # Show detailed logs for placement
                )

                # --- Check and Cancel the Test Order ---
                if buy_order_response and 'orderId' in buy_order_response:
                    order_id_to_cancel = buy_order_response['orderId']
                    print(f"\nSuccessfully placed dummy buy order with ID: {order_id_to_cancel}")
                    
                    # Wait a moment for the order to register on the exchange
                    print("Waiting 2 seconds before checking open orders...")
                    time.sleep(2) 
                    
                    # Verify the order is open
                    print(f"\n--- Checking Open Orders for {test_symbol} ---")
                    open_orders = wrapper.get_open_orders(symbol=test_symbol, verbose=True)
                    
                    # Find our specific order in the list
                    order_found = False
                    if open_orders:
                         for o in open_orders:
                             if o.get('orderId') == order_id_to_cancel:
                                 order_found = True
                                 print(f"Found test order {order_id_to_cancel} in open orders list.")
                                 break
                    
                    # Attempt to cancel if found
                    if order_found:
                         print(f"\n--- Attempting to Cancel Test Order {order_id_to_cancel} ---")
                         cancel_response = wrapper.cancel_order(orderId=order_id_to_cancel, symbol=test_symbol, verbose=True)
                         if cancel_response and "warning" not in cancel_response:
                              print(f"Cancellation request for order {order_id_to_cancel} appears successful.")
                         elif cancel_response and "warning" in cancel_response:
                               print(f"Cancellation attempt returned warning (likely already inactive): {cancel_response['warning']}")
                         else:
                              print(f"Cancellation request for order {order_id_to_cancel} failed (check logs).")
                         
                         # Verify cancellation by checking open orders again
                         print("Waiting 1 second before re-checking open orders...")
                         time.sleep(1)
                         wrapper.get_open_orders(symbol=test_symbol, verbose=True) 
                    else:
                         print(f"\nTest order {order_id_to_cancel} was NOT found in the open orders list after placement. It might have failed placement, filled unexpectedly, or there was an API delay/error.")

                else:
                     print("\nFailed to place the test limit buy order (check logs above for validation or API errors).")

            except Exception as order_err:
                 print(f"\nAn error occurred during the order placement/cancellation test: {order_err}")
                 wrapper_logger.error("Error during order test sequence:", exc_info=True)
                 
        else:
            print("\nCould not get order book, skipping order placement test.")

        # --- Example: Cancel All Open Orders (Use with extreme caution!) ---
        # print(f"\n--- Example: Attempting to Cancel ALL Open Orders for {test_symbol} ---")
        # confirmation = input(f"WARNING: This will cancel ALL open orders for {test_symbol}. Type 'YES' to confirm: ")
        # if confirmation == 'YES':
        #     cancel_all_response = wrapper.cancel_all_open_orders(symbol=test_symbol, verbose=True)
        #     if cancel_all_response is not None:
        #          print(f"Cancelled {len(cancel_all_response)} orders.")
        #     else:
        #          print("Failed to cancel all orders (check logs).")
        # else:
        #      print("Cancellation of all orders aborted.")

    # --- Handle Specific Errors During Example ---
    except ValueError as ve:
        wrapper_logger.error(f"Example script configuration or value error: {ve}")
    except ConnectionError as ce:
        wrapper_logger.error(f"Example script connection error: {ce}")
    except (ClientError, ServerError) as b_err:
         # Catch API errors that might occur outside specific method calls (e.g., during init)
         wrapper_logger.error(f"Example script encountered a Binance API Error: Code={b_err.error_code}, Msg={b_err.error_message}")
    except Exception as e:
        # Catch any other unexpected errors during the example run
        wrapper_logger.error(f"An unexpected error occurred during the example execution: {e}", exc_info=True)

    wrapper_logger.info("\n--- Binance API Wrapper Example Finished ---")