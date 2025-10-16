from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import logging
import math
import plotly.graph_objects as go
from collections import deque
import io  # For capturing logs

from rltrader.envs import TwoSidedMarketEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# --- Configuration (Use your existing config) ---
# NOTE: Ensure this config matches the one used by the environment logic if modified elsewhere
config = {
    # --- Data and Environment ---
    "csv_path": str((DATA_DIR / "orderbook_trimmed_small.csv").resolve()),
    #"csv_path":"/home/gaen/Documents/billions_db/orderbooks/binance/futures/ethusdc/28-Mar-2025/binance_futures_ethusdc_orderbook_28-Mar-2025.csv", # <<< --- UPDATE THIS PATH --- >>>
    # "env_path": "hft_env", # Not typically needed if importing directly
    # "env_class": "TwoSidedMarketEnv", # Not typically needed if importing directly

    # --- Core Simulation Parameters ---
    "initial_capital": 20000.0,
    "max_steps": 5000, # Max steps to load from CSV (can be overridden by dataset size)
    "episode_length": 400, # Max steps per training episode

    # --- Market Microstructure ---
    "order_book_levels": 9,
    "tick_size": 0.01,
    "lot_size": 0.005,

    # --- Agent Actions & Constraints ---
    "price_offset_ticks": 10,       # Range for placing orders relative to BBO (scaled from action)
    "max_order_volume": 2.5,        # Max volume per order (scaled from action)
    "max_active_orders": 10,
    "max_inventory": 5.0,           # Max absolute inventory allowed
    "allowed_aggressiveness_ticks": 5, # Max ticks aggressive placement allowed (e.g., crossing spread)

    # --- Latency & Costs ---
    "latency_steps_long": 0,        # Steps until BUY order is active
    "latency_steps_short": 0,       # Steps until SELL order is active
    "transaction_cost_long": 0.0000, # Proportional cost for BUY execution (e.g., 0.0004 = 0.04%)
    "transaction_cost_short": 0.0000, # Proportional cost for SELL execution
    "taker_penalty": 0.000,         # Additional penalty per unit volume for aggressive (taker) orders

    # --- Reward Components ---
    "inventory_penalty": 0.0000,     # Quadratic penalty factor for holding inventory
    "invalid_order_penalty": 0.0,    # Penalty for placing invalid orders (e.g., too far)
    "activity_bonus": 0.00,         # Bonus per unit volume executed (maker incentive)
    "quoting_reward_enabled": True,
    "quoting_reward_amount": 0.00001, # Reward per side for tight quotes
    "quoting_reward_max_ticks": 5,   # Max ticks from BBO for quoting reward

    # --- Action Space Features ---
    "explicit_cancel_enabled": True,
    "explicit_cancel_threshold": 0.7,  # Action[4] > threshold triggers cancel
    "explicit_cancel_penalty": 0.00001,# Penalty for explicit cancel action
    "explicit_cancel_clears_pending": True, # Cancel pending orders too?
    "do_nothing_threshold": -0.9,      # Action[5] > threshold skips market actions (range -1 to 1)

    # --- Observation Normalization Scaling ---
    "obs_qty_norm_scale": 1.0, # Multiply normalized quantities by this factor
    "obs_price_norm_scale": 100.0, # Divide normalized prices (in ticks) by this factor
}


# --- Logging Setup ---
# Custom handler to capture logs for display in Streamlit
class StreamlitLogHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level=level)
        # Use deque in session state to store logs persistently across reruns
        if 'log_records' not in st.session_state:
             st.session_state.log_records = deque(maxlen=200) # Store last 200 log messages
        self.log_records = st.session_state.log_records

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_records.append(msg)
        except Exception:
            self.handleError(record)

# Configure root logger
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
streamlit_handler = StreamlitLogHandler(level=logging.INFO) # Set desired level here
streamlit_handler.setFormatter(log_formatter)

# Get the root logger and add our handler (avoid removing others if possible)
root_logger = logging.getLogger()
# Clear existing handlers ONLY if they are our specific handler type to prevent duplicates on script rerun
# This is safer than clearing all handlers
for handler in root_logger.handlers[:]:
    if isinstance(handler, StreamlitLogHandler):
        root_logger.removeHandler(handler)
root_logger.addHandler(streamlit_handler)
root_logger.setLevel(logging.INFO) # Set root logger level

# Set level for the environment's logger specifically if needed
# logging.getLogger('TwoSidedMarketEnv').setLevel(logging.DEBUG) # Example

logging.info("Streamlit app started. Logging configured.") # Test message


# --- Helper Functions ---
@st.cache_resource # Cache the environment instance
def load_environment(config_dict):
    """Loads or reloads the HFT environment."""
    try:
        # Instantiate the environment
        env_instance = TwoSidedMarketEnv(config_dict)
        logging.info("HFT Environment loaded successfully via load_environment.")
        return env_instance
    except FileNotFoundError:
        st.error(f"Fatal Error: CSV file not found at {config_dict['csv_path']}. Please check the path.")
        logging.error(f"CSV file not found at {config_dict['csv_path']}")
        return None
    except ValueError as e:
        st.error(f"Fatal Error: Configuration or Data Error - {e}")
        logging.error(f"Configuration or Data Error: {e}")
        return None
    except Exception as e:
        st.error(f"Fatal Error: An unexpected error occurred during environment loading: {e}")
        logging.exception("Unexpected error during environment loading:")
        return None

def format_price(price):
    return f"{price:.2f}" if price is not None and not (isinstance(price, float) and math.isnan(price)) else "N/A"

def format_volume(volume):
     return f"{volume:.4f}" if volume is not None and not (isinstance(volume, float) and math.isnan(volume)) else "N/A"

def format_pnl(pnl):
    return f"{pnl:,.2f}"

def create_order_book_df(bids, asks, levels):
    data = []
    # Ensure bids and asks are numpy arrays before accessing shape
    bids_len = bids.shape[0] if isinstance(bids, np.ndarray) else 0
    asks_len = asks.shape[0] if isinstance(asks, np.ndarray) else 0
    max_len = max(bids_len, asks_len, levels)

    for i in range(max_len):
        bid_qty = bids[i, 1] if bids_len > i else None
        bid_prc = bids[i, 0] if bids_len > i else None
        ask_prc = asks[i, 0] if asks_len > i else None
        ask_qty = asks[i, 1] if asks_len > i else None
        data.append({
            'Bid Qty': format_volume(bid_qty),
            'Bid Price': format_price(bid_prc),
            'Ask Price': format_price(ask_prc),
            'Ask Qty': format_volume(ask_qty)
        })
    return pd.DataFrame(data)


def create_depth_chart(bids, asks, mid_price):
    fig = go.Figure()
    # Check if bids/asks are valid numpy arrays
    has_bids = isinstance(bids, np.ndarray) and bids.size > 0
    has_asks = isinstance(asks, np.ndarray) and asks.size > 0

    if has_bids:
        bid_prices = bids[:, 0]
        bid_qtys = bids[:, 1]
        # Sort bids descending for cumulative sum display
        bid_indices = np.argsort(bid_prices)[::-1]
        bid_prices_sorted = bid_prices[bid_indices]
        bid_qtys_sorted = bid_qtys[bid_indices]
        bid_cum_qtys = np.cumsum(bid_qtys_sorted)
        fig.add_trace(go.Scatter(
            x=bid_prices_sorted, y=bid_cum_qtys, mode='lines', name='Bids',
            line=dict(color='green', shape='hv'), fill='tozeroy',
            hovertext=[f"Price: {p:.2f}<br>CumQty: {q:.4f}" for p, q in zip(bid_prices_sorted, bid_cum_qtys)],
            hoverinfo="text"
        ))

    if has_asks:
        ask_prices = asks[:, 0]
        ask_qtys = asks[:, 1]
        # Sort asks ascending for cumulative sum display
        ask_indices = np.argsort(ask_prices)
        ask_prices_sorted = ask_prices[ask_indices]
        ask_qtys_sorted = ask_qtys[ask_indices]
        ask_cum_qtys = np.cumsum(ask_qtys_sorted)
        fig.add_trace(go.Scatter(
            x=ask_prices_sorted, y=ask_cum_qtys, mode='lines', name='Asks',
            line=dict(color='red', shape='hv'), fill='tozeroy',
            hovertext=[f"Price: {p:.2f}<br>CumQty: {q:.4f}" for p, q in zip(ask_prices_sorted, ask_cum_qtys)],
            hoverinfo="text"
        ))

    # Determine reasonable price range
    all_prices = []
    if has_bids: all_prices.extend(bids[:, 0])
    if has_asks: all_prices.extend(asks[:, 0])

    price_min, price_max = None, None
    if all_prices:
        valid_prices = [p for p in all_prices if p is not None and not math.isnan(p)]
        if valid_prices:
            if mid_price and not math.isnan(mid_price):
                # Calculate spread based on valid prices only
                min_vp, max_vp = min(valid_prices), max(valid_prices)
                spread = (max_vp - min_vp) if len(valid_prices) > 1 else config['tick_size'] * 20
                price_min = mid_price - spread * 1.5
                price_max = mid_price + spread * 1.5
            else:
                price_min = min(valid_prices) * 0.995
                price_max = max(valid_prices) * 1.005
        else: # No valid prices found
             price_min = config.get('tick_size', 0.01) * 0.9
             price_max = price_min * 1.2
    else:
        price_min = config.get('tick_size', 0.01) * 0.9
        price_max = price_min * 1.2 # Default fallback range

    fig.update_layout(
        title='Order Book Depth Chart', xaxis_title='Price', yaxis_title='Cumulative Quantity',
        xaxis_range=[price_min, price_max] if price_min is not None else None,
        hovermode='closest', # Changed hovermode
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def get_order_df(orders):
    if not orders:
        return pd.DataFrame(columns=['ID', 'Type', 'Price', 'Volume', 'Placed Step', 'Target Step', 'Taker@Act'])
    df = pd.DataFrame(orders)
    if not df.empty:
        df['Type'] = df['is_buy'].apply(lambda x: 'BUY' if x else 'SELL')
        # Include target_step and taker status if they exist
        cols_to_show = ['id', 'Type', 'price', 'volume', 'timestamp_placed']
        col_names = ['ID', 'Type', 'Price', 'Volume', 'Placed Step']
        if 'target_step' in df.columns:
             cols_to_show.append('target_step')
             col_names.append('Target Step')
        if 'is_taker_at_activation' in df.columns:
             cols_to_show.append('is_taker_at_activation')
             col_names.append('Taker@Act')


        df = df[cols_to_show]
        df.columns = col_names
        df['Price'] = df['Price'].apply(format_price)
        df['Volume'] = df['Volume'].apply(format_volume)
        # Format Taker@Act if present
        if 'Taker@Act' in df.columns:
            df['Taker@Act'] = df['Taker@Act'].apply(lambda x: str(x) if x is not None else 'N/A')

        return df.sort_values(by='Placed Step').reset_index(drop=True)
    else:
         return pd.DataFrame(columns=['ID', 'Type', 'Price', 'Volume', 'Placed Step', 'Target Step', 'Taker@Act'])


def create_historical_price_chart(price_history, order_history):
    """Creates the interactive historical price chart."""
    fig = go.Figure()
    if not price_history:
        fig.update_layout(title="Historical Prices (No data yet)")
        return fig

    df_price = pd.DataFrame(price_history)
    # Ensure 'step' column exists and is numeric before plotting
    if 'step' not in df_price.columns or not pd.api.types.is_numeric_dtype(df_price['step']):
         logging.warning("Historical price data missing 'step' column or it's not numeric.")
         df_price['step'] = range(len(df_price)) # Add step index if missing


    # Plot Mid Price, Best Bid, Best Ask
    if 'mid_price' in df_price.columns:
        fig.add_trace(go.Scatter(x=df_price['step'], y=df_price['mid_price'], mode='lines', name='Mid Price', line=dict(color='blue')))
    if 'best_bid' in df_price.columns:
        fig.add_trace(go.Scatter(x=df_price['step'], y=df_price['best_bid'], mode='lines', name='Best Bid', line=dict(color='rgba(0,255,0,0.5)'))) # Lighter green
    if 'best_ask' in df_price.columns:
        fig.add_trace(go.Scatter(x=df_price['step'], y=df_price['best_ask'], mode='lines', name='Best Ask', line=dict(color='rgba(255,0,0,0.5)'))) # Lighter red

    # Plot Order Placements
    if order_history:
        df_orders = pd.DataFrame(order_history)
        if not df_orders.empty and 'step' in df_orders.columns and 'price' in df_orders.columns and 'is_buy' in df_orders.columns:
            buy_orders = df_orders[df_orders['is_buy']]
            sell_orders = df_orders[~df_orders['is_buy']]

            if not buy_orders.empty:
                fig.add_trace(go.Scatter(
                    x=buy_orders['step'], y=buy_orders['price'], mode='markers', name='Buy Order Placed',
                    marker=dict(color='darkgreen', size=8, symbol='triangle-up'),
                    hovertext=[f"Buy Placed<br>Step: {s}<br>Price: {p:.2f}" for s, p in zip(buy_orders['step'], buy_orders['price'])],
                    hoverinfo="text"
                ))
            if not sell_orders.empty:
                 fig.add_trace(go.Scatter(
                    x=sell_orders['step'], y=sell_orders['price'], mode='markers', name='Sell Order Placed',
                    marker=dict(color='darkred', size=8, symbol='triangle-down'),
                    hovertext=[f"Sell Placed<br>Step: {s}<br>Price: {p:.2f}" for s, p in zip(sell_orders['step'], sell_orders['price'])],
                    hoverinfo="text"
                ))

    fig.update_layout(
        title='Historical Price & Order Placements',
        xaxis_title='Environment Step',
        yaxis_title='Price',
        hovermode='x unified' # Shows hover for all traces at a step
    )
    return fig

# --- /End Helper Functions ---


# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Interactive HFT Simulation")
st.title("🤖 Interactive High-Frequency Trading Environment")

# --- Load Environment ---
# Use the config dictionary defined above
env = load_environment(config)
if env is None:
    st.warning("Environment loading failed. Please check logs and config.")
    st.stop()
else:
    logging.info("Environment check in main script: OK.")


# --- Initialize Session State (More Robustly) ---
if 'env_initialized' not in st.session_state:
    logging.info("First run: Initializing environment state in Streamlit session.")
    try:
        obs, info = env.reset()
        st.session_state.env_initialized = True
        st.session_state.observation = obs
        st.session_state.info = info
        st.session_state.reward = 0.0
        st.session_state.terminated = False
        st.session_state.truncated = False
        st.session_state.last_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        st.session_state.step_count = 0
        # Initialize history lists
        st.session_state.price_history = []
        st.session_state.order_placement_history = []
        # Add initial state to history (optional, start chart from step 0)
        initial_price_data = {
            'step': info.get('current_step', 0),
            'mid_price': info.get('mid_price'),
            'best_bid': info.get('best_bid'),
            'best_ask': info.get('best_ask')
        }
        # Filter out None values before appending
        initial_price_data_clean = {k: v for k, v in initial_price_data.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
        if initial_price_data_clean.get('mid_price') is not None: # Only add if we have valid price data
            st.session_state.price_history.append(initial_price_data_clean)

        # Ensure log_records deque exists (handled by handler init, but double-check)
        if 'log_records' not in st.session_state:
            st.session_state.log_records = deque(maxlen=200)

        logging.info("Session state initialized.")

    except Exception as e:
        st.error(f"Error during initial environment reset: {e}")
        logging.exception("Error during initial env reset:")
        st.stop()


# --- Sidebar for Controls ---
st.sidebar.header("⚙️ Controls & Actions")

# --- Action Sliders ---
st.sidebar.markdown("**Manual Action Vector Override (Range: -1.0 to 1.0)**")
st.sidebar.caption("_(Used only when 'Step ▶️' is clicked)_")
action_values = {}
action_labels = [
    "Buy Offset Signal", "Sell Offset Signal",
    "Buy Size Signal", "Sell Size Signal",
    "Explicit Cancel Signal", "Do Nothing Signal"
]
action_tooltips = [
    "Action[0]: Positive->more passive buy, Negative->more aggressive buy",
    "Action[1]: Positive->more passive sell, Negative->more aggressive sell",
    "Action[2]: Controls buy order size (-1: 0%, +1: 100%)",
    "Action[3]: Controls sell order size (-1: 0%, +1: 100%)",
    f"Action[4]: > {config['explicit_cancel_threshold']:.2f} triggers cancel",
    f"Action[5]: > {config['do_nothing_threshold']:.2f} overrides other actions"
]

for i in range(env.action_space.shape[0]):
    key = f"action_{i}"
    # Use last_action from session state as default if available and step_count > 0
    default_value = float(st.session_state.last_action[i]) if st.session_state.step_count > 0 else 0.0
    # Update slider value based on session state or keep current slider value
    current_slider_value = st.session_state.get(key, default_value)

    action_values[i] = st.sidebar.slider(
        action_labels[i], -1.0, 1.0, current_slider_value, 0.01,
        help=action_tooltips[i], key=key
    )
manual_action_array = np.array(list(action_values.values()), dtype=np.float32)

# --- Quick Action Buttons ---
st.sidebar.markdown("**Quick Actions:**")
btn_cols1 = st.sidebar.columns(2)
btn_cols2 = st.sidebar.columns(2)

step_pressed = btn_cols1[0].button("Step ▶️", use_container_width=True, help="Execute action defined by sliders.", disabled=st.session_state.terminated or st.session_state.truncated)
random_pressed = btn_cols1[1].button("Random 🎲", use_container_width=True, help="Execute a random action.", disabled=st.session_state.terminated or st.session_state.truncated)
do_nothing_pressed = btn_cols2[0].button("Do Nothing 💤", use_container_width=True, help="Execute an action forcing 'Do Nothing'.", disabled=st.session_state.terminated or st.session_state.truncated)
cancel_all_pressed = btn_cols2[1].button("Cancel All ❌", use_container_width=True, help="Execute an action forcing 'Explicit Cancel'.", disabled=st.session_state.terminated or st.session_state.truncated)

repeat_pressed = st.sidebar.button("Repeat Last Action 🔄", use_container_width=True, help="Execute the same action vector as the last step.", disabled=st.session_state.terminated or st.session_state.truncated or st.session_state.step_count == 0)
reset_pressed = st.sidebar.button("Reset Environment 🔁", use_container_width=True)


# --- Main Display Area ---
info = st.session_state.info # Get current info

# Top Metrics
st.header("📊 Current State")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Env Step", info.get('current_step', 'N/A'))
m2.metric("Episode Step", info.get('total_steps_elapsed', 'N/A'))
m3.metric("MTM ($)", format_pnl(info.get('mtm', 0.0)))
m4.metric("Episode PnL ($)", format_pnl(info.get('episode_pnl', 0.0)))
m5.metric("Net Inventory", format_volume(info.get('net_inventory', 0.0)))
m6.metric("Cash ($)", format_pnl(info.get('cash', 0.0)))

# Status Notifications
status_container = st.container()
if st.session_state.terminated:
    status_container.error(f"🔴 Episode Terminated! Reason: (Check Logs tab). Final PnL: {format_pnl(info.get('episode_pnl', 0.0))}")
if st.session_state.truncated:
    status_container.warning(f"🟠 Episode Truncated! (Max steps reached). Final PnL: {format_pnl(info.get('episode_pnl', 0.0))}")
# Use info from the *previous* step to show what *happened*
if info.get('do_nothing_triggered', False):
    status_container.info("💤 'Do Nothing' action was triggered in the previous step.")
if info.get('explicit_cancel_triggered', False):
    status_container.info("❌ 'Explicit Cancel' action was triggered in the previous step.")

# Tabs for detailed views
tab_hist, tab_lob, tab_orders, tab_obs, tab_details, tab_logs = st.tabs([
    "📊 History", "📈 Order Book", "🛒 My Orders", "🧠 Observation", "📝 Details", "📜 Logs"
])

with tab_hist:
    st.subheader("Price History & Order Placements")
    # Create chart using data from session state
    price_chart_fig = create_historical_price_chart(
        st.session_state.get('price_history', []),
        st.session_state.get('order_placement_history', [])
    )
    st.plotly_chart(price_chart_fig, use_container_width=True)

with tab_lob:
    st.subheader("Limit Order Book (LOB)")
    lob_col1, lob_col2 = st.columns([1, 2])
    with lob_col1:
        st.markdown("**Book Levels:**")
        # Use env directly as it holds the *current* LOB state
        df_lob = create_order_book_df(env.bids, env.asks, config['order_book_levels'])
        st.dataframe(df_lob, height=350, use_container_width=True)
        st.markdown(f"""
        **Best Bid:** {format_price(info.get('best_bid'))} | **Best Ask:** {format_price(info.get('best_ask'))} | **Mid:** {format_price(info.get('mid_price'))} | **Spread:** {format_price(info.get('spread'))}
        """)
    with lob_col2:
        st.markdown("**Depth Chart:**")
        # Use env directly for current depth
        depth_fig = create_depth_chart(env.bids, env.asks, info.get('mid_price'))
        st.plotly_chart(depth_fig, use_container_width=True)

with tab_orders:
    st.subheader("My Orders & Positions")
    ord_col1, ord_col2 = st.columns(2)
    with ord_col1:
        st.markdown("**Active Orders:**")
        # Use env directly for current orders
        df_active = get_order_df(env.active_orders)
        st.dataframe(df_active, height=200, use_container_width=True)
    with ord_col2:
        st.markdown("**Pending Orders (Latency):**")
        # Use env directly for current orders
        df_pending = get_order_df(env.pending_orders)
        st.dataframe(df_pending, height=200, use_container_width=True)

    st.markdown(f"""
    **Positions:**
    Long: {format_volume(info.get('long_position'))} @ Avg Cost: {format_price(info.get('long_avg_cost'))} |
    Short: {format_volume(info.get('short_position'))} @ Avg Cost: {format_price(info.get('short_avg_cost'))}
    """)

# --- Observation Tab ---
with tab_obs:
    st.subheader("Observation Vector Breakdown")
    obs = st.session_state.observation
    obs_space = env.observation_space
    info = st.session_state.info # Get current info for denormalization

    st.markdown(f"**Shape:** `{obs_space.shape}` | **Type:** `{obs.dtype}`")
    st.markdown(f"**Raw Observation Vector (Last Step):**")
    st.code(f"{np.round(obs, 4)}")

    st.divider()
    st.markdown("**Unpacked & Denormalized Observation:**")

    # --- Denormalization Setup ---
    # Get necessary values from config and current info
    tick_size = config["tick_size"]
    max_order_vol = max(1e-9, config["max_order_volume"])
    max_inv = max(1e-9, config["max_inventory"])
    initial_capital = max(1.0, config["initial_capital"])
    price_scale = config["obs_price_norm_scale"]
    qty_scale = config["obs_qty_norm_scale"]
    max_active = config["max_active_orders"]
    lob_levels = config["order_book_levels"]

    # Get a safe midprice for denormalization reference
    mid_price_safe = info.get('mid_price')
    if mid_price_safe is None or math.isnan(mid_price_safe) or mid_price_safe <= 0:
        # Try last valid midprice from env if available
        if hasattr(env, '_last_valid_midprice') and not math.isnan(env._last_valid_midprice):
             mid_price_safe = env._last_valid_midprice
        else:
             mid_price_safe = 1.0 # Absolute fallback
        st.caption(f"(Using fallback midprice {mid_price_safe:.2f} for denormalization)")
    else:
        st.caption(f"(Using current midprice {mid_price_safe:.2f} for denormalization)")


    # --- Define Observation Segments ---
    portfolio_size = 9
    orders_prices_size = max_active
    orders_volumes_size = max_active
    orders_size = orders_prices_size + orders_volumes_size
    book_size = 4 * lob_levels
    market_size = 1
    total_size = portfolio_size + orders_size + book_size + market_size

    if len(obs) != total_size:
        st.error(f"Observation vector length mismatch! Expected {total_size}, got {len(obs)}. Cannot unpack accurately.")
    else:
        # --- Portfolio Segment ---
        with st.expander("Portfolio Features (Indices 0-8)", expanded=True):
            st.markdown(f"`obs[0]` **MTM (Normalized):** `{obs[0]:.4f}` -> **MTM ($):** `{(obs[0] * initial_capital):,.2f}`")
            st.markdown(f"`obs[1]` **Inventory (Normalized):** `{obs[1]:.4f}` -> **Inventory:** `{(obs[1] * max_inv):.4f}` *(Note: Based on clipped normalized value)*")
            st.markdown(f"`obs[2]` **Active Orders Count (Normalized):** `{obs[2]:.4f}` -> **Count:** `{int(round(obs[2] * max_active))}`")
            st.markdown(f"`obs[3]` **Long Position (Normalized):** `{obs[3]:.4f}` -> **Long Pos:** `{(obs[3] * max_inv):.4f}` *(Clipped)*")
            st.markdown(f"`obs[4]` **Short Position (Normalized):** `{obs[4]:.4f}` -> **Short Pos:** `{(obs[4] * max_inv):.4f}` *(Clipped)*")

            # Denormalize costs
            long_cost_ticks = obs[5] * price_scale
            long_cost = mid_price_safe + long_cost_ticks * tick_size
            st.markdown(f"`obs[5]` **Long Avg Cost (Norm Scaled Ticks from Mid):** `{obs[5]:.4f}` -> **Avg Cost ($):** `{long_cost:.2f}` *(Clipped)*")

            short_cost_ticks = obs[6] * price_scale
            short_cost = mid_price_safe + short_cost_ticks * tick_size
            st.markdown(f"`obs[6]` **Short Avg Cost (Norm Scaled Ticks from Mid):** `{obs[6]:.4f}` -> **Avg Cost ($):** `{short_cost:.2f}` *(Clipped)*")

            # Denormalize BBO deviations
            bid_dev_ticks = obs[7] * price_scale
            bid_approx = mid_price_safe + bid_dev_ticks * tick_size
            st.markdown(f"`obs[7]` **Best Bid Dev (Norm Scaled Ticks from Mid):** `{obs[7]:.4f}` -> **Best Bid ($ Approx):** `{bid_approx:.2f}`")

            ask_dev_ticks = obs[8] * price_scale
            ask_approx = mid_price_safe + ask_dev_ticks * tick_size
            st.markdown(f"`obs[8]` **Best Ask Dev (Norm Scaled Ticks from Mid):** `{obs[8]:.4f}` -> **Best Ask ($ Approx):** `{ask_approx:.2f}`")

        # --- Active Orders Segment ---
        start_idx = portfolio_size
        with st.expander(f"Active Orders Features (Indices {start_idx}-{start_idx + orders_size - 1})"):
            price_start = start_idx
            vol_start = start_idx + orders_prices_size
            for i in range(max_active):
                price_idx = price_start + i
                vol_idx = vol_start + i

                # Denormalize Price
                price_ticks = obs[price_idx] * price_scale
                order_price = mid_price_safe + price_ticks * tick_size

                # Denormalize Volume
                vol_norm = obs[vol_idx] / qty_scale # Reverse scaling
                order_vol = vol_norm * max_order_vol # Reverse normalization
                order_type = "BUY" if order_vol > 0 else "SELL" if order_vol < 0 else "NONE"
                order_vol_abs = abs(order_vol)

                st.markdown(f"**Order Slot {i+1}:**")
                st.markdown(f"  - `obs[{price_idx}]` Price (Norm Scaled Ticks): `{obs[price_idx]:.4f}` -> **Price ($):** `{order_price:.2f}` *(Clipped)*")
                st.markdown(f"  - `obs[{vol_idx}]` Volume (Norm Scaled): `{obs[vol_idx]:.4f}` -> **Volume:** `{order_vol_abs:.4f}` ({order_type}) *(Clipped)*")
            st.caption("*(Orders are padded with zeros if fewer than max_active_orders)*")


        # --- LOB Segment ---
        start_idx = portfolio_size + orders_size
        with st.expander(f"Order Book Features (Indices {start_idx}-{start_idx + book_size - 1})"):
            for level in range(lob_levels):
                bid_p_idx = start_idx + level * 4
                bid_q_idx = bid_p_idx + 1
                ask_p_idx = bid_q_idx + 1
                ask_q_idx = ask_p_idx + 1

                # Denormalize Bid Price
                bid_p_ticks = obs[bid_p_idx] * price_scale
                bid_price = mid_price_safe + bid_p_ticks * tick_size
                # Denormalize Bid Qty
                bid_q_norm = obs[bid_q_idx] / qty_scale
                bid_qty = bid_q_norm * max_order_vol

                # Denormalize Ask Price
                ask_p_ticks = obs[ask_p_idx] * price_scale
                ask_price = mid_price_safe + ask_p_ticks * tick_size
                # Denormalize Ask Qty
                ask_q_norm = obs[ask_q_idx] / qty_scale
                ask_qty = ask_q_norm * max_order_vol

                st.markdown(f"**Level {level+1}:**")
                st.markdown(f"  - `obs[{bid_p_idx}]` Bid Price (Norm Scaled Ticks): `{obs[bid_p_idx]:.4f}` -> **Bid Price ($):** `{bid_price:.2f}` *(Clipped)*")
                st.markdown(f"  - `obs[{bid_q_idx}]` Bid Qty (Norm Scaled): `{obs[bid_q_idx]:.4f}` -> **Bid Qty:** `{bid_qty:.4f}` *(Clipped)*")
                st.markdown(f"  - `obs[{ask_p_idx}]` Ask Price (Norm Scaled Ticks): `{obs[ask_p_idx]:.4f}` -> **Ask Price ($):** `{ask_price:.2f}` *(Clipped)*")
                st.markdown(f"  - `obs[{ask_q_idx}]` Ask Qty (Norm Scaled): `{obs[ask_q_idx]:.4f}` -> **Ask Qty:** `{ask_qty:.4f}` *(Clipped)*")
            st.caption("*(Levels are padded with zeros if book depth is less than order_book_levels)*")

        # --- Market Segment ---
        start_idx = portfolio_size + orders_size + book_size
        with st.expander(f"Market Features (Index {start_idx})"):
            spread_ticks = obs[start_idx] # Spread is already in ticks, no price_scale needed
            spread = spread_ticks * tick_size
            st.markdown(f"`obs[{start_idx}]` **Spread (Normalized Ticks):** `{obs[start_idx]:.4f}` -> **Spread ($):** `{spread:.2f}`")

# --- End Observation Tab ---


with tab_details:
    st.subheader("Additional Details")
    details_col1, details_col2 = st.columns(2)
    with details_col1:
        st.markdown("**Last Step Reward & Penalties:**")
        st.metric("Total Reward", f"{st.session_state.reward:.6f}")
        st.markdown(f"- Quoting Reward: `{info.get('quoting_reward_step', 0.0):.6f}`")
        # Add other potential reward components if you track them in `info`
        # Example: Assuming env adds these to info dict
        st.markdown(f"- Realized PnL (Fills): `{info.get('realized_pnl_fills', 'N/A')}`")
        st.markdown(f"- Activity Bonus: `{info.get('activity_bonus', 'N/A')}`")
        st.markdown(f"- Risk Penalty: `{info.get('risk_penalty', 'N/A')}`")
        st.markdown(f"- Taker Penalty (Applied at Activation): `{info.get('taker_penalty_applied', 'N/A')}`")
        st.markdown(f"- Explicit Cancel Penalty: `{info.get('explicit_cancel_penalty', 'N/A')}`")
        st.markdown(f"_(Add relevant components to TwoSidedMarketEnv._get_info() if needed)_")

    with details_col2:
        st.markdown("**Environment Configuration:**")
        with st.expander("View Config JSON"):
            st.json(config)

with tab_logs:
    st.subheader("Application & Environment Logs")
    st.markdown("Most recent logs appear first.")
    # Retrieve logs from session state deque
    log_messages = st.session_state.get('log_records', [])
    # Display in a text area, newest first
    log_text = "\n".join(list(log_messages)[::-1]) # Reverse order
    st.text_area("Logs", value=log_text, height=400, key="log_display", disabled=True)

# --- /End Main Display Area ---


# --- Action Handling Logic ---
action_to_take = None
action_source = "N/A" # Track where the action came from

if reset_pressed:
    logging.info("Reset button pressed. Re-initializing state.")
    try:
        obs, info = env.reset()
        st.session_state.observation = obs
        st.session_state.info = info
        st.session_state.reward = 0.0
        st.session_state.terminated = False
        st.session_state.truncated = False
        st.session_state.last_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        st.session_state.step_count = 0
        # Clear histories and add reset state
        st.session_state.price_history = []
        st.session_state.order_placement_history = []
        # Clear logs on reset, add a marker
        if 'log_records' in st.session_state:
            st.session_state.log_records.clear()
        else:
            st.session_state.log_records = deque(maxlen=200)
        st.session_state.log_records.append("--- Environment Reset ---")
        # Add initial state to history
        initial_price_data = {
            'step': info.get('current_step', 0),
            'mid_price': info.get('mid_price'),
            'best_bid': info.get('best_bid'),
            'best_ask': info.get('best_ask')
        }
        initial_price_data_clean = {k: v for k, v in initial_price_data.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
        if initial_price_data_clean.get('mid_price') is not None:
             st.session_state.price_history.append(initial_price_data_clean)


        st.success("Environment Reset!")
        st.rerun() # Rerun immediately
    except Exception as e:
        st.error(f"Error during environment reset: {e}")
        logging.exception("Error during env reset:")


# Determine which action to take based on button press priority
if not st.session_state.terminated and not st.session_state.truncated:
    if random_pressed:
        action_to_take = env.action_space.sample()
        action_source = "Random"
    elif do_nothing_pressed:
        # Force do_nothing signal high, others neutral/off
        action_to_take = np.array([0.0, 0.0, -1.0, -1.0, -1.0, 1.0], dtype=np.float32)
        action_source = "Do Nothing Button"
    elif cancel_all_pressed:
         # Force cancel signal high, others neutral/off
        action_to_take = np.array([0.0, 0.0, -1.0, -1.0, 1.0, -1.0], dtype=np.float32)
        action_source = "Cancel All Button"
    elif repeat_pressed:
        if st.session_state.step_count > 0:
            action_to_take = st.session_state.last_action
            action_source = "Repeat Last Action Button"
        else:
            st.warning("Cannot repeat action, no steps taken yet.")
    elif step_pressed:
        action_to_take = manual_action_array
        action_source = "Manual Sliders"

# --- Execute Step if an action was determined ---
if action_to_take is not None:
    if not st.session_state.terminated and not st.session_state.truncated:
        logging.info(f"Executing step {st.session_state.step_count + 1} with action from {action_source}: {np.round(action_to_take, 3)}")
        try:
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action_to_take)

            # --- Update State ---
            st.session_state.observation = obs
            st.session_state.reward = reward
            st.session_state.terminated = terminated
            st.session_state.truncated = truncated
            st.session_state.info = info
            st.session_state.last_action = action_to_take # Store the action *actually* taken
            st.session_state.step_count += 1

            # --- Update History ---
            # Price History
            current_price_data = {
                'step': info.get('current_step'),
                'mid_price': info.get('mid_price'),
                'best_bid': info.get('best_bid'),
                'best_ask': info.get('best_ask')
            }
            # Only add if step number is valid and price exists
            if current_price_data['step'] is not None:
                 current_price_data_clean = {k: v for k, v in current_price_data.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
                 if current_price_data_clean.get('mid_price') is not None: # Check if essential price is valid
                     st.session_state.price_history.append(current_price_data_clean)


            # Order Placement History (Check if info contains placement details)
            # *** Requires modification in TwoSidedMarketEnv's step/place_order to return placed order details in info dict ***
            # Example: if env returns info['placed_orders_details'] = [{'step': s, 'price': p, 'is_buy': b}, ...]
            placed_orders = info.get('placed_orders_details', [])
            if placed_orders:
                logging.info(f"Detected {len(placed_orders)} placed orders in step {info.get('current_step')}.")
                for order_detail in placed_orders:
                     # Add step number if missing from detail (should ideally be provided by env)
                     if 'step' not in order_detail:
                         order_detail['step'] = info.get('current_step')
                     st.session_state.order_placement_history.append(order_detail)


            # --- Log Outcome ---
            logging.info(f"Step {info.get('current_step')} Result: Reward={reward:.6f}, Term={terminated}, Trunc={truncated}, MTM={info.get('mtm', 0.0):.2f}")

            # Rerun to update UI immediately
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred during env.step(): {e}")
            logging.exception("Error during step execution:")
    else:
        st.warning("Cannot step, episode has already ended. Please reset.")


# --- Footer ---
st.divider()
# Display the actual last action taken from session state
last_action_display = np.round(st.session_state.last_action, 3) if st.session_state.step_count > 0 else "N/A"
st.markdown(f"<small>Session Step Count: {st.session_state.step_count} | Last Action Source: {action_source} | Last Action Vector: {last_action_display} </small>", unsafe_allow_html=True)
