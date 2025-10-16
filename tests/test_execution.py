from pathlib import Path

import pytest
import numpy as np
import os
import pandas as pd
import math # Added for isnan checks

pytest.importorskip("gymnasium")

# Use stable import path
from rltrader.envs import TwoSidedMarketEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Helper to create a dummy CSV if it doesn't exist
def create_dummy_csv(path: Path | None = None, levels=5):
    if path is None:
        path = DATA_DIR / "dummy_lob_data.csv"
    path = Path(path)
    if not os.path.exists(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Creating dummy CSV at {path}")
        cols = ["timestamp"]
        for i in range(1, levels + 1):
            cols += [f"bid{i}", f"bidqty{i}", f"ask{i}", f"askqty{i}"]
        # Create one row of plausible data
        dummy_data = {
            "timestamp": [pd.Timestamp.now().isoformat()],
        }
        bid_start, ask_start = 100.0, 100.1
        qty_val = 1.0
        for i in range(1, levels + 1):
             dummy_data[f"bid{i}"] = [bid_start - (i-1)*0.1]
             dummy_data[f"bidqty{i}"] = [qty_val]
             dummy_data[f"ask{i}"] = [ask_start + (i-1)*0.1]
             dummy_data[f"askqty{i}"] = [qty_val]

        df = pd.DataFrame(dummy_data, columns=cols)
        df.to_csv(path, index=False)
    return str(path)

# Default configuration for tests
DEFAULT_CONFIG = {
    "csv_path": create_dummy_csv(), # Use helper to ensure file exists
    "initial_capital": 10000.0,
    "order_book_levels": 5,
    "max_order_volume": 10.0,
    "latency_steps_long": 0, # Simplify for execution tests
    "latency_steps_short": 0,
    "tick_size": 0.01,
    "lot_size": 0.01,
    "max_active_orders": 10,
    "inventory_penalty": 0.01,
    "transaction_cost_long": 0.0001, # 0.01%
    "transaction_cost_short": 0.0001,
    "max_inventory": 20.0,
    "invalid_order_penalty": 0.0, # Not testing placement here
    "activity_bonus": 0.0,
    "taker_penalty": 0.0005, # Penalty cost per unit volume (affects reward, not _execute_orders return)
    "price_offset_ticks": 5,
    "allowed_aggressiveness_ticks": 2,
    "quoting_reward_enabled": False, # Focus on execution
    "quoting_reward_amount": 0.0,
    "quoting_reward_max_ticks": 0,
    "explicit_cancel_enabled": True,
    "explicit_cancel_threshold": 0.0,
    "explicit_cancel_penalty": 0.0,
    "explicit_cancel_clears_pending": True,
    "do_nothing_threshold": 1.1, # Disable 'do nothing' override
    "episode_length": 100, # Not strictly needed for _execute_orders tests
    "obs_qty_norm_scale": 1.0, # Multiply normalized quantities by this factor
    "obs_price_norm_scale": 100.0, # Divide normalized prices (in ticks) by this factor
}

@pytest.fixture
def setup_env():
    """Provides a fresh two-sided market environment instance for each test."""
    config = DEFAULT_CONFIG.copy()
    env = TwoSidedMarketEnv(config)
    env.reset() # Initialize internal states
    # Set a default valid market state for convenience
    env.bids = np.array([[100.00, 5.0], [99.99, 10.0]], dtype=np.float32)
    env.asks = np.array([[100.10, 5.0], [100.11, 10.0]], dtype=np.float32)
    env.best_bid = 100.00
    env.best_ask = 100.10
    env.midprice = 100.05
    env._last_valid_midprice = 100.05
    env.cash = env.config["initial_capital"]
    env.long_position = 0.0 # Explicitly reset positions for clarity
    env.short_position = 0.0
    env.long_avg_cost = 0.0
    env.short_avg_cost = 0.0
    env.last_executed_volume = 0.0
    return env

# --- Helper to Assert Taker Cancellation ---
def assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl):
    """Asserts the state after a taker order was cancelled."""
    assert realized_pnl == 0.0 , "Realized PnL should be 0 for cancelled taker orders"
    assert env.cash == initial_cash, "Cash should not change from cancelled taker in _execute_orders"
    assert env.long_position == initial_long, "Long position should not change from cancelled taker"
    assert env.short_position == initial_short, "Short position should not change from cancelled taker"
    assert env.active_orders == [], "Active orders list should be empty after taker cancellation"
    assert env.last_executed_volume == 0.0, "Executed volume should be 0 for cancelled taker"

# --- Helper to Assert No Fill ---
def assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, order_vol, realized_pnl):
     assert realized_pnl == 0.0, "Realized PnL should be 0 for unfilled orders"
     assert env.cash == initial_cash, "Cash should not change for unfilled orders"
     assert env.long_position == initial_long, "Long position should not change for unfilled orders"
     assert env.short_position == initial_short, "Short position should not change for unfilled orders"
     assert len(env.active_orders) == 1, "Active order list should contain the one unfilled order" # Key fix here
     assert env.active_orders[0]["id"] == order_id, "The correct unfilled order ID should remain"
     assert env.active_orders[0]["volume"] == pytest.approx(order_vol), "Volume of unfilled order should be unchanged"
     assert env.last_executed_volume == 0.0, "Executed volume should be 0 for unfilled orders"

# --- Test Categories ---

# 1. Basic Buy Fills (Using MAKER prices)
def test_execute_buy_maker_full_fill_single_level(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # MAKER price (below ask 100.10)
    buy_vol = 3.0
    fill_price = 100.10 # Will fill against best ask
    # Ensure LOB allows the fill
    env.asks = np.array([[fill_price, 5.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()

    expected_cost_val = buy_vol * fill_price
    expected_tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - expected_tx_cost

    assert realized_pnl == 0.0, "PnL should be 0 when opening long"
    assert env.cash == pytest.approx(expected_cash), "Cash calculation mismatch in single level buy fill"
    assert env.long_position == pytest.approx(buy_vol), "Long position mismatch in single level buy fill"
    assert env.long_avg_cost == pytest.approx(fill_price), "Avg cost mismatch in single level buy fill"
    assert env.short_position == 0.0
    assert env.active_orders == [], "Order should be removed after full fill"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Executed volume mismatch in single level buy fill"

def test_execute_buy_maker_partial_fill_single_level(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # MAKER price
    buy_vol = 7.0 # More than available at first level
    fill_price = 100.10
    available_vol = 5.0
    # Ensure LOB allows the fill
    env.asks = np.array([[fill_price, available_vol]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    fill_vol = available_vol # Should fill 5.0

    realized_pnl = env._execute_orders()

    expected_cost_val = fill_vol * fill_price
    expected_tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - expected_tx_cost

    assert realized_pnl == 0.0, "PnL should be 0 when opening long (partial)"
    assert env.cash == pytest.approx(expected_cash), "Cash calculation mismatch in partial buy fill"
    assert env.long_position == pytest.approx(fill_vol), "Long position mismatch in partial buy fill"
    assert env.long_avg_cost == pytest.approx(fill_price), "Avg cost mismatch in partial buy fill"
    assert env.short_position == 0.0
    assert len(env.active_orders) == 1, "Order should remain after partial fill"
    assert env.active_orders[0]["id"] == 1
    assert env.active_orders[0]["volume"] == pytest.approx(buy_vol - fill_vol), "Remaining volume mismatch in partial fill"
    assert env.last_executed_volume == pytest.approx(fill_vol), "Executed volume mismatch in partial buy fill"

def test_execute_buy_maker_full_fill_multi_level(setup_env):
    env = setup_env
    buy_price_limit = 100.11 # MAKER price allowing fill up to 100.11
    buy_vol = 7.0 # Enough to take first two ask levels
    # Set LOB for multi-level fill
    price1, vol1 = 100.10, 5.0
    price2, vol2 = 100.11, 10.0
    env.asks = np.array([[price1, vol1], [price2, vol2]], dtype=np.float32)
    env.best_ask = price1
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    fill_vol1 = vol1
    fill_vol2 = buy_vol - fill_vol1 # 2.0

    realized_pnl = env._execute_orders()

    cost1 = fill_vol1 * price1
    cost2 = fill_vol2 * price2
    total_cost_val = cost1 + cost2
    tx_cost_adj = (cost1 * env.config["transaction_cost_long"]) + (cost2 * env.config["transaction_cost_long"]) # More precise
    expected_cash = initial_cash - total_cost_val - tx_cost_adj
    expected_avg_cost = total_cost_val / buy_vol

    assert realized_pnl == 0.0, "PnL should be 0 when opening long (multi-level)"
    assert env.cash == pytest.approx(expected_cash), "Cash calc mismatch multi-level buy"
    assert env.long_position == pytest.approx(buy_vol), "Position mismatch multi-level buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch multi-level buy"
    assert env.short_position == 0.0
    assert env.active_orders == [], "Order should be removed after multi-level full fill"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch multi-level buy"

def test_execute_buy_maker_partial_fill_multi_level(setup_env):
    env = setup_env
    buy_price_limit = 100.11 # MAKER price
    buy_vol = 20.0 # More than available on first two levels
    # Set LOB
    price1, vol1 = 100.10, 5.0
    price2, vol2 = 100.11, 10.0
    env.asks = np.array([[price1, vol1], [price2, vol2]], dtype=np.float32)
    env.best_ask = price1
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    fill_vol1 = vol1
    fill_vol2 = vol2
    total_filled_vol = fill_vol1 + fill_vol2 # 15.0

    realized_pnl = env._execute_orders()

    cost1 = fill_vol1 * price1
    cost2 = fill_vol2 * price2
    total_cost_val = cost1 + cost2
    tx_cost_adj = (cost1 * env.config["transaction_cost_long"]) + (cost2 * env.config["transaction_cost_long"]) # More precise
    expected_cash = initial_cash - total_cost_val - tx_cost_adj
    expected_avg_cost = total_cost_val / total_filled_vol

    assert realized_pnl == 0.0, "PnL should be 0 opening long (multi-level partial)"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch multi-level partial buy"
    assert env.long_position == pytest.approx(total_filled_vol), "Position mismatch multi-level partial buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch multi-level partial buy"
    assert len(env.active_orders) == 1, "Order should remain after multi-level partial buy"
    assert env.active_orders[0]["volume"] == pytest.approx(buy_vol - total_filled_vol), "Remaining vol mismatch multi-level partial buy"
    assert env.last_executed_volume == pytest.approx(total_filled_vol), "Exec vol mismatch multi-level partial buy"


# 2. Basic Sell Fills (Using MAKER prices)
def test_execute_sell_maker_full_fill_single_level(setup_env):
    env = setup_env
    sell_price_limit = 100.01 # MAKER price (above bid 100.00)
    sell_vol = 3.0
    fill_price = 100.00 # Will fill against best bid
    # Ensure LOB allows the fill
    env.bids = np.array([[fill_price, 5.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]

    # Pre-load long position to realize PnL
    initial_long_vol = 5.0
    initial_long_cost = 99.90
    env.long_position = initial_long_vol
    env.long_avg_cost = initial_long_cost
    initial_cash = env.cash # Get cash *after* setting position if relevant

    realized_pnl = env._execute_orders()

    expected_revenue_val = sell_vol * fill_price
    expected_tx_cost = expected_revenue_val * env.config["transaction_cost_short"]
    # PnL calculation is based on the fill price vs avg cost
    expected_fill_pnl = (fill_price - initial_long_cost) * sell_vol
    expected_cash = initial_cash + expected_revenue_val - expected_tx_cost

    assert realized_pnl == pytest.approx(expected_fill_pnl), "Realized PnL mismatch closing long"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch closing long"
    assert env.long_position == pytest.approx(initial_long_vol - sell_vol), "Long pos reduction mismatch"
    assert env.long_avg_cost == pytest.approx(initial_long_cost), "Long avg cost should be unchanged closing"
    assert env.short_position == 0.0
    assert env.active_orders == [], "Order should be removed after full fill closing long"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch closing long"

def test_execute_sell_maker_partial_fill_single_level(setup_env):
    env = setup_env
    sell_price_limit = 100.01 # MAKER price
    sell_vol = 7.0 # More than available at first level
    fill_price = 100.00
    available_vol = 5.0
    # Ensure LOB allows the fill
    env.bids = np.array([[fill_price, available_vol]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    fill_vol = available_vol # Should fill 5.0

    realized_pnl = env._execute_orders() # Goes directly short

    expected_revenue_val = fill_vol * fill_price
    expected_tx_cost = expected_revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_revenue_val - expected_tx_cost

    assert realized_pnl == 0.0, "PnL should be 0 when opening short"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch opening short partial"
    assert env.long_position == 0.0
    assert env.short_position == pytest.approx(fill_vol), "Short pos mismatch opening short partial"
    assert env.short_avg_cost == pytest.approx(fill_price), "Short avg cost mismatch opening short partial"
    assert len(env.active_orders) == 1, "Order should remain opening short partial"
    assert env.active_orders[0]["volume"] == pytest.approx(sell_vol - fill_vol), "Remaining vol mismatch opening short partial"
    assert env.last_executed_volume == pytest.approx(fill_vol), "Exec vol mismatch opening short partial"

def test_execute_sell_maker_full_fill_multi_level(setup_env):
    env = setup_env
    sell_price_limit = 99.99 # MAKER price allowing fill down to 99.99
    sell_vol = 7.0 # Enough to take first two bid levels
    # Set LOB
    price1, vol1 = 100.00, 5.0
    price2, vol2 = 99.99, 10.0
    env.bids = np.array([[price1, vol1], [price2, vol2]], dtype=np.float32)
    env.best_bid = price1
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    fill_vol1 = vol1
    fill_vol2 = sell_vol - fill_vol1 # 2.0

    realized_pnl = env._execute_orders() # Goes short

    rev1 = fill_vol1 * price1
    rev2 = fill_vol2 * price2
    total_rev_val = rev1 + rev2
    tx_cost_adj = (rev1 * env.config["transaction_cost_short"]) + (rev2 * env.config["transaction_cost_short"]) # More precise
    expected_cash = initial_cash + total_rev_val - tx_cost_adj
    expected_avg_cost = total_rev_val / sell_vol # Avg price received

    assert realized_pnl == 0.0, "PnL should be 0 opening short multi-level"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch opening short multi-level"
    assert env.short_position == pytest.approx(sell_vol), "Short pos mismatch opening short multi-level"
    assert env.short_avg_cost == pytest.approx(expected_avg_cost), "Short avg cost mismatch opening short multi-level"
    assert env.long_position == 0.0
    assert env.active_orders == [], "Order should be removed opening short multi-level"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch opening short multi-level"

def test_execute_sell_maker_partial_fill_multi_level_realizing_pnl(setup_env):
    env = setup_env
    sell_price_limit = 99.99 # MAKER price
    sell_vol = 20.0 # More than available on first two levels
    # Set LOB
    price1, vol1 = 100.00, 5.0
    price2, vol2 = 99.99, 10.0
    env.bids = np.array([[price1, vol1], [price2, vol2]], dtype=np.float32)
    env.best_bid = price1
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]

    # Pre-load long position
    initial_long_vol = 10.0
    initial_long_cost = 99.95
    env.long_position = initial_long_vol
    env.long_avg_cost = initial_long_cost
    initial_cash = env.cash

    fill_vol1 = vol1 # 5.0
    fill_vol2 = vol2 # 10.0
    total_filled_vol = fill_vol1 + fill_vol2 # 15.0

    realized_pnl = env._execute_orders()

    # Execution breakdown:
    # Fill 1: 5.0 @ 100.00 (closes 5.0 long)
    # Fill 2: 10.0 @ 99.99 (closes remaining 5.0 long, opens 5.0 short)
    fill1_long_close_vol = 5.0
    fill2_long_close_vol = 5.0
    fill2_short_open_vol = 5.0

    # PnL from fills closing the long position
    pnl_fill1 = (price1 - initial_long_cost) * fill1_long_close_vol
    pnl_fill2 = (price2 - initial_long_cost) * fill2_long_close_vol
    expected_fill_pnl = pnl_fill1 + pnl_fill2

    rev1 = fill_vol1 * price1
    rev2 = fill_vol2 * price2
    total_rev_val = rev1 + rev2
    tx_cost_adj = (rev1 * env.config["transaction_cost_short"]) + (rev2 * env.config["transaction_cost_short"]) # More precise
    expected_cash = initial_cash + total_rev_val - tx_cost_adj

    assert realized_pnl == pytest.approx(expected_fill_pnl), "PnL mismatch realizing partial multi-level"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch realizing partial multi-level"
    assert env.long_position == 0.0, "Long pos should be 0 after realizing partial multi-level"
    assert env.short_position == pytest.approx(fill2_short_open_vol), "Short pos mismatch realizing partial multi-level"
    assert env.short_avg_cost == pytest.approx(price2), "Short avg cost mismatch realizing partial multi-level"
    assert len(env.active_orders) == 1, "Order should remain realizing partial multi-level"
    assert env.active_orders[0]["volume"] == pytest.approx(sell_vol - total_filled_vol), "Remaining vol mismatch realizing partial multi-level"
    assert env.last_executed_volume == pytest.approx(total_filled_vol), "Exec vol mismatch realizing partial multi-level"


# 3. No Fills
def test_execute_buy_no_fill_price_too_low(setup_env):
    env = setup_env
    buy_price = 100.09 # Below best ask 100.10
    buy_vol = 3.0
    order_id = 1
    # LOB state from fixture is sufficient (ask at 100.10)
    env.active_orders = [{"id": order_id, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, buy_vol, realized_pnl)


def test_execute_sell_no_fill_price_too_high(setup_env):
    env = setup_env
    sell_price = 100.01 # Above best bid 100.00
    sell_vol = 3.0
    order_id = 1
    # LOB state from fixture is sufficient (bid at 100.00)
    env.active_orders = [{"id": order_id, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, sell_vol, realized_pnl)


def test_execute_buy_no_fill_empty_ask_book(setup_env):
    env = setup_env
    env.asks = np.array([], dtype=np.float32).reshape(0, 2) # Empty asks
    env.best_ask = np.inf # Or some indicator of no ask
    buy_price = 100.10 # Doesn't matter, no asks
    buy_vol = 3.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, buy_vol, realized_pnl)


def test_execute_sell_no_fill_empty_bid_book(setup_env):
    env = setup_env
    env.bids = np.array([], dtype=np.float32).reshape(0, 2) # Empty bids
    env.best_bid = -np.inf
    sell_price = 100.00
    sell_vol = 3.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, sell_vol, realized_pnl)

def test_execute_buy_no_fill_zero_qty_levels(setup_env):
    # Order should remain if it only encounters zero Qty levels it *could* have filled at
    env = setup_env
    env.asks = np.array([[100.10, 0.0], [100.11, 0.0]], dtype=np.float32) # Zero qty
    env.best_ask = 100.10
    buy_price = 100.11 # Price would allow fill if qty existed
    buy_vol = 3.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    # Use assert_no_fill which checks len(active_orders) == 1
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, buy_vol, realized_pnl)


def test_execute_sell_no_fill_zero_qty_levels(setup_env):
    env = setup_env
    env.bids = np.array([[100.00, 0.0], [99.99, 0.0]], dtype=np.float32) # Zero qty
    env.best_bid = 100.00
    sell_price = 99.99 # Price allows fill
    sell_vol = 3.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    # Use assert_no_fill which checks len(active_orders) == 1
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, sell_vol, realized_pnl)

# 4. Taker Penalties (Testing the cancellation effect)
def test_execute_taker_buy_at_ask_penalty_applied_order_cancelled(setup_env):
    env = setup_env
    buy_price = 100.10 # Exactly at best ask -> TAKER
    buy_vol = 3.0
    env.active_orders = [{"id": 1, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_taker_buy_above_ask_penalty_applied_order_cancelled(setup_env):
    env = setup_env
    buy_price = 100.11 # Above best ask -> TAKER
    buy_vol = 3.0
    env.active_orders = [{"id": 1, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_taker_sell_at_bid_penalty_applied_order_cancelled(setup_env):
    env = setup_env
    sell_price = 100.00 # Exactly at best bid -> TAKER
    sell_vol = 3.0
    env.active_orders = [{"id": 1, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_taker_sell_below_bid_penalty_applied_order_cancelled(setup_env):
    env = setup_env
    sell_price = 99.99 # Below best bid -> TAKER
    sell_vol = 3.0
    env.active_orders = [{"id": 1, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_mixed_maker_taker_orders(setup_env):
    env = setup_env
    maker_buy_price = 100.09 # MAKER (Below ask 100.10) -> Won't fill
    maker_buy_vol = 2.0
    taker_sell_price = 100.00 # TAKER (At bid 100.00) -> Cancelled
    taker_sell_vol = 1.0
    maker_sell_price = 100.01 # MAKER (Above bid 100.00) -> Won't fill
    maker_sell_vol = 2.0

    env.active_orders = [
        {"id": 1, "price": maker_buy_price, "volume": maker_buy_vol, "is_buy": True, "timestamp_placed": 0},
        {"id": 2, "price": taker_sell_price, "volume": taker_sell_vol, "is_buy": False, "timestamp_placed": 0}, # TAKER
        {"id": 3, "price": maker_sell_price, "volume": maker_sell_vol, "is_buy": False, "timestamp_placed": 0},
    ]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()

    assert realized_pnl == 0.0
    assert env.cash == initial_cash # No change from execution logic
    assert env.long_position == initial_long
    assert env.short_position == initial_short
    assert len(env.active_orders) == 2, "Only maker orders should remain"
    active_order_ids = {o['id'] for o in env.active_orders}
    assert active_order_ids == {1, 3}, "Incorrect maker orders remained"
    assert env.last_executed_volume == 0.0

def test_execute_zero_taker_penalty_still_cancels_taker(setup_env):
    env = setup_env
    env.config["taker_penalty"] = 0.0 # Disable penalty value
    buy_price = 100.10 # Taker price (at ask)
    buy_vol = 3.0
    env.active_orders = [{"id": 1, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)


# 5. Inventory Limits
def test_execute_buy_blocked_by_max_inventory(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    env.long_position = max_inv # Already at max inventory
    env.long_avg_cost = 100.0
    buy_price_limit = 100.09 # MAKER price
    buy_vol = 5.0
    order_id = 1
    # Setup LOB so it *would* fill if not for inventory limit
    fill_price = 100.10
    env.asks = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": order_id, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position # Should be max_inv
    initial_short = env.short_position

    realized_pnl = env._execute_orders()

    # Order should remain if fill attempt was blocked by inventory check
    assert realized_pnl == 0.0, "PnL should be 0 if fill blocked by inventory"
    assert env.cash == initial_cash, "Cash should be unchanged if fill blocked by inventory"
    assert env.long_position == initial_long, "Long pos should be unchanged if fill blocked by inventory"
    assert env.short_position == initial_short
    assert len(env.active_orders) == 1, "Order should remain if fill blocked by inventory"
    assert env.active_orders[0]['id'] == order_id
    assert env.active_orders[0]['volume'] == buy_vol
    assert env.last_executed_volume == 0.0, "Exec vol should be 0 if fill blocked by inventory"

def test_execute_buy_partially_blocked_by_max_inventory(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    current_inv = 18.0
    env.long_position = current_inv
    env.long_avg_cost = 100.0
    buy_price_limit = 100.09 # MAKER price
    buy_vol = 5.0 # Wants to buy 5, only 2 allowed (20 - 18)
    fill_price = 100.10
    env.asks = np.array([[fill_price, 10.0]], dtype=np.float32) # Enough liquidity
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    allowed_fill_vol = max_inv - current_inv # 2.0

    realized_pnl = env._execute_orders()

    expected_cost_val = allowed_fill_vol * fill_price
    expected_tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - expected_tx_cost
    expected_avg_cost = ((current_inv * env.long_avg_cost) + (allowed_fill_vol * fill_price)) / max_inv

    assert realized_pnl == 0.0, "PnL mismatch partial inventory block buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch partial inventory block buy"
    assert env.long_position == pytest.approx(max_inv), "Position mismatch partial inventory block buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch partial inventory block buy"
    assert len(env.active_orders) == 1, "Order should remain partial inventory block buy"
    assert env.active_orders[0]["volume"] == pytest.approx(buy_vol - allowed_fill_vol), "Remaining vol mismatch partial inventory block buy"
    assert env.last_executed_volume == pytest.approx(allowed_fill_vol), "Exec vol mismatch partial inventory block buy"

def test_execute_sell_blocked_by_min_inventory(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    env.short_position = max_inv # Already at max short inventory (-20 effective)
    env.short_avg_cost = 100.0
    sell_price_limit = 100.01 # MAKER price
    sell_vol = 5.0
    order_id = 1
    # Setup LOB so it *would* fill
    fill_price = 100.00
    env.bids = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": order_id, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position # Should be max_inv

    realized_pnl = env._execute_orders()

    # Order should remain if fill attempt was blocked by inventory check
    assert realized_pnl == 0.0, "PnL mismatch full inventory block sell"
    assert env.cash == initial_cash, "Cash mismatch full inventory block sell"
    assert env.long_position == initial_long
    assert env.short_position == initial_short, "Short pos mismatch full inventory block sell"
    assert len(env.active_orders) == 1, "Order should remain full inventory block sell"
    assert env.active_orders[0]['id'] == order_id
    assert env.active_orders[0]['volume'] == sell_vol
    assert env.last_executed_volume == 0.0, "Exec vol mismatch full inventory block sell"

def test_execute_sell_partially_blocked_by_min_inventory(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    current_net_inv = -18.0
    env.short_position = abs(current_net_inv)
    env.short_avg_cost = 100.0
    sell_price_limit = 100.01 # MAKER price
    sell_vol = 5.0 # Wants to sell 5, only 2 allowed (-18 -> -20)
    fill_price = 100.00
    env.bids = np.array([[fill_price, 10.0]], dtype=np.float32) # Enough liquidity
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    allowed_fill_vol = max_inv - abs(current_net_inv) # 2.0

    realized_pnl = env._execute_orders()

    expected_revenue_val = allowed_fill_vol * fill_price
    expected_tx_cost = expected_revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_revenue_val - expected_tx_cost
    expected_avg_cost = ((abs(current_net_inv) * env.short_avg_cost) + (allowed_fill_vol * fill_price)) / max_inv

    assert realized_pnl == 0.0, "PnL mismatch partial inventory block sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch partial inventory block sell"
    assert env.short_position == pytest.approx(max_inv), "Short pos mismatch partial inventory block sell"
    assert env.short_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch partial inventory block sell"
    assert len(env.active_orders) == 1, "Order should remain partial inventory block sell"
    assert env.active_orders[0]["volume"] == pytest.approx(sell_vol - allowed_fill_vol), "Rem vol mismatch partial inventory block sell"
    assert env.last_executed_volume == pytest.approx(allowed_fill_vol), "Exec vol mismatch partial inventory block sell"

def test_execute_buy_hits_inventory_limit_exactly(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    current_inv = 18.0
    env.long_position = current_inv
    env.long_avg_cost = 100.0
    buy_price_limit = 100.09 # MAKER price
    buy_vol = 2.0 # Exactly hits the limit
    fill_price = 100.10
    env.asks = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    allowed_fill_vol = buy_vol

    realized_pnl = env._execute_orders()

    expected_cost_val = allowed_fill_vol * fill_price
    expected_tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - expected_tx_cost
    expected_avg_cost = ((current_inv * env.long_avg_cost) + (allowed_fill_vol * fill_price)) / max_inv

    assert realized_pnl == 0.0, "PnL mismatch exact inventory limit buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch exact inventory limit buy"
    assert env.long_position == pytest.approx(max_inv), "Position mismatch exact inventory limit buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch exact inventory limit buy"
    assert env.active_orders == [], "Order should be gone exact inventory limit buy"
    assert env.last_executed_volume == pytest.approx(allowed_fill_vol), "Exec vol mismatch exact inventory limit buy"

def test_execute_sell_hits_inventory_limit_exactly(setup_env):
    env = setup_env
    max_inv = env.config["max_inventory"] # 20.0
    current_net_inv = -18.0
    env.short_position = abs(current_net_inv)
    env.short_avg_cost = 100.0
    sell_price_limit = 100.01 # MAKER price
    sell_vol = 2.0 # Exactly hits the limit
    fill_price = 100.00
    env.bids = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    allowed_fill_vol = sell_vol

    realized_pnl = env._execute_orders()

    expected_revenue_val = allowed_fill_vol * fill_price
    expected_tx_cost = expected_revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_revenue_val - expected_tx_cost
    expected_avg_cost = ((abs(current_net_inv) * env.short_avg_cost) + (allowed_fill_vol * fill_price)) / max_inv

    assert realized_pnl == 0.0, "PnL mismatch exact inventory limit sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch exact inventory limit sell"
    assert env.short_position == pytest.approx(max_inv), "Short pos mismatch exact inventory limit sell"
    assert env.short_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch exact inventory limit sell"
    assert env.active_orders == [], "Order should be gone exact inventory limit sell"
    assert env.last_executed_volume == pytest.approx(allowed_fill_vol), "Exec vol mismatch exact inventory limit sell"


# 6. Edge Cases & Configuration Variations
def test_execute_zero_transaction_cost(setup_env):
    env = setup_env
    env.config["transaction_cost_long"] = 0.0 # Disable cost
    buy_price_limit = 100.09 # MAKER price
    buy_vol = 3.0
    fill_price = 100.10
    env.asks = np.array([[fill_price, 5.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()

    expected_cost_val = buy_vol * fill_price
    expected_cash = initial_cash - expected_cost_val # No tx cost

    assert realized_pnl == 0.0, "PnL mismatch zero tx cost"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch zero tx cost"
    assert env.long_position == pytest.approx(buy_vol), "Position mismatch zero tx cost"
    assert env.active_orders == [], "Order should be gone zero tx cost"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch zero tx cost"

def test_execute_multiple_buy_orders_filled(setup_env):
    # Test if orders are processed correctly against depleting book levels
    env = setup_env
    buy_price_limit1 = 100.10 # Allows fill up to 100.10
    buy_vol1 = 2.0
    buy_price_limit2 = 100.11 # Allows fill up to 100.11
    buy_vol2 = 4.0
    # Setup LOB
    price_a1, vol_a1 = 100.10, 3.0
    price_a2, vol_a2 = 100.11, 5.0
    env.asks = np.array([[price_a1, vol_a1], [price_a2, vol_a2]], dtype=np.float32)
    env.best_ask = price_a1
    # Orders: Assumes order processing preserves list order for this test
    env.active_orders = [
        {"id": 1, "price": buy_price_limit1, "volume": buy_vol1, "is_buy": True, "timestamp_placed": 0},
        {"id": 2, "price": buy_price_limit2, "volume": buy_vol2, "is_buy": True, "timestamp_placed": 1},
    ]
    initial_cash = env.cash

    # Expected Execution:
    # Order 1 (id=1): price=100.10, vol=2.0. Fills 2.0 @ 100.10. LOB ask[0] becomes (100.10, 1.0)
    # Order 2 (id=2): price=100.11, vol=4.0. Fills 1.0 @ 100.10 (remaining). Fills 3.0 @ 100.11.
    fill1_vol, fill1_price = 2.0, 100.10
    fill2a_vol, fill2a_price = 1.0, 100.10
    fill2b_vol, fill2b_price = 3.0, 100.11
    total_vol_filled = fill1_vol + fill2a_vol + fill2b_vol # 6.0

    realized_pnl = env._execute_orders()

    cost1 = fill1_vol * fill1_price
    cost2a = fill2a_vol * fill2a_price
    cost2b = fill2b_vol * fill2b_price
    total_cost_val = cost1 + cost2a + cost2b
    tx_cost_adj = (cost1 * env.config["transaction_cost_long"]) + \
                  (cost2a * env.config["transaction_cost_long"]) + \
                  (cost2b * env.config["transaction_cost_long"]) # Per fill tx cost
    expected_cash = initial_cash - total_cost_val - tx_cost_adj

    # Avg cost calculation depends on _process_fill calls being sequential
    # Assume env calculates it correctly, just check final state
    expected_avg_cost = total_cost_val / total_vol_filled

    assert realized_pnl == 0.0, "PnL mismatch multi-order buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch multi-order buy"
    assert env.long_position == pytest.approx(total_vol_filled), "Position mismatch multi-order buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch multi-order buy"
    assert env.active_orders == [], "Orders should be gone multi-order buy"
    assert env.last_executed_volume == pytest.approx(total_vol_filled), "Exec vol mismatch multi-order buy"

def test_execute_multiple_sell_orders_partially_filled(setup_env):
    env = setup_env
    sell_price_limit1 = 100.00 # Allows fill at 100.00
    sell_vol1 = 4.0
    sell_price_limit2 = 99.99 # Allows fill down to 99.99
    sell_vol2 = 4.0
    # Set LOB
    price_b1, vol_b1 = 100.00, 5.0
    price_b2, vol_b2 = 99.99, 2.0 # Total 7.0 available
    env.bids = np.array([[price_b1, vol_b1], [price_b2, vol_b2]], dtype=np.float32)
    env.best_bid = price_b1
    env.active_orders = [
        {"id": 1, "price": sell_price_limit1, "volume": sell_vol1, "is_buy": False, "timestamp_placed": 0},
        {"id": 2, "price": sell_price_limit2, "volume": sell_vol2, "is_buy": False, "timestamp_placed": 1},
    ]
    initial_cash = env.cash

    # Expected Execution:
    # Order 1 (id=1): price=100.00, vol=4.0. Fills 4.0 @ 100.00. LOB bid[0] becomes (100.00, 1.0)
    # Order 2 (id=2): price=99.99, vol=4.0. Fills 1.0 @ 100.00 (remaining). Fills 2.0 @ 99.99. Total fill = 3.0. Remaining vol = 1.0
    fill1_vol, fill1_price = 4.0, 100.00
    fill2a_vol, fill2a_price = 1.0, 100.00
    fill2b_vol, fill2b_price = 2.0, 99.99
    order2_filled_vol = fill2a_vol + fill2b_vol # 3.0
    total_vol_filled = fill1_vol + order2_filled_vol # 7.0

    realized_pnl = env._execute_orders()

    rev1 = fill1_vol * fill1_price
    rev2a = fill2a_vol * fill2a_price
    rev2b = fill2b_vol * fill2b_price
    total_rev_val = rev1 + rev2a + rev2b
    tx_cost_adj = (rev1 * env.config["transaction_cost_short"]) + \
                  (rev2a * env.config["transaction_cost_short"]) + \
                  (rev2b * env.config["transaction_cost_short"]) # Per fill tx cost
    expected_cash = initial_cash + total_rev_val - tx_cost_adj
    # Avg cost calculation:
    expected_avg_cost = total_rev_val / total_vol_filled

    assert realized_pnl == 0.0, "PnL mismatch multi-order sell partial"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch multi-order sell partial"
    assert env.short_position == pytest.approx(total_vol_filled), "Short pos mismatch multi-order sell partial"
    assert env.short_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch multi-order sell partial"
    assert len(env.active_orders) == 1, "Incorrect remaining orders multi-order sell partial"
    assert env.active_orders[0]["id"] == 2, "Incorrect remaining order ID multi-order sell partial"
    assert env.active_orders[0]["volume"] == pytest.approx(sell_vol2 - order2_filled_vol), "Remaining vol mismatch multi-order sell partial"
    assert env.last_executed_volume == pytest.approx(total_vol_filled), "Exec vol mismatch multi-order sell partial"

def test_execute_buy_fill_against_crossed_book_uses_ask_MAKER(setup_env):
    # If buy order is MAKER relative to ask (e.g. below ask) it should still fill if book crossed
    env = setup_env
    # Simulate a crossed book state (Bid >= Ask)
    bid_p, bid_v = 100.15, 5.0
    ask_p, ask_v = 100.10, 5.0 # Ask is lower
    env.bids = np.array([[bid_p, bid_v]], dtype=np.float32)
    env.asks = np.array([[ask_p, ask_v]], dtype=np.float32)
    env.best_bid = bid_p
    env.best_ask = ask_p
    env.midprice = (bid_p + ask_p) / 2.0

    buy_price_limit = 100.09 # MAKER price (below the low ask)
    buy_vol = 3.0
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    # Execution should still happen against the ask price (100.10) as it's the best available fill
    fill_price = env.best_ask # 100.10 -- Fix typo here
    realized_pnl = env._execute_orders()

    expected_cost_val = buy_vol * fill_price # Use the actual fill_price
    expected_tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - expected_tx_cost

    assert realized_pnl == 0.0, "PnL mismatch crossed book maker buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch crossed book maker buy"
    assert env.long_position == pytest.approx(buy_vol), "Position mismatch crossed book maker buy"
    assert env.long_avg_cost == pytest.approx(fill_price), "Avg cost mismatch crossed book maker buy"
    assert env.active_orders == [], "Order should be gone crossed book maker buy"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch crossed book maker buy"

def test_execute_sell_fill_against_crossed_book_uses_bid_MAKER(setup_env):
    # If sell order is MAKER relative to bid (e.g. above bid) it should still fill if book crossed
    env = setup_env
    bid_p, bid_v = 100.15, 5.0 # Bid is higher
    ask_p, ask_v = 100.10, 5.0
    env.bids = np.array([[bid_p, bid_v]], dtype=np.float32)
    env.asks = np.array([[ask_p, ask_v]], dtype=np.float32)
    env.best_bid = bid_p
    env.best_ask = ask_p
    env.midprice = (bid_p + ask_p) / 2.0

    sell_price_limit = 100.16 # MAKER price (above the high bid)
    sell_vol = 3.0
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash

    # Execution should still happen against the bid price (100.15)
    fill_price = env.best_bid # 100.15
    realized_pnl = env._execute_orders()

    expected_revenue_val = sell_vol * fill_price
    expected_tx_cost = expected_revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_revenue_val - expected_tx_cost

    assert realized_pnl == 0.0, "PnL mismatch crossed book maker sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch crossed book maker sell"
    assert env.short_position == pytest.approx(sell_vol), "Short pos mismatch crossed book maker sell"
    assert env.short_avg_cost == pytest.approx(fill_price), "Avg cost mismatch crossed book maker sell"
    assert env.active_orders == [], "Order should be gone crossed book maker sell"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch crossed book maker sell"

# Taker check tests remain valid as they check the cancellation logic based on price vs BBO
def test_execute_buy_taker_check_uses_valid_bbo(setup_env):
    env = setup_env
    env.best_bid = 100.00; env.best_ask = 100.10 # Standard BBO
    buy_price = 100.10 # Taker price
    buy_vol = 1.0
    env.active_orders = [{"id": 1, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_sell_taker_check_uses_valid_bbo(setup_env):
    env = setup_env
    env.best_bid = 100.00; env.best_ask = 100.10 # Standard BBO
    sell_price = 100.00 # Taker price
    sell_vol = 1.0
    env.active_orders = [{"id": 1, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_realized_pnl_buy_closes_short(setup_env):
    env = setup_env
    initial_short_vol = 5.0
    initial_short_cost = 100.20
    env.short_position = initial_short_vol
    env.short_avg_cost = initial_short_cost
    initial_cash = env.cash

    buy_price_limit = 100.09 # MAKER price to ensure fill
    fill_price = 100.10 # Fill against this ask
    buy_vol = 3.0 # Close part of the short
    env.asks = np.array([[fill_price, 5.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]

    realized_pnl = env._execute_orders()

    # PnL only from the part closing the short position
    expected_fill_pnl = (initial_short_cost - fill_price) * buy_vol
    cost_val = buy_vol * fill_price
    tx_cost = cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - cost_val - tx_cost

    assert realized_pnl == pytest.approx(expected_fill_pnl), "PnL mismatch buy closes short"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch buy closes short"
    assert env.short_position == pytest.approx(initial_short_vol - buy_vol), "Short pos mismatch buy closes short"
    assert env.short_avg_cost == pytest.approx(initial_short_cost), "Short avg cost mismatch buy closes short"
    assert env.long_position == 0.0
    assert env.active_orders == [], "Order should be gone buy closes short"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch buy closes short"

def test_execute_realized_pnl_sell_closes_long(setup_env):
    env = setup_env
    initial_long_vol = 5.0
    initial_long_cost = 99.90
    env.long_position = initial_long_vol
    env.long_avg_cost = initial_long_cost
    initial_cash = env.cash

    sell_price_limit = 100.01 # MAKER price
    fill_price = 100.00 # Fill against this bid
    sell_vol = 3.0 # Close part of the long
    env.bids = np.array([[fill_price, 5.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]

    realized_pnl = env._execute_orders()

    # PnL only from the part closing the long position
    expected_fill_pnl = (fill_price - initial_long_cost) * sell_vol
    revenue_val = sell_vol * fill_price
    tx_cost = revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + revenue_val - tx_cost

    assert realized_pnl == pytest.approx(expected_fill_pnl), "PnL mismatch sell closes long"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch sell closes long"
    assert env.long_position == pytest.approx(initial_long_vol - sell_vol), "Long pos mismatch sell closes long"
    assert env.long_avg_cost == pytest.approx(initial_long_cost), "Long avg cost mismatch sell closes long"
    assert env.short_position == 0.0
    assert env.active_orders == [], "Order should be gone sell closes long"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch sell closes long"

def test_execute_buy_fill_clears_short_opens_long(setup_env):
    env = setup_env
    initial_short_vol = 2.0
    initial_short_cost = 100.20
    env.short_position = initial_short_vol
    env.short_avg_cost = initial_short_cost
    initial_cash = env.cash

    buy_price_limit = 100.09 # MAKER price
    fill_price = 100.10 # Fill against ask
    buy_vol = 5.0 # Buy more than needed to cover short
    env.asks = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    cover_vol = initial_short_vol # 2.0
    open_long_vol = buy_vol - cover_vol # 3.0

    realized_pnl = env._execute_orders()

    # PnL only from closing the short position
    expected_fill_pnl = (initial_short_cost - fill_price) * cover_vol
    cost_val = buy_vol * fill_price
    tx_cost = cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - cost_val - tx_cost

    assert realized_pnl == pytest.approx(expected_fill_pnl), "PnL mismatch clear short open long"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch clear short open long"
    assert env.short_position == 0.0, "Short pos should be 0 clear short open long"
    assert env.short_avg_cost == 0.0
    assert env.long_position == pytest.approx(open_long_vol), "Long pos mismatch clear short open long"
    assert env.long_avg_cost == pytest.approx(fill_price), "Avg cost mismatch clear short open long"
    assert env.active_orders == [], "Order should be gone clear short open long"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch clear short open long"

def test_execute_sell_fill_clears_long_opens_short(setup_env):
    env = setup_env
    initial_long_vol = 2.0
    initial_long_cost = 99.90
    env.long_position = initial_long_vol
    env.long_avg_cost = initial_long_cost
    initial_cash = env.cash

    sell_price_limit = 100.01 # MAKER price
    fill_price = 100.00 # Fill against bid
    sell_vol = 5.0 # Sell more than needed to close long
    env.bids = np.array([[fill_price, 10.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    close_vol = initial_long_vol # 2.0
    open_short_vol = sell_vol - close_vol # 3.0

    realized_pnl = env._execute_orders()

    # PnL only from closing the long position
    expected_fill_pnl = (fill_price - initial_long_cost) * close_vol
    revenue_val = sell_vol * fill_price
    tx_cost = revenue_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + revenue_val - tx_cost

    assert realized_pnl == pytest.approx(expected_fill_pnl), "PnL mismatch clear long open short"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch clear long open short"
    assert env.long_position == 0.0, "Long pos should be 0 clear long open short"
    assert env.long_avg_cost == 0.0
    assert env.short_position == pytest.approx(open_short_vol), "Short pos mismatch clear long open short"
    assert env.short_avg_cost == pytest.approx(fill_price), "Avg cost mismatch clear long open short"
    assert env.active_orders == [], "Order should be gone clear long open short"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch clear long open short"

def test_execute_buy_maker_fill_exact_level_qty(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # Maker
    fill_price = 100.10
    level_qty = 5.0
    buy_vol = level_qty # Exactly matches level 1 qty
    env.asks = np.array([[fill_price, level_qty]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()
    expected_cost_val = buy_vol * fill_price
    tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch exact level buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch exact level buy"
    assert env.long_position == pytest.approx(buy_vol), "Position mismatch exact level buy"
    assert env.active_orders == [], "Order should be gone exact level buy"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch exact level buy"

def test_execute_sell_maker_fill_exact_level_qty(setup_env):
    env = setup_env
    sell_price_limit = 100.01 # Maker
    fill_price = 100.00
    level_qty = 5.0
    sell_vol = level_qty
    env.bids = np.array([[fill_price, level_qty]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()
    expected_rev_val = sell_vol * fill_price
    tx_cost = expected_rev_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_rev_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch exact level sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch exact level sell"
    assert env.short_position == pytest.approx(sell_vol), "Position mismatch exact level sell"
    assert env.active_orders == [], "Order should be gone exact level sell"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch exact level sell"

def test_execute_buy_maker_priced_deep_fills_correct_level(setup_env):
    env = setup_env
    buy_price_limit = 100.12 # MAKER price allowing fill up to 100.12
    # LOB setup
    price_a1, vol_a1 = 100.10, 2.0
    price_a2, vol_a2 = 100.11, 3.0
    price_a3, vol_a3 = 100.12, 5.0
    env.asks = np.array([[price_a1, vol_a1], [price_a2, vol_a2], [price_a3, vol_a3]], dtype=np.float32)
    env.best_ask = price_a1
    buy_vol = 6.0 # Takes level 1 (2.0), level 2 (3.0), and 1.0 from level 3
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    fill1_vol, fill1_price = 2.0, price_a1
    fill2_vol, fill2_price = 3.0, price_a2
    fill3_vol, fill3_price = 1.0, price_a3
    total_filled = fill1_vol + fill2_vol + fill3_vol

    realized_pnl = env._execute_orders()

    cost1 = fill1_vol * fill1_price
    cost2 = fill2_vol * fill2_price
    cost3 = fill3_vol * fill3_price
    total_cost_val = cost1 + cost2 + cost3
    tx_cost_adj = (cost1 * env.config["transaction_cost_long"]) + \
                  (cost2 * env.config["transaction_cost_long"]) + \
                  (cost3 * env.config["transaction_cost_long"])
    expected_cash = initial_cash - total_cost_val - tx_cost_adj
    expected_avg_cost = total_cost_val / total_filled

    assert realized_pnl == 0.0, "PnL mismatch deep buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch deep buy"
    assert env.long_position == pytest.approx(total_filled), "Position mismatch deep buy"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch deep buy"
    assert env.active_orders == [], "Order should be gone deep buy"
    assert env.last_executed_volume == pytest.approx(total_filled), "Exec vol mismatch deep buy"

def test_execute_sell_maker_priced_deep_fills_correct_level(setup_env):
    env = setup_env
    sell_price_limit = 99.98 # MAKER price allowing fill down to 99.98
    # LOB setup
    price_b1, vol_b1 = 100.00, 2.0
    price_b2, vol_b2 = 99.99, 3.0
    price_b3, vol_b3 = 99.98, 5.0
    env.bids = np.array([[price_b1, vol_b1], [price_b2, vol_b2], [price_b3, vol_b3]], dtype=np.float32)
    env.best_bid = price_b1
    sell_vol = 6.0 # Takes level 1 (2.0), level 2 (3.0), and 1.0 from level 3
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash

    fill1_vol, fill1_price = 2.0, price_b1
    fill2_vol, fill2_price = 3.0, price_b2
    fill3_vol, fill3_price = 1.0, price_b3
    total_filled = fill1_vol + fill2_vol + fill3_vol

    realized_pnl = env._execute_orders()

    rev1 = fill1_vol * fill1_price
    rev2 = fill2_vol * fill2_price
    rev3 = fill3_vol * fill3_price
    total_rev_val = rev1 + rev2 + rev3
    tx_cost_adj = (rev1 * env.config["transaction_cost_short"]) + \
                  (rev2 * env.config["transaction_cost_short"]) + \
                  (rev3 * env.config["transaction_cost_short"])
    expected_cash = initial_cash + total_rev_val - tx_cost_adj
    expected_avg_cost = total_rev_val / total_filled

    assert realized_pnl == 0.0, "PnL mismatch deep sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch deep sell"
    assert env.short_position == pytest.approx(total_filled), "Position mismatch deep sell"
    assert env.short_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch deep sell"
    assert env.active_orders == [], "Order should be gone deep sell"
    assert env.last_executed_volume == pytest.approx(total_filled), "Exec vol mismatch deep sell"

def test_execute_buy_maker_very_small_volume_fill(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # Maker
    fill_price = 100.10
    buy_vol = 0.01 # Minimum lot size
    env.asks = np.array([[fill_price, 1.0]], dtype=np.float32)
    env.best_ask = fill_price
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()
    expected_cost_val = buy_vol * fill_price
    tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch tiny buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch tiny buy"
    assert env.long_position == pytest.approx(buy_vol), "Position mismatch tiny buy"
    assert env.active_orders == [], "Order should be gone tiny buy"
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch tiny buy"

def test_execute_sell_maker_very_small_volume_fill(setup_env):
    env = setup_env
    sell_price_limit = 100.01 # Maker
    fill_price = 100.00
    sell_vol = 0.01
    env.bids = np.array([[fill_price, 1.0]], dtype=np.float32)
    env.best_bid = fill_price
    env.active_orders = [{"id": 1, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash

    realized_pnl = env._execute_orders()
    expected_rev_val = sell_vol * fill_price
    tx_cost = expected_rev_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_rev_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch tiny sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch tiny sell"
    assert env.short_position == pytest.approx(sell_vol), "Position mismatch tiny sell"
    assert env.active_orders == [], "Order should be gone tiny sell"
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch tiny sell"


def test_execute_mixed_orders_only_buy_fills(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # MAKER price -> Fills @ 100.10
    fill_price_ask = 100.10
    buy_vol = 2.0
    sell_price_limit = 100.01 # MAKER price (Above best bid 100.00) -> Won't fill
    sell_vol = 3.0
    order_buy_id = 1
    order_sell_id = 2
    # Set LOB where only buy can fill
    env.asks = np.array([[fill_price_ask, 5.0]], dtype=np.float32)
    env.best_ask = fill_price_ask
    env.bids = np.array([[100.00, 5.0]], dtype=np.float32) # Standard bid
    env.best_bid = 100.00
    env.active_orders = [
        {"id": order_buy_id, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0},
        {"id": order_sell_id, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 1},
    ]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()

    expected_cost_val = buy_vol * fill_price_ask
    tx_cost = expected_cost_val * env.config["transaction_cost_long"]
    expected_cash = initial_cash - expected_cost_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch mixed only buy"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch mixed only buy"
    assert env.long_position == pytest.approx(initial_long + buy_vol), "Long pos mismatch mixed only buy"
    assert env.short_position == initial_short
    assert len(env.active_orders) == 1, "Sell order should remain mixed only buy"
    assert env.active_orders[0]["id"] == order_sell_id
    assert env.active_orders[0]["volume"] == pytest.approx(sell_vol)
    assert env.last_executed_volume == pytest.approx(buy_vol), "Exec vol mismatch mixed only buy"

def test_execute_mixed_orders_only_sell_fills(setup_env):
    env = setup_env
    buy_price_limit = 100.09 # MAKER price (Below best ask 100.10) -> Won't fill
    buy_vol = 2.0
    sell_price_limit = 100.01 # MAKER price -> Fills @ 100.00
    fill_price_bid = 100.00
    sell_vol = 3.0
    order_buy_id = 1
    order_sell_id = 2
    # Set LOB where only sell can fill
    env.asks = np.array([[100.10, 5.0]], dtype=np.float32) # Standard ask
    env.best_ask = 100.10
    env.bids = np.array([[fill_price_bid, 5.0]], dtype=np.float32)
    env.best_bid = fill_price_bid
    env.active_orders = [
        {"id": order_buy_id, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0},
        {"id": order_sell_id, "price": sell_price_limit, "volume": sell_vol, "is_buy": False, "timestamp_placed": 1},
    ]
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()

    expected_rev_val = sell_vol * fill_price_bid
    tx_cost = expected_rev_val * env.config["transaction_cost_short"]
    expected_cash = initial_cash + expected_rev_val - tx_cost

    assert realized_pnl == 0.0, "PnL mismatch mixed only sell"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch mixed only sell"
    assert env.long_position == initial_long
    assert env.short_position == pytest.approx(initial_short + sell_vol), "Short pos mismatch mixed only sell"
    assert len(env.active_orders) == 1, "Buy order should remain mixed only sell"
    assert env.active_orders[0]["id"] == order_buy_id
    assert env.active_orders[0]["volume"] == pytest.approx(buy_vol)
    assert env.last_executed_volume == pytest.approx(sell_vol), "Exec vol mismatch mixed only sell"

def test_execute_buy_taker_locked_book_at_ask(setup_env):
    env = setup_env
    env.bids = np.array([[100.00, 5.0]], dtype=np.float32)
    env.asks = np.array([[100.00, 5.0]], dtype=np.float32) # Locked book
    env.best_bid = 100.00; env.best_ask = 100.00
    buy_price = 100.00 # Price is >= ask (100.00) -> Taker
    buy_vol = 1.0
    env.active_orders = [{"id": 1, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_sell_taker_locked_book_at_bid(setup_env):
    env = setup_env
    env.bids = np.array([[100.00, 5.0]], dtype=np.float32)
    env.asks = np.array([[100.00, 5.0]], dtype=np.float32) # Locked book
    env.best_bid = 100.00; env.best_ask = 100.00
    sell_price = 100.00 # Price is <= bid (100.00) -> Taker
    sell_vol = 1.0
    env.active_orders = [{"id": 1, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;
    realized_pnl = env._execute_orders()
    assert_taker_cancelled(env, initial_cash, initial_long, initial_short, realized_pnl)

def test_execute_buy_maker_locked_book_below_ask(setup_env):
    # Maker order should just not fill in a locked book if price is passive
    env = setup_env
    env.bids = np.array([[100.00, 5.0]], dtype=np.float32)
    env.asks = np.array([[100.00, 5.0]], dtype=np.float32) # Locked book
    env.best_bid = 100.00; env.best_ask = 100.00
    buy_price = 99.99 # Price is < ask (100.00) -> Maker (won't fill)
    buy_vol = 1.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": buy_price, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, buy_vol, realized_pnl)


def test_execute_sell_maker_locked_book_above_bid(setup_env):
    env = setup_env
    env.bids = np.array([[100.00, 5.0]], dtype=np.float32)
    env.asks = np.array([[100.00, 5.0]], dtype=np.float32) # Locked book
    env.best_bid = 100.00; env.best_ask = 100.00
    sell_price = 100.01 # Price is > bid (100.00) -> Maker (won't fill)
    sell_vol = 1.0
    order_id = 1
    env.active_orders = [{"id": order_id, "price": sell_price, "volume": sell_vol, "is_buy": False, "timestamp_placed": 0}]
    initial_cash = env.cash
    initial_long = env.long_position; initial_short = env.short_position;

    realized_pnl = env._execute_orders()
    assert_no_fill(env, initial_cash, initial_long, initial_short, order_id, sell_vol, realized_pnl)

def test_execute_buy_with_nan_in_ask_book_skips_level(setup_env):
    env = setup_env
    buy_price_limit = 100.12 # MAKER price allowing fill up to 100.12
    # LOB setup with NaN
    price_a1, vol_a1 = 100.10, 2.0
    price_a3, vol_a3 = 100.12, 5.0
    env.asks = np.array([[price_a1, vol_a1], [np.nan, 3.0], [price_a3, vol_a3]], dtype=np.float32) # NaN price on level 2
    env.best_ask = price_a1
    buy_vol = 4.0 # Takes level 1 (2.0) and 2.0 from level 3, skipping NaN level
    env.active_orders = [{"id": 1, "price": buy_price_limit, "volume": buy_vol, "is_buy": True, "timestamp_placed": 0}]
    initial_cash = env.cash

    fill1_vol, fill1_price = 2.0, price_a1
    fill3_vol, fill3_price = 2.0, price_a3
    total_filled = fill1_vol + fill3_vol

    realized_pnl = env._execute_orders()

    cost1 = fill1_vol * fill1_price
    cost3 = fill3_vol * fill3_price
    total_cost_val = cost1 + cost3
    tx_cost_adj = (cost1 * env.config["transaction_cost_long"]) + \
                  (cost3 * env.config["transaction_cost_long"])
    expected_cash = initial_cash - total_cost_val - tx_cost_adj
    expected_avg_cost = total_cost_val / total_filled

    assert realized_pnl == 0.0, "PnL mismatch nan skip"
    assert env.cash == pytest.approx(expected_cash), "Cash mismatch nan skip"
    assert env.long_position == pytest.approx(total_filled), "Position mismatch nan skip"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Avg cost mismatch nan skip"
    assert env.active_orders == [], "Order should be gone nan skip"
    assert env.last_executed_volume == pytest.approx(total_filled), "Exec vol mismatch nan skip"

# Use corrected tests from previous step
def test_execute_avg_cost_update_multiple_buys(setup_env):
    env = setup_env
    initial_cash = env.cash
    tx_cost_rate = env.config["transaction_cost_long"]

    # --- Step 1: Buy 2.0 @ 100.09 (Maker Price) ---
    buy_price_limit_1 = 100.09
    fill_price_1 = 100.09 # Assume LOB allows fill at limit
    buy_vol_1 = 2.0
    env.asks = np.array([[fill_price_1, 5.0]], dtype=np.float32)
    env.best_ask = fill_price_1
    env.active_orders = [{"id": 1, "price": buy_price_limit_1, "volume": buy_vol_1, "is_buy": True, "timestamp_placed": 0}]

    pnl1 = env._execute_orders()
    cost1 = buy_vol_1 * fill_price_1
    tx_cost1 = cost1 * tx_cost_rate
    cash_after_1 = initial_cash - cost1 - tx_cost1

    assert pnl1 == 0.0, "Step 1 PnL should be 0"
    assert env.long_position == pytest.approx(buy_vol_1), "Step 1 Long position incorrect"
    assert env.long_avg_cost == pytest.approx(fill_price_1), "Step 1 Avg cost incorrect"
    assert env.cash == pytest.approx(cash_after_1), "Step 1 Cash incorrect"
    assert env.active_orders == [], "Step 1 Order should be filled"

    # --- Step 2: Buy 3.0 @ 100.15 (Maker Price) ---
    buy_price_limit_2 = 100.15
    fill_price_2 = 100.15 # Assume LOB allows fill at limit
    buy_vol_2 = 3.0
    env.asks = np.array([[fill_price_2, 5.0]], dtype=np.float32)
    env.best_ask = fill_price_2
    env.active_orders = [{"id": 2, "price": buy_price_limit_2, "volume": buy_vol_2, "is_buy": True, "timestamp_placed": 1}]

    pnl2 = env._execute_orders()
    cost2 = buy_vol_2 * fill_price_2
    tx_cost2 = cost2 * tx_cost_rate
    cash_after_2 = cash_after_1 - cost2 - tx_cost2

    expected_total_vol = buy_vol_1 + buy_vol_2
    expected_total_cost_value = (buy_vol_1 * fill_price_1) + (buy_vol_2 * fill_price_2)
    expected_avg_cost = expected_total_cost_value / expected_total_vol

    assert pnl2 == 0.0, "Step 2 PnL should be 0"
    assert env.long_position == pytest.approx(expected_total_vol), "Step 2 Long position incorrect"
    assert env.long_avg_cost == pytest.approx(expected_avg_cost), "Step 2 Avg cost incorrect"
    assert env.cash == pytest.approx(cash_after_2), "Step 2 Cash incorrect"
    assert env.active_orders == [], "Step 2 Order should be filled"

def test_execute_avg_cost_update_close_partial_then_add(setup_env):
    env = setup_env
    tx_cost_rate_buy = env.config["transaction_cost_long"]
    tx_cost_rate_sell = env.config["transaction_cost_short"]

    # --- Initial State: Long 5.0 @ 99.90 ---
    initial_long_vol = 5.0
    initial_long_cost = 99.90
    env.long_position = initial_long_vol
    env.long_avg_cost = initial_long_cost
    initial_cash = env.cash

    # --- Step 1: Sell 2.0 @ 100.01 (MAKER, fills @ 100.00) ---
    sell_price_limit_1 = 100.01
    fill_price_1 = 100.00
    sell_vol_1 = 2.0
    env.bids = np.array([[fill_price_1, 5.0]], dtype=np.float32)
    env.best_bid = fill_price_1
    env.active_orders = [{"id": 1, "price": sell_price_limit_1, "volume": sell_vol_1, "is_buy": False, "timestamp_placed": 0}]

    pnl1 = env._execute_orders()
    revenue1 = sell_vol_1 * fill_price_1
    tx_cost1 = revenue1 * tx_cost_rate_sell
    cash_after_1 = initial_cash + revenue1 - tx_cost1
    expected_pnl1 = (fill_price_1 - initial_long_cost) * sell_vol_1

    assert pnl1 == pytest.approx(expected_pnl1), "Step 1 PnL incorrect"
    assert env.long_position == pytest.approx(initial_long_vol - sell_vol_1), "Step 1 Long position incorrect"
    assert env.long_avg_cost == pytest.approx(initial_long_cost), "Step 1 Avg cost incorrect"
    assert env.cash == pytest.approx(cash_after_1), "Step 1 Cash incorrect"
    assert env.active_orders == [], "Step 1 Order should be filled"

    # --- Step 2: Buy 4.0 @ 100.09 (MAKER, fills @ 100.10) ---
    buy_price_limit_2 = 100.09
    fill_price_2 = 100.10
    buy_vol_2 = 4.0
    # Use position and avg cost state *after* step 1
    long_pos_after_step1 = env.long_position
    long_avg_cost_after_step1 = env.long_avg_cost

    env.asks = np.array([[fill_price_2, 5.0]], dtype=np.float32)
    env.best_ask = fill_price_2
    env.active_orders = [{"id": 2, "price": buy_price_limit_2, "volume": buy_vol_2, "is_buy": True, "timestamp_placed": 1}]

    pnl2 = env._execute_orders()
    cost2 = buy_vol_2 * fill_price_2
    tx_cost2 = cost2 * tx_cost_rate_buy
    cash_after_2 = cash_after_1 - cost2 - tx_cost2

    expected_final_long_vol = long_pos_after_step1 + buy_vol_2
    expected_final_cost_value = (long_pos_after_step1 * long_avg_cost_after_step1) + (buy_vol_2 * fill_price_2)
    expected_final_avg_cost = expected_final_cost_value / expected_final_long_vol

    assert pnl2 == 0.0, "Step 2 PnL should be 0"
    assert env.long_position == pytest.approx(expected_final_long_vol), "Step 2 Long position incorrect"
    assert env.long_avg_cost == pytest.approx(expected_final_avg_cost), "Step 2 Avg cost incorrect"
    assert env.cash == pytest.approx(cash_after_2), "Step 2 Cash incorrect"
    assert env.active_orders == [], "Step 2 Order should be filled"


def test_execute_pnl_with_zero_initial_avg_cost(setup_env):
    env = setup_env
    initial_cash = env.cash
    tx_cost_rate_sell = env.config["transaction_cost_short"]
    tx_cost_rate_buy = env.config["transaction_cost_long"]
    # Start flat
    assert env.long_position == 0.0; assert env.long_avg_cost == 0.0
    assert env.short_position == 0.0; assert env.short_avg_cost == 0.0

    # --- Step 1: Sell 3 @ 100.01 (MAKER, fills @ 100.00) ---
    sell_price_limit_1 = 100.01
    fill_price_1 = 100.00
    sell_vol_1 = 3.0
    env.bids = np.array([[fill_price_1, 5.0]], dtype=np.float32); env.best_bid = fill_price_1
    env.active_orders = [{"id": 1, "price": sell_price_limit_1, "volume": sell_vol_1, "is_buy": False, "timestamp_placed": 0}]

    pnl1 = env._execute_orders()
    rev1 = sell_vol_1 * fill_price_1
    tx_cost1 = rev1 * tx_cost_rate_sell
    cash_after_1 = initial_cash + rev1 - tx_cost1

    assert pnl1 == 0.0, "Step 1 PnL should be 0 opening short"
    assert env.short_position == pytest.approx(sell_vol_1), "Step 1 Short position incorrect"
    assert env.short_avg_cost == pytest.approx(fill_price_1), "Step 1 Short avg cost incorrect"
    assert env.cash == pytest.approx(cash_after_1), "Step 1 Cash incorrect"
    assert env.active_orders == [], "Step 1 Order should be filled"

    # --- Step 2: Buy 3 @ 99.89 (MAKER, fills @ 99.90) ---
    buy_price_limit_2 = 99.89
    fill_price_2 = 99.90
    buy_vol_2 = 3.0
    # Need to use the short_avg_cost established in Step 1
    current_short_avg_cost = env.short_avg_cost # Should be fill_price_1 (100.00)

    env.asks = np.array([[fill_price_2, 5.0]], dtype=np.float32); env.best_ask = fill_price_2
    env.active_orders = [{"id": 2, "price": buy_price_limit_2, "volume": buy_vol_2, "is_buy": True, "timestamp_placed": 1}]

    pnl2 = env._execute_orders()
    cost2 = buy_vol_2 * fill_price_2
    tx_cost2 = cost2 * tx_cost_rate_buy
    cash_after_2 = cash_after_1 - cost2 - tx_cost2

    # PnL calculation uses the avg cost *before* the fill happens
    expected_pnl2 = (current_short_avg_cost - fill_price_2) * buy_vol_2

    assert pnl2 == pytest.approx(expected_pnl2), "Step 2 PnL incorrect closing short"
    assert env.short_position == 0.0, "Step 2 Short position should be 0"
    assert env.short_avg_cost == 0.0
    assert env.long_position == 0.0
    assert env.long_avg_cost == 0.0
    assert env.cash == pytest.approx(cash_after_2), "Step 2 Cash incorrect"
    assert env.active_orders == [], "Step 2 Order should be filled"


def test_execute_no_active_orders(setup_env):
    env = setup_env
    env.active_orders = []
    initial_cash = env.cash
    initial_long = env.long_position
    initial_short = env.short_position

    realized_pnl = env._execute_orders()

    assert realized_pnl == 0.0
    assert env.cash == initial_cash
    assert env.long_position == initial_long
    assert env.short_position == initial_short
    assert env.last_executed_volume == 0.0

# --- End of Tests ---
