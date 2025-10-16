import socket
import threading
import time
import json
from collections import deque

# Order and order book structures
class OrderSide:
    BUY = 'BUY'
    SELL = 'SELL'

class OrderType:
    LIMIT = 'LIMIT'
    MARKET = 'MARKET'

class Order:
    def __init__(self, order_id, side, order_type, price, quantity, timestamp):
        self.order_id = order_id
        self.side = side
        self.order_type = order_type
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

class Trade:
    def __init__(self, trade_id, buy_order_id, sell_order_id, price, quantity, timestamp):
        self.trade_id = trade_id
        self.buy_order_id = buy_order_id
        self.sell_order_id = sell_order_id
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp

# OrderBook that handles limit orders and market orders
class OrderBook:
    def __init__(self):
        # Buy orders are stored in descending order (highest price first)
        self.buy_orders = {}
        # Sell orders are stored in ascending order (lowest price first)
        self.sell_orders = {}
        self.order_map = {}

    def add_order(self, order):
        if order.side == OrderSide.BUY:
            if order.price not in self.buy_orders:
                self.buy_orders[order.price] = deque()
            self.buy_orders[order.price].append(order)
        elif order.side == OrderSide.SELL:
            if order.price not in self.sell_orders:
                self.sell_orders[order.price] = deque()
            self.sell_orders[order.price].append(order)

        # Add order to the order map for easy cancellation
        self.order_map[order.order_id] = order

    def cancel_order(self, order_id):
        if order_id not in self.order_map:
            return False
        order = self.order_map[order_id]
        if order.side == OrderSide.BUY:
            if order.price in self.buy_orders:
                self.buy_orders[order.price].remove(order)
                if not self.buy_orders[order.price]:
                    del self.buy_orders[order.price]
        elif order.side == OrderSide.SELL:
            if order.price in self.sell_orders:
                self.sell_orders[order.price].remove(order)
                if not self.sell_orders[order.price]:
                    del self.sell_orders[order.price]
        del self.order_map[order_id]
        return True

    def get_best_price(self, side):
        if side == OrderSide.BUY:
            if self.sell_orders:
                best_price = min(self.sell_orders.keys())
                return best_price, self.sell_orders[best_price][0].quantity
        elif side == OrderSide.SELL:
            if self.buy_orders:
                best_price = max(self.buy_orders.keys())
                return best_price, self.buy_orders[best_price][0].quantity
        return None, 0

# Matching Engine
class MatchingEngine:
    def __init__(self):
        self.order_book = OrderBook()
        self.next_trade_id = 1
        self.current_timestamp = 1

    def process_order(self, order):
        self.current_timestamp += 1
        order.timestamp = self.current_timestamp

        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.MARKET:
                self.process_market_order(order)
            else:
                self.process_limit_buy_order(order)
        elif order.side == OrderSide.SELL:
            if order.order_type == OrderType.MARKET:
                self.process_market_order(order)
            else:
                self.process_limit_sell_order(order)

    def process_market_order(self, order):
        if order.side == OrderSide.BUY:
            # Market buy orders match with the best available sell orders
            while order.quantity > 0 and self.order_book.sell_orders:
                best_price, available_qty = self.order_book.get_best_price(OrderSide.BUY)
                if available_qty == 0:
                    break
                trade_qty = min(order.quantity, available_qty)
                self.record_trade(order.order_id, self.order_book.sell_orders[best_price][0].order_id, best_price, trade_qty)
                order.quantity -= trade_qty
                self.order_book.sell_orders[best_price][0].quantity -= trade_qty
                if self.order_book.sell_orders[best_price][0].quantity == 0:
                    self.order_book.cancel_order(self.order_book.sell_orders[best_price][0].order_id)
        elif order.side == OrderSide.SELL:
            # Market sell orders match with the best available buy orders
            while order.quantity > 0 and self.order_book.buy_orders:
                best_price, available_qty = self.order_book.get_best_price(OrderSide.SELL)
                if available_qty == 0:
                    break
                trade_qty = min(order.quantity, available_qty)
                self.record_trade(self.order_book.buy_orders[best_price][0].order_id, order.order_id, best_price, trade_qty)
                order.quantity -= trade_qty
                self.order_book.buy_orders[best_price][0].quantity -= trade_qty
                if self.order_book.buy_orders[best_price][0].quantity == 0:
                    self.order_book.cancel_order(self.order_book.buy_orders[best_price][0].order_id)

    def process_limit_buy_order(self, order):
        while order.quantity > 0 and self.order_book.sell_orders:
            best_price, available_qty = self.order_book.get_best_price(OrderSide.BUY)
            if best_price and available_qty > 0:
                if best_price > order.price:
                    break  # No more acceptable sell orders

                trade_qty = min(order.quantity, available_qty)
                self.record_trade(order.order_id, self.order_book.sell_orders[best_price][0].order_id, best_price, trade_qty)
                order.quantity -= trade_qty
                self.order_book.sell_orders[best_price][0].quantity -= trade_qty
                if self.order_book.sell_orders[best_price][0].quantity == 0:
                    self.order_book.cancel_order(self.order_book.sell_orders[best_price][0].order_id)

        if order.quantity > 0:
            self.order_book.add_order(order)

    def process_limit_sell_order(self, order):
        while order.quantity > 0 and self.order_book.buy_orders:
            best_price, available_qty = self.order_book.get_best_price(OrderSide.SELL)
            if best_price and available_qty > 0:
                if best_price < order.price:
                    break  # No more acceptable buy orders

                trade_qty = min(order.quantity, available_qty)
                self.record_trade(self.order_book.buy_orders[best_price][0].order_id, order.order_id, best_price, trade_qty)
                order.quantity -= trade_qty
                self.order_book.buy_orders[best_price][0].quantity -= trade_qty
                if self.order_book.buy_orders[best_price][0].quantity == 0:
                    self.order_book.cancel_order(self.order_book.buy_orders[best_price][0].order_id)

        if order.quantity > 0:
            self.order_book.add_order(order)

    def record_trade(self, buy_order_id, sell_order_id, price, quantity):
        trade = Trade(self.next_trade_id, buy_order_id, sell_order_id, price, quantity, self.current_timestamp)
        self.next_trade_id += 1
        print(f"Trade executed: TradeID {trade.trade_id}, BuyOrder {trade.buy_order_id}, SellOrder {trade.sell_order_id}, Price {trade.price}, Quantity {trade.quantity}")

    def get_order_book_visualization(self):
        order_book_state = {
            "buy_orders": {},
            "sell_orders": {}
        }
        for price, orders in self.order_book.buy_orders.items():
            order_book_state["buy_orders"][price] = len(orders)
        for price, orders in self.order_book.sell_orders.items():
            order_book_state["sell_orders"][price] = len(orders)
        return json.dumps(order_book_state, indent=2)

    def send_execution_result(self, order_id, result):
        print(f"Execution result for order {order_id}: {result}")


# Socket Server for Receiving Orders
def handle_order_server(engine, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(5)
    print(f"Order server listening on port {port}...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")

        data = client_socket.recv(1024).decode('utf-8')
        order_data = json.loads(data)
        order = Order(
            order_data['order_id'],
            order_data['side'],
            order_data['order_type'],
            order_data['price'],
            order_data['quantity'],
            0
        )
        engine.process_order(order)
        engine.send_execution_result(order.order_id, "Executed successfully")
        client_socket.sendall(engine.get_order_book_visualization().encode('utf-8'))
        client_socket.close()

# Socket Server for Visualizing the Order Book
def handle_visualization_server(engine, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', port))
    server_socket.listen(5)
    print(f"Visualization server listening on port {port}...")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address}")
        client_socket.sendall(engine.get_order_book_visualization().encode('utf-8'))
        client_socket.close()


if __name__ == "__main__":
    engine = MatchingEngine()

    # Order server running on port 8080
    order_thread = threading.Thread(target=handle_order_server, args=(engine, 8080))
    order_thread.start()

    # Visualization server running on port 9090
    visualization_thread = threading.Thread(target=handle_visualization_server, args=(engine, 9090))
    visualization_thread.start()
