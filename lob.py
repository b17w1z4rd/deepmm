"""
lob.py - a limit order book with price-time priority.

The microstructure layer: resting limit orders on each side, matched against
incoming marketable orders best-price-first and, within a price level,
first-in-first-out (time priority). This is what actually turns a market maker's
quotes into fills, and it's where queue position and adverse selection live.

Prices are in integer ticks to keep comparisons exact.
"""
from collections import deque
import itertools


class Order:
    __slots__ = ("id", "side", "price", "qty")
    def __init__(self, oid, side, price, qty):
        self.id, self.side, self.price, self.qty = oid, side, price, qty


class OrderBook:
    def __init__(self):
        self.bids = {}                 # price -> deque[Order]   (buy side)
        self.asks = {}                 # price -> deque[Order]   (sell side)
        self.orders = {}               # id -> Order (live limit orders)
        self._ids = itertools.count(1)

    # ---- queries ----
    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return (b + a) / 2 if b is not None and a is not None else None

    def depth(self, side, price):
        book = self.bids if side == "buy" else self.asks
        return sum(o.qty for o in book.get(price, ()))

    # ---- limit orders ----
    def add_limit(self, side, price, qty):
        """Rest a limit order (assumed non-marketable). Returns its id."""
        oid = next(self._ids)
        o = Order(oid, side, price, qty)
        book = self.bids if side == "buy" else self.asks
        book.setdefault(price, deque()).append(o)
        self.orders[oid] = o
        return oid

    def cancel(self, oid):
        o = self.orders.pop(oid, None)
        if o is None:
            return False
        book = self.bids if o.side == "buy" else self.asks
        q = book.get(o.price)
        if q and o in q:
            q.remove(o)
            if not q:
                del book[o.price]
        return True

    # ---- marketable orders ----
    def market(self, side, qty):
        """
        Execute a market order for `qty`, walking the opposite book best-first
        with time priority. Returns list of (price, qty, resting_order_id) fills.
        `side` is the aggressor's side: a market 'buy' lifts asks.
        """
        book = self.asks if side == "buy" else self.bids
        take_best = min if side == "buy" else max
        fills = []
        while qty > 0 and book:
            px = take_best(book)
            q = book[px]
            while qty > 0 and q:
                o = q[0]
                traded = min(qty, o.qty)
                o.qty -= traded
                qty -= traded
                fills.append((px, traded, o.id))
                if o.qty == 0:
                    q.popleft()
                    self.orders.pop(o.id, None)
            if not q:
                del book[px]
        return fills


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ok = True
    def check(name, cond):
        global ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    b = OrderBook()
    a1 = b.add_limit("sell", 101, 5)     # ask 5 @ 101 (first in queue)
    a2 = b.add_limit("sell", 101, 3)     # ask 3 @ 101 (behind a1)
    b.add_limit("sell", 102, 10)         # ask 10 @ 102
    b.add_limit("buy", 99, 4)

    check("best bid/ask", b.best_bid() == 99 and b.best_ask() == 101)
    check("mid", b.mid() == 100.0)
    check("depth aggregates a level", b.depth("sell", 101) == 8)

    # a market buy of 6 should take all of a1 (5) then 1 from a2 (time priority)
    fills = b.market("buy", 6)
    check("price-time priority fills", fills == [(101, 5, a1), (101, 1, a2)])
    check("partial fill remains", b.orders[a2].qty == 2 and b.depth("sell", 101) == 2)

    # walk to the next level
    fills = b.market("buy", 4)
    check("walks to next price level", fills == [(101, 2, a2), (102, 2, 3)])
    check("best ask now 102", b.best_ask() == 102)

    # cancel
    bid = b.add_limit("buy", 98, 7)
    check("cancel removes liquidity", b.cancel(bid) and b.depth("buy", 98) == 0)

    print("\n" + ("all order-book tests passed" if ok else "SOME TESTS FAILED"))
    raise SystemExit(0 if ok else 1)
