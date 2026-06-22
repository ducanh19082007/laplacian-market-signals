import time
import shutil
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Any
import json


try:
    from .IngestionPipeline import OrderBookDashboard
    from .DataProcessing import ExchangeRateGraph
except ImportError:
    from IngestionPipeline import OrderBookDashboard
    from DataProcessing import ExchangeRateGraph


# NOTE: Binance and Coinbase Advanced Trade use real WebSocket depth/book streams.
# Kraken uses the v2 WebSocket API (wss://ws.kraken.com/v2).
#
#
# Author: Anh Duc Le


@dataclass
class BrokerConfig:
    name: str
    stream_url: str
    pairs: List[str]
    payload_extractor: Optional[Any] = None
    initial_message: Optional[Any] = None
    debug: bool = False


class MultiBrokerOrderBook:
    def __init__(
        self,
        brokers: List[BrokerConfig],
        refresh_interval: float = 0.5,
        assets: Optional[List[str]] = None,
        transfer_cost: float = 0.0,
    ):
        self.dashboards = []
        self.refresh_interval = refresh_interval
        # assets enables the live graph view; without it run_live() just shows the table.
        self.assets = [a.lower() for a in assets] if assets else None
        self.transfer_cost = transfer_cost

        for broker in brokers:
            dashboard = OrderBookDashboard(
                broker.pairs,
                broker.stream_url,
                broker_name=broker.name,
                refresh_interval=refresh_interval,
                show=False,
                payload_extractor=broker.payload_extractor,
                initial_message=broker.initial_message,
                debug=broker.debug,
            )
            dashboard.run()
            self.dashboards.append((broker, dashboard))

    def get_all_pairs(self) -> List[str]:
        pairs = set()
        for broker, _ in self.dashboards:
            pairs.update(pair.lower() for pair in broker.pairs)
        return sorted(pairs)

    def snapshot(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        snapshot = {}
        for pair in self.get_all_pairs():
            snapshot[pair] = {}
            for broker, dashboard in self.dashboards:
                bid, ask = dashboard.get_best_prices(pair)
                snapshot[pair][broker.name] = {"bid": bid, "ask": ask}
        return snapshot

    def aggregate_best_prices(self, pair_name: str) -> Dict[str, str]:
        bids = []
        asks = []
        for _, dashboard in self.dashboards:
            bid, ask = dashboard.get_best_prices(pair_name)
            try:
                bids.append(float(bid))
            except (ValueError, TypeError):
                pass
            try:
                asks.append(float(ask))
            except (ValueError, TypeError):
                pass

        return {
            "best_bid": max(bids) if bids else "N/A",
            "best_ask": min(asks) if asks else "N/A",
        }

    def print_table(self) -> None:
        brokers = [broker.name for broker, _ in self.dashboards]
        header = (
            ["PAIR"]
            + [f"{name} BID" for name in brokers]
            + [f"{name} ASK" for name in brokers]
            + ["BEST BID", "BEST ASK"]
        )
        column_width = 14

        print(" | ".join(h.upper().ljust(column_width) for h in header))
        print("-" * (len(header) * (column_width + 3)))

        snapshot = self.snapshot()
        for pair in self.get_all_pairs():
            row = [pair.upper().ljust(column_width)]
            for broker_name in brokers:
                row.append(str(snapshot[pair][broker_name]["bid"]).ljust(column_width))
            for broker_name in brokers:
                row.append(str(snapshot[pair][broker_name]["ask"]).ljust(column_width))
            aggregated = self.aggregate_best_prices(pair)
            row.append(str(aggregated["best_bid"]).ljust(column_width))
            row.append(str(aggregated["best_ask"]).ljust(column_width))
            print(" | ".join(row))

    def build_graph(self) -> Optional["ExchangeRateGraph"]:
        """Reshape the current snapshot into a log-weighted exchange-rate graph."""
        if not self.assets:
            return None
        graph = ExchangeRateGraph(self.assets, transfer_cost=self.transfer_cost)
        graph.build_from_snapshot(self.snapshot())
        graph.log_transform()
        return graph

    def print_arbitrage(self) -> None:
        """Build the graph from the live snapshot and report any arbitrage cycle."""
        graph = self.build_graph()
        if graph is None:
            return
        print("\n--- Graph (asset x venue) Arbitrage ---")
        cycle = graph.find_arbitrage()
        if cycle:
            path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
            ret = graph.cycle_return(cycle)
            print(f"Cycle : {path}")
            print(f"Return: {ret:.8f}  ({(ret - 1) * 100:+.4f}%)")
        else:
            print("No arbitrage cycle (market is arb-free or feed still warming up).")

    def run_live(self, clear_screen: bool = True) -> None:
        """
        Refresh the order-book table each tick.

        clear_screen=True wipes the terminal every tick (snapshot view).
        Set it False if you want the table to scroll and accumulate instead.
        """
        try:
            while True:
                if clear_screen:
                    print("\033[H\033[J", end="")
                print(f"--- Multi-Broker Order Book ({time.strftime('%H:%M:%S')}) ---")
                self.print_table()
                self.print_arbitrage()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            for _, dashboard in self.dashboards:
                dashboard.is_running = False
            print("\nMulti-broker monitoring stopped.")

    def _exchange_rate_box(self) -> List[str]:
        """
        Render the current best exchange rate per pair as a bordered box
        (list of equal-width lines), refreshed from the live snapshot.
        """
        rows = []
        for pair in self.get_all_pairs():
            agg = self.aggregate_best_prices(pair)
            rows.append((pair.upper(), str(agg["best_bid"]), str(agg["best_ask"])))

        pair_w = max([len(r[0]) for r in rows] + [len("PAIR")])
        bid_w = max([len(r[1]) for r in rows] + [len("BEST BID")])
        ask_w = max([len(r[2]) for r in rows] + [len("BEST ASK")])

        title = f"LIVE EXCHANGE RATES  {time.strftime('%H:%M:%S')}"
        header = f"{'PAIR'.ljust(pair_w)}  {'BEST BID'.ljust(bid_w)}  {'BEST ASK'.ljust(ask_w)}"
        inner_w = max(len(title), len(header))

        lines = ["┌─" + "─" * inner_w + "─┐",
                 "│ " + title.ljust(inner_w) + " │",
                 "├─" + "─" * inner_w + "─┤",
                 "│ " + header.ljust(inner_w) + " │"]
        for pair, bid, ask in rows:
            body = f"{pair.ljust(pair_w)}  {bid.ljust(bid_w)}  {ask.ljust(ask_w)}"
            lines.append("│ " + body.ljust(inner_w) + " │")
        lines.append("└─" + "─" * inner_w + "─┘")
        return lines

    @staticmethod
    def _render_side_by_side(left: List[str], right: List[str], gap: int = 4) -> str:
        """Lay two blocks of lines next to each other, right block truncated to fit."""
        left_w = max((len(l) for l in left), default=0)
        term_w = shutil.get_terminal_size((120, 40)).columns
        right_w = max(term_w - left_w - gap, 10)

        out = []
        for i in range(max(len(left), len(right))):
            l = (left[i] if i < len(left) else "").ljust(left_w)
            r = right[i] if i < len(right) else ""
            if len(r) > right_w:
                r = r[: right_w - 1] + "…"
            out.append(l + " " * gap + r)
        return "\n".join(out)

    def stream_arbitrage(
        self,
        only_on_change: bool = True,
        show_box: bool = True,
        history: int = 20,
    ) -> None:
        """
        Continuous arbitrage feed.

        Both modes clear the screen every tick so old text never accumulates.
        show_box=False -> redraw a plain header + the last `history` detections.
        show_box=True  -> redraw a live exchange-rate box on the left and the
                          most recent `history` detections on the right, so the
                          rates update in place beside the loop.

        only_on_change=True records a detection only when the cycle changes,
        so the list grows only on genuinely new opportunities. The detections
        deque keeps a sliding window of the last `history` lines (oldest rolls
        off as new ones arrive).
        """
        if not self.assets:
            print("stream_arbitrage needs `assets` set on the order book.")
            return

        log: deque = deque(maxlen=history)
        last_signature = None
        try:
            while True:
                graph = self.build_graph()
                new_line = None
                if graph is not None:
                    cycle = graph.find_arbitrage()
                    if cycle:
                        signature = tuple(cycle)
                        if not only_on_change or signature != last_signature:
                            ret = graph.cycle_return(cycle)
                            path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
                            new_line = (
                                f"[{time.strftime('%H:%M:%S')}] "
                                f"{(ret - 1) * 100:+.4f}%  ({ret:.8f})  {path}"
                            )
                            last_signature = signature
                    else:
                        last_signature = None

                if new_line:
                    log.append(new_line)

                # Clear the screen every tick so previous frames never pile up.
                print("\033[H\033[J", end="")
                if show_box:
                    right = ["ARBITRAGE DETECTIONS", "-" * 20]
                    right += list(log) if log else ["(none yet)"]
                    print(self._render_side_by_side(self._exchange_rate_box(), right))
                else:
                    print(f"--- Live Arbitrage Stream (refresh {self.refresh_interval}s, Ctrl-C to stop) ---")
                    for line in (log or ["(none yet)"]):
                        print(line)

                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            for _, dashboard in self.dashboards:
                dashboard.is_running = False
            print("\nArbitrage stream stopped.")


class URL_methods:

    # -------------------------------------------------------------------------
    # Binance
    # -------------------------------------------------------------------------

    @staticmethod
    def make_binance_depth_url(pairs: List[str], depth: int = 5, interval: str = "100ms") -> str:
        return (
            "wss://stream.binance.com:9443/stream?streams="
            + "/".join(f"{p}@depth{depth}@{interval}" for p in pairs)
        )

    # -------------------------------------------------------------------------
    # Coinbase Advanced Trade
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_coinbase_pair(pair: str, assets: List[str]) -> str:
        """Convert 'ethbtc' → 'ETH-BTC' using the known assets list to split."""
        pair = pair.upper()
        assets = [a.upper() for a in assets]
        for quote in assets:
            if pair.endswith(quote):
                base = pair[: -len(quote)]
                if base:
                    return f"{base}-{quote}"
        return pair

    @staticmethod
    def make_coinbase_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Coinbase Advanced Trade WebSocket subscription for the level2 channel.

        Key fix vs the old Coinbase Exchange API:
          - Uses "channel" (singular) instead of "channels" (list).
          - Targets wss://advanced-trade-ws.coinbase.com, not the legacy Pro endpoint.
        """
        formatted_pairs = [URL_methods.binance_to_coinbase_pair(p, assets) for p in pairs]
        return {
            "type": "subscribe",
            "product_ids": [p for p in formatted_pairs if "-" in p],
            "channel": "level2",   # ← singular; "channels" was the old Exchange API
        }

    @staticmethod
    def make_coinbase_payload_extractor():
        """
        Return a *stateful* closure that maintains the full order book across messages.

        Why stateful?
            Coinbase sends a one-time snapshot then only incremental diffs.
            A stateless extractor would discard the snapshot on the first update,
            leaving the book empty.  The closure accumulates price-level changes
            and always returns the current full book to IngestionPipeline.

        Coinbase Advanced Trade l2_data message shape:
        {
          "channel": "l2_data",
          "events": [{
            "type": "snapshot" | "update",
            "product_id": "BTC-USD",
            "updates": [
              {"side": "bid"|"offer", "price_level": "...", "new_quantity": "..."},
              ...
            ]
          }]
        }

        Note: Coinbase uses "offer" (not "sell" or "ask") for the ask side.
        """
        # books[key] = {"bids": {price_str: qty_str}, "asks": {price_str: qty_str}}
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Only handle l2_data; silently ignore subscriptions/heartbeats
            if data.get("channel") != "l2_data":
                return None
            events = data.get("events", [])
            if not events:
                return None

            for event in events:
                event_type = event.get("type")          # "snapshot" or "update"
                product_id = event.get("product_id", "")
                if not product_id:
                    continue

                key = product_id.replace("-", "").lower()   # "BTC-USD" → "btcusd"

                if event_type == "snapshot":
                    # Full replacement — clear and repopulate
                    books[key] = {"bids": {}, "asks": {}}
                    for upd in event.get("updates", []):
                        price = upd.get("price_level")
                        qty   = upd.get("new_quantity")
                        side  = upd.get("side")
                        if price is None or qty is None:
                            continue
                        if side == "bid":
                            books[key]["bids"][price] = qty
                        elif side == "offer":          # ← Coinbase says "offer", not "ask"
                            books[key]["asks"][price] = qty

                elif event_type == "update":
                    if key not in books:
                        books[key] = {"bids": {}, "asks": {}}
                    for upd in event.get("updates", []):
                        price = upd.get("price_level")
                        qty   = upd.get("new_quantity")
                        side  = upd.get("side")
                        if price is None or qty is None:
                            continue
                        try:
                            qty_f = float(qty)
                        except (ValueError, TypeError):
                            continue
                        if side == "bid":
                            if qty_f == 0:
                                books[key]["bids"].pop(price, None)
                            else:
                                books[key]["bids"][price] = qty
                        elif side == "offer":
                            if qty_f == 0:
                                books[key]["asks"].pop(price, None)
                            else:
                                books[key]["asks"][price] = qty

                book = books.get(key)
                if book:
                    # Return as list-of-tuples; _normalize_quote handles (price, qty) tuples
                    return key, {
                        "bids": list(book["bids"].items()),
                        "asks": list(book["asks"].items()),
                    }

            return None

        return extractor

    # -------------------------------------------------------------------------
    # Kraken  (v2 WebSocket API — replaces the old IBKR mock)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_kraken_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ETH/BTC'.

        Kraken uses a slash-separated format with uppercase asset codes.
        Note: Kraken does not list BNB, so BNB pairs will be accepted by this
        converter but will be rejected/ignored by the Kraken server.
        """
        pair_upper = pair.upper()
        assets_upper = [a.upper() for a in assets]
        for quote in assets_upper:
            if pair_upper.endswith(quote):
                base = pair_upper[: -len(quote)]
                if base:
                    return f"{base}/{quote}"
        return pair_upper

    @staticmethod
    def make_kraken_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Kraken v2 WebSocket subscription message for the book channel.

        Kraken v2 shape:
        {
          "method": "subscribe",
          "params": {"channel": "book", "symbol": ["ETH/BTC", ...], "depth": 10}
        }

        After subscribing, Kraken sends one snapshot per symbol, then incremental updates.
        """
        formatted = [URL_methods.binance_to_kraken_pair(p, assets) for p in pairs]
        return {
            "method": "subscribe",
            "params": {
                "channel": "book",
                "symbol": [p for p in formatted if "/" in p],
                "depth": 10,
            },
        }

    @staticmethod
    def make_kraken_payload_extractor():
        """
        Return a *stateful* closure for the Kraken v2 book channel.

        Kraken v2 book message shape:
        {
          "channel": "book",
          "type": "snapshot" | "update",
          "data": [{
            "symbol": "ETH/BTC",
            "bids": [{"price": 0.0603, "qty": 12.5}, ...],
            "asks": [{"price": 0.0604, "qty": 8.0},  ...],
            "checksum": 1234567890
          }]
        }

        qty == 0 means remove that price level (same convention as Coinbase).
        """
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore heartbeats, subscription acks, etc.
            if data.get("channel") != "book":
                return None
            msg_type = data.get("type")
            if msg_type not in ("snapshot", "update"):
                return None

            data_list = data.get("data", [])
            if not data_list:
                return None

            entry  = data_list[0]
            symbol = entry.get("symbol", "")
            if not symbol:
                return None

            key = symbol.replace("/", "").lower()   # "ETH/BTC" → "ethbtc"

            if msg_type == "snapshot":
                books[key] = {"bids": {}, "asks": {}}
                for b in entry.get("bids", []):
                    books[key]["bids"][str(b["price"])] = str(b["qty"])
                for a in entry.get("asks", []):
                    books[key]["asks"][str(a["price"])] = str(a["qty"])

            elif msg_type == "update":
                if key not in books:
                    books[key] = {"bids": {}, "asks": {}}
                for b in entry.get("bids", []):
                    price = str(b["price"])
                    qty   = b["qty"]
                    if qty == 0:
                        books[key]["bids"].pop(price, None)
                    else:
                        books[key]["bids"][price] = str(qty)
                for a in entry.get("asks", []):
                    price = str(a["price"])
                    qty   = a["qty"]
                    if qty == 0:
                        books[key]["asks"].pop(price, None)
                    else:
                        books[key]["asks"][price] = str(qty)

            book = books.get(key)
            if book:
                return key, {
                    "bids": list(book["bids"].items()),
                    "asks": list(book["asks"].items()),
                }
            return None

        return extractor


if __name__ == "__main__":
    quote_priority = ["btc", "eth", "xrp", "sol"]
    assets         = ["btc", "eth", "xrp", "sol"]

    def make_pair(a: str, b: str) -> str:
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    # SOL/XRP has no native order book on any of the three brokers — drop it entirely.
    # All other cross-pairs (xrpbtc, xrpeth, soleth, ethbtc, solbtc) exist on at least
    # Binance + Kraken, so they still produce a useful BEST BID / BEST ASK.
    GLOBALLY_UNSUPPORTED = {"solxrp", "xrpsol"}

    my_pairs = [
        make_pair(a, b)
        for a, b in combinations(assets, 2)
        if make_pair(a, b) not in GLOBALLY_UNSUPPORTED
    ]

    # Coinbase Advanced Trade quotes XRP only against USD/USDC — no XRP-BTC or XRP-ETH.
    # Subscribing to them causes server-side rejections and wastes a channel slot.
    # We still keep them in my_pairs so the table rows appear (showing N/A under Coinbase,
    # which is correct), and BEST BID/ASK aggregates from Binance + Kraken for those rows.
    COINBASE_UNSUPPORTED = {"xrpbtc", "xrpeth"}
    coinbase_pairs = [p for p in my_pairs if p not in COINBASE_UNSUPPORTED]
    #when put into a graph, we doesnt need it to be a complete graph of K4 now, the edges now are enough
    

    binance = BrokerConfig(
        name="Binance",
        stream_url=URL_methods.make_binance_depth_url(my_pairs),
        pairs=my_pairs,
        payload_extractor=None,
    )

    coinbase = BrokerConfig(
        name="Coinbase Adv.",
        stream_url="wss://advanced-trade-ws.coinbase.com",
        pairs=my_pairs,                 # full list for table display (N/A where missing)
        payload_extractor=URL_methods.make_coinbase_payload_extractor(),
        initial_message=URL_methods.make_coinbase_subscription_message(coinbase_pairs, assets),
        debug=False,
    )

    kraken = BrokerConfig(
        name="Kraken",
        stream_url="wss://ws.kraken.com/v2",                # Kraken v2 WebSocket
        pairs=my_pairs,
        payload_extractor=URL_methods.make_kraken_payload_extractor(),     # stateful closure
        initial_message=URL_methods.make_kraken_subscription_message(my_pairs, assets),
        debug=False,
    )

    broker_configs = [binance, coinbase, kraken]

    coinbase_sub = URL_methods.make_coinbase_subscription_message(my_pairs, assets)
    kraken_sub   = URL_methods.make_kraken_subscription_message(my_pairs, assets)
    print(f"Coinbase subscription:\n{json.dumps(coinbase_sub, indent=2)}\n")
    print(f"Kraken subscription:\n{json.dumps(kraken_sub, indent=2)}\n")

    multi_broker = MultiBrokerOrderBook(
        broker_configs,
        refresh_interval=0.05,
        assets=assets,          # enables the live (asset x venue) graph + arbitrage view
    )

    # Live exchange-rate box on the left + arbitrage detections on the right.
    # Set show_box=False for a plain scrolling arbitrage log instead.
    multi_broker.stream_arbitrage(only_on_change=True, show_box=True)

    # Snapshot order-book table instead (clears the screen each tick):
    # multi_broker.run_live()