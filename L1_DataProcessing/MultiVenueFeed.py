import time
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Any
import json


try:
    from .IngestionPipeline import OrderBookDashboard
except ImportError:
    from IngestionPipeline import OrderBookDashboard


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
    def __init__(self, brokers: List[BrokerConfig], refresh_interval: float = 0.5):
        self.dashboards = []
        self.refresh_interval = refresh_interval

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

    def run_live(self) -> None:
        try:
            while True:
                print("\033[H\033[J", end="")
                print(f"--- Multi-Broker Order Book ({time.strftime('%H:%M:%S')}) ---")
                self.print_table()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            for _, dashboard in self.dashboards:
                dashboard.is_running = False
            print("\nMulti-broker monitoring stopped.")


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

    multi_broker = MultiBrokerOrderBook(broker_configs, refresh_interval=0.5)
    multi_broker.run_live()