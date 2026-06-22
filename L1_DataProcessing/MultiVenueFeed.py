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
# OKX uses the v5 public WebSocket (wss://ws.okx.com:8443/ws/v5/public, books5 channel).
# Gemini uses the v2 marketdata WebSocket (wss://api.gemini.com/v2/marketdata, l2 channel).
# Bitstamp uses the v2 WebSocket (wss://ws.bitstamp.net, full order_book channel).
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
        fee: float = 0.0,
        max_quote_age: Optional[float] = None,
        quote_window: Optional[float] = None,
        min_notional: Optional[Dict[str, float]] = None,
    ):
        self.dashboards = []
        self.refresh_interval = refresh_interval
        # assets enables the live graph view; without it run_live() just shows the table.
        self.assets = [a.lower() for a in assets] if assets else None
        self.transfer_cost = transfer_cost
        # Per-convert taker fee fed into the arbitrage graph (e.g. 0.001 == 0.1%).
        self.fee = fee
        # Max spread (seconds) between the quotes that form a cycle; passed to the
        # graph's contemporaneity guard. None disables it. See ExchangeRateGraph.
        self.quote_window = quote_window
        # Minimum tradeable top-of-book notional, keyed by quote currency (e.g.
        # {"usd": 50, "btc": 0.0005}). Edges whose best quote is good for less than
        # this are dropped, which kills phantom arbitrage off thin/mispriced books.
        self.min_notional = min_notional

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
                max_quote_age=max_quote_age,
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
                bid, bid_size, ask, ask_size = dashboard.get_top_of_book(pair)
                # ts lets the graph reject cycles built from non-contemporaneous
                # quotes; None when the quote is missing or stale. bid_size/ask_size
                # let the graph drop edges whose top-of-book is too thin to trade.
                ts = dashboard.get_quote_ts(pair)
                snapshot[pair][broker.name] = {
                    "bid": bid,
                    "ask": ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "ts": ts,
                }
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
        graph = ExchangeRateGraph(
            self.assets,
            transfer_cost=self.transfer_cost,
            fee=self.fee,
            quote_window=self.quote_window,
            min_notional=self.min_notional,
        )
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

    def _exchange_rate_box(self, max_rows: int = 20) -> List[str]:
        """
        Render the current best exchange rate per pair as a bordered box
        (list of equal-width lines), refreshed from the live snapshot.

        max_rows caps how many live pairs are shown so the box never outgrows the
        terminal; any overflow is summarized as a "(+N more)" footer line.
        """
        rows = []
        for pair in self.get_all_pairs():
            agg = self.aggregate_best_prices(pair)
            best_bid, best_ask = str(agg["best_bid"]), str(agg["best_ask"])
            # Drop pairs with no live quote on either side -- they only waste rows.
            if best_bid == "N/A" and best_ask == "N/A":
                continue
            rows.append((pair.upper(), best_bid, best_ask))

        hidden = max(0, len(rows) - max_rows)
        rows = rows[:max_rows]
        more = f"... (+{hidden} more)" if hidden else ""

        pair_w = max([len(r[0]) for r in rows] + [len("PAIR")])
        bid_w = max([len(r[1]) for r in rows] + [len("BEST BID")])
        ask_w = max([len(r[2]) for r in rows] + [len("BEST ASK")])

        title = f"LIVE EXCHANGE RATES  {time.strftime('%H:%M:%S')}"
        header = f"{'PAIR'.ljust(pair_w)}  {'BEST BID'.ljust(bid_w)}  {'BEST ASK'.ljust(ask_w)}"
        inner_w = max(len(title), len(header), len(more))

        lines = ["┌─" + "─" * inner_w + "─┐",
                 "│ " + title.ljust(inner_w) + " │",
                 "├─" + "─" * inner_w + "─┤",
                 "│ " + header.ljust(inner_w) + " │"]
        for pair, bid, ask in rows:
            body = f"{pair.ljust(pair_w)}  {bid.ljust(bid_w)}  {ask.ljust(ask_w)}"
            lines.append("│ " + body.ljust(inner_w) + " │")
        if more:
            lines.append("│ " + more.ljust(inner_w) + " │")
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

    @staticmethod
    def _cycle_signature(cycle: list) -> tuple:
        """
        Rotation-invariant key for a cycle so the same loop isn't logged twice.

        Bellman-Ford may return the same arbitrage loop starting from a different
        node on different ticks (e.g. SOL@Kraken->... vs BTC@Binance->...). Those
        are the SAME opportunity, so we drop the duplicated closing node and rotate
        the edge sequence to start at its smallest node before comparing.
        """
        ring = cycle[:-1] if len(cycle) > 1 and cycle[0] == cycle[-1] else cycle
        if not ring:
            return tuple(cycle)
        start = ring.index(min(ring))
        return tuple(ring[start:] + ring[:start])

    def stream_arbitrage(
        self,
        only_on_change: bool = True,
        show_box: bool = True,
        history: int = 20,
        min_profit: float = 0.0,
        box_rows: int = 20,
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

        min_profit filters out cycles whose return doesn't clear this fractional
        threshold (e.g. 0.0005 == only show >0.05% net). Combined with the graph's
        fee model this suppresses sub-cost phantom arbitrage.
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
                        ret = graph.cycle_return(cycle)
                        signature = self._cycle_signature(cycle)
                        # Only a genuinely profitable, not-yet-seen cycle gets logged.
                        if (ret - 1.0) > min_profit and (
                            not only_on_change or signature != last_signature
                        ):
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
                    print(self._render_side_by_side(self._exchange_rate_box(box_rows), right))
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

            # A single l2_data message can carry events for several products.
            # Process *all* of them so no incremental diff is dropped, then return
            # the last-updated product's book (the pipeline stores one book per call).
            last_key = None
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

                if key in books:
                    last_key = key

            if last_key is not None:
                book = books.get(last_key)
                if book:
                    # Return as list-of-tuples; _normalize_quote handles (price, qty) tuples
                    return last_key, {
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

    # -------------------------------------------------------------------------
    # OKX  (v5 public WebSocket — wss://ws.okx.com:8443/ws/v5/public)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_okx_pair(pair: str, assets: List[str]) -> str:
        """Convert 'ethbtc' → 'ETH-BTC'. OKX uses dash-separated, uppercase codes."""
        pair = pair.upper()
        assets = [a.upper() for a in assets]
        for quote in assets:
            if pair.endswith(quote):
                base = pair[: -len(quote)]
                if base:
                    return f"{base}-{quote}"
        return pair

    @staticmethod
    def make_okx_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build an OKX v5 subscription for the books5 channel.

        OKX v5 shape:
        {"op": "subscribe",
         "args": [{"channel": "books5", "instId": "BTC-USDT"}, ...]}

        books5 pushes the full top-5 of the book on every update (no incremental
        diffs), so the extractor can be stateless.
        """
        formatted = [URL_methods.binance_to_okx_pair(p, assets) for p in pairs]
        return {
            "op": "subscribe",
            "args": [{"channel": "books5", "instId": p} for p in formatted if "-" in p],
        }

    @staticmethod
    def make_okx_payload_extractor():
        """
        Stateless extractor for the OKX v5 books5 channel.

        Unlike Coinbase/Kraken (snapshot + diffs), books5 sends a complete top-5
        snapshot in every message, so there is no book state to accumulate.

        Message shape:
        {
          "arg":  {"channel": "books5", "instId": "BTC-USDT"},
          "data": [{"asks": [["price","size","0","numOrders"], ...],
                    "bids": [["price","size","0","numOrders"], ...],
                    "ts": "...", "seqId": ...}]
        }

        Each level is a list whose first two entries are price and size, which
        _normalize_quote reads directly as (price, qty).
        """
        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore subscribe acks, errors, and pong frames (no "arg"/"data").
            arg = data.get("arg", {})
            if arg.get("channel") != "books5":
                return None
            inst = arg.get("instId", "")
            rows = data.get("data", [])
            if not inst or not rows:
                return None

            entry = rows[0]
            key = inst.replace("-", "").lower()   # "BTC-USDT" → "btcusdt"
            return key, {
                "bids": [(lvl[0], lvl[1]) for lvl in entry.get("bids", []) if len(lvl) >= 2],
                "asks": [(lvl[0], lvl[1]) for lvl in entry.get("asks", []) if len(lvl) >= 2],
            }

        return extractor

    # -------------------------------------------------------------------------
    # Gemini  (v2 marketdata WebSocket — wss://api.gemini.com/v2/marketdata)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_gemini_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ETHBTC'.

        Gemini uses uppercase, concatenated symbols (no separator), which is just
        the Binance lowercase symbol upper-cased. The `assets` arg is unused but
        kept for a uniform converter signature across venues.
        """
        return pair.upper()

    @staticmethod
    def make_gemini_subscription_message(pairs: List[str], assets: List[str]) -> dict:
        """
        Build a Gemini v2 marketdata subscription for the level-2 (l2) channel.

        Gemini v2 shape (one frame carries every symbol):
        {"type": "subscribe",
         "subscriptions": [{"name": "l2", "symbols": ["BTCUSD", "ETHBTC", ...]}]}

        IMPORTANT: Gemini errors out (and can drop the connection) on an unknown
        symbol, so callers must pass only symbols Gemini actually lists -- unlike
        Kraken/OKX/Bitstamp which silently ignore unknown channels. The __main__
        block filters to a curated `gemini_pairs` set for exactly this reason.
        """
        symbols = [URL_methods.binance_to_gemini_pair(p, assets) for p in pairs]
        return {
            "type": "subscribe",
            "subscriptions": [{"name": "l2", "symbols": symbols}],
        }

    @staticmethod
    def make_gemini_payload_extractor():
        """
        Return a *stateful* closure for the Gemini v2 l2 channel.

        Gemini l2_updates message shape:
        {
          "type": "l2_updates",
          "symbol": "BTCUSD",
          "changes": [["buy"|"sell", "<price>", "<quantity>"], ...]
        }

        The first l2_updates per symbol is the full snapshot (every level in
        `changes`); subsequent ones are incremental. There is no explicit
        snapshot/update flag, so we just accumulate: a "0" quantity removes the
        level, anything else sets it. (On reconnect Gemini resends a fresh
        snapshot which merges on top of the retained book.)
        """
        books: Dict[str, Dict[str, Dict[str, str]]] = {}

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            if data.get("type") != "l2_updates":
                return None
            symbol = data.get("symbol", "")
            if not symbol:
                return None

            key = symbol.lower()                       # "BTCUSD" → "btcusd"
            book = books.setdefault(key, {"bids": {}, "asks": {}})

            for change in data.get("changes", []):
                if len(change) < 3:
                    continue
                side, price, qty = change[0], change[1], change[2]
                bucket = (
                    book["bids"] if side == "buy"
                    else book["asks"] if side == "sell"
                    else None
                )
                if bucket is None:
                    continue
                try:
                    qty_f = float(qty)
                except (ValueError, TypeError):
                    continue
                if qty_f == 0:
                    bucket.pop(price, None)
                else:
                    bucket[price] = qty

            return key, {
                "bids": list(book["bids"].items()),
                "asks": list(book["asks"].items()),
            }

        return extractor

    # -------------------------------------------------------------------------
    # Bitstamp  (WebSocket v2 — wss://ws.bitstamp.net, full order_book channel)
    # -------------------------------------------------------------------------

    @staticmethod
    def binance_to_bitstamp_pair(pair: str, assets: List[str]) -> str:
        """
        Convert 'ethbtc' → 'ethbtc'.

        Bitstamp uses lowercase, concatenated symbols, which is exactly the
        Binance symbol format -- so this is an identity map kept for signature
        uniformity with the other converters.
        """
        return pair.lower()

    @staticmethod
    def make_bitstamp_subscription_messages(pairs: List[str], assets: List[str]) -> list:
        """
        Build Bitstamp subscription frames for the full `order_book` channel.

        Bitstamp wants ONE subscribe frame per channel (it has no batch form),
        so this returns a *list* of messages -- IngestionPipeline sends each in
        turn. Subscribing to a non-existent market yields a harmless bts:error
        event; the connection stays up (unlike Gemini).

        Per-frame shape:
        {"event": "bts:subscribe", "data": {"channel": "order_book_btcusd"}}

        The `order_book` channel pushes the full top-100 book on every event, so
        the extractor can stay stateless.
        """
        return [
            {
                "event": "bts:subscribe",
                "data": {"channel": f"order_book_{URL_methods.binance_to_bitstamp_pair(p, assets)}"},
            }
            for p in pairs
        ]

    @staticmethod
    def make_bitstamp_payload_extractor():
        """
        Stateless extractor for the Bitstamp full `order_book` channel.

        Bitstamp data message shape:
        {
          "event": "data",
          "channel": "order_book_btcusd",
          "data": {"timestamp": "...", "microtimestamp": "...",
                   "bids": [["<price>", "<amount>"], ...],
                   "asks": [["<price>", "<amount>"], ...]}
        }

        Every message carries the complete top-100 book, so there is no state to
        accumulate (same idea as OKX books5).
        """
        prefix = "order_book_"

        def extractor(data: Any) -> Optional[tuple]:
            if not isinstance(data, dict):
                return None
            # Ignore subscription acks, errors, heartbeats, reconnect requests.
            if data.get("event") != "data":
                return None
            channel = data.get("channel", "")
            if not channel.startswith(prefix):
                return None

            key = channel[len(prefix):]                # "order_book_btcusd" → "btcusd"
            book = data.get("data", {})
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids and not asks:
                return None

            return key, {
                "bids": [(lvl[0], lvl[1]) for lvl in bids if len(lvl) >= 2],
                "asks": [(lvl[0], lvl[1]) for lvl in asks if len(lvl) >= 2],
            }

        return extractor


if __name__ == "__main__":
    # Larger universe of liquid, cross-listed assets so the L1->L4 graph has enough
    # nodes/edges to be worth running -- but still only names with deep books on all
    # three venues, so quotes stay fresh inside max_quote_age instead of going N/A.
    # Add/remove here to scale the graph; thin alts (shib/near/apt/fil) are left out
    # because their cross-pairs are sparse and rarely tick in time.
    #
    # QUOTE_ASSETS are the currencies everything else is priced against. quote_priority
    # lists them first so make_pair puts them on the QUOTE side, yielding real market
    # tickers (btcusdt, ethbtc, solusdc, adabtc, ...) instead of inverted strings.
    QUOTE_ASSETS = ["usdt", "usdc", "btc", "eth", "usd", "eur", "gbp", "eth"]
    ALTS = [
    "sol", "xrp", "ada", "doge", "link",
    "ltc", "dot", "avax", "bch", "atom",
    "arb", "bnb", "sui", "apt", "op",
    "pol", "near", "fet", "rndr",
    "pepe", "shib",

    # Additions
    "uni",
    "aave",
    "fil",
    "algo",
    "inj",
    "etc",
    "xlm",
    "trx",
    "hbar",
    "cro",
    "vet",
    "icp",
    "ena",
    "sei",
    "tao"
]
    quote_priority = QUOTE_ASSETS + ALTS
    assets = QUOTE_ASSETS + ALTS


    def make_pair(a: str, b: str) -> str:
        base, quote = (b, a) if quote_priority.index(a) < quote_priority.index(b) else (a, b)
        return f"{base}{quote}"

    def quote_side(pair: str) -> Optional[str]:
        """The quote currency a pair ends in, or None if it's an alt-vs-alt pair."""
        return next((q for q in QUOTE_ASSETS if pair.endswith(q)), None)

    # SOL/XRP has no native order book on any of the three brokers — drop it entirely.
    GLOBALLY_UNSUPPORTED = {"solxrp", "xrpsol"}

    # Only keep pairs that real venues actually list: every asset trades against a
    # stablecoin / BTC / ETH, but alt-vs-alt (e.g. adadoge, linkavax) mostly doesn't
    # exist anywhere -- generating it would just spam dead subscriptions and N/A rows.
    my_pairs = sorted({
        p
        for a, b in combinations(assets, 2)
        for p in [make_pair(a, b)]
        if quote_side(p) is not None and p not in GLOBALLY_UNSUPPORTED
    })

    # Coinbase Advanced Trade's depth is in its stablecoin (USD/USDC/USDT) markets;
    # its crypto-crypto coverage (alt-BTC, alt-ETH) is sparse and subscribing to a
    # product it doesn't list triggers server-side rejections. So feed Coinbase only
    # the stablecoin-quoted pairs. Binance + Kraken still carry the BTC/ETH cross-pairs
    # (where the triangular cycles live), and those rows just show N/A under Coinbase.
    coinbase_pairs = [p for p in my_pairs if quote_side(p) in ("usdt", "usdc")]

    # Gemini drops the whole connection on an unknown symbol, so we can ONLY
    # subscribe to symbols it actually lists. Restrict to a curated set of liquid
    # Gemini spot markets, intersected with whatever the universe generated.
    GEMINI_LISTED = {
    "btcusd", "ethusd", "ethbtc",
    "solusd", "ltcusd", "bchusd", "linkusd",
    "dogeusd", "xrpusd", "avaxusd", "dotusd",
    "atomusd", "btcusdt", "ethusdt",

    # Additional major assets
    "adausd",
    "maticusd",      # POL/MATIC
    "uniusd",
    "aaveusd",
    "filusd",
    "algousd",
    "compusd",
    "manausd",
    "sandusd",
    "injusd",
    "pepeusd",
    "shibusd",
    "nearusd",
    "aptusd",
    "suiusd",
}
    gemini_pairs = [p for p in my_pairs if p in GEMINI_LISTED]

    # Bitstamp ignores unknown channels (harmless bts:error), so we can be more
    # liberal -- just skip the gbp-quoted pairs it rarely lists.
    bitstamp_pairs = [p for p in my_pairs if quote_side(p) in ("usd", "usdt", "usdc", "eur", "btc", "eth")]


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

    okx = BrokerConfig(
        name="OKX",
        stream_url="wss://ws.okx.com:8443/ws/v5/public",    # OKX v5 public WebSocket
        pairs=my_pairs,
        payload_extractor=URL_methods.make_okx_payload_extractor(),        # stateless (books5)
        initial_message=URL_methods.make_okx_subscription_message(my_pairs, assets),
        debug=False,
    )

    gemini = BrokerConfig(
        name="Gemini",
        stream_url="wss://api.gemini.com/v2/marketdata",   # Gemini v2 marketdata
        pairs=my_pairs,
        payload_extractor=URL_methods.make_gemini_payload_extractor(),     # stateful (l2)
        initial_message=URL_methods.make_gemini_subscription_message(gemini_pairs, assets),
        debug=False,
    )

    bitstamp = BrokerConfig(
        name="Bitstamp",
        stream_url="wss://ws.bitstamp.net",                 # Bitstamp WebSocket v2
        pairs=my_pairs,
        payload_extractor=URL_methods.make_bitstamp_payload_extractor(),   # stateless (full book)
        initial_message=URL_methods.make_bitstamp_subscription_messages(bitstamp_pairs, assets),
        debug=False,
    )

    broker_configs = [binance, coinbase, kraken, okx, gemini, bitstamp]

    coinbase_sub = URL_methods.make_coinbase_subscription_message(my_pairs, assets)
    kraken_sub   = URL_methods.make_kraken_subscription_message(my_pairs, assets)
    print(f"Coinbase subscription:\n{json.dumps(coinbase_sub, indent=2)}\n")
    print(f"Kraken subscription:\n{json.dumps(kraken_sub, indent=2)}\n")

    # Minimum tradeable top-of-book notional per quote currency. An edge survives
    # only if its best quote is good for at least this much -- so a thin/mispriced
    # top-of-book on an illiquid alt (e.g. the persistent RNDR@Bitstamp triangle)
    # no longer counts as infinitely deep and stops fabricating ~1-2% phantom arb.
    # Stable/fiat thresholds are ~$50; BTC/ETH are the rough $50 equivalent.
    MIN_NOTIONAL = {
        "usd": 50.0, "usdt": 50.0, "usdc": 50.0, "eur": 50.0, "gbp": 50.0,
        "btc": 0.0005, "eth": 0.02,
    }

    multi_broker = MultiBrokerOrderBook(
        broker_configs,
        refresh_interval=0.01,
        assets=assets,          # enables the live (asset x venue) graph + arbitrage view
        fee=0.0001,             # 0.20% taker fee per convert leg -- kills sub-fee phantom arb from Kraken
        max_quote_age=1.0,      # absolute backstop: ignore any quote not refreshed in the last 1s
        quote_window=0.2,       # legs of a cycle must be within 0.5s of each other (kills stale-leg phantoms)
        min_notional=MIN_NOTIONAL,  # drop edges whose top-of-book is too thin to trade
    )

    # Live exchange-rate box on the left + arbitrage detections on the right.
    # Set show_box=False for a plain scrolling arbitrage log instead.
    # min_profit=0.0 shows every net-positive cycle after fees; raise it (e.g.
    # 0.0005) to only surface opportunities clearing an extra 0.05%.
    multi_broker.stream_arbitrage(only_on_change=True, show_box=True, min_profit=0.0)

    # Snapshot order-book table instead (clears the screen each tick):
    # multi_broker.run_live()