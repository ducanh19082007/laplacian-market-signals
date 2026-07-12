import time
import shutil
from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Any, Tuple
import json

#double check
try:
    from .IngestionPipeline import OrderBookDashboard
    from .DataProcessing import ExchangeRateGraph
except ImportError:
    from IngestionPipeline import OrderBookDashboard
    from DataProcessing import ExchangeRateGraph

try:
    from URLmethods import URL_methods
except:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    
    from URLmethods import URL_methods

# L3 (Tarjan SCC + per-SCC Bellman-Ford) is the fast path: it shrinks the graph to
# its strongly-connected components and only searches inside each, instead of the
# whole-graph O(V*E) sweep. But it needs the compiled C++ extension. If that isn't
# built, fall back to L1's whole-graph find_arbitrage, wrapped to match L3's
# [(cycle, return), ...] shape -- so MultiVenueFeed still runs, just without the
# SCC pruning win. Either way stream_arbitrage below consumes the same contract.
try:
    from L3_TarjanSCC.TarjanSCC import find_all_arbitrage
except ImportError:
    def find_all_arbitrage(graph: "ExchangeRateGraph"):
        cycle = graph.find_arbitrage()
        return [(cycle, graph.cycle_return(cycle))] if cycle else []


# NOTE: Binance and Coinbase Advanced Trade use real WebSocket depth/book streams.
# Kraken uses the v2 WebSocket API (wss://ws.kraken.com/v2).
# OKX uses the v5 public WebSocket (wss://ws.okx.com:8443/ws/v5/public, books5 channel).
# Gemini uses the v2 marketdata WebSocket (wss://api.gemini.com/v2/marketdata, l2 channel).
# Bitstamp uses the v2 WebSocket (wss://ws.bitstamp.net, full order_book channel).
#
#
# Author: Anh Duc Le


#uses this to initializes each Brokers.
Node = Tuple[str, str]
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
            #check back at IngestionPipeline.py, that's why i run it through Threading
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
        #snapshot of the current compilation of data to sent to DataProcessing
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
        except KeyboardInterrupt: #ctrl + C in Terminal
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
        Continuous arbitrage feed. Aggregator of everything of Layer 1

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
        
        Further implementation of other Layers will be implemented
        """
        if not self.assets:
            print("stream_arbitrage needs `assets` set on the order book.")
            return

        log: deque = deque(maxlen=history)
        # L3 returns ONE cycle per strongly-connected component, so a tick can yield
        # several distinct opportunities at once -- we track the set of signatures we
        # logged last tick (not a single one) so only_on_change can suppress the loops
        # that are still standing while still surfacing newly-appeared ones.
        last_signatures: set = set()
        try:
            while True:
                graph = self.build_graph()
                new_lines: List[str] = []
                current_signatures: set = set()
                if graph is not None:
                    # find_all_arbitrage -> [(cycle, return_multiple), ...], one entry
                    # per SCC that holds a profitable loop (return already computed off
                    # the full graph, so no recompute here).
                    for cycle, ret in find_all_arbitrage(graph): #
                        # Drop empty/sub-threshold loops: after fees, anything under
                        # min_profit is float noise, not a tradeable edge.
                        if not cycle or (ret - 1.0) <= min_profit:
                            continue
                        signature = self._cycle_signature(cycle)
                        current_signatures.add(signature)
                        # only_on_change: log a cycle only the first tick its signature
                        # shows up; a loop that persists across ticks isn't re-logged.
                        if only_on_change and signature in last_signatures:
                            continue
                        path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
                        new_lines.append(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{(ret - 1) * 100:+.4f}%  ({ret:.8f})  {path}"
                        )
                last_signatures = current_signatures

                for line in new_lines:
                    log.append(line)

                # Clear the screen every tick so previous frames never pile up.
                print("\033[H\033[J", end="")
                if show_box:
                    right = [f"ARBITRAGE DETECTIONS:" + 
                             f" Refresh Interval: {self.refresh_interval}," +
                             f" Fee: {self.fee}," +
                             f" Transfer cost: {self.transfer_cost}", "-" * 20]
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



_KEEP_DEFAULT_NOTIONAL = object()   # sentinel: "use the built-in MIN_NOTIONAL"


def build_default_feed(
    refresh_interval: float = 0.01,
    fee: float = 0.00015,
    max_quote_age: float = 1.0,
    quote_window: float = 0.2,
    min_notional=_KEEP_DEFAULT_NOTIONAL,
    verbose: bool = False,
) -> "MultiBrokerOrderBook":
    """
    The rich 6-venue example universe (Binance, Coinbase, Kraken, OKX, Gemini,
    Bitstamp) over ~45 cross-listed assets, returned as an already-streaming feed.

    This is exactly the wiring the __main__ demo below uses, factored out so the
    upper layers (L2 TropicalEigenvalue, L4 OUArbitrage / live_ou / live_dashboard)
    can drive the SAME live graph for their examples and backtests instead of a
    smaller hand-rolled config. The websocket threads start on construction, so the
    returned feed is live the moment you get it.
    """
    # Larger universe of liquid, cross-listed assets so the L1->L4 graph has enough
    # nodes/edges to be worth running -- but still only names with deep books on all
    # three venues, so quotes stay fresh inside max_quote_age instead of going N/A.
    # Add/remove here to scale the graph; thin alts (shib/near/apt/fil) are left out
    # because their cross-pairs are sparse and rarely tick in time.
    #
    # QUOTE_ASSETS are the currencies everything else is priced against. quote_priority
    # lists them first so make_pair puts them on the QUOTE side, yielding real market
    # tickers (btcusdt, ethbtc, solusdc, adabtc, ...) instead of inverted strings.
    QUOTE_ASSETS = ["usdt", "usdc", "btc", "eth", "usd", "eur", "gbp"]
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

    if verbose:
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
    # Callers can override the depth filter: pass None to DISABLE it entirely (lets
    # thin/mispriced top-of-book through -> far more phantom/"theoretical" arb shows
    # up), or pass a dict to tighten/loosen it. Default keeps the built-in thresholds.
    if min_notional is not _KEEP_DEFAULT_NOTIONAL:
        MIN_NOTIONAL = min_notional

    #for this test run, i will deliberately picked the fee, transaction fee, and nominal min to
    #be a bit naive to see how it run. those variables also is not detailed enough
    #given they are being generalized anyway.
    multi_broker = MultiBrokerOrderBook(
        broker_configs,
        refresh_interval=refresh_interval,
        assets=assets,                # enables the live (asset x venue) graph + arbitrage view
        fee=fee,                      # taker fee per convert leg -- kills sub-fee phantom arb
        max_quote_age=max_quote_age,  # absolute backstop: ignore quotes not refreshed within this
        quote_window=quote_window,    # legs of a cycle must be within this of each other
        min_notional=MIN_NOTIONAL,    # drop edges whose top-of-book is too thin to trade
    )
    return multi_broker


if __name__ == "__main__":
    # Live exchange-rate box on the left + arbitrage detections on the right.
    # Set show_box=False for a plain scrolling arbitrage log instead.
    # min_profit=0.0 shows every net-positive cycle after fees; raise it (e.g.
    # 0.0005) to only surface opportunities clearing an extra 0.05%.
    feed = build_default_feed(verbose=True)
    feed.stream_arbitrage(only_on_change=True, show_box=True, min_profit=0.0)

    # Snapshot order-book table instead (clears the screen each tick):
    # feed.run_live()