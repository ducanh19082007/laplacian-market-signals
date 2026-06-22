"""
Turn the live multi-venue exchange-rate comparison into a graph for graph-theory work.

Nodes  : (asset, broker)            e.g. ("eth", "Binance")
Edges  : directed, one per achievable conversion, carrying the exchange rate.

The whole point is so triangular / cross-venue arbitrage becomes a graph problem:
once every edge weight is  -log(rate),  a profitable cycle (product of rates > 1)
turns into a NEGATIVE-sum cycle, which Bellman-Ford can detect.

No third-party deps -- this is just a reshaping of the snapshot dict produced by
MultiBrokerOrderBook.snapshot() plus a math.log over the edges.

Author: Anh Duc Le
"""

import math
from typing import Dict, List, Optional, Tuple

# A node is an (asset, broker) pair; an edge key is (from_node, to_node).
Node = Tuple[str, str]


def split_pair(pair: str, assets: List[str]) -> Optional[Tuple[str, str]]:
    """
    Split a concatenated pair like 'ethbtc' into (base, quote) = ('eth', 'btc').

    A pair string is base+quote with no separator, so we need the known asset
    list to find where the quote starts (same trick used in MultiVenueFeed).
    Price for 'ethbtc' is read as QUOTE per BASE  (i.e. BTC per ETH).
    """
    pair = pair.lower()
    assets = [a.lower() for a in assets]
    for quote in assets:
        if pair.endswith(quote):
            base = pair[: -len(quote)]
            if base and base in assets:
                return base, quote
    return None


def _to_float(value) -> Optional[float]:
    try:
        f = float(value)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


class ExchangeRateGraph:
    """
    Directed multi-venue exchange-rate graph.

    adjacency[u][v] = {
        "rate":   float,   # multiply your holding of u by this to get v
        "weight": float,   # -log(rate), filled in by log_transform()
        "kind":   "convert" | "transfer",
        "pair":   str,     # source order-book pair (convert edges only)
        "broker": str,     # venue the conversion happens on (convert edges only)
    }
    """

    def __init__(
        self,
        assets: List[str],
        transfer_cost: float = 0.0,
        fee: float = 0.0,
        quote_window: Optional[float] = None,
        min_notional: Optional[Dict[str, float]] = None,
    ):
        self.assets = [a.lower() for a in assets]
        # transfer_cost is the fractional cost of moving one asset between venues;
        # 0.0 => a perfect 1:1 transfer. e.g. 0.001 == 10 bps.
        self.transfer_cost = transfer_cost
        # fee is the fractional taker fee charged on each convert leg (e.g. 0.001
        # == 0.1%). Without it the graph is frictionless and reports sub-fee
        # "arbitrage" that no real trade could ever capture.
        self.fee = fee
        # quote_window (seconds) bounds how far apart, in time, the quotes that
        # build a cycle may be. None disables the guard; see build_from_snapshot.
        self.quote_window = quote_window
        # Minimum tradeable top-of-book notional per QUOTE currency, e.g.
        # {"usd": 50, "eur": 50, "btc": 0.0005}. A convert edge is only added if the
        # best quote on that side is good for at least this much (price * size, in
        # the quote currency). This is the guard against phantom arbitrage: a thin or
        # mispriced top-of-book on an illiquid market (the classic single-venue
        # triangle that "profits" by ~1-2% every tick) is good for a trivial amount,
        # so dropping it removes the loop. Missing/zero threshold => no filter for
        # that quote currency; missing size on a quote under an active threshold is
        # treated as untradeable. None disables the filter entirely.
        self.min_notional = {k.lower(): v for k, v in (min_notional or {}).items()}
        self.adjacency: Dict[Node, Dict[Node, dict]] = {}

    # ------------------------------------------------------------------ build
    def _add_edge(self, src: Node, dst: Node, rate: float, **attrs) -> None:
        if rate is None or rate <= 0:
            return
        self.adjacency.setdefault(src, {})
        self.adjacency.setdefault(dst, {})
        self.adjacency[src][dst] = {"rate": rate, "weight": None, **attrs}

    def build_from_snapshot(
        self, snapshot: Dict[str, Dict[str, Dict[str, str]]]
    ) -> "ExchangeRateGraph":
        """
        Build the graph from MultiBrokerOrderBook.snapshot():
            snapshot[pair][broker] = {"bid": ..., "ask": ...}

        For each broker that quotes a pair we add the two conversion directions,
        then we stitch venues together with same-asset transfer edges.
        """
        self.adjacency = {}
        brokers_per_asset: Dict[str, set] = {}

        # Contemporaneity guard. A triangular cycle assembled from quotes taken
        # seconds apart is the classic phantom: one leg drifts while the others
        # are stale, so the loop "profits" by exactly that drift (the tell is a
        # single-venue cycle whose % wanders tick to tick). We find the freshest
        # quote in the snapshot and drop any quote lagging it by more than
        # quote_window seconds, so every surviving edge -- and therefore every
        # cycle -- is near-contemporaneous. None disables the guard.
        newest_ts = None
        if self.quote_window is not None:
            all_ts = [
                q["ts"]
                for by_broker in snapshot.values()
                for q in by_broker.values()
                if isinstance(q, dict) and q.get("ts") is not None
            ]
            newest_ts = max(all_ts) if all_ts else None

        for pair, by_broker in snapshot.items():
            split = split_pair(pair, self.assets)
            if split is None:
                continue
            base, quote = split

            for broker, quote_dict in by_broker.items():
                bid = _to_float(quote_dict.get("bid"))
                ask = _to_float(quote_dict.get("ask"))

                # Drop quotes lagging the freshest one by more than the window;
                # a missing timestamp under an active guard counts as stale.
                if newest_ts is not None:
                    ts = quote_dict.get("ts")
                    if ts is None or (newest_ts - ts) > self.quote_window:
                        bid = ask = None

                # Reject crossed/locked books (bid >= ask). A real book always has
                # ask > bid; bid >= ask means the two sides came from out-of-sync
                # updates or rounded display prices. Left in, this fabricates a
                # same-venue round-trip "arbitrage" (sell at bid, rebuy at ask),
                # which is the phantom the detector was reporting.
                if bid is not None and ask is not None and bid >= ask:
                    bid = ask = None

                # Depth filter: drop a side whose best quote is good for less than
                # min_notional[quote] (price * size, in the quote currency). A thin
                # or stale top-of-book priced 1-2% off -- common on illiquid alts --
                # is good for a trivial amount; treated as infinitely deep it
                # fabricates the persistent single-venue triangle. Only applies when
                # a threshold is configured for this quote currency.
                min_q = self.min_notional.get(quote)
                if min_q:
                    bid_size = _to_float(quote_dict.get("bid_size"))
                    ask_size = _to_float(quote_dict.get("ask_size"))
                    if bid is not None and (bid_size is None or bid * bid_size < min_q):
                        bid = None
                    if ask is not None and (ask_size is None or ask * ask_size < min_q):
                        ask = None

                base_node: Node = (base, broker)
                quote_node: Node = (quote, broker)

                # Each convert leg loses the taker fee, so you keep (1 - fee) of it.
                keep = 1.0 - self.fee
                # Sell BASE -> receive QUOTE at the bid (quote per base).
                if bid is not None:
                    self._add_edge(base_node, quote_node, bid * keep,
                                   kind="convert", pair=pair, broker=broker)
                # Buy BASE with QUOTE -> pay the ask, so 1/ask base per quote.
                if ask is not None:
                    self._add_edge(quote_node, base_node, (1.0 / ask) * keep,
                                   kind="convert", pair=pair, broker=broker)

                if bid is not None or ask is not None:
                    brokers_per_asset.setdefault(base, set()).add(broker)
                    brokers_per_asset.setdefault(quote, set()).add(broker)

        self._add_transfer_edges(brokers_per_asset)
        return self

    def _add_transfer_edges(self, brokers_per_asset: Dict[str, set]) -> None:
        """Same asset across venues -> 1:1 transfer (minus transfer_cost)."""
        rate = 1.0 - self.transfer_cost
        for asset, brokers in brokers_per_asset.items():
            brokers = sorted(brokers)
            for a in brokers:
                for b in brokers:
                    if a == b:
                        continue
                    self._add_edge((asset, a), (asset, b), rate, kind="transfer")

    # -------------------------------------------------------------- transform
    def log_transform(self) -> "ExchangeRateGraph":
        """
        Set weight = -ln(rate) on every edge.

        Why: along a path the rates multiply, so the log-weights ADD. A cycle is
        profitable iff product(rate) > 1  <=>  sum(-ln rate) < 0, i.e. a negative
        cycle -- exactly what Bellman-Ford detects.
        """
        for nbrs in self.adjacency.values():
            for attrs in nbrs.values():
                attrs["weight"] = -math.log(attrs["rate"])
        return self

    # ----------------------------------------------------------- graph theory
    def nodes(self) -> List[Node]:
        return sorted(self.adjacency.keys())

    def edges(self) -> List[Tuple[Node, Node, dict]]:
        return [(u, v, a) for u, nbrs in self.adjacency.items() for v, a in nbrs.items()]

    def find_arbitrage(self) -> Optional[List[Node]]:
        """
        Bellman-Ford over the log-weights. Returns one node cycle whose rates
        multiply to > 1 (an arbitrage loop), or None if the market is arb-free.

        Call log_transform() first.
        """
        if any(a["weight"] is None for _, _, a in self.edges()):
            self.log_transform()

        nodes = self.nodes()
        if not nodes:
            return None

        dist = {n: 0.0 for n in nodes}          # 0 init => detects any neg cycle
        pred: Dict[Node, Optional[Node]] = {n: None for n in nodes}
        edges = self.edges()

        updated = None
        for _ in range(len(nodes)):
            updated = None
            for u, v, a in edges:
                if dist[u] + a["weight"] < dist[v] - 1e-12:
                    dist[v] = dist[u] + a["weight"]
                    pred[v] = u
                    updated = v
            if updated is None:
                return None  # converged, no negative cycle

        # `updated` sits on or downstream of a negative cycle; walk back len(nodes)
        # steps so we're guaranteed to land ON the cycle. Every node visited here
        # has a predecessor (it was relaxed), so pred[...] is never None.
        assert updated is not None  # loop only exits here if a relaxation happened
        node: Node = updated
        for _ in range(len(nodes)):
            prev = pred[node]
            assert prev is not None
            node = prev

        start = node
        cycle: List[Node] = [start]
        cur = pred[start]
        while cur is not None and cur != start:
            cycle.append(cur)
            cur = pred[cur]
        cycle.append(start)
        cycle.reverse()
        return cycle

    def cycle_return(self, cycle: List[Node]) -> float:
        """Product of rates around a node cycle (>1 means profit). 0 if broken."""
        product = 1.0
        for u, v in zip(cycle, cycle[1:]):
            edge = self.adjacency.get(u, {}).get(v)
            if edge is None:
                return 0.0
            product *= edge["rate"]
        return product

    # ------------------------------------------------------------------ debug
    @staticmethod
    def fmt(node: Node) -> str:
        asset, broker = node
        return f"{asset.upper()}@{broker}"

    def summary(self) -> str:
        lines = [f"Graph: {len(self.adjacency)} nodes, {len(self.edges())} edges"]
        for u, v, a in self.edges():
            w = "n/a" if a["weight"] is None else f"{a['weight']:+.6f}"
            lines.append(
                f"  {self.fmt(u):>16} -> {self.fmt(v):<16} "
                f"rate={a['rate']:.8f}  w={w}  ({a['kind']})"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    # Tiny static snapshot so this runs without a live feed.
    # Rates are rigged so eth->btc->eth across venues loops to a profit.
    assets = ["btc", "eth", "xrp", "sol"]
    snapshot = {
        "ethbtc": {
            "Binance": {"bid": "0.0610", "ask": "0.0611"},
            "Kraken":  {"bid": "0.0600", "ask": "0.0601"},
        },
        "xrpbtc": {
            "Binance": {"bid": "0.00001200", "ask": "0.00001201"},
        },
        "xrpeth": {
            "Binance": {"bid": "0.00019600", "ask": "0.00019650"},
        },
    }

    graph = ExchangeRateGraph(assets, transfer_cost=0.0).build_from_snapshot(snapshot)
    graph.log_transform()
    print(graph.summary())

    cycle = graph.find_arbitrage()
    if cycle:
        path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
        ret = graph.cycle_return(cycle)
        print(f"\nArbitrage cycle: {path}")
        print(f"Return multiple : {ret:.8f}  ({(ret - 1) * 100:+.4f}%)")
    else:
        print("\nNo arbitrage cycle found.")
