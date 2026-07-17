"""
Turn the live multi-venue exchange-rate comparison into a graph for graph-theory work.

Nodes  : (asset, broker)            e.g. ("eth", "Binance")
Edges  : directed, one per achievable conversion, carrying the exchange rate.

The whole point is so triangular / cross-venue arbitrage becomes a graph problem:
once every edge weight is  -log(rate),  a profitable cycle (product of rates > 1)
turns into a NEGATIVE-sum cycle, which Bellman-Ford can detect.

No third-party deps -- this is just a reshaping of the snapshot dict produced by
MultiBrokerOrderBook.snapshot() plus a math.log over the edges.

Furthermore, further implementation will be made here as L2 to L4 will be implemented
since the goal is not only to analyse the market but uses the analysis to reduce the computation
of graph theory, which notoriously known with L4.

Author: Anh Duc Le
"""

import math
from typing import Dict, List, Optional, Tuple

# A node is an (asset, broker) pair; an edge key is (from_node, to_node).
Node = Tuple[str, str]


def split_pair(pair: str, assets: List[str]) -> Optional[Tuple[str, str]]:
    """
    Split a concatenated pair like 'ethbtc' into (base, quote) = ('eth', 'btc').
    for a edge of (base, quote) in E as G = (V, E) is accounted

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
    
    Idea: created a adjacent matrix which each element of the matrix represents the weights of
    that relationship between each node's weights with -log(rate ( 1 - fee)) *actually

    adjacency[u][v] = {
        "rate":   float,   # multiply your holding of u by this to get v
        "weight": float,   # -log(rate) or -log(rate(1 - fee(%))), filled in by log_transform()
        "kind":   "convert" | "transfer",
        "pair":   str,     # source order-book pair (convert edges only)
        "broker": str,     # venue the conversion happens on (convert edges only)
    }
    """

    def __init__(
        self,
        assets: List[str],
        transfer_cost: float = 0.0,
        fee: float|dict=0.0,
        quote_window: Optional[float] = None,
        min_notional: Optional[Dict[str, float]] = None,
    ):
        self.assets = [a.lower() for a in assets] 
        
        # transfer_cost is the fractional cost of moving one asset between venues;
        # 0.0 => a perfect 1:1 transfer. e.g. 0.001 == 10 bps., like BTC from Binance
        # to BTC from Kraken need a cost to transfer.
        self.transfer_cost = transfer_cost
        
        # fee is the fractional taker fee charged on each convert leg (e.g. 0.001
        # == 0.1%). Without it the graph is frictionless and reports sub-fee
        # "arbitrage" that no real trade could ever capture.
        #technically, the weight would be Ln(Rn(1 - fee(%) ))
        #
        # fee may be EITHER a scalar (uniform across venues) OR a dict keyed by broker
        # name -- e.g. {"Binance": 0.001, "Coinbase Adv.": 0.006, "default": 0.001} --
        # so cross-venue cycles accrue each venue's REAL asymmetric taker fee. Fees are
        # static on the tick timescale, so this is a table set once, never fetched per
        # edge in the loop. Crucially, the fee is applied to a SEPARATE net rate while
        # each edge also keeps the fee-free `rate_raw`, so the fee stays a re-tunable
        # OFFLINE knob instead of being baked irreversibly into the gathered data.
        self.fee = fee
        
        # quote_window (seconds) bounds how far apart, in time, the quotes that
        # build a cycle may be. None disables the guard; see build_from_snapshot.
        #same with self.max_quote_age in IngestionPipeline
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
        
        #If this algorithm triggers a trade to capture a 5-cent profit, 
        # but you spend $1.50 in fixed network transaction overhead to execute it, 
        # you didn't make a profit—you just paid $1.45 for the privilege of trading.
        #remember, other than the venues-transaction-fees and the current-transaction-fee
        #there always a fixed fee, and the min_nominal is use to account on this.
        
        #to put thing simple, it is the smallest posible transaction size
        self.min_notional = {k.lower(): v for k, v in (min_notional or {}).items()}
        
        self.adjacency: Dict[Node, Dict[Node, dict]] = {} #the adjacent matrix, our output



    def _fee_for(self, broker: str) -> float:
        """
        Taker fee for a convert leg on `broker`. `self.fee` is either a scalar (uniform)
        or a dict keyed by broker name with an optional "default"; an unlisted venue
        falls back to "default" then 0.0. Kept tiny and pure so it's cheap to call per
        edge -- the table itself is loaded once, upstream.
        """
        fee = self.fee
        if isinstance(fee, dict):
            return fee.get(broker, fee.get("default", 0.0))
        return fee or 0.0

    def _add_edge(self, src: Node, dst: Node, rate: float, **attrs) -> None:
        if rate is None or rate <= 0:
            return
        self.adjacency.setdefault(src, {})
        self.adjacency.setdefault(dst, {})
        self.adjacency[src][dst] = {"rate": rate, "weight": None, **attrs}

    def build_from_snapshot(self, snapshot: Dict[str, Dict[str, Dict[str, str]]] ) -> "ExchangeRateGraph":
        """
        Build the graph from MultiBrokerOrderBook.snapshot():
            snapshot[pair][broker] = {"bid": ..., "ask": ...}

        For each broker that quotes a pair we add the two conversion directions,
        then we stitch venues together with same-asset transfer edges.
        
        Output on self.adjacent examples:
        self.adjacency = {
            ("eth", "Binance"): {
                ("btc", "Binance"): {"rate": 0.0610,        "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Binance"},
                ("xrp", "Binance"): {"rate": 5089.058...,   "weight": None, "kind": "convert", "pair": "xrpeth",  "broker": "Binance"},
                ("eth", "Kraken"):  {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
            ("btc", "Binance"): {
                ("eth", "Binance"): {"rate": 16.3666...,    "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Binance"},
                ("xrp", "Binance"): {"rate": 83263.94...,   "weight": None, "kind": "convert", "pair": "xrpbtc", "broker": "Binance"},
                ("btc", "Kraken"):  {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
            ("eth", "Kraken"): {
                ("btc", "Kraken"):  {"rate": 0.0600,        "weight": None, "kind": "convert", "pair": "ethbtc", "broker": "Kraken"},
                ("eth", "Binance"): {"rate": 1.0,           "weight": None, "kind": "transfer"},
            },
        }
        this is technically a 3x3 but there is no ethKraken to ethBinance so it will be None
        """
        self.adjacency = {}
        brokers_per_asset: Dict[str, set] = {}

        # Contemporaneity guard. A triangular cycle assembled from quotes taken
        # seconds apart is the classic phantom: one leg drifts while the others
        # are stale, so the loop "profits" by exactly that drift (the tell is a
        # single-venue cycle whose % wanders tick to tick). We find the freshest
        # quote in the snapshot and drop any quote lagging it by more than
        # quote_window seconds, so every surviving edge is near-contemporaneous. 
        # None disables the guard.
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
                    if ts is None or (newest_ts - ts) > self.quote_window: #this is timesheet so it is suppose to be in time
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

                # Each convert leg loses THIS venue's taker fee, so you keep (1 - fee_ij).
                # `rate` is net-of-fee (what feeds Bellman-Ford / cycle_return, i.e. the
                # EXECUTABLE view); `rate_raw` is the fee-free market rate and `fee` the
                # rate actually applied -- stored so net λ / STRESSED can be re-derived
                # offline under a different fee assumption without re-gathering.
                fee_ij = self._fee_for(broker)
                keep = 1.0 - fee_ij
                # Sell BASE -> receive QUOTE at the bid (quote per base).
                if bid is not None:
                    self._add_edge(base_node, quote_node, bid * keep,
                                   kind="convert", pair=pair, broker=broker,
                                   rate_raw=bid, fee=fee_ij)
                # Buy BASE with QUOTE -> pay the ask, so 1/ask base per quote.
                if ask is not None:
                    self._add_edge(quote_node, base_node, (1.0 / ask) * keep,
                                   kind="convert", pair=pair, broker=broker,
                                   rate_raw=1.0 / ask, fee=fee_ij)

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
                    # rate_raw=1.0: a transfer is fee-FREE structurally (same asset), so
                    # it must not distort raw λ; the transfer_cost lives only in the net
                    # `rate`, mirroring how convert edges split raw vs net.
                    self._add_edge((asset, a), (asset, b), rate, kind="transfer",
                                   rate_raw=1.0, fee=self.transfer_cost)

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

    # --------------------------------------------------- graph theory
    def nodes(self) -> List[Node]: #compilation of all the nodes
        return sorted(self.adjacency.keys())

    def edges(self) -> List[Tuple[Node, Node, dict]]: #compilation of all the edges
        """
        examples of the output:
        [
        # Edge 1: From ETH to BTC
        (
            ("eth", "Binance"),        # u (source node)
            ("btc", "Binance"),        # v (destination node)
            {                          # a (attributes dictionary)
                "rate": 0.060939, 
                "weight": 2.7978505, 
                "kind": "convert", 
                "pair": "ethbtc", 
                "broker": "Binance"
            }
        ),
        
        # Edge 2: From BTC to ETH
        (
            ("btc", "Binance"),        # u (source node)
            ("eth", "Binance"),        # v (destination node)
            {                          # a (attributes dictionary)
                "rate": 16.3502455, 
                "weight": -2.7942426, 
                "kind": "convert", 
                "pair": "ethbtc", 
                "broker": "Binance"
            }
        )
    ]
        """
        return [(u, v, a) for u, nbrs in self.adjacency.items() for v, a in nbrs.items()]

    def subgraph(self, nodes) -> "ExchangeRateGraph":
        """
        Return a NEW graph holding only `nodes` and the edges whose BOTH endpoints
        are in `nodes`. Non-destructive: self (the full graph) is left untouched.

        This is the L3 -> L1 reduction. L3 (Tarjan SCC) hands us the node set of one
        strongly-connected component; a profitable cycle must live ENTIRELY inside a
        single SCC, so running find_arbitrage() on this smaller adjacency finds the
        same cycle while relaxing far fewer edges -- turning the O(|V|*|E|) sweep over
        the whole graph into a sum over tiny components. Edge attr dicts (incl. the
        already-computed -ln(rate) weight) are shared by reference, so there is no
        rebuild and no log_transform() recompute. *hasnt implemented L3 yet but this is the subgraph.
        """
        keep = set(nodes)
        sub = ExchangeRateGraph(self.assets, transfer_cost=self.transfer_cost, fee=self.fee)
        sub.adjacency = {
            u: {v: a for v, a in nbrs.items() if v in keep}
            for u, nbrs in self.adjacency.items()
            if u in keep
        }
        return sub

    def cycle_return(self, cycle: List[Node]) -> float:
        """Product of rates around a node cycle (>1 means profit). 0 if broken."""
        product = 1.0
        for u, v in zip(cycle, cycle[1:]):
            edge = self.adjacency.get(u, {}).get(v) #in case the loop itself have 
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
    #this example is without ts and hence without quote_window
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
    
    snapshot1 = {
    "ethbtc": {
        "Binance": {"bid": "0.0610", "ask": "0.0611","ts": 1000.0,},
        "Kraken": {"bid": "0.0600", "ask": "0.0601","ts": 1000.676767,},
    },
    "xrpbtc": {
        "Binance": {"bid": "0.00001200", "ask": "0.00001201", "ts": 1000.0,},
    },
    "xrpeth": {
        "Binance": {"bid": "0.00019600", "ask": "0.00019650", "ts": 1000.0,},
    },
}

    # Detection lives in L3 (Tarjan SCC + Bellman-Ford, C++). Imported here inside
    # __main__, not at module top: a top-level import would be circular, since L3
    # imports ExchangeRateGraph from this module. Put the repo root on sys.path so
    # this resolves whether run as `python -m L1_DataProcessing.DataProcessing`
    # (root already on path) or by file path `python .../DataProcessing.py`
    # (only this dir is on path) -- the same dance the other layers use.
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from L3_TarjanSCC.TarjanSCC import find_all_arbitrage

    graph = ExchangeRateGraph(assets, transfer_cost=0.0, quote_window=0.1).build_from_snapshot(snapshot)
    graph.log_transform()
    print(graph.summary())

    found = find_all_arbitrage(graph)
    if found:
        for cycle, ret in found:
            path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
            print(f"\nArbitrage cycle: {path}")
            print(f"Return multiple : {ret:.8f}  ({(ret - 1) * 100:+.4f}%)")
    else:
        print("\nNo arbitrage cycle found.")
        
        
    graph1 = ExchangeRateGraph(assets, transfer_cost=0.0, quote_window=0.1).build_from_snapshot(snapshot1)
    graph1.log_transform()
    print(graph1.summary())

    found1 = find_all_arbitrage(graph1)
    if found1:
        for cycle1, ret1 in found1:
            path1 = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle1)
            print(f"\nArbitrage cycle: {path1}")
            print(f"Return multiple : {ret1:.8f}  ({(ret1 - 1) * 100:+.4f}%)")
    else:
        print("\nNo arbitrage cycle found.")
