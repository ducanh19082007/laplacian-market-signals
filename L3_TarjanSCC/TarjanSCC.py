# L3 — Tarjan SCC + per-component Bellman-Ford

# In L1 subgraph() vs Tarjan SCC — two halves of the same step.
# They aren't competing things; they're producer and consumer. Tarjan decides
# which nodes form a strongly-connected component (a cycle can only ever live
# inside one), and a Bellman-Ford sweep over each component finds the arbitrage
# loop. The key fact it exploits: a cycle must live entirely inside one SCC, so
# any arbitrage loop can never span two SCCs, and singletons (size-1 components
# that can't be in a cycle) are thrown away for free.
#
# Option B: BOTH Tarjan AND the per-SCC Bellman-Ford run in C++ (TarjanSCC.cpp),
# because Bellman-Ford is the O(V*E) hot path and wants to be on the same side of
# the FFI boundary as the SCC loop that drives it. Python's only jobs here are:
#   1. flatten the graph to an integer edge list (C++ wants ints, not (asset,
#      broker) tuples),
#   2. call the compiled module,
#   3. map the returned int-id cycles back to node tuples.
#
# This REPLACES the older Option-A plan (Tarjan returns node-sets -> Python
# subgraph() -> Python find_arbitrage()). subgraph()/find_arbitrage() in L1 stay
# as the pure-Python reference path for testing and for when the .so isn't built.
#
# Round trip:
#   graph.log_transform()                 # weights = -ln(rate), done in L1
#   find_all_arbitrage(graph)
#     -> C++ Tarjan + Bellman-Ford
#     -> [(cycle, return_multiple), ...]   # one entry per SCC that held a loop

import os
import sys
from typing import List, Tuple

# Make this module importable however it's invoked. The compiled extension sits
# next to this file; L1 lives at the repo root. Putting BOTH on sys.path lets
#   python L3_TarjanSCC/TarjanSCC.py     (script: only L3 dir is on path)
#   python -m L3_TarjanSCC.TarjanSCC     (module: only repo root is on path)
#   from L3_TarjanSCC.TarjanSCC import find_all_arbitrage   (package import)
# all resolve the same way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Keep the two imports separate so the error message points at the real cause:
# the "not built" hint must only fire for a genuinely missing .so, not for an
# L1 import-path problem (which the old coupled try/except wrongly blamed on C++).
try:
    import tarjan_arb  # the compiled C++ extension built next to this file, if it says error, dont worry abt it
except ImportError as exc:  # pragma: no cover - guidance, not logic
    raise ImportError(
        "L3 C++ extension 'tarjan_arb' not built. Run:\n"
        "    cd L3_TarjanSCC && python setup.py build_ext --inplace"
    ) from exc

from L1_DataProcessing.DataProcessing import ExchangeRateGraph

Node = Tuple[str, str]


def find_all_arbitrage(graph: ExchangeRateGraph) -> List[Tuple[List[Node], float]]:
    """
    Find one arbitrage cycle per strongly-connected component, via the C++ module.

    `graph` is an L1 ExchangeRateGraph. Returns a list of (cycle, return_multiple)
    pairs, where cycle is a closed node path like
        [("btc","Binance"), ("eth","Binance"), ("eth","Kraken"), ("btc","Binance")]
    and return_multiple > 1 means profit. Empty list => market is arb-free.

    log_transform() is ensured first, exactly like L1's find_arbitrage().
    """
    # Ensure -ln(rate) weights exist (mirrors find_arbitrage's lazy transform).
    if any(a["weight"] is None for _, _, a in graph.edges()):
        graph.log_transform()

    nodes = graph.nodes()                      # stable, sorted node order
    if not nodes:
        return []

    node_to_id = {node: i for i, node in enumerate(nodes)}

    # Flatten to (u_id, v_id, weight). We hand C++ only the precomputed weight;
    # all the fee/freshness/depth filtering already happened in L1's build step.
    edges = [
        (node_to_id[u], node_to_id[v], a["weight"])
        for u, v, a in graph.edges()
    ]

    raw_cycles = tarjan_arb.find_all_arbitrage(len(nodes), edges)

    results: List[Tuple[List[Node], float]] = []
    for id_path in raw_cycles:
        cycle = [nodes[i] for i in id_path]    # ints back to (asset, broker)
        results.append((cycle, graph.cycle_return(cycle)))
    return results


if __name__ == "__main__":
    # Smoke test against L1's own rigged snapshot.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from L1_DataProcessing.DataProcessing import ExchangeRateGraph

    assets = ["btc", "eth", "xrp", "sol"]
    snapshot = {
        "ethbtc": {
            "Binance": {"bid": "0.0610", "ask": "0.0611"},
            "Kraken":  {"bid": "0.0600", "ask": "0.0601"},
        },
        "xrpbtc": {"Binance": {"bid": "0.00001200", "ask": "0.00001201"}},
        "xrpeth": {"Binance": {"bid": "0.00019600", "ask": "0.00019650"}},
    }

    graph = ExchangeRateGraph(assets, transfer_cost=0.0).build_from_snapshot(snapshot)
    #graph.log_transform()

    found = find_all_arbitrage(graph)
    if not found:
        print("No arbitrage cycle found.")
    for cycle, ret in found:
        path = " -> ".join(ExchangeRateGraph.fmt(n) for n in cycle)
        print(f"Arbitrage cycle: {path}")
        print(f"Return multiple : {ret:.8f}  ({(ret - 1) * 100:+.4f}%)")
