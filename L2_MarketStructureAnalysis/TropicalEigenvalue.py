#karp algo simplification/ :) https://commons.lib.jmu.edu/cgi/viewcontent.cgi?article=1303&context=honors201019
#also tropical eigenvalue = max (max SCC_i) the graph will take from the Tarjan SCC
#interpretation:
#   In the MAX-PLUS (tropical) semiring, the eigenvalue of a weighted digraph's
#   matrix is exactly the MAXIMUM CYCLE MEAN of the graph. Put the edge weights in
#   PROFIT space, w_ij = ln(rate_ij), and a cycle's total weight is
#       sum(ln rate) = ln(product of rates) = ln(return_multiple),
#   so the cycle MEAN (total / length) is the average per-hop log-return of that
#   loop. The maximum cycle mean is therefore the single best per-hop return any
#   loop in the market offers -- the "top loop return rate" -- and its exp() is the
#   geometric-mean rate per hop.
#
#   Relationship to L2's OTHER tropical number (the README / Engine gate):
#       the engine's Structure.min_cycle_mean is the MIN-PLUS eigenvalue of the
#       -ln(rate) weights, and
#           min_cyclemean(-ln rate) = -( max_cyclemean(+ln rate) ) = -eigenvalue.
#       So this file's `eigenvalue` and the engine gate are the same object read
#       with opposite sign; .min_cycle_mean() below hands the engine its version.
#
#   Why per-SCC: a cycle lives ENTIRELY inside one strongly-connected component
#   (L3's whole premise), so the max cycle mean over the graph is the max over the
#   per-SCC max cycle means. We reuse that decomposition here -- Tarjan first, then
#   Karp's maximum-cycle-mean DP on each non-trivial SCC -- and take the max.
#
#   This module is the profit-space, max-cycle-mean sibling of GraphLaplacian.py; it
#   feeds L4's OU arbitrage predictor (L4_Regime&RiskEngine/OUArbitrage.py), which
#   models the eigenvalue time-series as a mean-reverting process.
#
# Author: Anh Duc Le

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Same import dance the other layers use, so this runs as a script, as -m, or as a
# package import.
if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ..L1_DataProcessing.DataProcessing import ExchangeRateGraph, Node
except ImportError:
    from L1_DataProcessing.DataProcessing import ExchangeRateGraph, Node

NEG_INF = float("-inf")


@dataclass
class SCCEigen:
    """The max-plus eigenvalue of ONE strongly-connected component."""
    nodes: List[Node]                      # the SCC's node set (frozen order)
    eigenvalue: float                      # max cycle mean of ln(rate) inside it (-inf if no loop)
    cycle: List[Node] = field(default_factory=list)   # the closed argmax loop
    return_multiple: float = 0.0           # product of rates around `cycle` (>1 => profit)

    @property
    def per_hop_rate(self) -> float:
        """exp(eigenvalue): geometric-mean rate per hop of the best loop here."""
        return math.exp(self.eigenvalue) if self.eigenvalue > NEG_INF else 0.0


@dataclass
class TropicalResult:
    """Whole-graph result: the tropical eigenvalue is the max over all SCCs."""
    eigenvalue: float                      # tropical (max-plus) eigenvalue = max SCC max-cycle-mean
    cycle: List[Node] = field(default_factory=list)   # the graph's top loop
    return_multiple: float = 0.0           # product of rates around the top loop
    per_scc: List[SCCEigen] = field(default_factory=list)

    @property
    def has_cycle(self) -> bool:
        return self.eigenvalue > NEG_INF and len(self.cycle) > 1

    @property
    def per_hop_rate(self) -> float:
        return math.exp(self.eigenvalue) if self.eigenvalue > NEG_INF else 0.0


class TropicalEigenvalue:
    """
    Max-plus eigenvalue (maximum cycle mean) of an ExchangeRateGraph.

    Pipeline mirrors L3's split but stays in pure Python so L2 has no dependency on
    the compiled .so:
        1. freeze a node ordering (dict -> integer ids), like GraphLaplacian does,
        2. iterative Tarjan SCC (same logic as TarjanSCC.cpp, transcribed to Python),
        3. Karp's maximum-cycle-mean DP on each non-trivial SCC,
        4. eigenvalue = max over SCCs; keep the winning loop.

    Edge weights are w_ij = ln(rate_ij) -- PROFIT space, so a bigger cycle mean is a
    better loop. (L1 stores -ln(rate) in edge["weight"]; we read edge["rate"]
    directly and log it ourselves, so log_transform() need NOT have run.)
    """

    def __init__(self, graph: "ExchangeRateGraph") -> None:
        self.graph = graph

        # STEP 1: freeze node ordering (dict keys -> integer ids), same as Laplacian.
        self.nodes: List[Node] = graph.nodes()
        self.index: Dict[Node, int] = {node: i for i, node in enumerate(self.nodes)}
        self.n = len(self.nodes)

        # Weighted adjacency in profit space: adj[u] = [(v, ln(rate_uv)), ...].
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(self.n)]
        for u, v, a in graph.edges():
            rate = a.get("rate")
            if rate is None or rate <= 0:
                continue
            self.adj[self.index[u]].append((self.index[v], math.log(rate)))

        # Filled by compute().
        self.result: Optional[TropicalResult] = None

    # ------------------------------------------------------------------ Tarjan
    def sccs(self) -> Tuple[List[int], int]:
        """
        Iterative Tarjan SCC. Direct Python transcription of TarjanSCC.cpp: an
        explicit work stack replaces recursion so a deep graph can't blow the stack.
        Returns (comp, num_sccs) where comp[node_id] is that node's SCC id.
        """
        UNVISITED = -1
        n, adj = self.n, self.adj
        idx = [UNVISITED] * n         # DFS discovery order
        low = [0] * n                 # lowlink
        on_stack = [False] * n
        comp = [UNVISITED] * n        # result: SCC id per node
        scc_stack: List[int] = []     # Tarjan's component stack
        next_index = 0
        next_scc = 0

        for root in range(n):
            if idx[root] != UNVISITED:
                continue
            # frame = [node, edge_i]  -- how far through this node's adjacency we are.
            call: List[List[int]] = [[root, 0]]

            while call:
                f = call[-1]
                v = f[0]

                if f[1] == 0:                     # first visit to v
                    idx[v] = low[v] = next_index
                    next_index += 1
                    scc_stack.append(v)
                    on_stack[v] = True

                recursed = False
                while f[1] < len(adj[v]):
                    w = adj[v][f[1]][0]
                    f[1] += 1
                    if idx[w] == UNVISITED:
                        call.append([w, 0])       # "recurse" into w
                        recursed = True
                        break
                    elif on_stack[w]:             # back edge to a node still on stack
                        low[v] = min(low[v], idx[w])
                if recursed:
                    continue

                # Done with v's edges: SCC root iff lowlink == index. Pop the component.
                if low[v] == idx[v]:
                    while True:
                        u = scc_stack.pop()
                        on_stack[u] = False
                        comp[u] = next_scc
                        if u == v:
                            break
                    next_scc += 1

                call.pop()                        # "return" from dfs(v)
                if call:                          # propagate lowlink to parent
                    parent = call[-1][0]
                    low[parent] = min(low[parent], low[v])

        return comp, next_scc

    # -------------------------------------------------------------------- Karp
    def _karp_max_cycle_mean(
        self, members: List[int]
    ) -> Tuple[float, List[int]]:
        """
        Karp's maximum-cycle-mean on ONE strongly-connected component.

        Karp's theorem (max form): with D_k(v) = max weight of a walk of EXACTLY k
        edges from a fixed source s to v,
            lambda* = max_v  min_{0<=k<=m-1}  ( D_m(v) - D_k(v) ) / (m - k),
        where m = |members|. Because the SCC is strongly connected, any s reaches
        every node, so the min/max are well defined. Returns (lambda*, cycle_ids)
        with cycle_ids a CLOSED loop of GLOBAL node ids ([start, ..., start]).

        (The classic statement is for the MINIMUM cycle mean; flipping every
        min<->max gives the maximum, which is the max-plus eigenvalue we want.)
        """
        m = len(members)
        if m < 2:
            return NEG_INF, []                     # singleton: no cycle possible

        # Relabel the SCC's nodes to local ids 0..m-1 and keep only internal edges.
        local = {g: i for i, g in enumerate(members)}
        in_scc = set(members)
        ladj: List[List[Tuple[int, float]]] = [[] for _ in range(m)]
        for g in members:
            for v, w in self.adj[g]:
                if v in in_scc:
                    ladj[local[g]].append((local[v], w))

        s = 0
        # D[k][v], parent P[k][v] for reconstruction.
        D = [[NEG_INF] * m for _ in range(m + 1)]
        P = [[-1] * m for _ in range(m + 1)]
        D[0][s] = 0.0
        for k in range(1, m + 1):
            Dk, Dk1, Pk = D[k], D[k - 1], P[k]
            for u in range(m):
                du = Dk1[u]
                if du == NEG_INF:
                    continue
                for v, w in ladj[u]:
                    cand = du + w
                    if cand > Dk[v]:
                        Dk[v] = cand
                        Pk[v] = u

        # lambda* = max_v min_k (D_m(v) - D_k(v)) / (m - k)
        best_lambda = NEG_INF
        best_v = -1
        Dm = D[m]
        for v in range(m):
            if Dm[v] == NEG_INF:
                continue
            worst = float("inf")                   # min over k for this v
            for k in range(m):                     # k = 0..m-1
                if D[k][v] == NEG_INF:
                    continue
                val = (Dm[v] - D[k][v]) / (m - k)
                if val < worst:
                    worst = val
            if worst > best_lambda:
                best_lambda = worst
                best_v = v

        if best_v < 0:
            return NEG_INF, []

        # Reconstruct: walk m parent-hops back from best_v. The m+1 visited nodes
        # must repeat (pigeonhole) -- the segment between the repeat is the optimal
        # cycle, guaranteed by Karp to attain lambda*.
        seq = [best_v]
        node = best_v
        for k in range(m, 0, -1):
            node = P[k][node]
            if node == -1:
                break
            seq.append(node)

        cycle_local: List[int] = []
        seen: Dict[int, int] = {}
        for i, nd in enumerate(seq):
            if nd in seen:
                cycle_local = seq[seen[nd]: i + 1]   # closed: seq[seen]==seq[i]
                break
            seen[nd] = i

        if not cycle_local:
            return best_lambda, []

        cycle_local.reverse()                      # seq is newest->oldest; make it forward
        cycle_ids = [members[i] for i in cycle_local]
        return best_lambda, cycle_ids

    # ----------------------------------------------------------------- compute
    def compute(self) -> TropicalResult:
        """Run Tarjan then Karp per SCC; the eigenvalue is the max over SCCs."""
        if self.n == 0:
            self.result = TropicalResult(NEG_INF)
            return self.result

        comp, num_sccs = self.sccs()
        buckets: List[List[int]] = [[] for _ in range(num_sccs)]
        for v in range(self.n):
            buckets[comp[v]].append(v)

        per_scc: List[SCCEigen] = []
        best = TropicalResult(NEG_INF)
        for members in buckets:
            if len(members) < 2:
                continue                           # singleton => no loop => skip
            lam, cyc_ids = self._karp_max_cycle_mean(members)
            if lam == NEG_INF:
                continue
            cycle = [self.nodes[i] for i in cyc_ids]
            ret = self.graph.cycle_return(cycle) if len(cycle) > 1 else 0.0
            scc = SCCEigen(
                nodes=[self.nodes[i] for i in members],
                eigenvalue=lam,
                cycle=cycle,
                return_multiple=ret,
            )
            per_scc.append(scc)
            if lam > best.eigenvalue:
                best = TropicalResult(
                    eigenvalue=lam, cycle=cycle, return_multiple=ret
                )

        best.per_scc = per_scc
        self.result = best
        return best

    # ------------------------------------------------------- convenience reads
    def eigenvalue(self) -> float:
        """The tropical (max-plus) eigenvalue = the top loop's per-hop log-return."""
        if self.result is None:
            self.compute()
        return self.result.eigenvalue

    def min_cycle_mean(self) -> float:
        """
        The engine/README's number: min-plus eigenvalue of the -ln(rate) weights,
        which equals -eigenvalue. < 0 => a profitable cycle exists (the L2 gate).
        Returns +inf when there is no cycle at all (nothing to gate on).
        """
        lam = self.eigenvalue()
        return float("inf") if lam == NEG_INF else -lam

    def is_arbitrage(self, fee: float = 0.0) -> bool:
        """
        True iff the best loop clears a per-hop taker fee. Net-profitable when the
        per-hop return beats the fee, i.e. eigenvalue > -ln(1 - fee). fee=0 reduces
        to "any loop with product of rates > 1".
        """
        return self.eigenvalue() > fee_threshold(fee)


def fee_threshold(fee: float) -> float:
    """
    Per-hop log-return an arbitrage loop must clear to be net profitable under a
    taker fee `fee` charged on every hop. product*(1-fee)^L > 1  <=>  cycle_mean >
    -ln(1-fee), independent of loop length L. This is the horizontal line L4 draws.
    """
    if fee <= 0.0:
        return 0.0
    if fee >= 1.0:
        return float("inf")
    return -math.log(1.0 - fee)


if __name__ == "__main__":
    # Smoke test against L1's own rigged snapshot (same one GraphLaplacian uses).
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

    trop = TropicalEigenvalue(graph)
    res = trop.compute()

    print("Node ordering:")
    for i, node in enumerate(trop.nodes):
        print(f"  {i}: {ExchangeRateGraph.fmt(node)}")

    comp, num = trop.sccs()
    print(f"\nTarjan found {num} SCC(s):")
    for scc_id in range(num):
        mem = [ExchangeRateGraph.fmt(trop.nodes[i]) for i in range(trop.n) if comp[i] == scc_id]
        print(f"  SCC {scc_id}: {mem}")

    print(f"\nTropical (max-plus) eigenvalue = {res.eigenvalue:+.8f}")
    print(f"  per-hop rate exp(lambda)    = {res.per_hop_rate:.8f}")
    print(f"  min_cycle_mean (engine gate)= {trop.min_cycle_mean():+.8f}")
    if res.has_cycle:
        path = " -> ".join(ExchangeRateGraph.fmt(n) for n in res.cycle)
        print(f"  top loop  : {path}")
        print(f"  return    : {res.return_multiple:.8f}  ({(res.return_multiple - 1) * 100:+.4f}%)")
    else:
        print("  no profitable loop (no non-trivial cycle).")

    for fee in (0.0, 0.001, 0.005, 0.02):
        print(f"  arbitrage @ fee {fee*100:>4.1f}%? {trop.is_arbitrage(fee)}  "
              f"(threshold lambda > {fee_threshold(fee):+.6f})")
